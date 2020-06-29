import collections
import concurrent
import dataclasses
import gc
import math
import multiprocessing
import os
import pathlib
import threading
import queue
import time
import typing

import pandas as pd
import PIL
import torch
import torch.nn.functional as F
import torchvision
import tqdm

from . import data
from . import model
from . import training
from . import statistics
from . import utils


PredictionOutcome = collections.namedtuple('PredictionOutcome', (
    'sample',  # grouped_filenames key
    'actual',
    'predicted',

    'topk_values',
    'topk_cats',
))


def _make_tensor_for_test(img, angle):
    img = torchvision.transforms.functional.rotate(
        img, angle=angle, resample=PIL.Image.BICUBIC)

    return torchvision.transforms.functional.to_tensor(img)

def test_combined_bicubic(
    img_path, m, label_manager,
    filedir='valid', rotate_deg=5, topk=None):

  model_device = next(iter(m.parameters())).device
  if topk is None:
    topk = min(5, label_manager.num_classes-1)

  grouped_filenames = collections.defaultdict(list)
  group_to_spacegroup = {}

  for filename in img_path.glob(f'{filedir}/*'):
    sample, _, space_group, _ = filename.name.split('.')
    grouped_filenames[sample].append(filename.name)
    space_group = int(space_group)
    group_to_spacegroup[sample] = space_group

  grouped_filenames = dict(grouped_filenames)

  num_correct, num_topk_correct = 0, 0
  outcomes = []

  with concurrent.futures.ProcessPoolExecutor() as exc:
    it = tqdm.tqdm(grouped_filenames.items())
    for group, filenames in it:
      space_group = group_to_spacegroup[group]

      images = []
      for filename in filenames:
        fullpath = str(img_path/filedir/filename)
        images.append(PIL.Image.open(fullpath).convert('L'))

      futures = []
      for img in images:
        for angle in range(0, 360, rotate_deg):
          f = exc.submit(_make_tensor_for_test, img, angle)
          futures.append(f)
      tensors = []
      for f in concurrent.futures.as_completed(futures):
        tensor = f.result()
        tensors.append(tensor)

      m.eval()
      with torch.no_grad():
        out = m(torch.stack(tensors).to(model_device))
        argmax_topk = F.softmax(out, dim=1).mean(dim=0).topk(k=topk)
        # If we take the log before mean, then we effectively multiply
        # the softmax probabilities together.
        # Empirically, this does not work as well as the above.
        # argmax_topk = F.softmax(out, dim=1).log().mean(dim=0).topk(k=topk)

      topk_cats = []
      for idx in argmax_topk.indices:
        topk_cats.append(label_manager.get_space_group(idx))

      if topk_cats[0] == space_group:
        num_correct += 1
      if space_group in topk_cats:
        num_topk_correct += 1

      outcomes.append(PredictionOutcome(
          sample=group,
          actual=space_group,
          predicted=topk_cats[0],
          topk_values=argmax_topk.values.tolist(),
          topk_cats=topk_cats
      ))

      accuracy = 100 * (num_correct / len(outcomes))
      topk_accuracy = 100 * (num_topk_correct / len(outcomes))
      it.set_description(f"Acc: {accuracy:.02f}% | {topk_accuracy:.02f}%")

  return outcomes


class CombinedBicubicTester:
  """Concurrent tester
  There seems to be a memory leak (maybe fragmentation?) when returning large
  tensors from a ProcessPoolExecutor. To mitigate this issue, spawn processes
  with limited lifetime and let the OS reclaim memory.
  """
  _num_tasks_per_process = 50

  def __init__(self,
      img_path, label_manager,
      filedir='valid',
      rotate_deg=5, rotation_interpolation=utils.RotationInterpolation.TORCH_BICUBIC,
      topk=None):

    grouped_filenames = collections.defaultdict(list)
    self._group_to_spacegroup = {}
    for filename in img_path.glob(f'{filedir}/*'):
      sample, _, space_group, _ = filename.name.split('.')
      grouped_filenames[sample].append(str(filename))
      space_group = int(space_group)
      self._group_to_spacegroup[sample] = space_group
    self._grouped_filenames = dict(grouped_filenames)

    self._label_manager = label_manager
    self._rotate_deg = rotate_deg
    self._rotation_interpolation = rotation_interpolation

    if topk is None:
      topk = min(5, label_manager.num_classes-1)
    self._topk = topk

  def _make_rotated_image_tensors(self, inp_q, out_q):
    for _ in range(self._num_tasks_per_process):
      try:
        group, filenames = inp_q.get_nowait()
      except queue.Empty:
        return

      images = []
      for filename in filenames:
        with PIL.Image.open(filename) as img:
          images.append(img.convert('L'))

      tensors = []
      for img in images:
        for angle in range(0, 360, self._rotate_deg):
          img_r = utils.rotate(img, angle, self._rotation_interpolation)
          tensors.append(torchvision.transforms.functional.to_tensor(img_r))

      out_q.put((group, torch.stack(tensors)))

  def _process_spawner(self, inp_q, out_q, num_procs):
    processes = []
    for _ in range(num_procs):
      p = multiprocessing.Process(
          target=self._make_rotated_image_tensors,
          args=(inp_q, out_q))
      p.start()
      processes.append(p)

    while not inp_q.empty():
      time.sleep(1)  # TODO: block on process signal instead of polling

      for idx, p in enumerate(processes):
        if p.is_alive(): continue

        p = multiprocessing.Process(
            target=self._make_rotated_image_tensors,
            args=(inp_q, out_q))
        p.start()
        processes[idx] = p

    for p in processes: p.join()

  @utils.defer
  def test(self, m, _defer=None, num_procs=None):
    arg_q = multiprocessing.Queue()
    tsr_q = multiprocessing.Queue(1)

    for group, filenames in self._grouped_filenames.items():
      arg_q.put((group, filenames))

    # Start a thread to spawn processes.
    if num_procs is None: num_procs = len(os.sched_getaffinity(0))
    proc_spawner_t = threading.Thread(
        target=self._process_spawner,
        args=(arg_q, tsr_q, num_procs))
    proc_spawner_t.start()
    _defer(proc_spawner_t.join)

    num_correct = 0
    outcomes = []

    m.eval()
    it = tqdm.tqdm(range(len(self._grouped_filenames)))
    for _ in it:
      group, tensors = tsr_q.get()
      space_group = self._group_to_spacegroup[group]

      with torch.no_grad():
        out = m(tensors.cuda())
        argmax_topk = F.softmax(out, dim=1).mean(dim=0).topk(k=self._topk)
        # If we take the log before mean, then we effectively multiply
        # the softmax probabilities together.
        # Empirically, this does not work as well as the above.
        # argmax_topk = F.softmax(out, dim=1).log().mean(dim=0).topk(k=topk)

      cat = self._label_manager.get_space_group(argmax_topk.indices[0])
      if cat == space_group: num_correct += 1

      topk_cats = []
      for idx in argmax_topk.indices:
        topk_cats.append(self._label_manager.get_space_group(idx))

      outcomes.append(PredictionOutcome(
          sample=group,
          actual=space_group,
          predicted=cat,
          topk_values=argmax_topk.values.tolist(),
          topk_cats=topk_cats
      ))

      accuracy = 100 * (num_correct / len(outcomes))
      it.set_description(f"Acc: {accuracy:.02f}%")
      it.update()

    assert tsr_q.empty()
    return outcomes


def train_and_test(
    data_name, model_params: model.ModelParams, tag,
    num_epochs, max_lr,
    batch_size=512, lsuv_iterations=4,
    use_mixup=True, p_erase=0.8, rotation_interpolation=None,
    use_cuda=True,
    check_description=False,
):
  gc.collect()  # hack to reclaim gpu memory

  description = (
      f"{tag}: {data_name} | {model_params} | {num_epochs} epochs @ lr={max_lr}"
      f"\np_erase={100*p_erase:.0f}% | rot_interp={rotation_interpolation}"
      f"\nlsuv_its={lsuv_iterations} | mixup={use_mixup}"
  )
  print(description)
  if check_description: return

  cbed_data = data.CbedData(
      data.get_img_path(data_name),
      batch_size=batch_size,
      p_erase=p_erase, rotation_interpolation=rotation_interpolation,
      pin_memory=use_cuda,
  )

  m = model.make_1chan_model(model_params, cbed_data.label_manager.num_classes)
  if use_cuda: m = m.cuda()
  model.lsuv(m, cbed_data, iterations=lsuv_iterations)

  trainer = training.Trainer(
      comment=f"{data_name}_{tag}",
      description=description,
      cbed_data=cbed_data,
      model=m,
      loss_fn=F.cross_entropy,
      use_mixup=use_mixup,
  )

  trainer.train_model(num_epochs, max_lr)

  outcomes = test_combined_bicubic(
      cbed_data.img_path, trainer.model, cbed_data.label_manager, rotate_deg=360)

  outcomes_df = pd.DataFrame(outcomes)
  outcomes_df.to_csv(pathlib.Path(trainer.save_filedir)/'outcomes.csv')
  outcomes_df.to_json(pathlib.Path(trainer.save_filedir)/'outcomes.json')

  return outcomes
