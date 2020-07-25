import collections
import concurrent
import dataclasses
import functools
import gc
import io
import math
import multiprocessing
import os
import pathlib
import threading
import queue
import shutil
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


def test_combined_bicubic(
    img_path, m, label_manager,
    filter_spacegroups=None,
    filedir='valid', topk=None, test_cbed_slices=('0', '1', '2')):

  model_device = next(iter(m.parameters())).device
  if topk is None:
    topk = min(5, label_manager.num_classes-1)

  grouped_filenames = collections.defaultdict(list)
  group_to_spacegroup = {}

  for filename in img_path.glob(f'{filedir}/*'):
    sample, cbed_slice, space_group, _ext = filename.name.split('.')
    if cbed_slice not in test_cbed_slices: continue
    space_group = int(space_group)
    if filter_spacegroups and space_group not in filter_spacegroups:
      continue
    grouped_filenames[sample].append(filename.name)
    group_to_spacegroup[sample] = space_group

  grouped_filenames = dict(grouped_filenames)

  num_correct, num_topk_correct = 0, 0
  outcomes = []

  it = tqdm.tqdm(grouped_filenames.items())
  for group, filenames in it:
    space_group = group_to_spacegroup[group]

    images = []
    for filename in filenames:
      fullpath = str(img_path/filedir/filename)
      images.append(PIL.Image.open(fullpath).convert('L'))

    tensors = []
    for img in images:
      tensors.append(torchvision.transforms.functional.to_tensor(img))

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

def _get_model_and_label_manager(group_to_model, g1, g2):
  if (g1, g2) in group_to_model:
    return group_to_model[(g1, g2)]
  if (g2, g1) in group_to_model:
    return group_to_model[(g2, g1)]
  return None, None

TestPFResult = collections.namedtuple('TestPFResult', (
    'sample',
    'space_group',  # actual label
    'main_preds',  # vector of len K (A, B, C, ...), space groups
    'main_probs',  # vector of len K (pA, pB, pC, ...), sum should be 1
    'm2_probs',    # map of len (K choose 2): {(0, 1): pAvB, (0, 2): pAvC, (1, 2): pBvC, ...}
))


def probability_flow(
    base_probabilities: typing.List[float],
    # pairwise_probabilities is map of (A, B): p, where p is the likelihood of A vs B.
    # should have length N choose 2, and A < B
    pairwise_probabilities: typing.Dict[typing.Tuple[int, int], float],
    pct_flow: typing.Union[float, typing.List[float]],
):
  n = len(base_probabilities)
  assert sum(base_probabilities) - 1 < 1e-5
  assert isinstance(pairwise_probabilities, dict)
  for a in range(n):
    for b in range(a+1, n):
      assert (a, b) in pairwise_probabilities
  assert len(pairwise_probabilities) == (n * (n-1)) // 2

  # Creating the matrix once and using linear combinations is much faster than
  # creating the matrix each time in the loop with flow_mat[a][b] = p * pct_flow / (n-1).
  flow_mat = torch.zeros(n, n)
  for a in range(n):
    for b in range(n):
      if a == b: continue  # diagonals calculated in outer loop as 1-sum(row)
      if a < b: p = 1 - pairwise_probabilities[(a, b)]
      else: p = pairwise_probabilities[(b, a)]
      flow_mat[a][b] = p / (n-1)
    flow_mat[a][a] = 1 - flow_mat[a][:].sum()

  if isinstance(pct_flow, float): pct_flow = [pct_flow]
  ps = torch.tensor(base_probabilities)
  for pct in pct_flow:
    ps = utils.lin_comb(ps @ flow_mat, ps, alpha=pct)
  return ps


@torch.no_grad()
def test_combined_bicubic2(
    img_path, main_model, main_label_manager,
    group_to_model,  # map of (g1, g2) to (model, label_manager)
    filter_spacegroups=None,
    filedir='valid', topk=5):
  main_model.eval()
  for m2, _ in group_to_model.values():
    m2.eval()

  grouped_filenames = collections.defaultdict(list)
  group_to_spacegroup = {}
  for filename in img_path.glob(f'{filedir}/*'):
    sample, _cbed_slice, space_group, _ext = filename.name.split('.')
    space_group = int(space_group)
    # if filter_spacegroups and space_group not in filter_spacegroups:
    #   continue
    grouped_filenames[sample].append(filename.name)
    group_to_spacegroup[sample] = space_group
  grouped_filenames = dict(grouped_filenames)

  results_by_k = {2: [], 3: [], 4: [], 5: []}
  num_correct, num_topk_correct, num_its = 0, 0, 0
  counters = [0, 0, 0, 0, 0, 0, 0, 0]
  it = tqdm.tqdm(grouped_filenames.items())
  for group, filenames in it:
    num_its += 1
    space_group = group_to_spacegroup[group]

    tensors = []
    for filename in filenames:
      img = PIL.Image.open(str(img_path/filedir/filename)).convert('L')
      tensors.append(torchvision.transforms.functional.to_tensor(img))
    tensors = torch.stack(tensors).cuda()

    out = main_model(tensors)
    argmax_topk = F.softmax(out, dim=1).mean(dim=0).topk(k=topk)
    topk_cats = [main_label_manager.get_space_group(idx) for idx in argmax_topk.indices]

    if topk_cats[0] == space_group:
      num_correct += 1

    for k in (2, 3, 4, 5):
      counter_base = 2 * k - 4
      main_preds = [topk_cats[i] for i in range(k)]
      if all(pred in filter_spacegroups for pred in main_preds):
        counters[counter_base] += 1  # number of times the top-k was represented
        if main_preds[0] == space_group:
          counters[counter_base+1] += 1  # number of times the top-1 was correct given that this happened.

        main_probs = [argmax_topk.values[i].item() for i in range(k)]
        result_kwargs = {
            'sample': group,
            'space_group': space_group,
            'main_preds': main_preds,
            'main_probs': main_probs,
            'm2_probs': {}
        }
        for idxA, predA in enumerate(main_preds):
          for idxB, predB in enumerate(main_preds[idxA+1:], start=idxA+1):
            assert idxA < idxB

            m2, lm2 = _get_model_and_label_manager(group_to_model, predA, predB)
            argmax_topk2 = F.softmax(m2(tensors), dim=1).mean(dim=0).cpu().topk(k=2)
            topk_cats2 = [lm2.get_space_group(idx) for idx in argmax_topk2.indices]

            assert topk_cats2 in ([predA, predB], [predB, predA])
            if topk_cats2[0] == predA:
              pAvB = argmax_topk2.values[0].item()
            else:
              pAvB = argmax_topk2.values[1].item()
            result_kwargs['m2_probs'][(idxA, idxB)] = pAvB

        results_by_k[k].append(TestPFResult(**result_kwargs))

    if space_group in topk_cats: num_topk_correct += 1
    accuracy = 100 * (num_correct / num_its)
    topk_accuracy = 100 * (num_topk_correct / num_its)
    it.set_description(f"Acc: {accuracy:.02f}% | {topk_accuracy:.02f}% | {counters}")

  return results_by_k


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
    num_topk_correct = 0
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

      topk_cats = []
      for idx in argmax_topk.indices:
        topk_cats.append(self._label_manager.get_space_group(idx))

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

    assert tsr_q.empty()
    return outcomes


def train_and_test(
    data_params: data.DataParams, model_params: model.ModelParams, tag,
    num_epochs, max_lr,
    lsuv_iterations=4,
    use_mixup=True,
    use_cuda=True,
    num_workers=None,
    use_weighted_cross_entropy=False,
    check_description=False,
):
  gc.collect()  # hack to reclaim gpu memory

  data_name = data_params.img_path
  description = (
      f"{tag}: {data_name} {num_epochs} epochs @ lr={max_lr}"
      f"\n{data_params}"
      f"\n{model_params}"
      f"\nlsuv_its={lsuv_iterations} | mixup={use_mixup}"
  )
  print(description)
  if check_description: return

  cbed_data = data.CbedData(data_params, num_workers=num_workers, pin_memory=use_cuda)

  m = model.make_1chan_model(model_params, cbed_data.label_manager.num_classes)
  if use_cuda: m = m.cuda()
  model.lsuv(m, cbed_data, iterations=lsuv_iterations)

  if use_weighted_cross_entropy:
    weights = cbed_data.make_cross_entropy_weights()
    if use_cuda: weights = weights.cuda()
    loss_fn = functools.partial(F.cross_entropy, weight=weights)
  else:
    loss_fn = F.cross_entropy

  trainer = training.Trainer(
      comment=f"{data_name}_{tag}",
      description=description,
      cbed_data=cbed_data,
      model=m,
      loss_fn=loss_fn,
      use_mixup=use_mixup,
  )
  with open(pathlib.Path(trainer.save_filedir)/'description.txt', 'w') as f:
    f.write(description)

  trainer.train_model(num_epochs, max_lr)

  outcomes = test_combined_bicubic(
      cbed_data.img_path, trainer.model, cbed_data.label_manager,
      filter_spacegroups=data_params.filter_spacegroups,
      test_cbed_slices=data_params.cbed_slices)

  outcomes_df = pd.DataFrame(outcomes)
  outcomes_df.to_csv(pathlib.Path(trainer.save_filedir)/'outcomes.csv')
  outcomes_df.to_json(pathlib.Path(trainer.save_filedir)/'outcomes.json')

  return trainer.save_filedir, outcomes


def train_and_test_pairwise_task(g1, g2, train_with_validation=False):
  save_filedir, _ = train_and_test(
      data.DataParams(
          'img9', batch_size=512, p_erase=0.8,
          filter_spacegroups=[g1, g2],
          train_with_validation=train_with_validation,
      ),
      model.ModelParams(
          'resnet50',
          l1_width=20,
          use_333_input_conv=True,
          pool_downsample_ident=True,
      ),
      tag='16',
      num_epochs=40, max_lr=2e-2,
      # use_weighted_cross_entropy=True,
  )

  # load the save_filedir as a bytesIO zip file and return it
  shutil.make_archive(
      '/tmp/archive',
      format='gztar',
      root_dir=save_filedir,
  )

  buf = io.BytesIO()
  with open('/tmp/archive.tar.gz', 'rb') as f:
    buf.write(f.read())
  buf.seek(0)
  return buf
