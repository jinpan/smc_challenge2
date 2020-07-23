import collections
import concurrent
import dataclasses
import functools
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

TestBC2Result = collections.namedtuple('TestBC2Result', (
    'space_group',
    'main_preds',
    'main_probs',
    'm2_preds',
    'm2_probs',
))

TestBC2Result2 = collections.namedtuple('TestBC2Result2', (
    'space_group',
    'main_preds',  # vector of len 3 (A, B, C), space groups
    'main_probs',  # vector of len 3 (pA, pB, pC), sum should be 1
    # in general, N choose 2 of these
    # TODO: need to figure out ordering for general case
    'm2_probs',    # vector of len 3 (p(AvB); p(BvC), p(CvA))
    'debug',
))

TestBC2Result3 = collections.namedtuple('TestBC2Result3', (
    'space_group',
    'main_preds',  # vector of len 4 (A, B, C, D), space groups
    'main_probs',  # vector of len 4 (pA, pB, pC, pD), sum should be 1
    # in general, N choose 2 of these
    'm2_probs',    # map of len 6: {(0, 1): pAvB, (0, 2): pAvC, (0, 3): pAvD, (1, 2): pBvC, ...}
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


  results, results2, results3, results4 = [], [], [], []
  num_correct, num_topk_correct, num_its = 0, 0, 0
  counters = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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

    t1, t2, t3, t4, t5 = topk_cats[0], topk_cats[1], topk_cats[2], topk_cats[3], topk_cats[4]
    if t1 == space_group:
      num_correct += 1

    if t1 in filter_spacegroups and t2 in filter_spacegroups and t3 in filter_spacegroups:
      counters[4] += 1  # number of times the top 3 was represented
      if t1 == space_group:
        counters[5] += 1  # number of times the top-1 was correct given that this happened.

      # TODO: Abstract this into something intelligible and scalable to more classes
      m_a, lm_a = _get_model_and_label_manager(group_to_model, t1, t2)
      m_b, lm_b = _get_model_and_label_manager(group_to_model, t2, t3)
      m_c, lm_c = _get_model_and_label_manager(group_to_model, t3, t1)

      argmax_topk2 = F.softmax(m_a(tensors), dim=1).mean(dim=0).cpu().topk(k=2)
      topk_cats2 = [lm_a.get_space_group(idx) for idx in argmax_topk2.indices]
      if topk_cats2[0] == t1:
        assert topk_cats2[1] == t2
        p1v2 = argmax_topk2.values[0].item()
      else:
        assert topk_cats2[0] == t2
        p1v2 = argmax_topk2.values[1].item()

      argmax_topk3 = F.softmax(m_b(tensors), dim=1).mean(dim=0).cpu().topk(k=2)
      topk_cats3 = [lm_b.get_space_group(idx) for idx in argmax_topk3.indices]
      if topk_cats3[0] == t2:
        assert topk_cats3[1] == t3
        p2v3 = argmax_topk3.values[0].item()
      else:
        assert topk_cats3[0] == t3
        p2v3 = argmax_topk3.values[1].item()

      argmax_topk4 = F.softmax(m_c(tensors), dim=1).mean(dim=0).cpu().topk(k=2)
      topk_cats4 = [lm_c.get_space_group(idx) for idx in argmax_topk4.indices]
      if topk_cats4[0] == t3:
        assert topk_cats4[1] == t1
        p3v1 = argmax_topk4.values[0].item()
      else:
        assert topk_cats4[0] == t1
        p3v1 = argmax_topk4.values[1].item()

      results2.append(TestBC2Result2(
          space_group=space_group,
          main_preds=[t1, t2, t3],
          main_probs=[argmax_topk.values[0].item(), argmax_topk.values[1].item(), argmax_topk.values[2].item()],
          # p(1v2), p(2v3), p(3v1)
          m2_probs=[p1v2, p2v3, p3v1],
          debug=(argmax_topk2, topk_cats2, argmax_topk3, topk_cats3, argmax_topk4, topk_cats4)
      ))

    if t1 in filter_spacegroups and t2 in filter_spacegroups and t3 in filter_spacegroups and t4 in filter_spacegroups:
      counters[6] += 1  # number of times the top 3 was represented
      if t1 == space_group:
        counters[7] += 1  # number of times the top-1 was correct given that this happened.

      result_kwargs = {
          'space_group': space_group,
          'main_preds': [t1, t2, t3, t4],
          'main_probs': [
              argmax_topk.values[0].item(),
              argmax_topk.values[1].item(),
              argmax_topk.values[2].item(),
              argmax_topk.values[3].item(),
          ],
          'm2_probs': {}
      }
      ts = [t1, t2, t3, t4]
      for idx, tA in enumerate(ts):
        for tB in ts[idx+1:]: # tA is always before tB
          idxB = ts.index(tB)
          assert idx < idxB

          m2, lm2 = _get_model_and_label_manager(group_to_model, tA, tB)

          argmax_topk2 = F.softmax(m2(tensors), dim=1).mean(dim=0).cpu().topk(k=2)
          topk_cats2 = [lm2.get_space_group(idx) for idx in argmax_topk2.indices]
          if topk_cats2[0] == tA:
            assert topk_cats2[1] == tB
            pAvB = argmax_topk2.values[0].item()
          else:
            assert topk_cats2[0] == tB
            pAvB = argmax_topk2.values[1].item()

          result_kwargs['m2_probs'][(idx, idxB)] = pAvB

      results3.append(TestBC2Result3(**result_kwargs))

    if t1 in filter_spacegroups and t2 in filter_spacegroups and t3 in filter_spacegroups and t4 in filter_spacegroups and t5 in filter_spacegroups:
      counters[8] += 1  # number of times the top 4 was represented
      if t1 == space_group:
        counters[9] += 1  # number of times the top-1 was correct given that this happened.

      result_kwargs = {
          'space_group': space_group,
          'main_preds': [t1, t2, t3, t4, t5],
          'main_probs': [
              argmax_topk.values[0].item(),
              argmax_topk.values[1].item(),
              argmax_topk.values[2].item(),
              argmax_topk.values[3].item(),
              argmax_topk.values[4].item(),
          ],
          'm2_probs': {}
      }
      ts = [t1, t2, t3, t4, t5]
      for idx, tA in enumerate(ts):
        for tB in ts[idx+1:]: # tA is always before tB
          idxB = ts.index(tB)
          assert idx < idxB

          m2, lm2 = _get_model_and_label_manager(group_to_model, tA, tB)

          argmax_topk2 = F.softmax(m2(tensors), dim=1).mean(dim=0).cpu().topk(k=2)
          topk_cats2 = [lm2.get_space_group(idx) for idx in argmax_topk2.indices]
          if topk_cats2[0] == tA:
            assert topk_cats2[1] == tB
            pAvB = argmax_topk2.values[0].item()
          else:
            assert topk_cats2[0] == tB
            pAvB = argmax_topk2.values[1].item()

          result_kwargs['m2_probs'][(idx, idxB)] = pAvB
      results4.append(TestBC2Result3(**result_kwargs))

    m2, lm2 = _get_model_and_label_manager(group_to_model, t1, t2)
    if m2 is not None:
      counters[0] += 1  # number of times the correct group was in the top 2 (6539)
      if t1 == space_group:
        counters[1] += 1  # given counters[0], number of times the main model got it right (4719)

      argmax_topk2 = F.softmax(m2(tensors), dim=1).mean(dim=0).topk(k=2)
      topk_cats2 = [lm2.get_space_group(idx) for idx in argmax_topk2.indices]
      u1, u2 = topk_cats2[0], topk_cats2[1]
      if u1 == space_group:
        counters[2] += 1  # number of times the auxiliary model got it right by itself (4428)

      if t1 == u1:
        assert t2 == u2
        p0 = argmax_topk.values[0].item() * argmax_topk2.values[0].item()
        p1 = argmax_topk.values[1].item() * argmax_topk2.values[1].item()
      else:  # t1 != u1
        assert t2 != u2
        p0 = argmax_topk.values[0].item() * argmax_topk2.values[1].item()
        p1 = argmax_topk.values[1].item() * argmax_topk2.values[0].item()

      if p0 > p1: est = t1
      else: est = t2
      if est == space_group:
        counters[3] += 1  # number of times the joint prediction got it right

      results.append(TestBC2Result(
          space_group=space_group,
          main_preds=[t1, t2],
          main_probs=[argmax_topk.values[0].item(), argmax_topk.values[1].item()],
          m2_preds=[u1, u2],
          m2_probs=[argmax_topk2.values[0].item(), argmax_topk2.values[1].item()],
      ))

    if space_group in topk_cats: num_topk_correct += 1
    accuracy = 100 * (num_correct / num_its)
    topk_accuracy = 100 * (num_topk_correct / num_its)
    it.set_description(f"Acc: {accuracy:.02f}% | {topk_accuracy:.02f}% | {counters}")

  return results, results2, results3, results4


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

  return outcomes
