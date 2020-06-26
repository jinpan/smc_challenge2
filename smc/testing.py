import collections
import concurrent
import math

import PIL
import torch
import torch.nn.functional as F
import torchvision
import tqdm

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
    img_path, model, label_manager,
    filedir='valid', rotate_deg=5, topk=None,
    mean_and_std=None):

  if topk is None:
    topk = min(5, label_manager.num_classes-1)

  if mean_and_std is None:
    normalize_fn = lambda t: t
  else:
    normalize_fn = mean_and_std.to_transform_normalize()

  grouped_filenames = collections.defaultdict(list)
  group_to_spacegroup = {}

  for filename in img_path.glob(f'{filedir}/*'):
    sample, _, space_group, _ = filename.name.split('.')
    grouped_filenames[sample].append(filename.name)
    space_group = int(space_group)
    group_to_spacegroup[sample] = space_group

  grouped_filenames = dict(grouped_filenames)

  num_correct = 0
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
        tensor = normalize_fn(tensor)
        tensors.append(tensor)

      model.eval()
      with torch.no_grad():
        out = model(torch.stack(tensors).cuda())
        argmax_topk = F.softmax(out, dim=1).mean(dim=0).topk(k=topk)
        # If we take the log before mean, then we effectively multiply
        # the softmax probabilities together.
        # Empirically, this does not work as well as the above.
        # argmax_topk = F.softmax(out, dim=1).log().mean(dim=0).topk(k=topk)

      cat = label_manager.get_space_group(argmax_topk.indices[0])

      if cat == space_group:
        num_correct += 1

      topk_cats = []
      for idx in argmax_topk.indices:
        topk_cats.append(label_manager.get_space_group(idx))

      outcomes.append(PredictionOutcome(
          sample=group,
          actual=space_group,
          predicted=cat,
          topk_values=argmax_topk.values.tolist(),
          topk_cats=topk_cats
      ))

      accuracy = 100 * (num_correct / len(outcomes))
      it.set_description(f"Acc: {accuracy:.02f}%")

  return outcomes
