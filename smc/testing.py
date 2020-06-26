import collections
import math

import PIL
import torch
import torch.nn.functional as F
import torchvision
import tqdm

from . import utils



PredictionOutcome = collections.namedtuple('PredictionOutcome', (
    'sample',  # grouped_filenames key
    'actual',
    'predicted',

    'topk_values',
    'topk_cats',
))

def get_cat(label_manager, label_id):
  return str(label_manager._labels[label_id])


def test_combined_bicubic(
    img_path, model, label_manager,
    filedir='valid', rotate_deg=5, topk=None):

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

  angles_deg = list(range(0, 360, rotate_deg))
  angles_rad = list(map(math.radians, angles_deg))
  num_angles_per_image = len(angles_rad)
  angles_rad *= 3

  num_correct = 0
  outcomes = []

  it = tqdm.tqdm(grouped_filenames.items())
  for group, filenames in it:
    space_group = group_to_spacegroup[group]

    images = []
    for filename in filenames:
      fullpath = str(img_path/filedir/filename)
      img = PIL.Image.open(fullpath).convert('L')

      for angle in angles_deg:
        img_r = torchvision.transforms.functional.rotate(
          img, angle, resample=PIL.Image.BICUBIC)
        img_rt = torchvision.transforms.functional.to_tensor(img_r)
        images.append(img_rt)
    images_t_rotated = torch.stack(images).cuda()

    model.eval()
    with torch.no_grad():
      out = model(images_t_rotated)
      argmax_topk = F.softmax(out, dim=1).mean(dim=0).topk(k=topk)

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


def test_combined(
    img_path, model, label_manager,
    filedir='valid', rotate_deg=5, topk=None):

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

  angles_rad = [math.radians(deg) for deg in range(0, 360, rotate_deg)]
  num_angles_per_image = len(angles_rad)
  angles_rad *= 3

  num_correct = 0
  outcomes = []

  it = tqdm.tqdm(grouped_filenames.items())
  for group, filenames in it:
    space_group = group_to_spacegroup[group]

    images = []
    for filename in filenames:
      fullpath = str(img_path/filedir/filename)
      img = PIL.Image.open(fullpath).convert('L')
      img_t = torchvision.transforms.functional.to_tensor(img)
      images.append(img_t.cuda())

    images_t = torch.cat([
        img_t.repeat(num_angles_per_image, 1, 1, 1) for img_t in images
    ])
    images_t_rotated = utils.rotate(images_t, angles_rad)

    model.eval()
    with torch.no_grad():
      out = model(images_t_rotated)
      argmax_topk = F.softmax(out, dim=1).mean(dim=0).topk(k=topk)

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
