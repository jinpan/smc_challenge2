import collections
import dataclasses
import functools
import math
import os
import pathlib
import random
import re
import typing

import PIL
import torch
import torch.nn.functional as F
import torchvision
import tqdm

from . import statistics
from . import utils


_space_groups = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76,
    78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 91, 92, 94, 95, 96, 97, 98,
    99, 100, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 114, 115,
    116, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
    131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
    145, 146, 147, 148, 149, 150, 152, 154, 155, 156, 157, 158, 159, 160,
    161, 162, 163, 164, 165, 166, 167, 172, 173, 174, 176, 177, 179, 180,
    181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
    195, 196, 197, 198, 199, 208, 212, 213, 214, 215, 216, 217, 218, 219,
    220, 221, 223, 224, 225, 226, 227, 229, 230,
)
assert len(_space_groups) == 200
_space_group_to_canonical_id = {}
for idx, group in enumerate(_space_groups):
  _space_group_to_canonical_id[group] = idx


class LabelManager:
  def __init__(self):
    self._label_to_label_id = {}
    self._labels = []

    self._frozen = False

  def freeze(self):
    self._frozen = True

  def get_label_mappings(self, raw_labels):
    labels = set()
    for raw_label in raw_labels:
      label = int(raw_label)
      assert label in _space_group_to_canonical_id
      labels.add(label)

    mappings = {}

    if self._frozen:
      for label in labels:
        mappings[str(label)] = self._label_to_label_id.get(label)
      return mappings

    assert len(self._label_to_label_id) == 0
    for label in sorted(labels):
      label_id = len(self._labels)
      self._label_to_label_id[label] = label_id
      self._labels.append(label)
      mappings[str(label)] = label_id

    return mappings


  @property
  def num_classes(self):
    return len(self._labels)

  def get_space_group(self, label_id):
    return self._labels[label_id]

  def state_dict(self):
    return {
      # could reconstruct this from self._labels, but space savings is insignificant
      'label_to_label_id': self._label_to_label_id,
      'labels': self._labels,
      'frozen': self._frozen,
    }

  @classmethod
  def load(cls, data):
    inst = cls()

    inst._label_to_label_id = data['label_to_label_id']
    inst._labels = data['labels']
    inst._frozen = data['frozen']

    return inst


class CbedImageCacheDataset(torch.utils.data.Dataset):
  _spacegroup_re = re.compile(r'.*\.(0|1|2)\.(\d{1,3})\.png$')

  def __init__(self, roots, label_manager, image_loader_fn,
               filter_spacegroups=None,
               cbed_slices=('0','1','2'), lazy=True):
    self._filenames = []
    self._images = []
    self._labels = []
    self._image_loader_fn = image_loader_fn

    raw_labels = []
    filenames = []

    for root in roots:
      for filename in root.glob('*.png'):
        m = utils.must_match(self._spacegroup_re, filename.name)
        cbed_slice, space_group = m.group(1), m.group(2)
        if cbed_slice not in cbed_slices: continue
        if filter_spacegroups is not None and int(space_group) not in filter_spacegroups:
          continue

        filenames.append(filename)
        raw_labels.append(space_group)

    mappings = label_manager.get_label_mappings(set(raw_labels))

    dropped_labels = set()
    for filename, raw_label in utils.zip_eq(filenames, raw_labels):
      label = mappings.get(raw_label)
      if label is None:
        dropped_labels.add(raw_label)
        continue

      self._filenames.append(filename)
      self._labels.append(label)

      if lazy:
        img = None
      else:
        img = image_loader_fn(filename)
      self._images.append(img)

    if dropped_labels:
      print(f"WARNING: Dropped these labels: {dropped_labels}")

  def __getitem__(self, idx):
    img = self._images[idx]
    if img is None:
      img = self._image_loader_fn(self._filenames[idx])
      self._images[idx] = img
    return img, self._labels[idx]

  def __len__(self):
    return len(self._images)


class CbedDataset(torch.utils.data.Dataset):
  def __init__(self, roots, label_manager, image_loader_fn, cbed_slices, transform, filter_spacegroups):
    self._image_cache = CbedImageCacheDataset(
        roots, label_manager, image_loader_fn, filter_spacegroups, cbed_slices=cbed_slices)

    self._transform = transform

  def __getitem__(self, idx):
    image, label = self._image_cache[idx]
    return self._transform(image), label

  def __len__(self):
    return len(self._image_cache)


def get_img_path(data_name):
  rootdir = pathlib.Path(os.environ.get(
      '_SMC_IMG_DIR',
      '/home/jin/sandbox/smc_challenge2/images',
  ))
  return rootdir/data_name


@dataclasses.dataclass
class DataParams:
  img_path: str
  batch_size: int
  p_erase: float

  cbed_slices: typing.Tuple[str] = ('0','1','2')

  horizontal_flip: bool = False
  rotation_interpolation: utils.RotationInterpolation = None
  chans: str = 'L'
  filter_spacegroups: typing.Optional[typing.Container[int]] = None
  train_with_validation: bool = False

class CbedData:
  def __init__(self,
      params: DataParams,
      num_workers=None, pin_memory=True,
    ):
    self.img_path = get_img_path(params.img_path)

    self.label_manager = LabelManager()

    if num_workers is None:
      # number of cpus available to the current process
      num_workers = len(os.sched_getaffinity(0))
    self._num_workers = num_workers

    train_transforms, valid_transforms = [], []
    if not params.horizontal_flip and not params.rotation_interpolation:
      def image_loader_fn(fn):
        with PIL.Image.open(fn).convert(params.chans) as img:
          return torchvision.transforms.functional.to_tensor(img)
    else:
      image_loader_fn = lambda fn: PIL.Image.open(fn).convert(params.chans)

      if params.horizontal_flip:
        train_transforms.append(torchvision.transforms.RandomHorizontalFlip())
      if params.rotation_interpolation is not None:
        train_transforms.append(utils.RandomRotation(params.rotation_interpolation))

      train_transforms.append(torchvision.transforms.ToTensor())
      valid_transforms.append(torchvision.transforms.ToTensor())

    train_transforms.append(torchvision.transforms.RandomErasing(p=params.p_erase))


    train_paths = [self.img_path/'train']
    valid_paths = []
    if params.train_with_validation:
      train_paths.append(self.img_path/'valid')
      valid_paths.append(self.img_path/'test')
    else:
      valid_paths.append(self.img_path/'valid')

    self._train_set = CbedDataset(
        train_paths,
        self.label_manager,
        image_loader_fn,
        cbed_slices=params.cbed_slices,
        transform=torchvision.transforms.Compose(train_transforms),
        filter_spacegroups=params.filter_spacegroups,
    )
    self.label_manager.freeze()
    self._valid_set = CbedDataset(
        valid_paths,
        self.label_manager,
        image_loader_fn,
        cbed_slices=params.cbed_slices,
        transform=torchvision.transforms.Compose(valid_transforms),
        filter_spacegroups=params.filter_spacegroups,
    )

    self.train_loader = torch.utils.data.DataLoader(
        self._train_set,
        batch_size=params.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=self._num_workers,
    )
    self.valid_loader = torch.utils.data.DataLoader(
        self._valid_set,
        batch_size=params.batch_size,
        pin_memory=pin_memory,
        # don't set num_workers since we do minimal transformations on the validation set.
    )


  def make_cross_entropy_weights(self):
    train_set = self._train_set

    count_by_label_id = torch.zeros(self.label_manager.num_classes)
    for label_id, count in collections.Counter(train_set._image_cache._labels).items():
      count_by_label_id[label_id] = float(count)
    assert min(count_by_label_id) > 0

    return count_by_label_id.mean() / count_by_label_id
