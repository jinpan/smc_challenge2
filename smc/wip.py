"""
Partially fleshed out ideas, staged here.
"""


###############################################################################
# Dataset / Dataloader for test augmentation
# Allows us to pipeline generation of augmentated data with model evaluation
# for walltime speedup
# On pause because we need a mechanism to recombine slices back together
# .. perhaps by saving filenames
###############################################################################
class CbedTestAugmentationDataset(torch.utils.data.Dataset):
  def __init__(self, image_cache_dataset, rotate_deg=5):
    self._image_cache_dataset = image_cache_dataset
    self._rotate_angles_deg = list(range(0, 360, rotate_deg))

  def __getitem__(self, idx):

    base_idx = idx // self.num_angles
    img, label = self._image_cache_dataset[base_idx]
    angle = self._rotate_angles_deg[idx % self.num_angles]
    img_r = torchvision.transforms.functional.rotate(
        img, angle, resample=PIL.Image.BICUBIC)

    return img_r, label

  def __len__(self):
    return self.num_images * self.num_angles
  
  @property
  def num_angles(self):
    return len(self._rotate_angles_deg)

  @property
  def num_images(self):
    return len(self._image_cache_dataset)


class CbedTestAugmentationBatchSampler:
  def __init__(self, aug_dataset):
    self._aug_dataset = aug_dataset
  
  def __iter__(self):
    num_angles = self._aug_dataset.num_angles
    for idx in range(0, len(self._aug_dataset), num_angles):
      yield list(range(idx, idx+num_angles))

  def __len__(self):
    return self._aug_dataset.num_images

def make_test_augmentation_dataloader(cbed_data, rotate_deg=5):
    image_cache_dataset = self._valid_set._image_cache
    aug_dataset = CbedTestAugmentationDataset(
        image_cache_dataset,
        rotate_deg=rotate_deg)

    return torch.utils.data.DataLoader(
        aug_dataset,
        num_workers=self._num_workers,
        pin_memory=True,
    )

###############################################################################
# / Dataset / Dataloader for test augmentation
###############################################################################