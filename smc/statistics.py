import collections

import torchvision


class MeanAndStd(
    collections.namedtuple('MeanAndStd', ('mean', 'std'))):

  def to_transform_normalize(self):
    return torchvision.transforms.Normalize(mean=self.mean, std=self.std)


IMG9_stats = MeanAndStd(
    mean=0.047377074244559954,
    # std=0.05467088726489753,
    std=0.5,  # fake std
)
