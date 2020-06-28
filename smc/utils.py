import enum
import functools
import math
import random

import cv2
from matplotlib import pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torchvision


def set_seed(n):
  random.seed(n)
  torch.manual_seed(n)


@enum.unique
class RotationInterpolation(enum.Enum):
  TORCH_BICUBIC = 1
  OPENCV_CUBIC = 2
  OPENCV_LANCZOS4 = 3


def rotate(img, deg, rotation_interpolation):
  if rotation_interpolation == RotationInterpolation.TORCH_BICUBIC:
    return torchvision.transforms.functional.rotate(
        img, deg, resample=PIL.Image.BICUBIC)

  if rotation_interpolation in (
      RotationInterpolation.OPENCV_CUBIC,
      RotationInterpolation.OPENCV_LANCZOS4,
  ):
    flags = {
        RotationInterpolation.OPENCV_CUBIC: cv2.INTER_CUBIC,
        RotationInterpolation.OPENCV_LANCZOS4: cv2.INTER_LANCZOS4,
    }[rotation_interpolation]

    w, h = img.size
    cx, cy = w//2, h//2

    cv2_img = np.array(img)
    M = cv2.getRotationMatrix2D((cx, cy), angle=deg, scale=1.)
    cv2_rot = cv2.warpAffine(cv2_img, M, (w, h), flags=flags)

    return PIL.Image.fromarray(cv2_rot)

  raise RuntimeError("Unhandled rotation interpolation")


class RandomRotation:
  def __init__(self, rotation_interpolation):
    self._rotation_interpolation = rotation_interpolation

  def __call__(self, img):
    deg = int(360 * random.random())
    return rotate(img, deg, self._rotation_interpolation)


# Based on https://towerbabbel.com/go-defer-in-python/
def defer(func):
  @functools.wraps(func)
  def func_wrapper(*args, **kwargs):
    deferred = []
    def _defer_handler(f, *a, **kw): deferred.append((f, a, kw))
    try:
      return func(*args, _defer=_defer_handler, **kwargs)
    finally:
      for fn, a, kw in reversed(deferred): fn(*a, **kw)
  return func_wrapper


def must_match(regex, string):
  m = regex.match(string)
  if m is None:
    raise RuntimeError("failed to match")
  return m


def imshow(image_t, ax=None, vmin=-3, vmax=3):
  np_img = image_t.cpu().numpy()

  ax = ax or plt.gca()

  if np_img.shape[0] == 1:
    imshow_img = np_img[0]
  else:
    imshow_img = np.transpose(np_img, (1, 2, 0))

  ax.imshow(imshow_img, cmap='gray', vmin=vmin, vmax=vmax)


def lin_comb(a, b, alpha):
  return alpha * a + (1-alpha) * b


def zip_eq(a, b):
  it_a = iter(a)
  it_b = iter(b)

  while True:
    try:
      next_a = next(it_a)
      next_a_ok = True
    except StopIteration:
      next_a_ok = False

    try:
      next_b = next(it_b)
      next_b_ok = True
    except StopIteration:
      next_b_ok = False

    if next_a_ok and next_b_ok:
      yield (next_a, next_b)
    elif not next_a_ok and not next_b_ok:
      return
    else:
      raise RuntimeError("a and b are unequal lengths")
