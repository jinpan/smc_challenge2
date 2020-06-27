import functools
import math
import random

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def set_seed(n):
  random.seed(n)
  torch.manual_seed(n)


def rotate(images_t, angles_rad):
  batch_size, _, _, _ = images_t.shape  # BS x C x H x W
  assert len(angles_rad) == batch_size

  theta = torch.zeros(batch_size, 2, 3)
  for i, angle_rad in enumerate(angles_rad):
    alpha, beta = math.cos(angle_rad), math.sin(angle_rad)

    theta[i][0][0], theta[i][0][1] = alpha, beta
    theta[i][1][0], theta[i][1][1] = -beta, alpha

  grid = F.affine_grid(theta=theta.cuda(), size=images_t.shape, align_corners=False)
  images_t = F.grid_sample(images_t, grid, align_corners=False)

  return images_t

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
