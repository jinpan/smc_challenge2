import dataclasses
import functools
import typing

import numpy as np
import torch
from torch import nn
import torchvision

from . import data
from . import utils


# Much of this module is from torchvision.models.resnet, modified for single
# channel images

class BasicBlock(nn.Module):
  # changed base_width from 64 to 20
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
               base_width=20, dilation=1, norm_layer=None):
    super().__init__()
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d
    if groups != 1 or base_width != 20:
      raise ValueError('BasicBlock only supports groups=1 and base_width=20')
    if dilation > 1:
      raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    # Both self.conv1 and self.downsample layers downsample the input when stride != 1
    self.conv1 = torchvision.models.resnet.conv3x3(inplanes, planes, stride)
    self.bn1 = norm_layer(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = torchvision.models.resnet.conv3x3(planes, planes)
    self.bn2 = norm_layer(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out

class Bottleneck(nn.Module):
  # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
  # while original implementation places the stride at the first 1x1 convolution(self.conv1)
  # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
  # This variant is also known as ResNet V1.5 and improves accuracy according to
  # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

  # changed base_width from 64 to 20

  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
               base_width=20, dilation=1, norm_layer=None):
    super().__init__()
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d
    width = int(planes * (base_width / 20.)) * groups
    # Both self.conv2 and self.downsample layers downsample the input when stride != 1
    self.conv1 = torchvision.models.resnet.conv1x1(inplanes, width)
    self.bn1 = norm_layer(width)
    self.conv2 = torchvision.models.resnet.conv3x3(width, width, stride, groups, dilation)
    self.bn2 = norm_layer(width)
    self.conv3 = torchvision.models.resnet.conv1x1(width, planes * self.expansion)
    self.bn3 = norm_layer(planes * self.expansion)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out


class ResNet1Chan(nn.Module):
  def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
               groups=1, width_per_group=20, replace_stride_with_dilation=None,
               norm_layer=None):
    # changed width_per_group from 64 -> 20
    super().__init__()
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d
    self._norm_layer = norm_layer

    self.inplanes = 20
    self.dilation = 1
    if replace_stride_with_dilation is None:
      # each element in the tuple indicates if we should replace
      # the 2x2 stride with a dilated convolution instead
      replace_stride_with_dilation = [False, False, False]
    if len(replace_stride_with_dilation) != 3:
      raise ValueError("replace_stride_with_dilation should be None "
                       "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
    self.groups = groups
    self.base_width = width_per_group
    # changed initial input from 3->1
    self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = norm_layer(self.inplanes)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.layer1 = self._make_layer(block, 20, layers[0])  # 64 -> 20
    self.layer2 = self._make_layer(block, 40, layers[1], stride=2,  # 128 -> 40
                                   dilate=replace_stride_with_dilation[0])
    self.layer3 = self._make_layer(block, 80, layers[2], stride=2,  # 256 -> 80
                                   dilate=replace_stride_with_dilation[1])
    self.layer4 = self._make_layer(block, 160, layers[3], stride=2,  # 512 -> 160
                                   dilate=replace_stride_with_dilation[2])
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(160 * block.expansion, num_classes)  # 512 -> 160

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    if zero_init_residual:
      for m in self.modules():
        if isinstance(m, Bottleneck):
          nn.init.constant_(m.bn3.weight, 0)
        elif isinstance(m, BasicBlock):
          nn.init.constant_(m.bn2.weight, 0)

  def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
    norm_layer = self._norm_layer
    downsample = None
    previous_dilation = self.dilation
    if dilate:
      self.dilation *= stride
      stride = 1
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          torchvision.models.resnet.conv1x1(self.inplanes, planes * block.expansion, stride),
          norm_layer(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(block(self.inplanes, planes, groups=self.groups,
                          base_width=self.base_width, dilation=self.dilation,
                          norm_layer=norm_layer))

    return nn.Sequential(*layers)

  def _forward_impl(self, x):
    # See note [TorchScript super()]
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)

    return x

  def forward(self, x):
    return self._forward_impl(x)


def maybe_apply_dropout(model, dropout):
  if dropout is None:
    return

  linear = model.fc
  model.fc = nn.Sequential(nn.Dropout(p=dropout), linear)

def make_resnet18_1chan(num_classes=1000, dropout=None):
  model = ResNet1Chan(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
  maybe_apply_dropout(model, dropout)

  return model

def make_resnet34_1chan(num_classes=1000, dropout=None):
  model = ResNet1Chan(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
  maybe_apply_dropout(model, dropout)

  return model

def make_resnet50_1chan(num_classes=1000, dropout=None):
  model = ResNet1Chan(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
  maybe_apply_dropout(model, dropout)

  return model


###############################################################################
# LSUV
###############################################################################

@dataclasses.dataclass
class LSUV_Context:
  verbose: bool = False
  std_eps: float = 1e-8
  means: typing.List[float] = dataclasses.field(default_factory=list)
  stds: typing.List[float] = dataclasses.field(default_factory=list)

  iteration: int = dataclasses.field(default=-1, init=False)

  def next_iteration(self):
    self.iteration += 1
    self.means.append([])
    self.stds.append([])
    if self.verbose: print('=' * 80)

  def add_mean_std(self, mean, std):
    self.means[self.iteration].append(mean)
    self.stds[self.iteration].append(std)

  def summarize(self):
    print(f"Ran {self.iteration+1} iterations on {len(self.means[0])} layers:")
    for i, (means, stds) in enumerate(utils.zip_eq(self.means, self.stds)):
      print(
          f"it {i}: "
          f"avg mean: {np.mean(means):+0.2f} "
          f"std mean: {np.std(means):+0.2f} "
          f"fin mean: {means[-1]:+0.2f}"
      )
      print(
          f"      "
          f"avg std : {np.mean(stds):+0.2f} "
          f"std std : {np.std(stds):+0.2f} "
          f"fin std : {stds[-1]:+0.2f}"
      )

  def hook(self):
    def _hook(m, _input, out):
      mean, std = out.mean().item(), out.std().item()
      self.add_mean_std(mean, std)

      if self.verbose: print(f"{type(m).__name__} | mean: {mean:0.2f} | std: {std:0.2f}")

      assert isinstance(m.weight, torch.Tensor)
      m.weight.mul_(1. / (std + self.std_eps))

      has_bias = getattr(m, 'bias') is not None
      if has_bias:
        m.bias.sub_(mean)
    return _hook


def get_linear_mods(model):
  linear_layers = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)

  stack = [model]
  while stack:
    m = stack.pop(0)
    if isinstance(m, linear_layers):
      yield m
    else:
      stack.extend(m.children())


@utils.defer
def lsuv(model, dat, iterations=2, verbose=False, _defer=None):
  # dat is data.CbedData or a tensor batch.
  if isinstance(dat, data.CbedData):
    dat, _ = next(iter(dat.train_loader))
    dat = dat.to(next(model.parameters()).device)
  assert isinstance(dat, torch.Tensor)
  assert len(dat.shape) == 4
  assert len(dat.shape[0]) >= 1000
  assert len(dat.shape[1]) == 1

  ctx = LSUV_Context(verbose=verbose)

  for m in get_linear_mods(model):
    handle = m.register_forward_hook(ctx.hook())
    _defer(lambda h: h.remove(), handle)

  model.train()
  with torch.no_grad():
    for _ in range(iterations):
      ctx.next_iteration()
      model(dat)

  ctx.summarize()

  return ctx
