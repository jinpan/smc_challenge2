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
               base_width=20, l1_width=20):
    super().__init__()
    if groups != 1:
      raise ValueError('BasicBlock only supports groups=1')
    # Both self.conv1 and self.downsample layers downsample the input when stride != 1
    self.conv1 = torchvision.models.resnet.conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = torchvision.models.resnet.conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
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

  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
               base_width=20, l1_width=20):
    super().__init__()
    width = int(planes * (base_width / float(l1_width))) * groups
    # Both self.conv2 and self.downsample layers downsample the input when stride != 1
    self.conv1 = torchvision.models.resnet.conv1x1(inplanes, width)
    self.bn1 = nn.BatchNorm2d(width)
    self.conv2 = torchvision.models.resnet.conv3x3(width, width, stride, groups)
    self.bn2 = nn.BatchNorm2d(width)
    self.conv3 = torchvision.models.resnet.conv1x1(width, planes * self.expansion)
    self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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
               groups=1, width_per_group=20, l1_width=20,
               use_333_input_conv=False,     # resnet-C
               pool_downsample_ident=False,  # resnet-D
               ):
    # changed width_per_group from 64 -> 20
    super().__init__()
    self._l1_width = l1_width
    self._pool_downsample_ident = pool_downsample_ident  # resnet-D

    self.inplanes = l1_width
    self.groups = groups
    self.base_width = width_per_group
    # changed initial input from 3->1 (nn.Conv2d(3, ...) -> nn.Conv2d(1, ...))
    if use_333_input_conv:  # resnet-C from bag of tricks paper
      half_inplanes = self.inplanes // 2
      input_conv = nn.Sequential(
          nn.Conv2d(1, half_inplanes, kernel_size=3, stride=2, padding=1, bias=False),
          nn.Conv2d(half_inplanes, half_inplanes, kernel_size=3, stride=1, padding=1, bias=False),
          nn.Conv2d(half_inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
      )
    else:
      input_conv = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
    self.input_stem = nn.Sequential(
        input_conv,
        nn.BatchNorm2d(self.inplanes),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )

    self.layer1 = self._make_layer(block, l1_width, layers[0])  # 64 -> l1_width
    self.layer2 = self._make_layer(block, 2*l1_width, layers[1], stride=2)  # 128 -> 2*l1_width
    self.layer3 = self._make_layer(block, 4*l1_width, layers[2], stride=2)  # 256 -> 4*l1_width
    self.layer4 = self._make_layer(block, 8*l1_width, layers[3], stride=2)  # 512 -> 8*l1_width
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(8*l1_width * block.expansion, num_classes)  # 512 -> 8*l1_width

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

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    outplanes = planes * block.expansion
    if stride != 1 or self.inplanes != outplanes:
      downsample_layers = []

      if stride != 1 and self._pool_downsample_ident:
        downsample_layers.extend([
            nn.AvgPool2d(kernel_size=stride, stride=stride),
            torchvision.models.resnet.conv1x1(self.inplanes, outplanes),
        ])
      else:
        downsample_layers.append(
            torchvision.models.resnet.conv1x1(self.inplanes, outplanes, stride)
        )

      downsample_layers.append(nn.BatchNorm2d(outplanes))
      downsample = nn.Sequential(*downsample_layers)

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                        base_width=self.base_width, l1_width=self._l1_width))
    self.inplanes = outplanes
    for _ in range(1, blocks):
      layers.append(block(self.inplanes, planes, groups=self.groups,
                          base_width=self.base_width, l1_width=self._l1_width))

    return nn.Sequential(*layers)

  def _forward_impl(self, x):
    # See note [TorchScript super()]
    x = self.input_stem(x)

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


@dataclasses.dataclass
class ModelParams:
  name: str
  l1_width: int = 20
  groups: int = None  # resnext
  width_per_group: int = None  # resnext
  zero_init_residual: bool = False
  use_333_input_conv: bool = False  # resnet-C
  pool_downsample_ident: bool = False  # resnet-D

def make_1chan_model(params: ModelParams, num_classes: int):
  name_to_fn = {
      'resnet18': make_resnet18_1chan,
      'resnet34': make_resnet34_1chan,
      'resnet50': make_resnet50_1chan,
      'resnet101': make_resnet101_1chan,
  }
  fn = name_to_fn[params.name]

  kwargs = {
      'num_classes': num_classes,
      'l1_width': params.l1_width,
      'zero_init_residual': params.zero_init_residual,
      'use_333_input_conv': params.use_333_input_conv,
      'pool_downsample_ident': params.pool_downsample_ident,
  }
  if params.groups is not None: kwargs['groups'] = params.groups
  if params.width_per_group is not None: kwargs['width_per_group'] = params.width_per_group

  return fn(**kwargs)


def make_resnet18_1chan(num_classes=1000, dropout=None, **kwargs):
  model = ResNet1Chan(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)
  maybe_apply_dropout(model, dropout)

  return model

def make_resnet34_1chan(num_classes=1000, dropout=None, **kwargs):
  model = ResNet1Chan(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
  maybe_apply_dropout(model, dropout)

  return model

def make_resnet50_1chan(num_classes=1000, dropout=None, **kwargs):
  model = ResNet1Chan(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
  maybe_apply_dropout(model, dropout)

  return model

def make_resnet101_1chan(num_classes=1000, dropout=None, **kwargs):
  model = ResNet1Chan(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)
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
  assert dat.shape[1] == 1

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
