import dataclasses
import pickle
import typing

import torch
import torch.utils.tensorboard
from torch import nn
import tqdm

from . import data

# Based on https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
# MIT license
class Lamb(torch.optim.Optimizer):
  def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0, adam=False):
    if not 0.0 <= lr:
      raise ValueError("Invalid learning rate: {}".format(lr))
    if not 0.0 <= eps:
      raise ValueError("Invalid epsilon value: {}".format(eps))
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
    defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    self.adam = adam
    super().__init__(params, defaults)

  def step(self, closure=None):
    """Performs a single optimization step.
    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    loss = None
    if closure is not None:
      loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None: continue
        grad = p.grad.data
        if grad.is_sparse:
          raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instad.')

        state = self.state[p]

        # State initialization
        if len(state) == 0:
          state['step'] = 0
          # Exponential moving average of gradient values
          state['exp_avg'] = torch.zeros_like(p.data)
          # Exponential moving average of squared gradient values
          state['exp_avg_sq'] = torch.zeros_like(p.data)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['betas']

        state['step'] += 1

        # Decay the first and second moment running average coefficient
        # m_t; exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        # v_t; exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

        # Paper v3 does not use debiasing.
        # bias_correction1 = 1 - beta1 ** state['step']
        # bias_correction2 = 1 - beta2 ** state['step']
        # Apply bias to lr to avoid broadcast.
        step_size = group['lr'] # * math.sqrt(bias_correction2) / bias_correction1

        weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)

        adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
        if group['weight_decay'] != 0:
          adam_step.add_(p.data, alpha=group['weight_decay'])

        adam_norm = adam_step.pow(2).sum().sqrt()
        if weight_norm == 0 or adam_norm == 0:
          trust_ratio = 1
        else:
          trust_ratio = weight_norm / adam_norm
        state['weight_norm'] = weight_norm
        state['adam_norm'] = adam_norm
        state['trust_ratio'] = trust_ratio
        if self.adam: trust_ratio = 1

        p.data.add_(adam_step, alpha=-step_size * trust_ratio)

    return loss


@dataclasses.dataclass
class Trainer:
  comment: str
  description: str

  cbed_data: data.CbedData
  model: nn.Module
  loss_fn: typing.Callable

  save_filename: str=''
  opt: typing.Optional[torch.optim.Optimizer] = None
  opt_sched: typing.Optional[torch.optim.lr_scheduler._LRScheduler] = None

  run_validation: bool = True

  def save(self):
    with open(f"{self.save_filename}.model", 'xb') as f:
      torch.save(self.model.state_dict(), f)
    with open(f"{self.save_filename}.label_manager", 'xb') as f:
      torch.save(self.cbed_data.label_manager.state_dict(), f)
    if self.opt:
      with open(f"{self.save_filename}.opt", 'xb') as f:
        torch.save(self.opt.state_dict(), f)


  @classmethod
  def load(self, model, save_filename):
    with open(f"{save_filename}.model", 'rb') as f:
      model.load_state_dict(torch.load(f))
    with open(f"{save_filename}.label_manager", 'rb') as f:
      label_manager = data.LabelManager.load(torch.load(f))

    return label_manager


  def train_model(self, epochs, lr, wd=0.01):
    if self.opt is None:
      self.opt = Lamb(self.model.parameters(), lr=lr, weight_decay=wd)
    if self.opt_sched is None:
      self.opt_sched = torch.optim.lr_scheduler.OneCycleLR(
          self.opt, max_lr=lr,
          steps_per_epoch=len(self.cbed_data.train_loader), epochs=epochs)

    with torch.utils.tensorboard.SummaryWriter(comment=f"__{self.comment}") as tbw:
      print(f"Logging to {tbw.log_dir}")
      tbw.add_text('desc', self.description)
      if self.save_filename:
        tbw.add_text('save', self.save_filename)

      for epoch in tqdm.tqdm(range(epochs)):
        self.model.train()
        self.train_epoch(tbw, epoch)

        if self.run_validation:
          self.model.eval()
          self.validate(tbw, epoch)

    self.save()

  def train_epoch(self, tbw, epoch):
    losses = []

    for xb, yb in self.cbed_data.train_loader:
      xb, yb = xb.cuda(), yb.cuda()

      preds = self.model(xb)
      loss = self.loss_fn(preds, yb)
      losses.append(loss)
      loss.backward()

      self.opt.step()
      self.opt.zero_grad()

      self.opt_sched.step()

    tbw.add_scalar('Loss/train', torch.stack(losses).mean().item(), epoch)


  @torch.no_grad()
  def validate(self, tbw, epoch):
    def loss(preds, yb):
      return self.loss_fn(preds, yb)
    def accuracy(preds, yb):
      preds = preds.ar

    topk = min(5, self.cbed_data.label_manager.num_classes - 1)
    topk_key = f'Accuracy/top_{topk}'

    metric_data = {
      'Loss/valid': [],
      'Accuracy/valid': [],
      topk_key: [],
    }

    for xb, yb in self.cbed_data.valid_loader:
      xb, yb = xb.cuda(), yb.cuda()

      preds = self.model(xb)

      metric_data['Loss/valid'].append(self.loss_fn(preds, yb))

      metric_data['Accuracy/valid'].append((preds.argmax(dim=1)==yb).float().mean())

      preds_topk = preds.topk(topk, dim=1).indices
      matches = (preds_topk == yb.view(-1, 1)).float().max(dim=1).values
      metric_data[topk_key].append(matches.float().mean())

    for name, vals in metric_data.items():
      avg_val = torch.stack(vals).mean().item()
      tbw.add_scalar(name, avg_val, epoch)

