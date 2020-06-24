import collections
import concurrent.futures
import dataclasses
import functools
import io
import multiprocessing
import pathlib
import random
import re
import typing

import cv2
import h5py
import IPython
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import PIL
import png
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import tqdm


def notify(msg):
    js = f'new Notification("{msg}")'
    return IPython.display.Javascript(js)

def fix_seeds(n):
    random.seed(n)
    torch.manual_seed(n)


ImageScalingConfig = collections.namedtuple('ImageScalingConfig', (
    'type',  # log, root
    'type_param1',  # for log, this is the floor.  for root, this is the kth root.
    'type_param2',  # for log and root, this is the multiplier.
    'size',  # (width, height)
    # TODO: parameters for middle out
))

ImageProducerConfig = collections.namedtuple('ImageProducerConfig', (
    'h5_root',
    'output_dir',

    'scale_config',  # ImageScalingConfig
))


class ImageScaler:
    def __init__(self, config):
        assert isinstance(config, ImageScalingConfig)
        self._config = config

        middleout_avg_slope = -0.029357852439725347
        image_center = (511/2, 511/2)
        self._middleout_mat = np.zeros((512, 512))
        for row in range(512):
            for col in range(512):
                dist_sq = (row - image_center[0]) ** 2 + (col - image_center[0]) ** 2
                dist = int(dist_sq ** 0.5)

                self._middleout_mat[row][col] = middleout_avg_slope * dist
    
    def scale(self, img):
        # Scale the image magnitudes
        if self._config.type == 'log':
            floor = self._config.type_param1
            scaled_img = np.maximum(floor, np.log(img)) - floor
            scaled_img *= self._config.type_param2
        elif self._config.type == 'log_middleout':
            log_img = np.log(img) - 0.5 * self._middleout_mat
            floor = self._config.type_param1
            scaled_img = np.maximum(floor, log_img) - floor
            scaled_img *= self._config.type_param2
        elif self._config.type == 'root':
            scaled_img = img ** (1. / self._config.type_param1)
            scaled_img *= self._config.type_param2
        else:
            raise RuntimeError(f"Invalid scaling config: {self._config.type}")

        scaled_img = np.clip(scaled_img, 0, 255)
        
        # Scale the image dimensions
        # cv2.resize documention recommends INTER_AREA for shrinking an image.
        scaled_img = cv2.resize(scaled_img, (self._config.size), interpolation=cv2.INTER_AREA)
        
        return scaled_img


class ImageProducer:
    def __init__(self, config):
        assert isinstance(config, ImageProducerConfig)
        self._config = config

        self._image_scaler = ImageScaler(config.scale_config)

    def _save_image(self, cbed_stack, filename):
        cbed_stack = [self._image_scaler.scale(cbed_slice) for cbed_slice in cbed_stack]

        flattened = np.concatenate(cbed_stack, axis=1)
        png_img = png.from_array(flattened.astype(np.uint8), 'L')

        if isinstance(filename, io.IOBase):
            png_img.write(filename)
        else:
            png_img.save(filename)

    def _save_image_separately(self, cbed_stack, filenames):
        for cbed_slice, filename in zip(cbed_stack, filenames):
            cbed_slice = self._image_scaler.scale(cbed_slice)

            png_img = png.from_array(cbed_slice.astype(np.uint8), 'L')

            if isinstance(filename, io.IOBase):
                png_img.write(filename)
            else:
                png_img.save(filename)

    def _process_filename(self, disk_semaphore, filename, combine_stack):
        m = re.match(r'^batch_(dev|train|test)_(\d{1,3}).h5$', filename.name)
        assert m is not None

        data_class = m.group(1)
        data_batch_id = m.group(2)

        output_dir = pathlib.Path(self._config.output_dir)
        rel_output_dir = {
            'dev': 'valid',
            'train': 'train',
            'test': 'test',
        }[data_class]
        output_dir = output_dir/rel_output_dir

        h5buf = io.BytesIO()
        with disk_semaphore:
            with open(filename, 'rb') as f:
                h5buf.write(f.read())
        h5buf.seek(0)
        
        ok_space_groups = [
            # for img11
            # '9', '164', '123', '186', '146', '7', '191', '33', '19', '74', '6', '141', '140',
            # '167', '189', '13', '71', '61', '160', '36', '136', '88', '60', '230', '64', '55',
            # '122', '176', '127', '58',
            # for img12
            # '14', '15',
        ]

        with h5py.File(h5buf, 'r') as h5f:
            for group in h5f.values():
                cbed = group['cbed_stack']
                space_group = group.attrs['space_group'].decode()
                
                if ok_space_groups and space_group not in ok_space_groups:
                    continue
                
                if combine_stack:
                    fn = output_dir/f'{data_batch_id}_{group.name[1:]}.{space_group}.png'
                    self._save_image(cbed, fn)
                else:
                    filenames = [
                        output_dir/f'{data_batch_id}_{group.name[1:]}.0.{space_group}.png',
                        output_dir/f'{data_batch_id}_{group.name[1:]}.1.{space_group}.png',
                        output_dir/f'{data_batch_id}_{group.name[1:]}.2.{space_group}.png',
                    ]
                    self._save_image_separately(cbed, filenames)

    def produce(self, combine_stack, parallelism=1):
        output_dir = pathlib.Path(self._config.output_dir)
        output_dir.mkdir()
        for subdir in ('train', 'valid', 'test'):
            (output_dir/subdir).mkdir()
        
        with open(output_dir/'config.txt', 'w') as f:
            f.write(str(self._config))

        path = pathlib.Path(self._config.h5_root)
        with concurrent.futures.ProcessPoolExecutor(parallelism) as exec:
            with multiprocessing.Manager() as m:
                disk_semaphore = m.Semaphore(1)
                futures = []

                for filename in path.glob('*/*.h5'):
                    futures.append(exec.submit(self._process_filename, disk_semaphore, filename, combine_stack))
                
                for _ in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    pass


# https://towerbabbel.com/go-defer-in-python/
def defer(func):
    @functools.wraps(func)
    def func_wrapper(*args, **kwargs):
        deferred = []
        try:
            return func(*args, defer=lambda f: deferred.append(f), **kwargs)
        finally:
            for fn in reversed(deferred):
                fn()
    return func_wrapper


class ModelOptBuf:
    def __init__(self):
        self.model = io.BytesIO()
        self.opt = io.BytesIO()
    
    def save(self, model, opt, filename=None):
        if filename is None:
            torch.save(model.state_dict(), self.model)
            torch.save(opt.state_dict(), self.opt)
        else:
            with open(filename+'.model', 'wb') as f:
                torch.save(model.state_dict(), f)
            with open(filename+'.opt', 'wb') as f:
                torch.save(opt.state_dict(), f)
    
    def load(self, model, opt, filename=None):
        if filename is None:
            buf.model.seek(0);
            model.load_state_dict(torch.load(buf.model))
            
            if opt is not None:
                buf.opt.seek(0);
                opt.load_state_dict(torch.load(buf.opt))
        else:
            with open(filename+'.model', 'rb') as f:
                model.load_state_dict(torch.load(f))

            if opt is not None:
                with open(filename+'.opt', 'rb') as f:
                    opt.load_state_dict(torch.load(f))


def save_to_buf(model, opt):
    buf = ModelOptBuf(io.BytesIO(), io.BytesIO())
    
    torch.save(model.state_dict(), buf.model)
    torch.save(opt.state_dict(), buf.opt)
    
    return buf

def load_from_buf(buf, model, opt):
    buf.model.seek(0);
    buf.opt.seek(0);
    
    model.load_state_dict(torch.load(buf.model))
    opt.load_state_dict(torch.load(buf.opt))


@defer
def find_lr(
    train_dl, model, loss_fn, opt,
    defer=None, lr_start=1e-7, lr_end=10., beta=0.98):

    # Save the model and optimizer, and Restore them at the function exit
    buf = save_to_buf(model, opt)
    defer(lambda: load_from_buf(buf, model, opt))

    learning_rates = []
    avg_loss = 0.
    losses = []

    # TODO: this should really be independent of the number of training batches
    # and be governed by some external parameter.
    num_iter = len(train_dl) - 1  # Skip last iteration

    for idx, (xb, yb) in enumerate(train_dl):
        lr = lr_start * ((lr_end / lr_start) ** (idx / (num_iter - 1)))
        learning_rates.append(lr)

        out = model(xb)
        loss = loss_fn(out, yb)

        # Although we could do this in post, compute this online to be consistent
        # with how optimizers must compute online averages.
        avg_loss = avg_loss * beta + float(loss) * (1 - beta)  # EWMA
        losses.append(avg_loss / (1 - (beta ** (1 + idx)))) # debias initialization

        loss.backward()
        with torch.no_grad():
            opt.lr = lr
            opt.step()
            opt.zero_grad()
        
        # TODO: some notion of early termination

    return learning_rates, losses


@dataclasses.dataclass
class Learner:
    model: typing.Any
    loss_fn: typing.Any
    tbw: typing.Any

    train_loader: typing.Any
    valid_loader: typing.Any
    
    base_epoch: int=0
    topk: int=5

    def train_epoch(self, epoch, opt_scheduler):
    # def train_epoch(self, epoch):
        losses = []
        
        for xb, yb in self.train_loader:
            xb, yb = xb.cuda(), yb.cuda()
            preds = self.model(xb)
            loss = self.loss_fn(preds, yb)
            losses.append(loss)
            loss.backward()

            self.opt.step()
            self.opt.zero_grad()
            
            opt_scheduler.step()

        self.tbw.add_scalar('Loss/train', torch.stack(losses).mean().item(), epoch)
    

    def train_model(self, epochs, lr, weight_decay=0.01):
        self.opt = Lamb2(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        opt_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.opt,
            max_lr=lr,
            steps_per_epoch=len(self.train_loader),
            epochs=epochs,
        )

        for epoch in tqdm.tqdm(range(epochs)):
            epoch += self.base_epoch
            
            self.model.train()
            self.train_epoch(epoch, opt_scheduler)

            self.model.eval()
            self.validate(epoch)

        self.base_epoch += epochs
    
    @torch.no_grad()
    def validate(self, epoch):
        def loss(preds, yb):
            return self.loss_fn(preds, yb)
        def accuracy(preds, yb):
            preds = preds.argmax(dim=1)
            return (preds==yb).float().mean()
        def top_k_accuracy(preds, yb):
            preds = preds.topk(self.topk, dim=1).indices
            matches = (preds == yb.view(-1, 1)).float().max(dim=1).values
            return matches.float().mean()
            
        validations = [
            ('Loss/valid', loss, []),
            ('Accuracy/valid', accuracy, []),
            ('Accuracy/valid_top_k', top_k_accuracy, []),
        ]
        
        for xb, yb in self.valid_loader:
            xb, yb = xb.cuda(), yb.cuda()
            preds = self.model(xb)
            
            for _, valid_fn, outs in validations:
                outs.append(valid_fn(preds, yb))
        
        for fn_name, _, outs in validations:
            avg_out = torch.stack(outs).mean().item()
            self.tbw.add_scalar(fn_name, avg_out, epoch)
                

    def validate_epoch(self):
        accuracies = []
        for xb, yb in self.valid_loader:
            xb, yb = xb.cuda(), yb.cuda()
            preds = self.model(xb).argmax(dim=1)
            accuracies.append((preds==yb).float().mean())
        return torch.stack(accuracies).mean().item()
    
    def validate_epoch2(self):
        accuracies = []
        for xb, yb in self.valid_loader:
            xb, yb = xb.cuda(), yb.cuda()
            preds = self.model(xb).topk(5, dim=1).indices
            matches = (preds == yb.view(-1, 1)).float().max(dim=1).values
            accuracies.append(matches.float().mean())
        return torch.stack(accuracies).mean().item()


class AdamOptimizer(torch.optim.Optimizer):
    def __init__(self,
                 parameters, lr,
                 betas=(0.9, 0.999), eps=1e-8, wd=0.
        ):
        self.parameters = list(parameters)

        defaults = dict(
            lr=lr, betas=betas, eps=eps, wd=wd)
        
        super(AdamOptimizer, self).__init__(self.parameters, defaults)
    
    @torch.no_grad()
    def step(self):
        for pg in self.param_groups:
            for p in pg['params']:
                if p.grad is None: continue
                grad = p.grad
                
                state = self.state[p]
                if len(state) == 0:  # need to initialize state
                    state['step'] = 0
                    state['mean_est'] = 0.
                    state['var_est'] = 0.
                    
                state['step'] += 1
                
                b1, b2 = pg['betas']

                state['mean_est'] = b1 * state['mean_est'] + (1-b1) * grad
                state['var_est'] = b2 * state['var_est'] + (1-b2) * (grad * grad)

                debiased_mean_est = state['mean_est'] / (1 - b1 ** state['step'])
                debiased_var_est = state['var_est'] / (1 - b2 ** state['step'])
                
                lr, eps = pg['lr'], pg['eps']

                if pg['wd']:
                    p -= lr * pg['wd'] * p
                
                p -= lr * debiased_mean_est / (debiased_var_est.sqrt() + eps)
                
                
class LambOptimizer(torch.optim.Optimizer):
    def __init__(self,
                 parameters, lr,
                 betas=(0.9, 0.999), eps=1e-8, phis=(.0, 10.), wd=0.,
                 use_adam=False,
        ):
        self.parameters = list(parameters)

        defaults = dict(
            lr=lr, betas=betas, eps=eps, phis=phis, wd=wd, use_adam=use_adam)
        
        # self.preclamped_norms1 = []
        # self.preclamped_norms2 = []
        # self.preclamped_norms3 = []
        
        super(LambOptimizer, self).__init__(self.parameters, defaults)
    
    @torch.no_grad()
    def step(self):
        for pg in self.param_groups:
            for p in pg['params']:
                if p.grad is None: continue
                grad = p.grad
                
                state = self.state[p]
                if len(state) == 0:  # need to initialize state
                    state['step'] = 0
                    state['mean_est'] = 0.
                    state['var_est'] = 0.
                    
                state['step'] += 1
                
                b1, b2 = pg['betas']

                state['mean_est'] = b1 * state['mean_est'] + (1-b1) * grad
                state['var_est'] = b2 * state['var_est'] + (1-b2) * (grad * grad)

                debiased_mean_est = state['mean_est'] / (1 - b1 ** state['step'])
                debiased_var_est = state['var_est'] / (1 - b2 ** state['step'])
                
                lr, eps = pg['lr'], pg['eps']
                
                r = debiased_mean_est / (debiased_var_est.sqrt() + eps)
                q = r
                if pg['wd']:
                    q.add_(pg['wd'] * p)
                
                if pg['use_adam']:
                    trust_factor = 1.
                else:
                    trust_factor = (p.norm() / q.norm()).clamp(*pg['phis'])
                    
                p -= lr * trust_factor * q

                
# from https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
class Lamb2(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0, adam=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        self.adam = adam
        super(Lamb2, self).__init__(params, defaults)

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
                if p.grad is None:
                    continue
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
                # m_t
                # exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                # exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                # Paper v3 does not use debiasing.
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
                # Apply bias to lr to avoid broadcast.
                step_size = group['lr'] # * math.sqrt(bias_correction2) / bias_correction1

                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    # adam_step.add_(group['weight_decay'], p.data)
                    adam_step.add_(p.data, alpha=group['weight_decay'])

                adam_norm = adam_step.pow(2).sum().sqrt()
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                # p.data.add_(-step_size * trust_ratio, adam_step)
                p.data.add_(adam_step, alpha=-step_size * trust_ratio)

        return loss
                
    
class LabelManager():
    def __init__(self):
        self.raw_labels_to_label_id = {}
        self.raw_labels = []
        self.frozen = False
    
    def freeze(self):
        self.frozen = True
    
    def get_label_from_raw(self, raw_label):
        if raw_label not in self.raw_labels_to_label_id:
            if self.frozen:
                return None
            
            self.raw_labels_to_label_id[raw_label] = len(self.raw_labels)
            self.raw_labels.append(raw_label)
        
        return self.raw_labels_to_label_id[raw_label]
    
    
def load_image(filepath, chans='RGB'):
    # TODO: load this in as a single channel
    img = PIL.Image.open(filepath).convert(chans)
    
    return torchvision.transforms.functional.to_tensor(img)


def imshow(image_t):
    np_img = image_t.numpy()
    num_chans = np_img.shape[0]
    if num_chans == 1:
        plt.imshow(
            np.transpose(np_img[0], (0,1)),
            interpolation='nearest', cmap='gray')
    else:
        plt.imshow(np.transpose(np_img, (1,2,0)), interpolation='nearest')


class CbedDataset(torch.utils.data.Dataset):
    def __init__(self, root, label_manager, transform, chans='RGB'):
        self.images = []
        self.labels = []
        self.transform = transform
        
        for filename in tqdm.tqdm(root.glob('*.png')):
            m = re.match(r'.*\.(\d{1,3})\.png$', filename.name)
            raw_label = m.group(1)
            
            label = label_manager.get_label_from_raw(raw_label)
            if label is None:
                continue

            self.labels.append(label)
            
            img = PIL.Image.open(filename).convert(chans)
            self.images.append(img)

        super(CbedDataset, self).__init__()
    
    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        
        return self.transform(img), label

    def __len__(self):
        return len(self.images)


class CbedDataLoader:
    def __init__(self, img_path, batch_size=500, chans='RGB'):
        img_path = pathlib.Path(img_path)
        
        self.label_manager = LabelManager()
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(360., resample=PIL.Image.BICUBIC),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomErasing(p=0.8),
        ])
        self._train_set = CbedDataset(
            img_path/'train', self.label_manager, transform=train_transform, chans=chans)
        self.label_manager.freeze()
        
        valid_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        self._valid_set = CbedDataset(
            img_path/'valid', self.label_manager, valid_transform, chans=chans)
        
        self.train_loader = torch.utils.data.DataLoader(
            self._train_set,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
        )
        self.valid_loader = torch.utils.data.DataLoader(
            self._valid_set,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )
    

class AdaptiveConcatPool2d(torch.nn.Module):
    def __init__(self):
        super(AdaptiveConcatPool2d, self).__init__()
        output_size = 1
        self.ap = torch.nn.AdaptiveAvgPool2d(output_size)
        self.mp = torch.nn.AdaptiveMaxPool2d(output_size)
       
    def forward(self, xb):
        return torch.cat([self.mp(xb), self.ap(xb)], 1)


class Flatten(torch.nn.Module):
    def forward(self, xb):
        # bs = xb.size(0)
        # return xb.view(bs, -1)
        return xb.view(xb.size(0), -1)


def initialize_layers(model, base_init_fn):
    def init_fn(m):
        if isinstance(m, (torch.nn.BatchNorm1d, )): return
        try:
            first_param = next(iter(m.parameters()))
        except StopIteration: return  # no parameters
        if not first_param.requires_grad:
            return
        
        # Cribbed from fastai.  Ideally, we should not rely on the names of attributes
        # but it's unclear how to do this more generally (or if it's even necessary)
        if hasattr(m, 'weight'):
            base_init_fn(m.weight)
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0.)
        
    to_init = [model]
    while to_init:
        m = to_init.pop(0)
        if isinstance(m, torch.nn.Module):
            init_fn(m)
        to_init.extend(m.children())


def make_cnn_model(
    base_arch, num_classes,
    base_init_fn=torch.nn.init.kaiming_normal_,
    pretrained=True,
):
    
    body = base_arch(pretrained=pretrained)
    # Drop the last 2 layers from the resnet arch
    # TODO: assert that base_arch is a resnet arch
    body = torch.nn.Sequential(*list(body.children())[:-2])
    
    lin_filters = [1024, 512, num_classes]
    ps = [0.25, 0.5]
    actns = [torch.nn.ReLU(inplace=True), None]
    
    layers = [AdaptiveConcatPool2d(), Flatten()]
    for num_in, num_out, p, actn in zip(
        lin_filters[:-1], lin_filters[1:], ps, actns):
        layers.append(torch.nn.BatchNorm1d(num_in))
        layers.append(torch.nn.Dropout(p))
        layers.append(torch.nn.Linear(num_in, num_out))
        if actn is not None:
            layers.append(actn)
    
    head = torch.nn.Sequential(*layers)
    
    initialize_layers(head, base_init_fn)
    
    return torch.nn.Sequential(body, head)

def make_cnn_model1d(
    base_arch, num_classes,
    base_init_fn=torch.nn.init.kaiming_normal_,
    pretrained=True,
):
    
    body = base_arch(pretrained=pretrained)
    # Drop the last 2 layers from the resnet arch
    # TODO: assert that base_arch is a resnet arch
    body = torch.nn.Sequential(*list(body.children())[:-2])
    
    lin_filters = [320, 160, num_classes]
    ps = [0.25, 0.5]
    actns = [torch.nn.ReLU(inplace=True), None]
    
    layers = [AdaptiveConcatPool2d(), Flatten()]
    for num_in, num_out, p, actn in zip(
        lin_filters[:-1], lin_filters[1:], ps, actns):
        layers.append(torch.nn.BatchNorm1d(num_in))
        layers.append(torch.nn.Dropout(p))
        layers.append(torch.nn.Linear(num_in, num_out))
        if actn is not None:
            layers.append(actn)
    
    head = torch.nn.Sequential(*layers)
    
    initialize_layers(head, base_init_fn)

    return torch.nn.Sequential(body, head)


def freeze_body(model):
    # Freeze the body of the mode, excluding batchnorm units
    assert len(model) == 2
    
    stack = [model[0]]
    while stack:
        m = stack.pop(0)
        children = list(m.children())
        stack.extend(children)
        
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            continue
        
        if len(children) == 0:  # leaf node
            for p in m.parameters():
                p.requires_grad = False
                
def unfreeze_body(model):
    for p in model.parameters():
        p.requires_grad = True

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=20, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
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

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=20, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
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
        super(ResNet1Chan, self).__init__()
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
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
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
    
def make_resnet18_1chan(pretrained=False, num_classes=1000):
    assert pretrained == False
    return ResNet1Chan(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def make_resnet34_1chan(pretrained=False, num_classes=1000,
                        dropout=None):
    assert pretrained == False
    model = ResNet1Chan(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    
    if dropout is not None:
        linear = model.fc
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            linear,
        )
    
    return model

def make_resnet50_1chan(pretrained=False, num_classes=1000):
    assert pretrained == False
    return ResNet1Chan(
        Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


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

def test_validation_set(img_path, model, label_manager, filedir='valid', rotate_deg=5, topk=5):
    grouped_filenames = collections.defaultdict(list)
    group_to_spacegroup = {}

    for filename in img_path.glob(f'{filedir}/*'):
        sample, _, space_group, _ = filename.name.split('.')
        grouped_filenames[sample].append(filename.name)
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
                tensors.append(f.result())

            model.eval()
            with torch.no_grad():
                out = model(torch.stack(tensors).cuda())
                argmax_topk = F.softmax(out, dim=1).mean(dim=0).topk(k=topk)
                # argmax_topk = F.softmax(out, dim=1).log().mean(dim=0).topk(k=topk)

            cat = label_manager.raw_labels[argmax_topk.indices[0]]
            
            if str(cat) == space_group:
                num_correct += 1
                    
            topk_cats = []
            for idx in argmax_topk.indices:
                topk_cats.append(label_manager.raw_labels[idx])
                    
            outcomes.append(PredictionOutcome(
                sample=group,
                actual=space_group,
                predicted=cat,
                topk_values=argmax_topk.values.tolist(),
                topk_cats=topk_cats
            ))

            accuracy = 100 * (num_correct / len(outcomes))
            it.set_description(f"Acc: {accuracy:.02f}%")

    print(num_correct, len(grouped_filenames))

    return outcomes


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        
    @staticmethod
    def reduce_loss(loss, reduction='mean'):
        return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss
    
    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        
        return nll * (1-self.eps) + loss * (self.eps)
