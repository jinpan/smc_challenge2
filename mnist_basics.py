import dataclasses
import random
import os
import typing

import fastai
import fastai.datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.tensorboard
import torchvision
import PIL


def fix_seeds(n):
    random.seed(n)
    torch.manual_seed(n)


def image_filename_to_tensor(filename):
    img = PIL.Image.open(filename)
    tensor = torch.tensor(np.array(img))

    return tensor.float() / 255


def mnist_distance(a, b):
    return (a-b).abs().mean(dim=(-2, -1))


def l1_loss(pred, target):
    return (pred-target).abs().mean()


def rmse_loss(pred, target):
    return ((pred-target)**2).mean().sqrt()


def main_old():
    path = fastai.datasets.untar_data(fastai.datasets.URLs.MNIST_SAMPLE)

    train_threes = torch.stack(list(map(image_filename_to_tensor, (path/'train/3').ls())))
    train_sevens = torch.stack(list(map(image_filename_to_tensor, (path/'train/7').ls())))

    avg_threes = train_threes.mean(dim=0)
    avg_sevens = train_sevens.mean(dim=0)

    valid_threes = torch.stack(list(map(image_filename_to_tensor, (path/'valid/3').ls())))
    valid_sevens = torch.stack(list(map(image_filename_to_tensor, (path/'valid/7').ls())))

    # print(f"#valid 3s: {len(valid_threes)} | #valid 7s: {len(valid_sevens)}")

    def is_3(x):
        return mnist_distance(x, avg_threes) < mnist_distance(x, avg_sevens)

    print(f"given 3, p(3): {is_3(valid_threes).float().mean()}")
    print(f"given 7, p(7): {1-is_3(valid_sevens).float().mean()}")

    train_x = torch.cat([train_threes, train_sevens]).view(-1, 28*28)
    train_y = torch.tensor([1.] * len(train_threes) + [0.] * len(train_sevens))[:, None]

    valid_x = torch.cat([valid_threes, valid_sevens]).view(-1, 28*28)
    valid_y = torch.tensor([1.] * len(valid_threes) + [0.] * len(valid_sevens))[:, None]

    print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape)


def main(tbw):
    path = fastai.datasets.untar_data(fastai.datasets.URLs.MNIST_SAMPLE)

    train_threes = torch.stack(list(map(image_filename_to_tensor, (path/'train/3').ls())))
    train_sevens = torch.stack(list(map(image_filename_to_tensor, (path/'train/7').ls())))

    valid_threes = torch.stack(list(map(image_filename_to_tensor, (path/'valid/3').ls())))
    valid_sevens = torch.stack(list(map(image_filename_to_tensor, (path/'valid/7').ls())))

    train_x = torch.cat([train_threes, train_sevens]).view(-1, 28*28)
    train_y = torch.tensor([1.] * len(train_threes) + [0.] * len(train_sevens))[:, None]

    valid_x = torch.cat([valid_threes, valid_sevens]).view(-1, 28*28)
    valid_y = torch.tensor([1.] * len(valid_threes) + [0.] * len(valid_sevens))[:, None]

    train_loader = DataLoader(train_x, train_y, batch_size=256)
    valid_loader = DataLoader(valid_x, valid_y, batch_size=256)

    num_hidden_layers = 30

    fix_seeds(10914)
    model = MyNet(num_hidden_layers)
    params = list(model.parameters())
    for p in params:
        print(p.shape, p.mean().item(), p.std().item())
    opt = BasicOptim(model.parameters(), lr=1e-3)
    learner = Learner(model, mnist_loss, opt, tbw, train_loader, valid_loader)

    images, labels = next(iter(train_loader))
    grid = torchvision.utils.make_grid(images.view(-1, 1, 28, 28))
    tbw.add_image('images', grid, 0)
    tbw.add_graph(model, images)

    learner.train_model(20)


class MyNet(torch.nn.Module):
    def __init__(self, num_hidden_layers):
        super(MyNet, self).__init__()
        self.w1 = torch.nn.Parameter(torch.randn((28*28, num_hidden_layers)))
        self.b1 = torch.nn.Parameter(torch.randn(num_hidden_layers))

        # self.linear1 = torch.nn.Linear(28*28, num_hidden_layers)
        self.linear2 = torch.nn.Linear(num_hidden_layers, 1)

    def forward(self, xb):
        res = xb @ self.w1 + self.b1
        # res = self.linear1(xb)
        res.relu_()
        return self.linear2(res)


@dataclasses.dataclass
class Learner:
    model: typing.Any
    loss_fn: typing.Any
    opt: typing.Any
    tbw: typing.Any

    train_loader: typing.Any
    valid_loader: typing.Any

    def train_epoch(self):
        for xb, yb in self.train_loader:
            preds = self.model(xb)
            loss = self.loss_fn(preds, yb)
            loss.backward()

            self.opt.step()
            self.opt.zero_grad()

    def train_model(self, epochs):
        for epoch in range(epochs):
            self.train_epoch()

            with torch.no_grad():
                valid_acc = self.validate_epoch()
                self.tbw.add_scalar('Accuracy/valid', valid_acc, epoch)
                print(round(valid_acc, 4))

    def batch_accuracy(self, xb, yb):
        preds = self.model(xb).sigmoid()
        correct = (preds > 0.5) == yb
        return correct.float().mean()

    def validate_epoch(self):
        accuracies = []
        for xb, yb in self.valid_loader:
            accuracies.append(self.batch_accuracy(xb, yb))
        return torch.stack(accuracies).mean().item()


@dataclasses.dataclass
class BasicOptim:
    params: typing.Any
    lr: typing.Any

    def __post_init__(self):
        self.params = list(self.params)

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.grad = None


def mnist_loss(preds, targets):
    preds.sigmoid_()
    return torch.where(targets == 1, 1-preds, preds).mean()


@dataclasses.dataclass
class DataLoader:
    collection_x: typing.Any
    collection_y: typing.Any
    batch_size: int = 5
    shuffle: bool = True

    def __post_init__(self):
        assert len(self.collection_x) == len(self.collection_y)

    def __iter__(self):
        self.batch_num = 0

        if self.shuffle:
            permutations = list(range(len(self.collection_x)))
            random.shuffle(permutations)

            self.collection_x = [self.collection_x[i] for i in permutations]
            self.collection_y = [self.collection_y[i] for i in permutations]
        return self

    def __next__(self):
        start = self.batch_num * self.batch_size
        if start > len(self.collection_x):
            raise StopIteration()

        self.batch_num += 1

        sl = slice(start, start+self.batch_size)

        xb = torch.stack(self.collection_x[sl])
        yb = torch.stack(self.collection_y[sl])
        return xb, yb


if __name__ == '__main__':
    os.system('clear')
    print("=== Starting! ===")

    hparam_dict = {
        'a': 1,
        'b': 3,
    }
    metric_dict = {
        'hparam/accuracy': 6,
    }
    with torch.utils.tensorboard.SummaryWriter(comment="__my-comment") as tbw:
        tbw.add_text("desc", "my fancy description")
        tbw.add_hparams(hparam_dict, metric_dict)
        main(tbw)

    print("=== Finished! ===")
