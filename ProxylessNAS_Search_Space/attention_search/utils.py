import os
import sys
import shutil
import numpy as np
import time, datetime
import torch
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import pickle
import torchvision.datasets as datasets
from collections import defaultdict
from config import config
import copy
import cv2
import random
import PIL
from PIL import Image
import math
import torchvision.transforms as transforms

def broadcast(args, obj, src, group=torch.distributed.group.WORLD, async_op=False):
    print('local_rank:{}, obj:{}'.format(args.local_rank, obj))
    obj_tensor = torch.from_numpy(np.array(obj)).cuda()
    torch.distributed.broadcast(obj_tensor, src, group, async_op)
    obj = obj_tensor.cpu().numpy()
    print('local_rank:{}, tensor:{}'.format(args.local_rank, obj))
    return obj

class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss

def get_optimizer_schedule(model, args):
    all_parameters = model.parameters()
    weight_parameters = []
    for pname, p in model.named_parameters():
      if p.ndimension() == 4 or 'classifier.0.weight' in pname or 'classifier.0.bias' in pname:
          weight_parameters.append(p)
    weight_parameters_id = list(map(id, weight_parameters))
    other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))
    optimizer = torch.optim.SGD(
        [{'params' : other_parameters},
        {'params' : weight_parameters, 'weight_decay' : args.weight_decay}],
        args.learning_rate,
        momentum=args.momentum,
        )

    delta_iters = args.total_iters / (1.-args.min_lr / args.learning_rate)
    print('delta_iters={}'.format(delta_iters))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/delta_iters), last_epoch=-1)
    return optimizer, scheduler

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].reshape(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res

def save_checkpoint(state, save):
  if not os.path.exists(save):
    os.makedirs(save)
  filename = os.path.join(save, 'checkpoint_epoch_{0}.pth.tar'.format(state['epoch']))
  torch.save(state, filename)
  print('Save CheckPoint....')
  last_checkpoint = os.path.join(save, 'checkpoint.pth.tar')
  shutil.copyfile(filename, last_checkpoint)


def get_location(s, key):
    d = defaultdict(list)
    for k,va in [(v,i) for i,v in enumerate(s)]:
        d[k].append(va)
    return d[key]

def merge_ops(rngs):
    cand = []
    for rng in rngs:
        for r in rng:
          if isinstance(r, list):
            cand += r
            cand += [-3]
          else:
            cand.append(r)
        cand += [-2]
    cand = cand[:-1]
    return cand

def split_ops_(cand_):
  if -3 in cand_:
    idxs_ = get_location(cand_, -3)
    idxs_ = idxs_[:-1]
    tmp = []
    last_idx_ = 0
    for idx_ in idxs_:
      tmp.append(cand_[last_idx_:idx_])
      last_idx_ = idx_ + 1
    tmp.append(cand_[last_idx_:-1])
    cand_ = tmp
  return cand_

def split_ops(cand):
    cand = list(cand)
    ops = []
    idxs = get_location(cand, -2)
    last_idx = 0
    for idx in idxs:
        cand_ = cand[last_idx:idx]
        cand_ = split_ops_(cand_)
        ops.append(cand_)
        last_idx = idx+1
    cand_ = cand[last_idx:]
    cand_ = split_ops_(cand_)
    ops.append(cand_)
    return ops

def get_search_space_size(operations):
  comb_num = 1
  for j in range(len(operations)):
      comb_num *= len(operations[j])
  return comb_num

class ImageNetWithRandomLabels(datasets.ImageFolder):
  """ImageNet dataset, with support for randomly corrupt labels.
  Params
  ------
  rand_seed: int
    Default 0. numpy random seed.
  num_classes: int
    Default 1000. The number of classes in the dataset.
  """
  def __init__(self, rand_seed=0, num_classes=1000, **kwargs):
    super(ImageNetWithRandomLabels, self).__init__(**kwargs)
    self.n_classes = num_classes
    self.rand_seed = rand_seed
    self.random_labels()

  def random_labels(self):
    labels = np.array(self.targets)
    print('num_classes:{}, random labels num:{}, random seed:{}'.format(self.n_classes, len(labels), self.rand_seed))
    np.random.seed(self.rand_seed)
    rnd_labels = np.random.randint(0, self.n_classes, len(labels))
    # we need to explicitly cast the labels from npy.int64 to
    # builtin int type, otherwise pytorch will fail...
    labels = [int(x) for x in rnd_labels]

    self.targets = labels

class RandomResizedCrop(object):

    def __init__(self, scale=(0.08, 1.0), target_size: int = 224, max_attempts: int = 10):
        assert scale[0] <= scale[1]
        self.scale = scale
        assert target_size > 0
        self.target_size = target_size
        assert max_attempts > 0
        self.max_attempts = max_attempts

    def __call__(self, img):
        assert isinstance(img, PIL.Image.Image)
        img = np.asarray(img, dtype=np.uint8)
        H, W, C = img.shape

        well_cropped = False
        for _ in range(self.max_attempts):
            crop_area = (H * W) * random.uniform(self.scale[0], self.scale[1])
            crop_edge = round(math.sqrt(crop_area))
            dH = H - crop_edge
            dW = W - crop_edge
            crop_left = random.randint(min(dW, 0), max(dW, 0))
            crop_top = random.randint(min(dH, 0), max(dH, 0))
            if dH >= 0 and dW >= 0:
                well_cropped = True
                break

        crop_bottom = crop_top + crop_edge
        crop_right = crop_left + crop_edge
        if well_cropped:
            crop_image = img[crop_top:crop_bottom, :, :][:, crop_left:crop_right, :]

        else:
            roi_top = max(crop_top, 0)
            padding_top = roi_top - crop_top
            roi_bottom = min(crop_bottom, H)
            padding_bottom = crop_bottom - roi_bottom
            roi_left = max(crop_left, 0)
            padding_left = roi_left - crop_left
            roi_right = min(crop_right, W)
            padding_right = crop_right - roi_right

            roi_image = img[roi_top:roi_bottom, :, :][:, roi_left:roi_right, :]
            crop_image = cv2.copyMakeBorder(roi_image, padding_top, padding_bottom, padding_left, padding_right,
                                            borderType=cv2.BORDER_CONSTANT, value=0)

        random.choice([1])
        target_image = cv2.resize(crop_image, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        target_image = PIL.Image.fromarray(target_image.astype('uint8'))
        return target_image


class LighteningJitter(object):

    def __init__(self, eigen_vecs, eigen_values, max_eigen_jitter=0.1):
        self.eigen_vecs = np.array(eigen_vecs, dtype=np.float32)
        self.eigen_values = np.array(eigen_values, dtype=np.float32)
        self.max_eigen_jitter = max_eigen_jitter

    def __call__(self, img):
        assert isinstance(img, PIL.Image.Image)
        img = np.asarray(img, dtype=np.float32)
        img = np.ascontiguousarray(img / 255)

        cur_eigen_jitter = np.random.normal(scale=self.max_eigen_jitter, size=self.eigen_values.shape)
        color_purb = (self.eigen_vecs @ (self.eigen_values * cur_eigen_jitter)).reshape([1, 1, -1])
        img += color_purb
        img = np.ascontiguousarray(img * 255)
        img.clip(0, 255, out=img)
        img = PIL.Image.fromarray(np.uint8(img))
        return img

def get_train_transform():
    eigvec = np.array([
        [-0.5836, -0.6948, 0.4203],
        [-0.5808, -0.0045, -0.8140],
        [-0.5675, 0.7192, 0.4009]
    ])

    eigval = np.array([0.2175, 0.0188, 0.0045])

    transform = transforms.Compose([
        RandomResizedCrop(target_size=224, scale=(0.08, 1.0)),
        LighteningJitter(eigen_vecs=eigvec[::-1, :], eigen_values=eigval,
                         max_eigen_jitter=0.1),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
    ])
    return transform


def reset_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        if torch.cuda.is_available():
            mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        else:
            mask = Variable(torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x
