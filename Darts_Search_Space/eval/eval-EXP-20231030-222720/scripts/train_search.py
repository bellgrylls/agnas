import os
import sys
import time
import glob
import numpy as np
from numpy import random
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model_search import Network
from genotypes import PRIMITIVES
from genotypes import Genotype, Normal_Genotype, Reduce_Genotype

from torchvision.datasets import ImageFolder
import torchvision.transforms as T

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/sdb_new/wz/dataset/ISIC-2017', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels') 
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.7, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
   format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 3


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  logging.info(model)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  # train_transform, valid_transform = utils._data_transforms_cifar10(args)
  data_transform = {
        "train": T.Compose([
            T.Resize(size=(256, 256)),
            T.RandomResizedCrop(224),
            T.ToTensor(),
            T.Normalize(mean=127.5, std=127.5)]),
        "val": T.Compose([
            T.Resize(size=(256, 256)),
            T.RandomResizedCrop(224),
            T.ToTensor(),
            T.Normalize(mean=127.5, std=127.5)])}


  train_data = ImageFolder(args.data+'/train', transform=data_transform["train"])

  num_train = len(train_data)
  print("train-num:",num_train)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)


  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, criterion, optimizer, lr)
    logging.info('train_acc %f', train_acc)

    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, criterion, optimizer, lr):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  op_Attention_normal_all = []
  op_Attention_reduce_all = []

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = input.cuda()
    target = target.cuda(non_blocking=True)

    input_search, target_search = next(iter(valid_queue))
    input_search = input_search.cuda()
    target_search = target_search.cuda(non_blocking=True)

    optimizer.zero_grad()
    logits, op_Attention_normal, op_Attention_reduce = model(input)
    loss = criterion(logits, target)

    op_Attention_normal_all = np.sum([op_Attention_normal_all, op_Attention_normal], axis=0)
    op_Attention_reduce_all = np.sum([op_Attention_reduce_all, op_Attention_reduce], axis=0)
    
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1  = utils.accuracy(logits, target, topk=(1))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f', step, objs.avg, top1.avg)

      # logging.info('op_Attention_normal_all %s', op_Attention_normal_all)
      # logging.info('op_Attention_reduce_all %s', op_Attention_reduce_all)
      normal_genotype_0 = Normal_Genotype(normal=_parse(op_Attention_normal_all[0]), normal_concat=range(2, 6))
      normal_genotype_1 = Normal_Genotype(normal=_parse(op_Attention_normal_all[1]), normal_concat=range(2, 6))
      normal_genotype_2 = Normal_Genotype(normal=_parse(op_Attention_normal_all[2]), normal_concat=range(2, 6))
      normal_genotype_3 = Normal_Genotype(normal=_parse(op_Attention_normal_all[3]), normal_concat=range(2, 6))
      # normal_genotype_4 = Normal_Genotype(normal=_parse(op_Attention_normal_all[4]), normal_concat=range(2, 6))
      # normal_genotype_5 = Normal_Genotype(normal=_parse(op_Attention_normal_all[5]), normal_concat=range(2, 6))
      reduce_genotype_0 = Reduce_Genotype(reduce=_parse(op_Attention_reduce_all[0]), reduce_concat=range(2, 6))
      reduce_genotype_1 = Reduce_Genotype(reduce=_parse(op_Attention_reduce_all[1]), reduce_concat=range(2, 6))
      logging.info('Cell_0 = %s', normal_genotype_0)
      logging.info('Cell_1 = %s', normal_genotype_1)
      logging.info('Cell_2 = %s', reduce_genotype_0)
      logging.info('Cell_3 = %s', normal_genotype_2)
      logging.info('Cell_4 = %s', normal_genotype_3)
      logging.info('Cell_5 = %s', reduce_genotype_1)
      # logging.info('Cell_6 = %s', normal_genotype_4)
      # logging.info('Cell_7 = %s', normal_genotype_5)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  
  model.eval()

  op_Attention_normal_all = []
  op_Attention_reduce_all = []
  for step, (input, target) in enumerate(valid_queue):
    input = input.cuda()
    target = target.cuda(non_blocking=True)
    with torch.no_grad():
      logits, op_Attention_normal, op_Attention_reduce = model(input)
      loss = criterion(logits, target)
    
      op_Attention_normal_all = np.sum([op_Attention_normal_all, op_Attention_normal], axis=0)
      op_Attention_reduce_all = np.sum([op_Attention_reduce_all, op_Attention_reduce], axis=0)

    prec1 = utils.accuracy(logits, target, topk=(1))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f', step, objs.avg, top1.avg)

      # logging.info('op_Attention_normal_all %s', op_Attention_normal_all)
      # logging.info('op_Attention_reduce_all %s', op_Attention_reduce_all)
      normal_genotype_0 = Normal_Genotype(normal=_parse(op_Attention_normal_all[0]), normal_concat=range(2, 6))
      normal_genotype_1 = Normal_Genotype(normal=_parse(op_Attention_normal_all[1]), normal_concat=range(2, 6))
      normal_genotype_2 = Normal_Genotype(normal=_parse(op_Attention_normal_all[2]), normal_concat=range(2, 6))
      normal_genotype_3 = Normal_Genotype(normal=_parse(op_Attention_normal_all[3]), normal_concat=range(2, 6))
      # normal_genotype_4 = Normal_Genotype(normal=_parse(op_Attention_normal_all[4]), normal_concat=range(2, 6))
      # normal_genotype_5 = Normal_Genotype(normal=_parse(op_Attention_normal_all[5]), normal_concat=range(2, 6))
      reduce_genotype_0 = Reduce_Genotype(reduce=_parse(op_Attention_reduce_all[0]), reduce_concat=range(2, 6))
      reduce_genotype_1 = Reduce_Genotype(reduce=_parse(op_Attention_reduce_all[1]), reduce_concat=range(2, 6))
      logging.info('Cell_0 = %s', normal_genotype_0)
      logging.info('Cell_1 = %s', normal_genotype_1)
      logging.info('Cell_2 = %s', reduce_genotype_0)
      logging.info('Cell_3 = %s', normal_genotype_2)
      logging.info('Cell_4 = %s', normal_genotype_3)
      logging.info('Cell_5 = %s', reduce_genotype_1)
      # logging.info('Cell_6 = %s', normal_genotype_4)
      # logging.info('Cell_7 = %s', normal_genotype_5)

  return top1.avg, objs.avg


def _parse(Atten_weights):
  gene = []
  start =0
  n = 2
  for i in range (4):
    end = start + n
    A = Atten_weights[start:end].copy()
    edges = sorted(range(i + 2), key=lambda x: -max(A[x][k] for k in range(len(A[x])) if k != PRIMITIVES.index('none')))[:2]
    for j in edges:
      k_best = None
      for k in range(len(A[j])):
        if k != PRIMITIVES.index('none'):
          if k_best is None or A[j][k] > A[j][k_best]:
            k_best = k
      gene.append((PRIMITIVES[k_best], j))  
    start = end
    n += 1
  return gene


if __name__ == '__main__':
  main() 

