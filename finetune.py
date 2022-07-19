from __future__ import print_function
import sys
sys.path.append(".")
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from utils.utils import setup_seed, remove_state_dict_module

from torchvision.models.resnet import resnet18, resnet50
import time
from utils.utils import AverageMeter, logger
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SubsetRandomSampler
import copy
from dataset.customDataset import Custom_Dataset
import torch.distributed as dist
from utils.utils import accuracy, get_imagenet_root_split
from utils.utils import cosine_annealing, get_cifar10_data_split, get_cifar100_data_split, \
    get_food101_data_split, get_EuroSAT_data_split, get_iNaturalist_sub1000_data_split
from collections import OrderedDict

from functools import partial
from dataset.cifar10 import subsetCIFAR10, subsetCIFAR100

from utils.utils import fix_backbone, DistillCrossEntropy

import sklearn.metrics
from utils.optimizer import LARS
from pdb import set_trace


parser = argparse.ArgumentParser(description='PyTorch Imagenet Training')
parser.add_argument('experiment', type=str, help='exp name')
parser.add_argument('--model', default='res50', type=str, help='model name')
parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer type')
parser.add_argument('--data', default='', type=str, help='path to data')
parser.add_argument('--dataset', default='CUB200', type=str, help='')
parser.add_argument('--save_dir', default='checkpoints_tune', type=str, help='path to save checkpoint')
parser.add_argument('--out_feature', default='', type=str, help='path to save the feature, if not empty, '
                                                                'enter the feature outputting mode')
parser.add_argument('--batch-size', type=int, default=48, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.2, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--checkpoint', default='', type=str,
                    help='path to resume model')
parser.add_argument('--checkpoint_pretrain', default='', type=str,
                    help='path to pretrained model')
parser.add_argument('--resume', action='store_true',
                    help='if resume training')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='the start epoch number')
parser.add_argument('--log-interval', default=50, type=int,
                    help='display interval')
parser.add_argument('--decreasing_lr', default='3,6,9', help='decreasing strategy')
parser.add_argument('--customSplit', type=str, default='', help='custom split for training')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--fc_seed', type=int, default=1, help='random seed for fc initalization')
parser.add_argument('--num_workers', type=int, default=10, help='num workers')
parser.add_argument("--cosineLr", action='store_true', help="if use cosine lr schedule")
parser.add_argument("--fixBackbone", action='store_true', help="if fix backbone")
parser.add_argument("--fixBackbone_cnt", type=int, default=1, help="how many layers are fixed")
parser.add_argument('--tuneFromFirstFC', action='store_true', help="tune from first fc layer")

# model read settings
parser.add_argument('--cvt_state_dict', action='store_true', help='use for ss model')
parser.add_argument('--cvt_state_dict_moco', action='store_true', help='use for model from moco')

# distributed training
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--test_freq', default=1, help="test freq", type=int)
parser.add_argument('--late_test', action="store_true")

# DnA
parser.add_argument('--pretrain_low_rank', action='store_true', help='if use low_rank for pretraining')
parser.add_argument('--pretrain_low_rank_r_ratio', default=0, type=float, help='the value of small rank r')
parser.add_argument('--pretrain_low_rank_fix', action='store_true',
                    help='if fix low rank part and only tune sparsity part')
parser.add_argument('--pretrain_low_rank_merge_to_std_model', action='store_true',
                    help='if fix low rank part and only tune sparsity part')
parser.add_argument('--pretrain_low_rank_keep_noise', action='store_true',
                    help='if keep noise in decomposition')
parser.add_argument('--pretrain_low_rank_consecutive', action='store_true',
                    help='if keep weight consecutive')
parser.add_argument('--pretrain_low_rank_lora', action='store_true',
                    help='if pretrain employs lora')

# options for distillation
parser.add_argument('--distillation', action='store_true', help='if use distillation')
parser.add_argument('--distillation_checkpoint', default="", type=str)
parser.add_argument('--distillation_temp', default=0.1, type=float)

args = parser.parse_args()

if args.local_rank == -1:
    args.local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])

# settings
model_dir = os.path.join(args.save_dir, args.experiment)
if not os.path.exists(model_dir) and args.local_rank == 0:
    os.system("mkdir -p {}".format(model_dir))
use_cuda = torch.cuda.is_available()
setup_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# distributed
dist.init_process_group(backend="nccl", init_method="env://")
torch.cuda.set_device(args.local_rank)

rank = torch.distributed.get_rank()
log = logger(os.path.join(model_dir), local_rank=rank)
log.info(str(args))

world_size = torch.distributed.get_world_size()
print("employ {} gpus".format(world_size))

assert args.batch_size % world_size == 0
batch_size = args.batch_size // world_size

if args.dataset == 'imagenet' or args.dataset == 'iNaturalist' or args.dataset == 'iNaturalist_sub1000' or \
        args.dataset == 'cifar10_large' or args.dataset == 'cifar100_large' or args.dataset == 'food-101' or \
        args.dataset == 'EuroSAT':
    # setup data loader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
    # setup data loader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
else:
    raise ValueError("dataset {} does not exist".format(args.dataset))

if args.dataset == 'imagenet':
    root, txt_train, txt_val, txt_test, pathReplaceDict = get_imagenet_root_split(args.data, args.customSplit)

    train_datasets = Custom_Dataset(root=root, txt=txt_train, transform=transform_train, pathReplace=pathReplaceDict)
    val_datasets = Custom_Dataset(root=root, txt=txt_val, transform=transform_test, pathReplace=pathReplaceDict)
    test_datasets = Custom_Dataset(root=root, txt=txt_test, transform=transform_test, pathReplace=pathReplaceDict)

elif args.dataset == 'food-101':
    root, txt_train, txt_val, txt_test = get_food101_data_split(args.data, args.customSplit)

    train_datasets = Custom_Dataset(root=root, txt=txt_train, transform=transform_train)
    val_datasets = Custom_Dataset(root=root, txt=txt_val, transform=transform_test)
    test_datasets = Custom_Dataset(root=root, txt=txt_test, transform=transform_test)

elif args.dataset == 'EuroSAT':
    root, txt_train, txt_val, txt_test = get_EuroSAT_data_split(args.data, args.customSplit)

    train_datasets = Custom_Dataset(root=root, txt=txt_train, transform=transform_train)
    val_datasets = Custom_Dataset(root=root, txt=txt_val, transform=transform_test)
    test_datasets = Custom_Dataset(root=root, txt=txt_test, transform=transform_test)

elif args.dataset == 'iNaturalist_sub1000':
    root, txt_train, txt_val, txt_test = get_iNaturalist_sub1000_data_split(args.data, args.customSplit)

    train_datasets = Custom_Dataset(root=root, txt=txt_train, transform=transform_train)
    val_datasets = Custom_Dataset(root=root, txt=txt_val, transform=transform_test)
    test_datasets = Custom_Dataset(root=root, txt=txt_test, transform=transform_test)

elif args.dataset == "cifar10" or args.dataset == "cifar10_large":
        # the data distribution
        root, train_idx, val_idx = get_cifar10_data_split(args.data, args.customSplit)

        train_idx = list(np.load(train_idx))
        val_idx = list(np.load(val_idx))
        train_datasets = subsetCIFAR10(root=root, sublist=train_idx, transform=transform_train, download=True)
        val_datasets = subsetCIFAR10(root=root, sublist=val_idx, transform=transform_test, download=True)
        test_datasets = subsetCIFAR10(root=root, sublist=[], train=False, transform=transform_test, download=True)

elif args.dataset == "cifar100" or args.dataset == "cifar100_large":
        # the data distribution
        root, train_idx, val_idx = get_cifar100_data_split(args.data, args.customSplit)

        train_idx = list(np.load(train_idx))
        val_idx = list(np.load(val_idx))
        train_datasets = subsetCIFAR100(root=root, sublist=train_idx, transform=transform_train, download=True)
        val_datasets = subsetCIFAR100(root=root, sublist=val_idx, transform=transform_test, download=True)
        test_datasets = subsetCIFAR100(root=root, sublist=[], train=False, transform=transform_test, download=True)

else:
    raise ValueError("dataset of {} is not supported".format(args.dataset))

train_sampler = torch.utils.data.distributed.DistributedSampler(train_datasets, shuffle=True)
val_sampler = torch.utils.data.distributed.DistributedSampler(val_datasets, shuffle=False)
test_sampler = torch.utils.data.distributed.DistributedSampler(test_datasets, shuffle=False)

train_loader = torch.utils.data.DataLoader(train_datasets, num_workers=args.num_workers, batch_size=batch_size,
                                           sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(val_datasets, num_workers=2, batch_size=batch_size,
                                         sampler=val_sampler)
test_loader = torch.utils.data.DataLoader(test_datasets, num_workers=2, batch_size=batch_size,
                                          sampler=test_sampler)


if args.dataset == 'imagenet':
    num_class = 1000
elif args.dataset == 'food-101':
    num_class = 101
elif args.dataset == 'EuroSAT':
    num_class = 10
elif args.dataset == 'iNaturalist':
    num_class = 8142
elif args.dataset == 'iNaturalist_sub1000':
    num_class = 1000
elif args.dataset == 'cifar10' or args.dataset == 'cifar10_large':
    num_class = 10
elif args.dataset == 'cifar100' or args.dataset == 'cifar100_large':
    num_class = 100
else:
    raise NotImplementedError("no such dataset: {}".format(args.dataset))

if rank == 0:
    class_stat = [0 for _ in range(num_class)]
    try:
        for target in train_datasets.labels:
            class_stat[target] += 1
    except:
        for target in train_datasets.targets:
            class_stat[target] += 1
    print("class distribution in training set is {}".format(class_stat))


def train(args, model, device, train_loader, optimizer, epoch, log, world_size, scheduler, model_teacher=None):
    model.train()

    dataTimeAve = AverageMeter()
    totalTimeAve = AverageMeter()
    end = time.time()

    if args.fixBackbone:
        fix_backbone(model, log, args.fixBackbone_cnt, verbose=(epoch==1))

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cosineLr:
            scheduler.step()

        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        dataTime = time.time() - end
        dataTimeAve.update(dataTime)

        optimizer.zero_grad()

        logits = model(data)
        if model_teacher is None:
            loss = F.cross_entropy(logits, target)
        else:
            teacher_logits = model_teacher.eval()(data)
            loss = DistillCrossEntropy(T=args.distillation_temp)(logits, teacher_logits)

        loss.backward()
        optimizer.step()

        totalTime = time.time() - end
        totalTimeAve.update(totalTime)
        end = time.time()
        # print progress
        if batch_idx % args.log_interval == 0:
            log.info(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tData time: {:.3f} ({:.3f})\tTotal time: {:.3f} ({:.3f})'.format(
                    epoch, (batch_idx * train_loader.batch_size + len(data)) * world_size, len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(), dataTimeAve.val, dataTimeAve.avg,
                    totalTimeAve.val, totalTimeAve.avg))


def eval_test(model, device, loader, log, world_size, prefix='test', dataset='imagenet',
              customSplit=None, num_class=1000, out_feature=''):
    model.eval()
    test_loss = 0
    correct = 0
    whole = 0

    top1_avg = AverageMeter()
    top5_avg = AverageMeter()
    model.eval()

    perClassAccRight = [0 for i in range(num_class)]
    perClassAccWhole = [0 for i in range(num_class)]

    if out_feature != '':
        output_all = []
        target_all = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True).long()
            output = model(data)

            output_list = [torch.zeros_like(output) for _ in range(world_size)]
            target_list = [torch.zeros_like(target) for _ in range(world_size)]
            torch.distributed.all_gather(output_list, output)
            torch.distributed.all_gather(target_list, target)
            output = torch.cat(output_list, dim=0)
            target = torch.cat(target_list, dim=0)

            if out_feature != '':
                output_all.append(output.cpu().numpy())
                target_all.append(target.cpu().numpy())
                continue

            pred = output.max(1)[1].long()
            for cntClass in torch.unique(target):
                perClassAccRight[cntClass] += pred[target == cntClass].eq(
                    target[target == cntClass].view_as(pred[target == cntClass])).sum().item()
                perClassAccWhole[cntClass] += len(target[target == cntClass])

            try:
                test_loss += F.cross_entropy(output, target, size_average=False).item()
            except:
                set_trace()
            # pred = output.max(1, keepdim=True)[1]
            # correct += pred.eq(target.view_as(pred)).sum().item()
            # whole += len(target)
            top1, top5 = accuracy(output, target, topk=(1, 5))
            top1_avg.update(top1, data.shape[0])
            top5_avg.update(top5, data.shape[0])

    if out_feature != '':
        output_all = np.concatenate(output_all)
        target_all = np.concatenate(target_all)
        np.save(out_feature, {"feature": output_all, "label": target_all})

    classWiseAcc = np.array(perClassAccRight) / np.array(perClassAccWhole)
    accPerClassStr = ""
    for i in range(num_class):
        accPerClassStr += "{:.04} ".format(classWiseAcc[i])
    log.info('acc per class is {}'.format(accPerClassStr))

    test_loss /= len(loader.dataset)
    log.info('{}: Average loss: {:.4f}, Accuracy: top1 ({:.2f}%) top5 ({:.2f}%)'.format(prefix,
                                                                                        test_loss, top1_avg.avg,
                                                                                        top5_avg.avg))
    return test_loss, top1_avg.avg, top5_avg.avg


def main():

    if args.model == 'res18':
        model_fun = resnet18
    elif args.model == 'res50':
        model_fun = resnet50
    else:
        assert False
    kwargs = {"num_classes": num_class}

    if args.pretrain_low_rank:
        from models.general_framework import resnet_frame
        from models.general_framework.convs.low_rank_conv2d import low_rank_conv2d
        assert args.model == "res50"
        conv_layer = partial(low_rank_conv2d, r_ratio=args.pretrain_low_rank_r_ratio,
                             fix_low_rank=args.pretrain_low_rank_fix, keep_noise=args.pretrain_low_rank_keep_noise,
                             reshape_consecutive=args.pretrain_low_rank_consecutive,
                             lora_mode=args.pretrain_low_rank_lora)
        model_fun = resnet_frame.__dict__["resnet50"]
        kwargs = dict(conv_layer=conv_layer, num_classes=num_class)

        if args.pretrain_low_rank_merge_to_std_model:
            model_low_rank = model_fun(**kwargs)
            # create a std model instead
            model_fun = resnet50
            kwargs = dict(num_classes=num_class)

    model = model_fun(**kwargs).cuda()

    if args.tuneFromFirstFC:
        if not args.cvt_state_dict_moco:
            from utils.proj_head import proj_head_simclr
            ch = model.fc.in_features
            model.fc = nn.Sequential(proj_head_simclr(ch, finetuneMode=True), nn.Linear(ch, num_class))
            if args.out_feature != '':
                model.fc = nn.Sequential(proj_head_simclr(ch, finetuneMode=True), nn.Identity())
            if args.pretrain_low_rank and args.pretrain_low_rank_merge_to_std_model:
                model_low_rank.fc = nn.Sequential(proj_head_simclr(ch, finetuneMode=True), nn.Linear(ch, num_class))
                if args.out_feature != '':
                    model_low_rank.fc = nn.Sequential(proj_head_simclr(ch, finetuneMode=True), nn.Identity())
        else:
            dim_mlp = model.fc.weight.shape[1]
            ch = model.fc.in_features
            model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.Linear(ch, num_class))
            if args.out_feature != '':
                model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.Identity())
            if args.pretrain_low_rank and args.pretrain_low_rank_merge_to_std_model:
                model_low_rank.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.Linear(ch, num_class))
                if args.out_feature != '':
                    model_low_rank.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.Identity())
    else:
        assert args.out_feature == ''
    
    if args.dataset == "cifar10" or args.dataset == "cifar100":
        log.info("remove maxpooling and enlarge conv layer for small resolution")
        model.conv1 = nn.Conv2d(3, model.conv1.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        if args.pretrain_low_rank and args.pretrain_low_rank_merge_to_std_model:
            model_low_rank.conv1 = nn.Conv2d(3, model.conv1.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            model_low_rank.maxpool = nn.Identity()

    process_group = torch.distributed.new_group(list(range(world_size)))
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)

    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank, find_unused_parameters=True)

    model_teacher = None
    if args.distillation:
        if args.pretrain_low_rank: assert args.pretrain_low_rank_merge_to_std_model
        assert args.distillation_checkpoint != ""
        checkpoint_distill = torch.load(args.distillation_checkpoint, map_location="cpu")
        model_teacher = model_fun(**kwargs).cuda()

        if args.dataset == "cifar10" or args.dataset == "cifar100":
            log.info("remove maxpooling and enlarge conv layer for small resolution for teacher")
            model_teacher.conv1 = nn.Conv2d(3, model_teacher.conv1.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            model_teacher.maxpool = nn.Identity()

        if args.tuneFromFirstFC:
            assert not args.cvt_state_dict_moco
            from utils.proj_head import proj_head_simclr
            ch = model_teacher.fc.in_features
            model_teacher.fc = nn.Sequential(proj_head_simclr(ch, finetuneMode=True), nn.Linear(ch, num_class))
        model_teacher.load_state_dict(remove_state_dict_module(checkpoint_distill["state_dict"]))
        model_teacher = model_teacher.cuda()
        for param in model_teacher.parameters():
            param.requires_grad = False

    if args.checkpoint_pretrain != '':
        checkpoint = torch.load(args.checkpoint_pretrain, map_location="cpu")
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'P_state' in checkpoint:
            state_dict = checkpoint['P_state']
        else:
            state_dict = checkpoint

        if args.tuneFromFirstFC:
            if args.out_feature != '':
                in_features = 100
            else:
                in_features = model.module.fc[1].in_features
        else:
            in_features = model.module.fc.in_features

        setup_seed(args.fc_seed)
        if args.cvt_state_dict:
            small_resolution = False
            if args.dataset == "cifar10" or args.dataset == "cifar100":
                small_resolution = True

            state_dict = cvt_state_dict(state_dict, in_features, num_class,
                                        small_resolution=small_resolution, tuneFromFirstFC=args.tuneFromFirstFC)

        if args.cvt_state_dict_moco:
            state_dict = cvt_state_dict_moco(state_dict, in_features, num_class,
                                             tuneFromFirstFC=args.tuneFromFirstFC)

        if args.pretrain_low_rank and args.pretrain_low_rank_merge_to_std_model:
            # only implement for res50
            assert not (args.model != 'res50')
            model_low_rank.load_state_dict(cvt_state_dict_remove_module(state_dict))
            log.info("low rank training merging")
            for module in model_low_rank.modules():
                if isinstance(module, low_rank_conv2d):
                    module.merge()
            state_dict = cvt_state_dict_low_rank_to_std(model_low_rank.state_dict())
            state_dict = cvt_state_dict_add_module(state_dict)

        if args.out_feature != '':
            del state_dict['module.fc.1.weight']
            del state_dict['module.fc.1.bias']

        model.load_state_dict(state_dict)

        log.info('read checkpoint {}'.format(args.checkpoint))

    if args.out_feature != '':
        eval_test(model, device, test_loader, log, world_size,
                  prefix='test', dataset=args.dataset, num_class=num_class, out_feature=args.out_feature)
        return

    if args.pretrain_low_rank:
        if not args.pretrain_low_rank_fix:
            log.info("low rank training merging")
            for module in model.modules():
                if isinstance(module, low_rank_conv2d):
                    module.merge()

    params = model.parameters()

    optimizer = setup_optimizer(args.optimizer, params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, log=log)
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

    if args.cosineLr:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(step,
                                                    args.epochs * len(train_loader),
                                                    1,  # since lr_lambda computes multiplicative factor
                                                    1e-6 / args.lr,
                                                    warmup_steps=0)
        )
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

    start_epoch = args.start_epoch

    best_prec1 = 0

    if args.resume:
        if os.path.isfile(os.path.join(model_dir, 'best_model.pt')):
            checkpoint = torch.load(os.path.join(model_dir, 'best_model.pt'))
            model.load_state_dict(checkpoint['state_dict'])

            _, top1_vali_tacc, top5_vali_tacc = eval_test(model, device, val_loader, log, world_size,
                                                          prefix='vali', dataset=args.dataset, num_class=num_class)
            best_prec1 = top1_vali_tacc
            log.info("previous best prec1 is {}".format(best_prec1))

        if args.checkpoint != '':
            checkpoint = torch.load(args.checkpoint, map_location="cpu")
        elif os.path.isfile(os.path.join(model_dir, 'model.pt')):
            checkpoint = torch.load(os.path.join(model_dir, 'model.pt'))
        else:
             checkpoint = None

        if checkpoint is not None:
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

            if 'epoch' in checkpoint and 'optim' in checkpoint:
                start_epoch = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optim'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                log.info("resume the checkpoint {} from epoch {}".format(args.checkpoint, checkpoint['epoch']))
            else:
                raise ValueError("checkpoint broken")

        else:
            log.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            log.info("no available checkpoint, start from scratch or pretrain!!!!!!!!!!!")
            log.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        log.info("current lr is {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        train_sampler.set_epoch(epoch)

        train(args, model, device, train_loader, optimizer, epoch, log, world_size=world_size, scheduler=scheduler, model_teacher=model_teacher)

        # adjust learning rate for SGD
        if not args.cosineLr:
            scheduler.step()

        if args.late_test:
            start_test_epoch = int(0.8 * (args.epochs + 1))
        else:
            start_test_epoch = 0

        if rank == 0:
            # save checkpoint
            save_dict = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim': optimizer.state_dict(),
                'best_prec1': best_prec1,
                'scheduler': scheduler.state_dict(),
            }

            torch.save(save_dict, os.path.join(model_dir, 'model.pt'))

        if epoch % args.test_freq == 0 and epoch >= start_test_epoch:
            # evaluation on natural examples
            log.info('================================================================')
            _, top1_vali_tacc, top5_vali_tacc = eval_test(model, device, val_loader, log, world_size, dataset=args.dataset, prefix='vali', num_class=num_class)
            log.info('================================================================')

            if rank == 0:
                if epoch % 50 == 0:
                    torch.save(save_dict, os.path.join(model_dir, 'model_{}.pt'.format(epoch)))

                is_best = top1_vali_tacc > best_prec1
                best_prec1 = max(top1_vali_tacc, best_prec1)

                if is_best:
                    torch.save(save_dict, os.path.join(model_dir, 'best_model.pt'))
            torch.distributed.barrier()

    checkpoint = torch.load(os.path.join(model_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['state_dict'])

    _, _, test_top5_tacc = eval_test(model, device, test_loader, log, world_size, dataset=args.dataset, num_class=num_class)
    log.info("On the best_model, test top5 tacc is {}".format(test_top5_tacc))


def cvt_state_dict(state_dict, in_features, num_class, small_resolution, tuneFromFirstFC=False):
    state_dict_new = copy.deepcopy(state_dict)

    name_to_del = []
    for name, item in state_dict_new.items():
        if 'normalize' in name:
            name_to_del.append(name)
        if 'fc' in name:
            if tuneFromFirstFC:
                if "1" not in name:
                    name_to_del.append(name)
            else:
                name_to_del.append(name)
        if 'save' in name:
            name_to_del.append(name)
    for name in np.unique(name_to_del):
        del state_dict_new[name]

    # add module
    if 'module' not in list(state_dict_new.keys())[0]:
        state_dict_with_module = OrderedDict()
        for key, item in state_dict_new.items():
            state_dict_with_module["module." + key] = item
        state_dict_new = state_dict_with_module

    # zero init fc
    if tuneFromFirstFC:
        state_dict = state_dict_new
        state_dict_new = OrderedDict()
        for name, item in state_dict.items():
            if "fc" not in name:
                state_dict_new[name] = item
            else:
                state_dict_new[name.replace("fc.", "fc.0.")] = item

        state_dict_new['module.fc.1.weight'] = torch.zeros(num_class, in_features).normal_(mean=0.0, std=0.01).to(
            state_dict_new['module.conv1.weight'].device)
        state_dict_new['module.fc.1.bias'] = torch.zeros(num_class).to(state_dict_new['module.conv1.weight'].device)
    else:
        state_dict_new['module.fc.weight'] = torch.zeros(num_class, in_features).normal_(mean=0.0, std=0.01).to(
            state_dict_new['module.conv1.weight'].device)
        state_dict_new['module.fc.bias'] = torch.zeros(num_class).to(state_dict_new['module.conv1.weight'].device)

    # adjust the weight of conv1
    if small_resolution:
        shape = state_dict_new['module.conv1.weight'].shape
        if shape[-1] != 3:
            out_shape = [3, 3]
            state_dict_new['module.conv1.weight'] = F.interpolate(state_dict_new['module.conv1.weight'], out_shape)

    return state_dict_new


def cvt_state_dict_moco(state_dict, in_features, num_classes, tuneFromFirstFC=False):
    # rename moco pre-trained keys
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and (not "fc" in k) and (not "normalize" in k):
            # remove prefix
            state_dict[k.replace("module.encoder_q.", "module.")] = state_dict[k]

        if tuneFromFirstFC and "fc.0" in k:
            state_dict[k.replace("module.encoder_q.", "module.")] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    if not tuneFromFirstFC:
        state_dict['module.fc.weight'] = torch.zeros(num_classes, in_features). \
            normal_(mean=0.0, std=0.01).to(list(state_dict.items())[0][1].device)
        state_dict['module.fc.bias'] = torch.zeros(num_classes).to(list(state_dict.items())[0][1].device)
    else:
        state_dict['module.fc.1.weight'] = torch.zeros(num_classes, in_features). \
            normal_(mean=0.0, std=0.01).to(list(state_dict.items())[0][1].device)
        state_dict['module.fc.1.bias'] = torch.zeros(num_classes).to(list(state_dict.items())[0][1].device)

    return state_dict


def cvt_state_dict_low_rank_to_std(state_dict):
    state_dict_new = OrderedDict()
    for k, v in state_dict.items():
        if not ("lora" in k or "sparse" in k or "noise" in k):
            state_dict_new[k] = v
    return state_dict_new


def cvt_state_dict_SupSup(state_dict):
    state_dict_new = OrderedDict()
    for k, v in state_dict.items():
        if "scores" not in k:
            state_dict_new[k] = v
    return state_dict_new


def cvt_state_dict_remove_module(state_dict):
    # remove module
    state_dict_new = OrderedDict()
    for key, item in state_dict.items():
        state_dict_new[key.replace("module.", "")] = item
    return state_dict_new


def cvt_state_dict_add_module(state_dict):
    # remove module
    state_dict_new = OrderedDict()
    for key, item in state_dict.items():
        state_dict_new["module." + key] = item
    return state_dict_new


def setup_optimizer(optimizer_type, params, lr, momentum, weight_decay, log):
    log.info('INFO: Creating Optimizer: [{}]  LR: [{:.8f}] Momentum : [{:.8f}] Weight Decay: [{:.8f}]'
             .format(optimizer_type, lr, momentum, weight_decay))

    if optimizer_type == 'adam':
        optimizer_fun = torch.optim.Adam
        optimizer_params = {"lr": lr}
    elif optimizer_type == 'lars':
        optimizer_fun = LARS
        optimizer_params = {"lr": lr, "weight_decay": weight_decay}
    elif optimizer_type == 'sgd':
        optimizer_fun = torch.optim.SGD
        optimizer_params = {"lr": lr, "weight_decay": weight_decay, "momentum": momentum}
    else:
        raise NotImplementedError("no defined optimizer: {}".format(optimizer_type))

    optimizer = optimizer_fun(params, **optimizer_params)

    return optimizer


if __name__ == '__main__':
    main()
