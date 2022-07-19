# !/usr/bin/env python
import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np

from utils.loader import TwoCropsTransform, GaussianBlur

from utils.utils import get_imagenet_root_split, get_cifar10_data_split, get_cifar100_data_split, \
     get_food101_data_split,  get_EuroSAT_data_split, remove_state_dict_module, check_and_cvt_pretrain_type, \
     get_iNaturalist_sub1000_data_split, logger
from dataset.customDataset import Custom_Dataset
from dataset.cifar10 import subsetCIFAR10, subsetCIFAR100

from utils.optimizer import LARS
from utils.proj_head import proj_head_simclr


from functools import partial

from utils.utils import AverageMeter
from utils.utils import nt_xent, gather_features


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('experiment', type=str)
parser.add_argument('--save_dir', type=str, default="checkpoints_moco")
parser.add_argument('--data', metavar='DIR', default='',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--checkpoint_pretrain', default='', type=str,
                    help='the pretrained contrastive learning model. ')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--save_freq', default=100, type=int)
parser.add_argument('--optimizer', default="sgd", type=str)

# parallel training
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')


# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--color-jitter-strength', default=1.0, type=float,
                    help='augmentation color jittering strength')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--simclr-t', default=0.2, type=float,
                    help='softmax temperature (default: 0.2)')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

# options for simclr
parser.add_argument('--mlpout', default=128, type=int,
                    help='the output dimension of simclr')

# options about dataset
parser.add_argument('--dataset', default='ImageNet', type=str,
                    help='the dataset to employ')
parser.add_argument('--customSplit', default='', type=str,
                    help='custom split for training')

# options for low rank network
parser.add_argument('--low_rank', action='store_true', help='if use DnA during pre-training')
parser.add_argument('--low_rank_r_ratio', default=0, type=float, help='the value of small rank r')
parser.add_argument('--low_rank_alpha', default=10.0, type=float, help='the ratio ')
parser.add_argument('--low_rank_fix_sparse', action='store_true', help='if fix s when tunning low rank')
parser.add_argument('--low_rank_fix_low_rank', action='store_true', help='if fix U, V when tunning low rank')
parser.add_argument('--low_rank_tune_V', action='store_true', help='if only tune V')
parser.add_argument('--low_rank_tune_U', action='store_true', help='if only tune U')
parser.add_argument('--low_rank_tune_V_S', action='store_true', help='if only tune V and S')
parser.add_argument('--low_rank_tune_U_S', action='store_true', help='if only tune U and S')
parser.add_argument('--low_rank_tune_all', action='store_true', help='tune U V while fixing mask of S')
parser.add_argument('--low_rank_compress_step', default=1000, type=int, help='the step number for compressing')
parser.add_argument('--low_rank_lambda_s', default=0.01, type=float, help='the value of sparse threshold')
parser.add_argument('--low_rank_sparse_ratio', default=-1, type=float,
                    help='if sparse ratio is specified, we globally pick the largest r% weights and set it as sparse')
parser.add_argument('--low_rank_UV_lr_ratio', default=1, type=float, help='the lr employed for low rank part '
                                                                          'compared to sparse part')
parser.add_argument('--low_rank_only_decompose', action='store_true', help='decompose')
parser.add_argument('--low_rank_keep_noise', action='store_true', help='if keep the noise')
parser.add_argument('--low_rank_reshape_consecutive', action='store_true', help='if use reshape consecutive')
parser.add_argument('--low_rank_decompose_no_s', action='store_true', help='if use decompose without s')
parser.add_argument('--low_rank_lora_mode', action='store_true', help='if use lora mode for low rank')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.local_rank == -1:
        args.local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])

    print("distributing")
    dist.init_process_group(backend="nccl", init_method="env://")
    print("paired")

    torch.cuda.set_device(args.local_rank)

    gloal_rank = torch.distributed.get_rank()
    args.gpu = gloal_rank

    world_size = torch.distributed.get_world_size()
    print("employ {} gpus in total".format(world_size))

    logName = "log.txt"
    save_dir = os.path.join(args.save_dir, args.experiment)
    if not os.path.exists(save_dir):
        os.system("mkdir -p {}".format(save_dir))
    log = logger(path=save_dir, local_rank=gloal_rank, log_name=logName)
    log.info(str(args))

    main_worker(args.local_rank, gloal_rank, world_size, save_dir, log, args)


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


def main_worker(local_rank, global_rank, world_size, save_dir, log, args):

    # prepare dataset
    train_dataset = init_dataset(args, log)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    args.batch_size = int(args.batch_size / world_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    # create model
    log.info("=> creating model '{}'".format(args.arch))

    if args.low_rank:
        from models.general_framework import resnet_frame
        from models.general_framework.convs.low_rank_conv2d import low_rank_conv2d
        conv_layer = partial(low_rank_conv2d, lora_alpha=args.low_rank_alpha, r_ratio=args.low_rank_r_ratio,
                             fix_sparse=args.low_rank_fix_sparse, fix_low_rank=args.low_rank_fix_low_rank,
                             tune_U=args.low_rank_tune_U, tune_V=args.low_rank_tune_V,
                             tune_V_S=args.low_rank_tune_V_S, tune_U_S=args.low_rank_tune_U_S,
                             tune_all=args.low_rank_tune_all,
                             keep_noise=args.low_rank_keep_noise,
                             reshape_consecutive=args.low_rank_reshape_consecutive,
                             decompose_no_s=args.low_rank_decompose_no_s, lora_mode=args.low_rank_lora_mode)
        model = resnet_frame.__dict__[args.arch](conv_layer=conv_layer)
    else:
        from models.origin import resnet
        model = resnet.__dict__[args.arch]()

    in_dim = model.fc.in_features  # 2048
    model.fc = proj_head_simclr(in_dim, output_cnt=args.mlpout)

    if args.dataset == "cifar10" or args.dataset == "cifar100":
        log.info("remove maxpooling and enlarge conv layer for small resolution")
        model.conv1 = nn.Conv2d(3, model.conv1.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    process_group = torch.distributed.new_group(list(range(world_size)))
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)

    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    torch.cuda.set_device(args.local_rank)
    model.cuda(args.local_rank)
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank,
                                                      find_unused_parameters=False)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.local_rank)

    if args.checkpoint_pretrain != '':
        checkpoint = torch.load(args.checkpoint_pretrain, map_location="cpu")
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'P_state' in checkpoint:
            state_dict = checkpoint['P_state']
        else:
            state_dict = checkpoint

        state_dict = remove_state_dict_module(state_dict)
        state_dict = check_and_cvt_pretrain_type(state_dict, model.module.state_dict(), log)

    model_dict = model.module.state_dict()
    ori_model_keys_num = model_dict.keys().__len__()

    if args.dataset == "cifar10" or args.dataset == "cifar100":
        shape = state_dict['conv1.weight'].shape
        if shape[-1] != 3:
            out_shape = [3, 3]
            state_dict['conv1.weight'] = F.interpolate(state_dict['conv1.weight'], out_shape)

    overlap_state_dict = {k: v for k, v in state_dict.items() if k in model_dict.keys()}
    overlap_keys_num = overlap_state_dict.keys().__len__()

    model_dict.update(overlap_state_dict)

    model.module.load_state_dict(model_dict)

    log.info("Load SimCLR Pre-trained Model! [{}/{}]"
             .format(overlap_keys_num, ori_model_keys_num))

    log.info('read pretrain model {}'.format(args.checkpoint_pretrain))

    if args.low_rank:
        from low_rank import prepare_low_rank
        params = prepare_low_rank(model, args.low_rank_compress_step, args.low_rank_lambda_s,
                                  args.low_rank_r_ratio, args.checkpoint_pretrain, args.low_rank_keep_noise, log,
                                  args.dataset, args.lr * args.low_rank_UV_lr_ratio, args.low_rank_reshape_consecutive,
                                  args.low_rank_decompose_no_s, args.low_rank_lora_mode, args.low_rank_sparse_ratio)
        if args.low_rank_only_decompose:
            return
    else:
        params = model.parameters()

    optimizer = setup_optimizer(args.optimizer, params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, log=log)

    if args.resume:
        if os.path.isfile(os.path.join(save_dir, 'checkpoint.pth.tar')):
            log.info("=> loading checkpoint '{}'".format(os.path.join(save_dir, 'checkpoint.pth.tar')))
            if args.gpu is None:
                checkpoint = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.local_rank)
                checkpoint = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'), map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            log.info("=> loaded checkpoint '{}' (epoch {})"
                     .format(args.resume, checkpoint['epoch']))
        else:
            log.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            log.info("no available checkpoint, start from scratch!!!!!!!!!!!")
            log.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args, log)

        train_simclr(train_loader, model, optimizer, epoch, log, args, local_rank, world_size)

        if global_rank == 0:
            save_dict = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            save_checkpoint(save_dict, is_best=False,
                            filename=os.path.join(save_dir, 'checkpoint.pth.tar'.format(epoch + 1)))

            if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
                save_checkpoint(save_dict, is_best=False,
                                filename=os.path.join(save_dir, 'checkpoint_{}.pth.tar'.format(epoch + 1)))

            # remove checkpoint for resuming after training finished
            if (epoch + 1) == args.epochs and args.save_freq > 800:
                os.system("rm {}".format(os.path.join(save_dir, 'checkpoint.pth.tar'.format(epoch + 1))))


def init_dataset(args, log):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.dataset == "cifar10" or args.dataset == "cifar100":
        image_size = 32
    else:
        image_size = 224

    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        s = args.color_jitter_strength
        log.info("employed augmentation strength is {}".format(s))
        augmentation = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        assert False

    if args.dataset == "imagenet":
        # Data loading code
        root, txt_train, _, _, pathReplaceDict = get_imagenet_root_split(args.data, args.customSplit)
        train_dataset = Custom_Dataset(
            root,
            txt_train,
            TwoCropsTransform(augmentation),
            pre_load=False, pathReplace=pathReplaceDict)
    elif args.dataset == "cifar10" or args.dataset == "cifar10_large":
        # the data distribution
        root, train_idx, _ = get_cifar10_data_split(args.data, args.customSplit, ssl=True)

        train_idx = list(np.load(train_idx))
        train_dataset = subsetCIFAR10(root=root, sublist=train_idx, download=True,
                                      transform=TwoCropsTransform(augmentation))
    elif args.dataset == "cifar100" or args.dataset == "cifar100_large":
        # the data distribution
        root, train_idx, _ = get_cifar100_data_split(args.data, args.customSplit, ssl=True)

        train_idx = list(np.load(train_idx))
        train_dataset = subsetCIFAR100(root=root, sublist=train_idx, download=True,
                                       transform=TwoCropsTransform(augmentation))
    elif args.dataset == "food-101":
        # Data loading code
        root, txt_train, _, _ = get_food101_data_split(args.data, args.customSplit, ssl=True)
        train_dataset = Custom_Dataset(
            root,
            txt_train,
            TwoCropsTransform(augmentation),
            pre_load=False)
    elif args.dataset == "EuroSAT":
        # Data loading code
        root, txt_train, _, _ = get_EuroSAT_data_split(args.data, args.customSplit, ssl=True)
        train_dataset = Custom_Dataset(
            root,
            txt_train,
            TwoCropsTransform(augmentation),
            pre_load=False)
    elif args.dataset == "iNaturalist_sub1000":
        # Data loading code
        root, txt_train, _, _ = get_iNaturalist_sub1000_data_split(args.data, args.customSplit)
        train_dataset = Custom_Dataset(
            root,
            txt_train,
            TwoCropsTransform(augmentation),
            pre_load=False)
    else:
        raise ValueError("No such dataset: {}".format(args.dataset))

    return train_dataset


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, args, log):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr = cosine_annealing(epoch, args.epochs, lr, 1e-6, warmup_steps=10)
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    log.info("current lr is {}".format(lr))


def cosine_annealing(step, total_steps, lr_max, lr_min, warmup_steps=0):
    assert warmup_steps >= 0

    if step < warmup_steps:
        lr = lr_max * step / warmup_steps
    else:
        lr = lr_min + (lr_max - lr_min) * 0.5 * (
                    1 + np.cos((step - warmup_steps) / (total_steps - warmup_steps) * np.pi))

    return lr


def train_simclr(train_loader, model, optimizer, epoch, log, args, local_rank, world_size):
    losses = AverageMeter()
    losses.reset()
    data_time_meter = AverageMeter()
    train_time_meter = AverageMeter()

    end = time.time()

    for i, (inputs, _) in enumerate(train_loader):

        data_time = time.time() - end
        data_time_meter.update(data_time)

        inputs = torch.stack(inputs, dim=1)
        d = inputs.size()
        # print("inputs origin shape is {}".format(d))
        inputs = inputs.view(d[0] * 2, d[2], d[3], d[4]).cuda(non_blocking=True)

        model.train()

        features = model(inputs)

        features = gather_features(features, local_rank, world_size)

        loss = nt_xent(features, t=args.simclr_t)

        # normalize the loss
        loss = loss * world_size

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        losses.update(float(loss.detach().cpu() / world_size), inputs.shape[0])

        train_time = time.time() - end
        end = time.time()
        train_time_meter.update(train_time)

        # torch.cuda.empty_cache()
        if i % args.print_freq == 0 or i == len(train_loader) - 1:
            log.info('Epoch: [{0}][{1}/{2}]\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'data_time: {data_time.val:.2f} ({data_time.avg:.2f})\t'
                     'train_time: {train_time.val:.2f} ({train_time.avg:.2f})\t'.format(
                epoch, i, len(train_loader), loss=losses,
                data_time=data_time_meter, train_time=train_time_meter))


if __name__ == '__main__':
    main()
