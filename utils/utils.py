import torch
import torch.nn as nn
import os
import time
import numpy as np
import random
import torch.nn.functional as F
import re
from collections import OrderedDict
from pdb import set_trace


def gather_features(features, local_rank, world_size):
    features_list = [torch.zeros_like(features) for _ in range(world_size)]
    torch.distributed.all_gather(features_list, features)
    features_list[local_rank] = features
    features = torch.cat(features_list)
    return features


# loss
def pair_cosine_similarity(x, eps=1e-8):
    n = x.norm(p=2, dim=1, keepdim=True)
    return (x @ x.t()) / (n * n.t()).clamp(min=eps)


def nt_xent(x, t=0.5):
    # print("device of x is {}".format(x.device))

    x = pair_cosine_similarity(x)
    x = torch.exp(x / t)
    idx = torch.arange(x.size()[0])
    # Put positive pairs on the diagonal
    idx[::2] += 1
    idx[1::2] -= 1
    x = x[idx]
    # subtract the similarity of 1 from the numerator
    x = x.diag() / (x.sum(0) - torch.exp(torch.tensor(1 / t)))

    return -torch.log(x).mean()


def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class logger(object):
    def __init__(self, path, log_name="log.txt", local_rank=0):
        self.path = path
        self.local_rank = local_rank
        self.log_name = log_name

    def info(self, msg):
        if self.local_rank == 0:
            print(msg)
            with open(os.path.join(self.path, self.log_name), 'a') as f:
                f.write(msg + "\n")


def fix_bn(model, fixto):
    if fixto == 'nothing':
        # fix none
        # fix previous three layers
        pass
    elif fixto == 'layer1':
        # fix the first layer
        for name, m in model.named_modules():
            if not ("layer2" in name or "layer3" in name or "layer4" in name or "fc" in name):
                m.eval()
    elif fixto == 'layer2':
        # fix the previous two layers
        for name, m in model.named_modules():
            if not ("layer3" in name or "layer4" in name or "fc" in name):
                m.eval()
    elif fixto == 'layer3':
        # fix every layer except fc
        # fix previous four layers
        for name, m in model.named_modules():
            if not ("layer4" in name or "fc" in name):
                m.eval()
    elif fixto == 'layer4':
        # fix every layer except fc
        # fix previous four layers
        for name, m in model.named_modules():
            if not ("fc" in name):
                m.eval()
    else:
        assert False


class DistillCrossEntropy(nn.Module):
    def __init__(self, T):
        super(DistillCrossEntropy, self).__init__()
        self.T = T
        return

    def forward(self, inputs, target):
        """
        :param inputs: prediction logits
        :param target: target logits
        :return: loss
        """
        log_likelihood = - F.log_softmax(inputs / self.T, dim=1)
        sample_num, class_num = target.shape
        loss = torch.sum(torch.mul(log_likelihood, torch.softmax(target / self.T, dim=1)))/sample_num

        return loss


def change_batchnorm_momentum(module, value):
  if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
    module.momentum = value
  for name, child in module.named_children():
    change_batchnorm_momentum(child, value)


def get_negative_mask_to_another_branch(batch_size):
    negative_mask = torch.ones((batch_size, batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0

    return negative_mask


def nt_xent_only_compare_to_another_branch(x1, x2, t=0.5):
    out1 = F.normalize(x1, dim=-1)
    out2 = F.normalize(x2, dim=-1)
    d = out1.size()
    batch_size = d[0]

    neg = torch.exp(torch.mm(out1, out2.t().contiguous()) / t)

    mask = get_negative_mask_to_another_branch(batch_size).cuda()
    neg = neg.masked_select(mask).view(batch_size, -1)

    # pos score
    pos = torch.exp(torch.sum(out1 * out2, dim=-1) / t)

    # estimator g()
    Ng = neg.sum(dim=-1)

    # contrastive loss
    loss = (- torch.log(pos / (pos + Ng)))
    return loss.mean()


def nt_xent_compare_to_queue(out1, out2, queue, t=0.5, sampleWiseLoss=False):

    d = out1.size()
    batch_size = d[0]

    neg = torch.exp(torch.mm(out1, queue.clone().detach()) / t)

    # pos score
    pos = torch.exp(torch.sum(out1 * out2, dim=-1) / t)

    # estimator g()
    Ng = neg.sum(dim=-1)

    # contrastive loss
    loss = (- torch.log(pos / (pos + Ng)))

    if sampleWiseLoss:
        return loss
    else:
        return loss.mean()


def gatherFeatures(features, local_rank, world_size):
    features_list = [torch.zeros_like(features) for _ in range(world_size)]
    torch.distributed.all_gather(features_list, features)
    features_list[local_rank] = features
    features = torch.cat(features_list)
    return features


# loss
def pair_cosine_similarity(x, eps=1e-8):
    n = x.norm(p=2, dim=1, keepdim=True)
    return (x @ x.t()) / (n * n.t()).clamp(min=eps)


def nt_xent(x, t=0.5, sampleWiseLoss=False, return_prob=False):
    # print("device of x is {}".format(x.device))

    x = pair_cosine_similarity(x)
    x = torch.exp(x / t)
    idx = torch.arange(x.size()[0])
    # Put positive pairs on the diagonal
    idx[::2] += 1
    idx[1::2] -= 1
    x = x[idx]
    # subtract the similarity of 1 from the numerator
    x = x.diag() / (x.sum(0) - torch.exp(torch.tensor(1 / t)))

    if return_prob:
        return x.reshape(len(x) // 2, 2).mean(-1)

    sample_loss = -torch.log(x)

    if sampleWiseLoss:
        return sample_loss.reshape(len(sample_loss) // 2, 2).mean(-1)

    return sample_loss.mean()


def nt_xent_weak_compare(x, t=0.5, features2=None, easy_mining=0.9):
    if features2 is None:
        out = F.normalize(x, dim=-1)
        d = out.size()
        batch_size = d[0] // 2
        out = out.view(batch_size, 2, -1).contiguous()
        out_1 = out[:, 0]
        out_2 = out[:, 1]
    else:
        batch_size = x.shape[0]
        out_1 = F.normalize(x, dim=-1)
        out_2 = F.normalize(features2, dim=-1)

    # neg score
    out = torch.cat([out_1, out_2], dim=0)
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / t)

    mask = get_negative_mask(batch_size).cuda()
    neg = neg.masked_select(mask).view(2 * batch_size, -1)

    total_neg_num = neg.shape[1]
    hard_sample_num = max(int((1 - easy_mining) * total_neg_num), 1)
    score = neg
    threshold = score.kthvalue(hard_sample_num, dim=1, keepdim=True)[0]
    neg = (score <= threshold) * neg
    Ng = neg.sum(dim=-1)

    # pos score
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / t)
    pos = torch.cat([pos, pos], dim=0)

    # contrastive loss
    loss = (- torch.log(pos / (pos + Ng)))

    return loss.mean()


def focal_loss(prob, gamma):
    """Computes the focal loss"""
    loss = (1 - prob) ** gamma * (-torch.log(prob))
    return loss.mean()


def fix_focal_loss(prob, fix_prob, gamma):
    """Computes the focal loss"""
    loss = (1 - fix_prob) ** gamma * (-torch.log(prob))
    return loss.mean()


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


def nt_xent_instance_large(x, t=0.5, return_porbs=False):
    out = F.normalize(x, dim=-1)
    d = out.size()
    batch_size = d[0] // 2
    out = out.view(batch_size, 2, -1).contiguous()
    out_1 = out[:, 0]
    out_2 = out[:, 1]
    out = torch.cat([out_1, out_2], dim=0)

    # doesn't give gradient
    losses = []
    probs = []

    with torch.no_grad():
        for cnt in range(batch_size):
            # pos score
            pos = torch.exp(torch.sum(out_1[cnt] * out_2[cnt]) / t)
            pos = torch.stack([pos, pos], dim=0)

            Ng1 = torch.exp((out_1[cnt].unsqueeze(0) * out).sum(1) / t).sum() - torch.exp(torch.Tensor([1 / t,])).to(out.device)
            Ng2 = torch.exp((out_2[cnt].unsqueeze(0) * out).sum(1) / t).sum() - torch.exp(torch.Tensor([1 / t,])).to(out.device)
            Ng = torch.cat([Ng1, Ng2], dim=0)

            if not return_porbs:
                # contrastive loss
                losses.append(- torch.log(pos / Ng).mean())
            else:
                # contrastive loss
                probs.append((pos / Ng).mean())

    if not return_porbs:
        losses = torch.stack(losses)
        return losses
    else:
        probs = torch.stack(probs)
        return probs


def cosine_annealing(step, total_steps, lr_max, lr_min, warmup_steps=0):
    assert warmup_steps >= 0

    if step < warmup_steps:
        lr = lr_max * step / warmup_steps
    else:
        lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos((step - warmup_steps) / (total_steps - warmup_steps) * np.pi))

    return lr


def get_imagenet_root_path(root):

    pathReplaceDict = {}
    if os.path.isdir(root):
        pass
    elif os.path.isdir("/ssd1/bansa01/imagenet_final"):
        root = "/ssd1/bansa01/imagenet_final"
    elif os.path.isdir("/mnt/imagenet"):
        root = "/mnt/imagenet"
    elif os.path.isdir("/hdd3/ziyu/imagenet"):
        root = "/hdd3/ziyu/imagenet"
    elif os.path.isdir("/ssd2/invert/imagenet_final/"):
        root = "/ssd2/invert/imagenet_final/"
    elif os.path.isdir("/home/yucheng/imagenet"):
        root = "/home/yucheng/imagenet"
    elif os.path.isdir("/data/datasets/ImageNet_unzip"):
        root = "/data/datasets/ImageNet_unzip"
    elif os.path.isdir("/scratch/user/jiangziyu/imageNet"):
        root = "/scratch/user/jiangziyu/imageNet"
    elif os.path.isdir("/datadrive_d/imagenet"):
        root = "/datadrive_d/imagenet"
    elif os.path.isdir("/datadrive_c/imagenet"):
        root = "/datadrive_c/imagenet"
    elif os.path.isdir("/ssd2/tianlong/zzy/data/imagenet"):
        root = "/ssd2/tianlong/zzy/data/imagenet"
    elif os.path.isdir("/home/xinyu/dataset/imagenet2012"):
        root = "/home/xinyu/dataset/imagenet2012"
    elif os.path.isdir("/home/xueq13/scratch/ziyu/ImageNet/ILSVRC/Data/CLS-LOC"):
        root = "/home/xueq13/scratch/ziyu/ImageNet/ILSVRC/Data/CLS-LOC"
    elif os.path.isdir("/hdd1/ziyu/ImageNet/"):
        root = "/hdd1/ziyu/ImageNet/"
    elif os.path.isdir("/mnt/models/imagenet_new"):
        root = "/mnt/models/imagenet_new"
        pathReplaceDict = {"train/": "train_new/"}
    elif os.path.isdir("/mnt/models/imagenet"):
        root = "/mnt/models/imagenet"
    else:
        print("No dir for imagenet")
        assert False

    return root, pathReplaceDict


def get_imagenet_root_split(root, customSplit, domesticAnimalSplit=False):
    root, pathReplaceDict = get_imagenet_root_path(root)

    txt_train = "split/imagenet/imagenet_train.txt"
    txt_val = "split/imagenet/imagenet_val.txt"
    txt_test = "split/imagenet/imagenet_test.txt"

    if domesticAnimalSplit:
        txt_train = "split/imagenet/imagenet_domestic_train.txt"
        txt_val = "split/imagenet/imagenet_domestic_val.txt"
        txt_test = "split/imagenet/imagenet_domestic_test.txt"

    if customSplit != '':
        txt_train = "split/imagenet/{}.txt".format(customSplit)

    return root, txt_train, txt_val, txt_test, pathReplaceDict


def get_cifar10_data_split(root, customSplit, ssl=False, semi_fixMatch=False):
    assert int(ssl) + int(semi_fixMatch) <= 1
    # if ssl is True, use both train and val splits
    if os.path.isdir(root):
        root = root
    else:
        if os.path.isdir('../../data'):
            root = '../../data'
        elif os.path.isdir('/mnt/models/Ziyu/'):
            root = '/mnt/models/Ziyu/'
        else:
            assert False

    if ssl:
        assert customSplit == ''
        train_idx = "split/cifar10/trainValIdxList.npy"
        return root, train_idx, None

    train_idx = "split/cifar10/trainIdxList.npy"
    val_idx = "split/cifar10/valIdxList.npy"
    if customSplit != '':
        train_idx_cus = "split/cifar10/{}.npy".format(customSplit)

        if semi_fixMatch:
            return root, train_idx, val_idx, train_idx_cus
        else:
            train_idx = train_idx_cus
    else:
        assert not semi_fixMatch, "need custom split for semi_fixMatch"

    return root, train_idx, val_idx


def get_cifar100_data_split(root, customSplit, ssl=False, semi_fixMatch=False):
    assert int(ssl) + int(semi_fixMatch) <= 1

    if os.path.isdir(root):
        root = root
    else:
        if os.path.isdir('../../data'):
            root = '../../data'
        elif os.path.isdir('/mnt/models/dataset/'):
            root = '/mnt/models/dataset/'
        elif os.path.isdir('/mnt/models/'):
            root = '/mnt/models/'
        else:
            root = '.'

    if ssl:
        assert customSplit == ''
        train_idx = "split/cifar100/cifar100_trainValIdxList.npy"
        return root, train_idx, None

    train_idx = "split/cifar100/cifar100_trainIdxList.npy"
    val_idx = "split/cifar100/cifar100_valIdxList.npy"
    if customSplit != '':
        train_idx_cus = "split/cifar100/{}.npy".format(customSplit)

        if semi_fixMatch:
            return root, train_idx, val_idx, train_idx_cus
        else:
            train_idx = train_idx_cus
    else:
        assert not semi_fixMatch, "need custom split for semi_fixMatch"

    return root, train_idx, val_idx


def get_food101_path(root):
    if os.path.isdir(root):
        root = root
    else:
        if os.path.isdir("/hdd1/ziyu/MSR/data/food-101/images"):
            root = "/hdd1/ziyu/MSR/data/food-101/images"
        elif os.path.isdir('/mnt/models/food-101/images/'):
            root = '/mnt/models/food-101/images/'
        else:
            assert False

    return root


def get_food101_data_split(root, customSplit, ssl=False):
    root = get_food101_path(root)

    txt_train = "split/food-101/food101_train.txt"
    txt_val = "split/food-101/food101_val.txt"
    txt_test = "split/food-101/food101_test.txt"

    if customSplit != '':
        txt_train = "split/food-101/{}.txt".format(customSplit)

    if ssl:
        assert customSplit == ''
        train_idx = "split/food-101/food101_trainval.txt"
        return root, train_idx, None, None

    return root, txt_train, txt_val, txt_test


def get_iNaturalist_path(root):
    if os.path.isdir(root):
        root = root
    else:
        if os.path.isdir("/datadrive_c/ziyu/iNaturalist/"):
            root = "/datadrive_c/ziyu/iNaturalist/"
        elif os.path.isdir("/mnt/inaturalist/"):
            root = "/mnt/inaturalist/"
        elif os.path.isdir("/home/xueq13/scratch/ziyu/iNaturalist"):
            root = "/home/xueq13/scratch/ziyu/iNaturalist"
        elif os.path.isdir("/hdd1/ziyu/MSR/data"):
            root = "/hdd1/ziyu/MSR/data"
        elif os.path.isdir('/mnt/models/'):
            root = '/mnt/models/'
        else:
            assert False

    return root


def get_iNaturalist_data_split(root, customSplit):
    root = get_iNaturalist_path(root)

    txt_train = "split/iNaturalist/iNaturalist18_train.txt"
    txt_val = "split/iNaturalist/iNaturalist18_val.txt"
    txt_test = "split/iNaturalist/iNaturalist18_val.txt"

    if customSplit != '':
        txt_train = "split/iNaturalist/{}.txt".format(customSplit)

    return root, txt_train, txt_val, txt_test


def get_iNaturalist_sub1000_data_split(root, customSplit):
    root = get_iNaturalist_path(root)

    txt_train = "split/iNaturalist/iNaturalist18_sub1000_train.txt"
    txt_val = "split/iNaturalist/iNaturalist18_sub1000_val.txt"
    txt_test = "split/iNaturalist/iNaturalist18_sub1000_val.txt"

    if customSplit != '':
        txt_train = "split/iNaturalist/{}.txt".format(customSplit)

    return root, txt_train, txt_val, txt_test


def get_EuroSAT_path(root):
    if os.path.isdir(root):
        root = root
    else:
        if os.path.isdir("/hdd1/ziyu/MSR/data/EuroSAT/2750"):
            root = "/hdd1/ziyu/MSR/data/EuroSAT/2750"
        elif os.path.isdir("/mnt/models/Ziyu/EuroSAT/2750"):
            root = "/mnt/models/Ziyu/EuroSAT/2750"
        else:
            assert False

    return root


def get_EuroSAT_data_split(root, customSplit, ssl=False):
    root = get_EuroSAT_path(root)

    txt_train = "split/EuroSAT/EuroSAT_train.txt"
    txt_val = "split/EuroSAT/EuroSAT_val.txt"
    txt_test = "split/EuroSAT/EuroSAT_test.txt"

    if customSplit != '':
        txt_train = "split/EuroSAT/{}.txt".format(customSplit)

    if ssl:
        assert customSplit == ''
        train_idx = "split/EuroSAT/EuroSAT_trainval.txt"
        return root, train_idx, None, None

    return root, txt_train, txt_val, txt_test


def remove_state_dict_module(state_dict):
    # rename pre-trained keys
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.'):
            # remove prefix
            state_dict[k.replace("module.", "")] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

    return state_dict


def fix_backbone(model, log, fixBackbone_cnt=1, verbose=False):
    # fix every layer except fc
    # fix previous four layers
    log.info("fix backbone for last {} layers".format(fixBackbone_cnt))

    if fixBackbone_cnt == 1:
        key_words_last_layers = ["fc",]
    elif fixBackbone_cnt == 2:
        key_words_last_layers = ["fc", "layer4.2"]
    elif fixBackbone_cnt == 3:
        key_words_last_layers = ["fc", "layer4.2", "layer4.1"]
    else:
        raise ValueError("only support fixBackbone_cnt of [1,2,3], but get {}".format(fixBackbone_cnt))

    for name, param in model.named_parameters():
        contain_key_word_flag = False
        for key_word in key_words_last_layers:
            if key_word in name:
                contain_key_word_flag = True
                break

        if not contain_key_word_flag:
            if verbose:
                print("fix {}".format(name))
            param.requires_grad = False
        else:
            if verbose:
                print("free {}".format(name))

    for name, m in model.named_modules():
        contain_key_word_flag = False
        for key_word in key_words_last_layers:
            if key_word in name:
                contain_key_word_flag = True
                break

        if not contain_key_word_flag:
            m.eval()


def check_and_cvt_pretrain_type(pretrain_state_dict, model_state_dict, log):
    moco_type = False

    for key in pretrain_state_dict:
        if "encoder_q." in key:
            moco_type=True
            break

    if moco_type:
        log.info("#### cvt_moco_pretrain ####")
        return cvt_moco_pretrain(pretrain_state_dict, model_state_dict)
    else:
        return pretrain_state_dict


def cvt_moco_pretrain(pretrain_state_dict, model_state_dict):
    new_state_dict = OrderedDict()
    for key, item in pretrain_state_dict.items():
        if "encoder_q." in key:
            new_state_dict[key.replace("encoder_q.", "")] = item

    for key, item in model_state_dict.items():
        if "fc" in key:
            assert key not in new_state_dict
            new_state_dict[key] = item

    return new_state_dict

