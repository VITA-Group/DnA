import sys
import os
from os.path import join
import torch
import time
import numpy as np
from pdb import set_trace

import torch.nn.functional as F
from utils.utils import logger
sys.path.append(".")

from models.general_framework.convs.low_rank_conv2d import low_rank_conv2d


def reassign_sparsity(model, low_rank_sparse_ratio):
    weights = []
    for m in model.modules():
        if isinstance(m, low_rank_conv2d):
            assert not m.lora_mode
            assert m.keep_noise
            assert m.r > 0
            residual_all = m.sparse_weight.data * (1 - m.sparse_mask.data) + m.noise.data
            weights.append(residual_all.clone().cpu().detach())

    number_of_remaining_weights = torch.sum(torch.tensor([v.numel() for v in weights])).cpu().numpy()
    number_of_weights_to_prune_magnitude = np.ceil(low_rank_sparse_ratio * number_of_remaining_weights).astype(int)

    # Create a vector of all the unpruned weights in the model.
    weight_vector = np.concatenate([v.numpy().flatten() for v in weights])
    threshold = np.sort(np.abs(weight_vector))[min(number_of_weights_to_prune_magnitude, len(weight_vector) - 1)]

    for m in model.modules():
        if isinstance(m, low_rank_conv2d):
            m.reassign_sparsity(threshold)


def prepare_low_rank(model, compress_step, lambda_s, r_ratio, pretrain_ckpt, with_noise, log, dataset,
                     low_rank_UV_lr, low_rank_reshape_consecutive, low_rank_decompose_no_s, low_rank_lora_mode,
                     low_rank_sparse_ratio):
    params = []

    if low_rank_lora_mode:
        return model.parameters()

    decompose_ckpt_name = "decompose_step{}_s{}_Rratio{}.pth".format(compress_step, lambda_s, r_ratio)
    if low_rank_reshape_consecutive:
        decompose_ckpt_name = decompose_ckpt_name.replace(".pth", "_conse.pth")
    if low_rank_decompose_no_s:
        decompose_ckpt_name = decompose_ckpt_name.replace(".pth", "_noS.pth")
    if with_noise:
        decompose_ckpt_name = decompose_ckpt_name.replace(".pth", "_with_noise.pth")

    sparse_decomposition_ckpt = join(os.path.dirname(pretrain_ckpt), decompose_ckpt_name)

    print("target decomposition ckpt is {}".format(sparse_decomposition_ckpt))

    # load pretrain_ckpt
    if os.path.isfile(sparse_decomposition_ckpt):
        pass
    else:
        if dataset == "cifar10" or dataset == "cifar100":
            assert False, "Please conduct decompose at dataset with larger resolution"

        if torch.distributed.get_rank() == 0:
            # save pretrain_ckpt
            log_decompose = logger(os.path.dirname(pretrain_ckpt),
                                   decompose_ckpt_name.replace(".pth", ".txt"))

            # decompose
            sparse_weight_num_collect, sparse_weight_all_collect = 0, 0
            for m in model.modules():
                decompose = getattr(m, "decompose", None)
                if callable(decompose):
                    sparse_weight_num, sparse_weight_all = decompose(compress_step, lambda_s, log_decompose)
                    sparse_weight_num_collect += sparse_weight_num
                    sparse_weight_all_collect += sparse_weight_all
            log_decompose.info("overall sparsity is {}".format(sparse_weight_num_collect/sparse_weight_all_collect))

            torch.save(model.state_dict(), sparse_decomposition_ckpt)
        torch.distributed.barrier()
        time.sleep(3)

    log.info("load previous decomposition {}".format(sparse_decomposition_ckpt))
    state_dict = torch.load(sparse_decomposition_ckpt, map_location="cpu")

    # cvt stat_dict
    if dataset == "cifar10" or dataset == "cifar100":
        shape = state_dict['module.conv1.weight'].shape
        if shape[-1] != 3:
            out_shape = [3, 3]
            state_dict['module.conv1.weight'] = F.interpolate(state_dict['module.conv1.weight'], out_shape)
        if 'module.conv1.lora_A' in state_dict:
            del state_dict['module.conv1.lora_A']
            del state_dict['module.conv1.lora_B']
            del state_dict['module.conv1.sparse_weight']
            del state_dict['module.conv1.sparse_mask']
        if 'module.conv1.noise' in state_dict:
            del state_dict['module.conv1.noise']

    model.load_state_dict(state_dict)

    if low_rank_sparse_ratio > 0:
        log.info("re-assign the sparsity for decomposition")
        reassign_sparsity(model, low_rank_sparse_ratio)

    for name, param in model.named_parameters():
        if param.requires_grad:
            if "lora" in name:
                params.append({'params': param, "lr": low_rank_UV_lr})
            else:
                params.append({'params': param})

    return params

