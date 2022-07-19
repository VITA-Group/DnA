#  ------------------------------------------------------------------------------------------
#  change from lora code
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List
from pdb import set_trace


class LoRALayer():
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class low_rank_conv2d(nn.Conv2d, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            r: int = 0,
            r_ratio: float = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            merge_weights: bool = True,
            fix_low_rank: bool = False,
            fix_sparse: bool = False,
            tune_U: bool = False,
            tune_V: bool = False,
            tune_U_S: bool = False,
            tune_V_S: bool = False,
            keep_noise: bool = False,
            reshape_consecutive: bool = False,
            decompose_no_s: bool = False,
            tune_all: bool = False,
            lora_mode: bool = False,
            **kwargs
    ):
        self.reshape_consecutive = reshape_consecutive
        self.decompose_no_s = decompose_no_s
        if r == 0 and r_ratio > 0:
            sup_rank = min(in_channels * kernel_size, out_channels * kernel_size)
            r = min(int(in_channels * kernel_size * r_ratio), sup_rank)
            r = max(r, 1)

        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert type(kernel_size) is int
        # Actual trainable parameters
        if r > 0:
            # print("r is {}".format(r))
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels * kernel_size, r))
            )

            if not lora_mode:
                self.sparse_weight = nn.Parameter(
                    torch.zeros_like(self.weight)
                )
                self.sparse_mask = nn.Parameter(
                    torch.zeros_like(self.weight)
                )

            if keep_noise:
                self.noise = nn.Parameter(
                    torch.zeros_like(self.weight)
                )
                self.noise.requires_grad = False
            # self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            if not lora_mode:
                self.sparse_mask.requires_grad = False
            assert int(fix_low_rank) + int(fix_sparse) + int(tune_U) + int(tune_V) + int(tune_U_S) + int(tune_V_S) + int(tune_all) <= 1
            if fix_low_rank:
                self.lora_A.requires_grad = False
                self.lora_B.requires_grad = False
                self.sparse_weight.requires_grad = True
            if fix_sparse:
                self.lora_A.requires_grad = True
                self.lora_B.requires_grad = True
                self.sparse_weight.requires_grad = False
            if tune_U:
                self.lora_A.requires_grad = False
                self.lora_B.requires_grad = True
                self.sparse_weight.requires_grad = False
            if tune_V:
                self.lora_A.requires_grad = True
                self.lora_B.requires_grad = False
                self.sparse_weight.requires_grad = False
            if tune_U_S:
                self.lora_A.requires_grad = False
                self.lora_B.requires_grad = True
                self.sparse_weight.requires_grad = True
            if tune_V_S:
                self.lora_A.requires_grad = True
                self.lora_B.requires_grad = False
                self.sparse_weight.requires_grad = True
            if tune_all:
                self.lora_A.requires_grad = True
                self.lora_B.requires_grad = True
                self.sparse_weight.requires_grad = True

            self.keep_noise = keep_noise
            self.reset_parameters()

            self.lora_mode = lora_mode
            if self.lora_mode:
                assert not (keep_noise or self.reshape_consecutive)
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)
                self.lora_A.requires_grad = True
                self.lora_B.requires_grad = True

    def reassign_sparsity(self, threshold):
        assert not self.lora_mode
        assert self.keep_noise
        assert self.r > 0
        residual = self.sparse_weight.data * (1 - self.sparse_mask.data) + self.noise.data
        self.sparse_mask.data = 0 * self.sparse_mask.data + (residual.abs() < threshold).float()
        self.sparse_weight.data = residual * (1 - self.sparse_mask.data)
        self.noise.data = residual - self.sparse_weight.data

    def merge(self, ):
        self.merged = True
        if self.lora_mode:
            self.weight.data = self.lora_AB_to_weight() + self.weight
        else:
            self.weight.data = self.lora_AB_to_weight() + self.sparse_weight * (1 - self.sparse_mask.data)
            if self.keep_noise:
                self.weight.data += self.noise.data
        self.weight.requires_grad = True
        if hasattr(self, 'lora_A'):
            self.lora_A.requires_grad = False
            self.lora_B.requires_grad = False
        if hasattr(self, 'sparse_weight'):
            self.sparse_weight.requires_grad = False

    def lora_AB_to_weight(self):
        if self.reshape_consecutive:
            C_out, C_in, h, w = self.weight.shape
            consecutive_weight_shape = [C_out, h, w, C_in]
            weight = (self.lora_B @ self.lora_A).reshape(consecutive_weight_shape).permute(0, 3, 1, 2).contiguous()
        else:
            weight = (self.lora_B @ self.lora_A).view(self.weight.shape).contiguous()
        return weight

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            if self.lora_mode:
                weight = self.lora_AB_to_weight() + self.weight
            else:
                weight = self.lora_AB_to_weight() + self.sparse_weight * (1 - self.sparse_mask.data)
                if self.keep_noise:
                    weight += self.noise.data
            return F.conv2d(
                x, weight,
                self.bias, self.stride, self.padding, self.dilation, self.groups
            )
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def sparsity_regu(self, ):
        if hasattr(self, 'lora_A'):
            return torch.abs(self.sparse_weight).sum()
        else:
            return 0

    @torch.no_grad()
    def decompose(self, compress_step, lambda_s, log=None):
        if self.lora_mode:
            self.reset_parameters()
            return 1, 1

        residual_change = []

        print_fun = print if log is None else log.info

        sparse_weight_num = 0
        sparse_weight_all = 0

        if self.r > 0:
            if self.reshape_consecutive:
                # weight shape [C_out, C_in, h, w] -> [C_out, h, w, C_in]
                weight = self.weight.data.permute(0, 2, 3, 1).reshape(
                    self.lora_B.data.shape[0], self.lora_A.data.shape[1])
            else:
                weight = self.weight.data.reshape(self.lora_B.data.shape[0], self.lora_A.data.shape[1])
            print("weight.device is {}".format(weight.device))
            U = torch.randn((self.lora_B.data.shape[0], 1), device=weight.device)
            V = torch.randn((1, self.lora_A.data.shape[1]), device=weight.device)

            for rank in range(self.r):
                S = torch.zeros_like(weight)
                # print(rank)
                # print("Shape U is {}".format(U.shape))
                for _ in range(compress_step):
                    U = torch.qr((weight - S) @ V.T)[0]
                    V = U.T @ (weight - S)
                    S = weight - U @ V
                    q = lambda_s
                    sparse_mask = (S.abs() < q)
                    if self.decompose_no_s:
                        sparse_mask = torch.ones_like(sparse_mask, device=sparse_mask.device)
                    S[sparse_mask] = 0
                residual_change.append(torch.norm(weight - U @ V).item() / torch.norm(weight))

                E = weight - U @ V - S
                E_vector = torch.qr(E)[1][:1]
                if (rank < self.r - 1):
                    V = torch.cat([V, E_vector])

            print_fun("residual change of this layer for last 10 step is {}".format([float(r) for r in residual_change[-10:]]))
            print_fun("sparsity of this layer is {}".format(sparse_mask.float().mean()))

            sparse_weight_num += int(sparse_mask.sum())
            sparse_weight_all += int(sparse_mask.numel())

            self.lora_B.data = 0 * self.lora_B.data + U.contiguous()
            self.lora_A.data = 0 * self.lora_A.data + V.contiguous()

            if self.reshape_consecutive:
                C_out, C_in, h, w = self.weight.shape
                consecutive_weight_shape = [C_out, h, w, C_in]
                self.sparse_weight.data = 0 * self.sparse_weight.data + S.reshape(consecutive_weight_shape).permute(0, 3, 1, 2).contiguous()
                self.sparse_mask.data = 0 * self.sparse_mask.data + sparse_mask.reshape(consecutive_weight_shape).permute(0, 3, 1, 2).float().contiguous()
                if self.keep_noise:
                    self.noise.data = 0 * self.noise.data + \
                                        E.reshape(consecutive_weight_shape).permute(0, 3, 1, 2).float().contiguous()
            else:
                self.sparse_weight.data = 0 * self.sparse_weight.data + S.reshape(self.sparse_weight.data.shape).contiguous()
                self.sparse_mask.data = 0 * self.sparse_mask.data + sparse_mask.reshape(self.sparse_weight.data.shape).float().contiguous()
                if self.keep_noise:
                    self.noise.data = 0 * self.noise.data + \
                                        E.reshape(self.sparse_weight.data.shape).float().contiguous()

        return sparse_weight_num, sparse_weight_all