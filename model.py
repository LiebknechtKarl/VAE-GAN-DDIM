"""Simple, minimal implementation of Mamba in one file of PyTorch.

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""
from __future__ import annotations
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum


@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)


































class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        super().__init__()
        self.args = args
        
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)

        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Tie output projection to embedding weights.
                                                     # See "Weight Tying" paper

    def forward(self, input_ids):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            logits: shape (b, l, vocab_size)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """
        # input_ids     torch.Size([1, 4])     tensor([[46, 31834, 310, 253]])
        x = self.embedding(input_ids)       
        # x      torch.Size([1, 4, 1024])
        # tensor([[[-0.2568,  0.0735, -0.3682,  ...,  0.0758, -0.0457, -0.0427],
        #         [ 0.1295, -0.0636, -0.0089,  ..., -0.0984,  0.2529, -0.0201],
        #         [-0.1934, -0.0655, -0.0858,  ..., -0.1334, -0.0401, -0.0631],
        #         [-0.1710,  0.1047, -0.1179,  ...,  0.1142, -0.0250, -0.2144]]])

        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        logits = self.lm_head(x)

        return logits

    
    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        """Load pretrained weights from HuggingFace into model.
    
        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'
                            
        Returns:
            model: Mamba model with weights loaded
    
        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file
        
        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))
        
        
        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
        
        config_data = load_config_hf(pretrained_model_name)
        args = ModelArgs(
            d_model=config_data['d_model'],
            n_layer=config_data['n_layer'],
            vocab_size=config_data['vocab_size']
        )
        model = Mamba(args)
        
        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict)
        
        return model


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)
        

    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297
            
            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
            
        """
        output = self.mixer(self.norm(x)) + x

        return output
            








class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)
        
    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
    
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)
        
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (b, l, d) = x.shape
        # x    torch.Size([1, 4, 1024])
        # tensor([[[-0.3369,  0.1116, -0.4718,  ...,  0.0827, -0.0534, -0.0473],
        #         [ 0.1262, -0.0717, -0.0084,  ..., -0.0798,  0.2196, -0.0165],
        #         [-0.2903, -0.1138, -0.1258,  ..., -0.1665, -0.0536, -0.0799],
        #         [-0.2661,  0.1887, -0.1793,  ...,  0.1478, -0.0347, -0.2813]]])
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        # x_and_res
        # tensor([[[ 0.6588, -0.5707, -0.7355,  ...,  0.1671, -0.2068,  0.1302],
        #         [-0.8075,  0.1066,  0.2913,  ..., -0.2311, -0.0270,  0.6466],
        #         [ 0.0576, -0.0879,  0.2239,  ...,  0.2617, -0.2169, -0.2590],
        #         [-0.0176, -0.0465, -0.2361,  ...,  0.1896, -0.0926,  0.3989]]])        
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)
        # x.shape   torch.Size([1, 4, 2048])            res.shape   torch.Size([1, 4, 2048])
        # x
        # tensor([[[ 0.6588, -0.5707, -0.7355,  ...,  0.1874, -0.6609,  0.2027],
        #         [-0.8075,  0.1066,  0.2913,  ..., -0.4321, -0.2292, -0.0294],
        #         [ 0.0576, -0.0879,  0.2239,  ...,  0.2812,  0.0410,  0.3539],
        #         [-0.0176, -0.0465, -0.2361,  ...,  0.0505,  0.0583,  0.3007]]])   
        # res
        # tensor([[[-0.2294, -0.6511,  0.5866,  ...,  0.1671, -0.2068,  0.1302],
        #         [ 0.6936, -0.7012, -0.5469,  ..., -0.2311, -0.0270,  0.6466],
        #         [ 0.4481, -0.2030, -0.1675,  ...,  0.2617, -0.2169, -0.2590],
        #         [ 0.1232,  0.3344,  0.1085,  ...,  0.1896, -0.0926,  0.3989]]])
        x = rearrange(x, 'b l d_in -> b d_in l')        # 变维度   x.shape   torch.Size([1, 4, 2048])    --->  torch.Size([1, 2048, 4])
        x = self.conv1d(x)[:, :, :l]      # -----> torch.Size([1, 2048, 4])
        x = rearrange(x, 'b d_in l -> b l d_in')        # -----> x.shape    torch.Size([1, 4, 2048])
        
        x = F.silu(x)       # x.shape   torch.Size([1, 4, 2048])

        y = self.ssm(x)
        
        y = y * F.silu(res)
        
        output = self.out_proj(y)

        return output

    
    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (d_in, n) = self.A_log.shape            # A 随机初始化的数        (d_in, n)         torch.Size([2048, 16])

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()
        # x.shape
        # torch.Size([1, 4, 2048])
        # tensor([[[-0.0744,  0.0433,  0.0008,  ...,  0.0026, -0.0721,  0.0182],
        #         [ 0.0093,  0.0972,  0.0919,  ...,  0.0285, -0.0301, -0.0045],
        #         [-0.0932,  0.0495, -0.0258,  ..., -0.0461, -0.0480,  0.0222],
        #         [-0.0449,  0.0674, -0.0227,  ...,  0.0381, -0.0567, -0.0152]]])
        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        # [self.args.dt_rank, n, n]             [64, 16, 16]
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)    # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        #  x.shape torch.Size([1, 4, 2048])             delta.shape           torch.Size([1, 4, 2048])
        # tensor([[[0.2397, 0.2011, 0.0755,  ..., 0.0616, 0.2725, 0.1076],
        #          [0.1168, 0.0991, 0.0334,  ..., 0.0241, 0.1041, 0.0333],
        #          [0.0754, 0.0665, 0.0602,  ..., 0.0334, 0.0840, 0.0935],
        #          [0.0960, 0.0563, 0.0059,  ..., 0.0054, 0.0822, 0.0484]]])
        # A [2048, 16]      B [1, 4, 16]      C [1, 4, 16]      D.shape torch.Size([2048])
        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        
        return y

    
    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
    
        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)
    
        Returns:
            output: shape (b, l, d_in)
    
        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
            
        """

        #  u .shape torch.Size([1, 4, 2048])             delta.shape           torch.Size([1, 4, 2048])
        # tensor([[[0.2397, 0.2011, 0.0755,  ..., 0.0616, 0.2725, 0.1076],
        #          [0.1168, 0.0991, 0.0334,  ..., 0.0241, 0.1041, 0.0333],
        #          [0.0754, 0.0665, 0.0602,  ..., 0.0334, 0.0840, 0.0935],
        #          [0.0960, 0.0563, 0.0059,  ..., 0.0054, 0.0822, 0.0484]]])
        # A [2048, 16]      B [1, 4, 16]      C [1, 4, 16]      D.shape torch.Size([2048])
        (b, l, d_in) = u.shape
        n = A.shape[1]
        
        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        # delta.shape    torch.Size([1, 4, 2048])           A.shape     torch.Size([2048, 16])
        # torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n')).shape   torch.Size([1, 4, 2048, 16])
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        # deltaB_u.shape        torch.Size([1, 4, 2048, 16])
        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []    
        for i in range(l):          # l = 4
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        # len(ys)       4     ys[0].shape   torch.Size([1, 2048])
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
        
        y = y + u * D
        # y.shape       torch.Size([1, 4, 2048])
        return y


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
        
