from einops import reduce
from typing import TypeVar
Tensor = TypeVar("torch.tensor")

import math
from inspect import isfunction
from typing import Any, Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from packaging import version

from collections import OrderedDict
from torch import nn, Tensor
from torch.jit.annotations import Tuple, List, Dict, Optional



class FPN_down_g(nn.Module):
    def __init__(self, in_channel_list, out_channel_list):
        super(FPN_down_g, self).__init__()
        self.inner_layer = nn.ModuleList()
        self.out_layer = nn.ModuleList()
        for i, in_channel in enumerate(in_channel_list[:-1]):
            self.inner_layer.append(GroupConv(in_channel, out_channel_list[i], 1).cuda())
            self.out_layer.append(GroupConv(out_channel_list[i], out_channel_list[i], kernel_size=3, padding=1).cuda())
    def forward(self, x):
        features_down = []
        prev_feature = x[0]
        for i in range(len(x) - 1):
            current_feature = x[i + 1]
            prev_feature = self.inner_layer[i](prev_feature)
            size = (prev_feature.shape[2] // 2, prev_feature.shape[3] // 2)
            prev_feature = F.interpolate(prev_feature, size=size)
            prev_n_current = prev_feature + current_feature
            prev_feature = self.out_layer[i](prev_n_current)
            features_down.append(prev_feature)
        return features_down


class GroupConv(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size, stride=1,padding=0) -> None:
        super(GroupConv, self).__init__()
        self.conv = nn.Conv2d(3*in_channels, 3*out_channels, kernel_size, stride, padding,groups=3)
    def forward(self, data: Tensor, **kwargs) -> Tensor:
        data = torch.concat(torch.chunk(data,3,dim=-1),dim=1)
        data = self.conv(data)
        data = torch.concat(torch.chunk(data,3,dim=1),dim=-1)
        return data

class FPN_up_g(nn.Module):
    def __init__(self, in_channel_list, out_channel_list):
        super(FPN_up_g, self).__init__()
        self.inner_layer = nn.ModuleList()
        self.out_layer = nn.ModuleList()
        self.depth = len(out_channel_list)
        for i, in_channel in enumerate(in_channel_list[:-1]):
            self.inner_layer.append(GroupConv(in_channel, out_channel_list[i], 1).cuda())
            self.out_layer.append(GroupConv(out_channel_list[i], out_channel_list[i], kernel_size=3, padding=1).cuda())

    def forward(self, x):
        features_up = []
        prev_feature = x[0]
        for i in range(self.depth):
            prev_feature = self.inner_layer[i](prev_feature)
            size = (prev_feature.shape[2] * 2, prev_feature.shape[3] * 2)
            prev_feature = F.interpolate(prev_feature, size=size)
            current_feature = x[i + 1]
            prev_n_current = prev_feature + current_feature
            prev_feature = self.out_layer[i](prev_n_current)
            features_up.append(prev_feature)

        return features_up[::-1]

class ExtraFPNBlock(nn.Module):
    def forward(
            self,
            results: List[Tensor],
            x: List[Tensor],
            names: List[str],
    ) -> Tuple[List[Tensor], List[str]]:
        pass


class FeaturePyramidNetwork(nn.Module):
    def __init__(
            self,
            in_channels_list: List[int],
            out_channels: int,
            extra_blocks: Optional[ExtraFPNBlock] = None,
    ):
        super(FeaturePyramidNetwork, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        if extra_blocks is not None:
            assert isinstance(extra_blocks, ExtraFPNBlock)
        self.extra_blocks = extra_blocks

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = 0
        for m in self.inner_blocks:
            num_blocks += 1
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.inner_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = 0
        for m in self.layer_blocks:
            num_blocks += 1
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.layer_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Computes the FPN for a set of feature maps.

        Arguments:
            x (OrderedDict[Tensor]): feature maps for each feature level.

        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        results = []
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out


class LastLevelMaxPool(ExtraFPNBlock):
    """
    Applies a max_pool2d on top of the last feature map
    """

    def forward(
            self,
            x: List[Tensor],
            y: List[Tensor],
            names: List[str],
    ) -> Tuple[List[Tensor], List[str]]:
        names.append("pool")
        x.append(F.max_pool2d(x[-1], 1, 2, 0))
        return x, names


class LastLevelP6P7(ExtraFPNBlock):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(
            self,
            p: List[Tensor],
            c: List[Tensor],
            names: List[str],
    ) -> Tuple[List[Tensor], List[str]]:
        p5, c5 = p[-1], c[-1]
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        p.extend([p6, p7])
        names.extend(["p6", "p7"])
        return p, names


if version.parse(torch.__version__) >= version.parse("2.0.0"):
    SDP_IS_AVAILABLE = True
    from torch.backends.cuda import SDPBackend, sdp_kernel

    BACKEND_MAP = {
        SDPBackend.MATH: {
            "enable_math": True,
            "enable_flash": False,
            "enable_mem_efficient": False,
        },
        SDPBackend.FLASH_ATTENTION: {
            "enable_math": False,
            "enable_flash": True,
            "enable_mem_efficient": False,
        },
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False,
            "enable_flash": False,
            "enable_mem_efficient": True,
        },
        None: {"enable_math": True, "enable_flash": True, "enable_mem_efficient": True},
    }
else:
    from contextlib import nullcontext

    SDP_IS_AVAILABLE = False
    sdp_kernel = nullcontext
    BACKEND_MAP = {}
    print(
        f"No SDP backend available, likely because you are running in pytorch versions < 2.0. In fact, "
        f"you are using PyTorch {torch.__version__}. You might want to consider upgrading."
    )

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False
    print("no module 'xformers'. Processing without...")

def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)

class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        ctx.gpu_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(),
            "dtype": torch.get_autocast_gpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad(), torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b c (h w)")
        w_ = torch.einsum("bij,bjk->bik", q, k)

        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, "b c h w -> b c (h w)")
        w_ = rearrange(w_, "b i j -> b j i")
        h_ = torch.einsum("bij,bjk->bik", v, w_)
        h_ = rearrange(h_, "b c (h w) -> b c h w", h=h)
        h_ = self.proj_out(h_)

        return x + h_


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        backend=None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.backend = backend

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        h = self.heads

        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            n_cp = x.shape[0] // n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        ## old
        """
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        """
        ## new
        with sdp_kernel(**BACKEND_MAP[self.backend]):
            # print("dispatching into backend", self.backend, "q/k/v shape: ", q.shape, k.shape, v.shape)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask
            )  # scale is dim_head ** -0.5 per default

        del q, k, v
        out = rearrange(out, "b h n d -> b n (h d)", h=h)

        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(
        self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, **kwargs
    ):
        super().__init__()
        print(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{heads} heads with a dimension of {dim_head}."
        )
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            # n_cp = x.shape[0]//n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )
        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=None, op=self.attention_op
        )

        # TODO: Use this directly in the attention operation, as a bias
        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention,  # ampere
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
        attn_mode="softmax",
        sdp_backend=None,
    ):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        if attn_mode != "softmax" and not XFORMERS_IS_AVAILABLE:
            print(
                f"Attention mode '{attn_mode}' is not available. Falling back to native attention. "
                f"This is not a problem in Pytorch >= 2.0. FYI, you are running with PyTorch version {torch.__version__}"
            )
            attn_mode = "softmax"
        elif attn_mode == "softmax" and not SDP_IS_AVAILABLE:
            print(
                "We do not support vanilla attention anymore, as it is too expensive. Sorry."
            )
            if not XFORMERS_IS_AVAILABLE:
                assert (
                    False
                ), "Please install xformers via e.g. 'pip install xformers==0.0.16'"
            else:
                print("Falling back to xformers efficient attention.")
                attn_mode = "softmax-xformers"
        attn_cls = self.ATTENTION_MODES[attn_mode]
        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            assert sdp_backend is None or isinstance(sdp_backend, SDPBackend)
        else:
            assert sdp_backend is None
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
            backend=sdp_backend,
        )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            backend=sdp_backend,
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        if self.checkpoint:
            print(f"{self.__class__.__name__} is using checkpointing")

    def forward(
        self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0
    ):
        kwargs = {"x": x}

        if context is not None:
            kwargs.update({"context": context})

        if additional_tokens is not None:
            kwargs.update({"additional_tokens": additional_tokens})

        if n_times_crossframe_attn_in_self:
            kwargs.update(
                {"n_times_crossframe_attn_in_self": n_times_crossframe_attn_in_self}
            )

        # return mixed_checkpoint(self._forward, kwargs, self.parameters(), self.checkpoint)
        if context is None:
            return checkpoint(
                self._forward, [x], self.parameters(), self.checkpoint
            )
        else:
            return checkpoint(
                self._forward, [x, context], self.parameters(), self.checkpoint
            )

    def _forward(
        self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0
    ):
        x = (
            self.attn1(
                self.norm1(x),
                context=context if self.disable_self_attn else None,
                additional_tokens=additional_tokens,
                n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self
                if not self.disable_self_attn
                else 0,
            )
            + x
        )
        x = (
            self.attn2(
                self.norm2(x), context=context, additional_tokens=additional_tokens
            )
            + x
        )
        x = self.ff(self.norm3(x)) + x
        return x


class BasicTransformerSingleLayerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention  # on the A100s not quite as fast as the above version
        # (todo might depend on head_dim, check, falls back to semi-optimized kernels for dim!=[16,32,64,128])
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        attn_mode="softmax",
    ):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim,
        )
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        if context is None:
            return checkpoint(
                self._forward, [x], self.parameters(), self.checkpoint
            )
        else:
            return checkpoint(
                self._forward, [x, context], self.parameters(), self.checkpoint
            )

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context) + x
        x = self.ff(self.norm2(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        attn_type="softmax",
        use_checkpoint=True,
        # sdp_backend=SDPBackend.FLASH_ATTENTION
        sdp_backend=None,
    ):
        super().__init__()
        print(
            f"constructing {self.__class__.__name__} of depth {depth} w/ {in_channels} channels and {n_heads} heads"
        )
        from omegaconf import ListConfig

        if exists(context_dim) and not isinstance(context_dim, (list, ListConfig)):
            context_dim = [context_dim]
        if exists(context_dim) and isinstance(context_dim, list):
            if depth != len(context_dim):
                print(
                    f"WARNING: {self.__class__.__name__}: Found context dims {context_dim} of depth {len(context_dim)}, "
                    f"which does not match the specified 'depth' of {depth}. Setting context_dim to {depth * [context_dim[0]]} now."
                )
                # depth does not match context dims.
                assert all(
                    map(lambda x: x == context_dim[0], context_dim)
                ), "need homogenous context_dim to match depth automatically"
                context_dim = depth * [context_dim[0]]
        elif context_dim is None:
            context_dim = [None] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    attn_mode=attn_type,
                    checkpoint=use_checkpoint,
                    sdp_backend=sdp_backend,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            # self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            if i > 0 and len(context) == 1:
                i = 0  # use same context for each block
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class Lrpe(nn.Module):
    def __init__(
            self,
            num_heads=8,
            embed_dim=64,
    ):
        super().__init__()
        d = num_heads * embed_dim

        self.index = torch.empty(0)
        self.theta = nn.Parameter(10000 ** (-2 / d *
                                            torch.arange(d)).reshape(
            num_heads, 1, -1))

    def forward(self, x, offset=0):
        # x: b, h, n, d
        # offset: for k, v cache
        n = x.shape[-2]
        if self.index.shape[0] < n:
            self.index = torch.arange(n).reshape(1, -1, 1).to(x)
        index = self.index[:, :n] + offset
        theta = self.theta * index
        x = torch.concat([x * torch.cos(theta), x * torch.sin(theta)], dim=-1)

        return x

class NormLinearAttention(nn.Module):
    def __init__(
            self,
            query_dim,
            heads,
            dropout=0.0,
            context_dim=0,
            bias=False,
            use_lrpe=True,
            layer=0,
            **kwargs
    ):
        super().__init__()

        hidden_dim = query_dim
        bias = bias,
        self.n_head = heads
        self.use_lrpe = use_lrpe

        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.qkvu_proj = nn.Linear(hidden_dim, 4 * hidden_dim, bias=bias)
        if self.use_lrpe:
            self.lrpe = Lrpe(num_heads=self.n_head, embed_dim=hidden_dim // self.n_head)
        self.act = F.silu
        self.norm = nn.LayerNorm(hidden_dim)
        # self.norm = SimpleRMSNorm(hidden_dim)

        if layer >= 16:
            self.forward_type = 'right'
        else:
            self.forward_type = 'left'

        self.clip = True
        self.eps = 1e-5

    def abs_clamp(self, t):
        min_mag = 1e-2
        max_mag = 100
        sign = t.sign()
        return t.abs_().clamp_(min_mag, max_mag) * sign

    def forward_right(
            self,
            x,
    ):
        # x: b n d
        n = x.shape[-2]
        # linear map
        q, k, v, u = self.qkvu_proj(x).chunk(4, dim=-1)
        # reshape
        q, k, v = map(
            lambda x: rearrange(x, 'b n (h d) -> b h n d', h=self.n_head),
            [q, k, v])
        # act
        q = self.act(q)
        k = self.act(k)

        # q = F.normalize(q, dim=-1)
        # k = F.normalize(k, dim=-1)

        # lrpe
        if self.use_lrpe:
            offset = 0
            q = self.lrpe(q, offset=offset)
            k = self.lrpe(k, offset=offset)

        kv = torch.einsum("... n d, ... n e -> ... d e", k, v)
        # if self.clip:
        # kv = self.abs_clamp(kv)
        output = torch.einsum('... n d, ... d e -> ... n e', q, kv)

        # Z = 1/(torch.einsum("nlhi,nlhi->nlh", q, k.cumsum(1)) + self.eps)

        # print('right:', q.shape, k.shape, v.shape, kv.shape)

        # reshape
        output = rearrange(output, 'b h n d -> b n (h d)')
        # normalize
        output = self.norm(output)

        # gate
        output = u * output
        # outproj
        output = self.out_proj(output)

        return output

    def forward_left(
            self,
            x,
    ):
        # x: b n d
        n = x.shape[-2]
        # linear map
        q, k, v, u = self.qkvu_proj(x).chunk(4, dim=-1)
        # reshape
        q, k, v = map(
            lambda x: rearrange(x, 'b n (h d) -> b h n d', h=self.n_head),
            [q, k, v])
        # act
        q = self.act(q)
        k = self.act(k)

        # q = F.normalize(q, dim=-1)
        # k = F.normalize(k, dim=-1)

        # lrpe
        if self.use_lrpe:
            offset = 0
            q = self.lrpe(q, offset=offset)
            k = self.lrpe(k, offset=offset)

        qk = torch.einsum("... m d, ... n d -> ... m n", q, k)
        # if self.clip:
        # qk = self.abs_clamp(qk)

        output = torch.einsum('... m n, ... n e -> ... m e', qk, v)

        # print('left:', q.shape, k.shape, v.shape, qk.shape)

        # reshape
        output = rearrange(output, 'b h n d -> b n (h d)')
        # normalize
        output = self.norm(output)

        # gate
        output = u * output
        # outproj
        output = self.out_proj(output)

        return output

    def forward(self, x, context, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        if self.forward_type == 'left':
            return self.forward_left(x)
        elif self.forward_type == 'right':
            return self.forward_right(x)

class BasicTransformerBlock_(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention,  # ampere
        'linear': NormLinearAttention,
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
        attn_mode="softmax",
        sdp_backend=None,
        layer = 0,
    ):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        if attn_mode != "softmax" and not XFORMERS_IS_AVAILABLE:
            print(
                f"Attention mode '{attn_mode}' is not available. Falling back to native attention. "
                f"This is not a problem in Pytorch >= 2.0. FYI, you are running with PyTorch version {torch.__version__}"
            )
            attn_mode = "softmax"
        elif attn_mode == "softmax" and not SDP_IS_AVAILABLE:
            print(
                "We do not support vanilla attention anymore, as it is too expensive. Sorry."
            )
            if not XFORMERS_IS_AVAILABLE:
                assert (
                    False
                ), "Please install xformers via e.g. 'pip install xformers==0.0.16'"
            else:
                print("Falling back to xformers efficient attention.")
                attn_mode = "softmax-xformers"
        attn_cls = self.ATTENTION_MODES[attn_mode]
        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            assert sdp_backend is None or isinstance(sdp_backend, SDPBackend)
        else:
            assert sdp_backend is None
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
            backend=sdp_backend,
            layer = layer,
        )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            backend=sdp_backend,
            layer = layer
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        if self.checkpoint:
            print(f"{self.__class__.__name__} is using checkpointing")

    def forward(
        self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0
    ):
        kwargs = {"x": x}

        if context is not None:
            kwargs.update({"context": context})

        if additional_tokens is not None:
            kwargs.update({"additional_tokens": additional_tokens})

        if n_times_crossframe_attn_in_self:
            kwargs.update(
                {"n_times_crossframe_attn_in_self": n_times_crossframe_attn_in_self}
            )

        # return mixed_checkpoint(self._forward, kwargs, self.parameters(), self.checkpoint)
        if context is None:
            return checkpoint(
                self._forward, [x], self.parameters(), self.checkpoint
            )
        else:
            return checkpoint(
                self._forward, [x, context], self.parameters(), self.checkpoint
            )

    def _forward(
        self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0
    ):
        x = (
            self.attn1(
                self.norm1(x),
                context=context if self.disable_self_attn else None,
                additional_tokens=additional_tokens,
                n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self
                if not self.disable_self_attn
                else 0,
            )
            + x
        )
        x = (
            self.attn2(
                self.norm2(x), context=context, additional_tokens=additional_tokens
            )
            + x
        )
        x = self.ff(self.norm3(x)) + x
        return x



class SpatialTransformer_(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        attn_type="softmax",
        use_checkpoint=True,
        # sdp_backend=SDPBackend.FLASH_ATTENTION
        sdp_backend=None,
        layer = 0,
    ):
        super().__init__()
        print(
            f"constructing {self.__class__.__name__} of depth {depth} w/ {in_channels} channels and {n_heads} heads"
        )
        from omegaconf import ListConfig

        if exists(context_dim) and not isinstance(context_dim, (list, ListConfig)):
            context_dim = [context_dim]
        if exists(context_dim) and isinstance(context_dim, list):
            if depth != len(context_dim):
                print(
                    f"WARNING: {self.__class__.__name__}: Found context dims {context_dim} of depth {len(context_dim)}, "
                    f"which does not match the specified 'depth' of {depth}. Setting context_dim to {depth * [context_dim[0]]} now."
                )
                # depth does not match context dims.
                assert all(
                    map(lambda x: x == context_dim[0], context_dim)
                ), "need homogenous context_dim to match depth automatically"
                context_dim = depth * [context_dim[0]]
        elif context_dim is None:
            context_dim = [None] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock_(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    attn_mode=attn_type,
                    checkpoint=use_checkpoint,
                    sdp_backend=sdp_backend,
                    layer=layer
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            # self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            if i > 0 and len(context) == 1:
                i = 0  # use same context for each block
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in

class GroupConv(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size, stride=1,padding=0) -> None:
        super(GroupConv, self).__init__()
        self.conv = nn.Conv2d(3*in_channels, 3*out_channels, kernel_size, stride, padding,groups=3)
    def forward(self, data: Tensor, **kwargs) -> Tensor:
        data = torch.concat(torch.chunk(data,3,dim=-1),dim=1)
        data = self.conv(data)
        data = torch.concat(torch.chunk(data,3,dim=1),dim=-1)
        return data

class GroupConvTranspose(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size, stride=1,padding=0,output_padding=0) -> None:
        super(GroupConvTranspose, self).__init__()
        self.conv = nn.ConvTranspose2d(3*in_channels, 3*out_channels, kernel_size, stride, padding,output_padding,groups=3)
    def forward(self, data: Tensor, **kwargs) -> Tensor:
        data = torch.concat(torch.chunk(data,3,dim=-1),dim=1)
        data = self.conv(data)
        data = torch.concat(torch.chunk(data,3,dim=1),dim=-1)
        return data

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def normalization(channels, group_layer_num=32):
    return GroupNorm32(group_layer_num, channels)
class ResBlock_g(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        dropout,
        out_channels=None,
        use_conv=False,
        dims=2,
        use_checkpoint=False,
        group_layer_num_in=32,
        group_layer_num_out=32,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels if out_channels else channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            normalization(channels, group_layer_num_in),
            nn.SiLU(),
            GroupConv(channels, self.out_channels, 3, padding=1)
            # conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels, group_layer_num_out),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                GroupConv(self.out_channels, self.out_channels, 3, padding=1)
                # conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            # self.skip_connection = conv_nd(
            #     dims, channels, self.out_channels, 3, padding=1
            # )
            self.skip_connection = GroupConv(channels, self.out_channels, 3, padding=1)
        else:
            # self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)
            self.skip_connection = GroupConv(channels, self.out_channels,1)
    def forward(self, x):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
             self._forward, [x], self.parameters(), self.use_checkpoint
        )

    def _forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        h = self.skip_connection(x) + h
        return h


class VAE(nn.Module):
    def __init__(self, vae_config) -> None:
        super(VAE, self).__init__()

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 256, 256])
        z_shape = vae_config.get("z_shape", [4, 64, 64])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)

        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape

        self.kl_std = kl_std
        self.kl_weight = kl_weight

        hidden_dims = [512, 512, 1024, 1024, 1024, 1024, 1024, 2 * self.z_shape[0]]
        # feature size:  64,  32,  16,   8,    4,    8,   16,       32
        feature_size = [64, 32, 16, 8, 4, 8, 16, 32]
        #

        hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
        # feature size:  16,    8,   4,   8,    16,  32,  64

        self.in_layer = nn.Sequential(ResBlock_g(
            32,
            dropout=0,
            out_channels=128,
            use_conv=True,
            dims=2,
            use_checkpoint=False,
            group_layer_num_in=1
        ),
            nn.BatchNorm2d(128),
            nn.SiLU())

        #
        # self.spatial_modulation = nn.Linear(128*3, 128*3)

        # Build Encoder
        self.encoders_down = nn.ModuleList()
        in_channels = 128
        for i, h_dim in enumerate(hidden_dims[:1]):
            stride = 2
            modules = []
            modules.append(
                nn.Sequential(
                    # nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                    GroupConv(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU(),
                    ResBlock_g(
                        h_dim,
                        dropout=0,
                        out_channels=h_dim,
                        use_conv=True,
                        dims=2,
                        use_checkpoint=False,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU()),
            )
            in_channels = h_dim
            self.encoders_down.append(nn.Sequential(*modules))

        for i, h_dim in enumerate(hidden_dims[1:5]):
            dim_head = h_dim // num_heads
            self.encoders_down.append(nn.Sequential(
                # nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                GroupConv(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.SiLU(),
                SpatialTransformer_(h_dim,
                                   num_heads,
                                   dim_head,
                                   depth=transform_depth,
                                   context_dim=h_dim,
                                   disable_self_attn=False,
                                   use_linear=True,
                                   attn_type="linear",
                                   use_checkpoint=True,
                                   layer=feature_size[i + 1]
                                   ),
                nn.BatchNorm2d(h_dim),
                nn.SiLU()
            ))
            in_channels = h_dim

        # self.encoder_fpn = FPN_down([512, 512, 1024, 1024], [512, 1024, 1024])
        self.encoder_fpn = FPN_down_g([512, 512, 1024, 1024], [512, 1024, 1024])
        self.encoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims[5:]):
            modules = []
            if i > 0:
                in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            modules.append(nn.Sequential(
                                         GroupConvTranspose(in_channels,
                                                            h_dim,
                                                            kernel_size=3,
                                                            stride=2,
                                                            padding=1,
                                                            output_padding=1),
                                         nn.BatchNorm2d(h_dim),
                                         nn.SiLU()))
            if i == 2:
                modules.append(nn.Sequential(ResBlock_g(
                    h_dim,
                    dropout=0,
                    out_channels=2 * z_shape[0],
                    use_conv=True,
                    dims=2,
                    use_checkpoint=False,
                    group_layer_num_in = 1,
                    group_layer_num_out = 1
                ),
                    nn.BatchNorm2d(2 * z_shape[0]),
                    nn.SiLU()))
                in_channels = z_shape[0]
            else:
                modules.append(nn.Sequential(SpatialTransformer_(h_dim,
                                                                num_heads,
                                                                dim_head,
                                                                depth=transform_depth,
                                                                context_dim=h_dim,
                                                                disable_self_attn=False,
                                                                use_linear=True,
                                                                attn_type="linear",
                                                                use_checkpoint=True,
                                                                layer=feature_size[i + 5]
                                                                ),
                                             nn.BatchNorm2d(h_dim),
                                             nn.SiLU()))
                in_channels = h_dim
            self.encoders_up.append(nn.Sequential(*modules))

        ## build decoder
        hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
        # feature size:  16,    8,   4,   8,    16,  32,  64

        feature_size_decoder = [16, 8, 4, 8, 16, 32, 64]

        self.decoder_in_layer = nn.Sequential(ResBlock_g(
            self.z_shape[0],
            dropout=0,
            out_channels=512,
            use_conv=True,
            dims=2,
            use_checkpoint=False,
            group_layer_num_in=1
        ),
            nn.BatchNorm2d(512),
            nn.SiLU())

        self.decoders_down = nn.ModuleList()
        in_channels = 512
        for i, h_dim in enumerate(hidden_dims_decoder[0:3]):
            dim_head = h_dim // num_heads
            stride = 2
            self.decoders_down.append(nn.Sequential(
                # nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                GroupConv(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.SiLU(),
                SpatialTransformer_(h_dim,
                                   num_heads,
                                   dim_head,
                                   depth=transform_depth,
                                   context_dim=h_dim,
                                   disable_self_attn=False,
                                   use_linear=True,
                                   attn_type="linear",
                                   use_checkpoint=True,
                                   layer=feature_size_decoder[i]
                                   ),
                nn.BatchNorm2d(h_dim),
                nn.SiLU()
            ))
            in_channels = h_dim

        # self.decoder_fpn = FPN_up([1024, 1024, 1024, 512], [1024, 1024, 512])
        self.decoder_fpn = FPN_up_g([1024, 1024, 1024, 512], [1024, 1024, 512])
        self.decoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims_decoder[3:]):
            modules = []
            if i > 0 and i < 4:
                in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            modules.append(nn.Sequential(
                                        GroupConvTranspose( in_channels,
                                                            h_dim,
                                                            kernel_size=3,
                                                            stride=2,
                                                            padding=1,
                                                            output_padding=1),
                                         nn.BatchNorm2d(h_dim),
                                         nn.SiLU()))
            if i < 4:
                modules.append(nn.Sequential(SpatialTransformer_(h_dim,
                                                                num_heads,
                                                                dim_head,
                                                                depth=transform_depth,
                                                                context_dim=h_dim,
                                                                disable_self_attn=False,
                                                                use_linear=True,
                                                                attn_type="linear",
                                                                use_checkpoint=True,
                                                                layer=feature_size_decoder[i + 3]
                                                                ),
                                             nn.BatchNorm2d(h_dim),
                                             nn.SiLU()))
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(ResBlock_g(
                    h_dim,
                    dropout=0,
                    out_channels=h_dim,
                    use_conv=True,
                    dims=2,
                    use_checkpoint=False,
                ),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU()))
                in_channels = h_dim
            self.decoders_up.append(nn.Sequential(*modules))

        self.decoders_up.append(nn.Sequential(
            GroupConvTranspose(in_channels,
                               in_channels,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            ResBlock_g(
                in_channels,
                dropout=0,
                out_channels=self.plane_shape[1],
                use_conv=True,
                dims=2,
                use_checkpoint=False,
            ),
            nn.BatchNorm2d(self.plane_shape[1]),
            nn.Tanh()))

    def encode(self, enc_input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :enc_input: (Tensor) Input tensor to encoder [B x D x resolution x resolution]
        :return: (Tensor) List of latent codes
        """
        result = enc_input
        if self.plane_dim == 5:
            result = torch.concat(torch.chunk(result,3,dim=1),dim=-1).squeeze(1)
        elif self.plane_dim == 4:
            result = torch.concat(torch.chunk(result,3,dim=1),dim=-1)

        feature = self.in_layer(result)
        features_down = []
        for i, module in enumerate(self.encoders_down):
            feature = module(feature)
            if i in [0, 1, 2, 3]:
                features_down.append(feature)

        features_down = self.encoder_fpn(features_down)

        # breakpoint()

        feature = self.encoders_up[0](feature)
        feature = torch.cat([feature, features_down[-1]], dim=1)
        feature = self.encoders_up[1](feature)
        feature = torch.cat([feature, features_down[-2]], dim=1)
        feature = self.encoders_up[2](feature)

        encode_channel = self.z_shape[0]
        mu = feature[:, :encode_channel, ...]
        log_var = feature[:, encode_channel:, ...]

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        x = self.decoder_in_layer(z)
        feature_down = [x]
        for i, module in enumerate(self.decoders_down):
            x = module(x)
            feature_down.append(x)
        feature_down = self.decoder_fpn(feature_down[::-1])
        for i, module in enumerate(self.decoders_up):
            if i in [1, 2, 3]:
                x = torch.cat([x, feature_down[-i]], dim=1)
                x = module(x)
            else:
                x = module(x)
        if self.plane_dim == 5:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0: plane_w].unsqueeze(1),
                              x[..., plane_w: plane_w * 2].unsqueeze(1),
                              x[..., plane_w * 2: plane_w * 3].unsqueeze(1), ], dim=1)
        elif self.plane_dim == 4:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0: plane_w],
                              x[..., plane_w: plane_w * 2],
                              x[..., plane_w * 2: plane_w * 3], ], dim=1)
        return x

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, data: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(data)
        z = self.reparameterize(mu, log_var)
        result = self.decode(z)

        return [result, data, mu, log_var, z]

    # only using VAE loss
    def loss_function(self,
                      *args) -> dict:
        mu = args[2]
        log_var = args[3]

        if self.kl_std == 'zero_mean':
            latent = self.reparameterize(mu, log_var)
            # print("latent shape: ", latent.shape) # (B, dim)
            l2_size_loss = torch.sum(torch.norm(latent, dim=-1))
            kl_loss = l2_size_loss / latent.shape[0]

        else:
            std = torch.exp(torch.clamp(0.5 * log_var, max=10)) + 1e-6
            gt_dist = torch.distributions.normal.Normal(torch.zeros_like(mu), torch.ones_like(std) * self.kl_std)
            sampled_dist = torch.distributions.normal.Normal(mu, std)
            # gt_dist = normal_dist.sample(log_var.shape)
            # print("gt dist shape: ", gt_dist.shape)

            kl = torch.distributions.kl.kl_divergence(sampled_dist, gt_dist)  # reversed KL
            kl_loss = reduce(kl, 'b ... -> b (...)', 'mean').mean()

        return self.kl_weight * kl_loss

    def sample(self,
               num_samples: int,
               **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """

        z_rollout_shape = [self.z_shape[0], self.z_shape[1], self.z_shape[2] * 3]
        gt_dist = torch.distributions.normal.Normal(torch.zeros(num_samples, *(z_rollout_shape)),
                                                    torch.ones(num_samples, *(z_rollout_shape)) * self.kl_std)

        z = gt_dist.sample().cuda()
        samples = self.decode(z)
        return samples, z

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def get_latent(self, x):
        '''
        given input x, return the latent code
        x:  [B x C x H x W]
        return: [B x latent_dim]
        '''
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z

# debug vae

# if __name__ == "__main__":
#     vae_config = {"kl_std": 0.25,
#                   "kl_weight": 0.001,
#                   "plane_shape": [3, 32, 128, 128],
#                   "z_shape": [4, 32, 32],
#                   "num_heads": 16,
#                   "transform_depth": 1}
#
#     vae_model = VAE(vae_config).cuda()
#
#     input_tensor = torch.randn(2, 3, 32, 128, 128).cuda()
#     out = vae_model(input_tensor)
#     loss = vae_model.loss_function(*out)
#     print("loss: {}".format(loss))
#     print("z shape: {}".format(out[-1].shape))
#     print("reconstruct shape: {}".format(out[0].shape))
#     samples = vae_model.sample(2)
#     print("samples shape: {}".format(samples[0].shape))