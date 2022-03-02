import os, math
import logging, warnings
# from einops import rearrange
from paddle import nn, Tensor
from functools import partial
from typing import Optional, Tuple
import collections.abc
from itertools import repeat

BN_MOMENTUM = 0.1

def _make_layer(
        block,
        inplanes,
        planes,
        blocks,
        input_resolution=None,
        num_heads=1,
        stride=1,
        window_size=7,
        halo_size=1,
        mlp_ratio=4.0,
        q_dilation=1,
        kv_dilation=1,
        sr_ratio=1,
        attn_type="msw",
    ):
        downsample = None

        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias_attr=False,
                ),
                nn.BatchNorm2D(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = []

        if isinstance(block, GeneralTransformerBlock):
            layers.append(
                block(
                    inplanes,
                    planes,
                    num_heads,
                    window_size,
                    halo_size,
                    mlp_ratio,
                    q_dilation,
                    kv_dilation,
                    sr_ratio,
                    attn_type,
                )
            )
        else:
            layers.append(block(inplanes, planes, stride, downsample))

        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

# region bottleneck_block.py
class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2D(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias_attr=False
        )
        self.bn2 = nn.BatchNorm2D(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2D(
            planes, planes * self.expansion, kernel_size=1, bias_attr=False
        )
        self.bn3 = nn.BatchNorm2D(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BottleneckDWP(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckDWP, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1,  bias_attr=False,)
        self.bn1 = nn.BatchNorm2D(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2D(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias_attr=False,
            groups=planes,
        )
        self.bn2 = nn.BatchNorm2D(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2D(
            planes, planes * self.expansion, kernel_size=1,  bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
# endregion



# region transformer_block.py
class GeneralTransformerBlock(nn.Layer):
    expansion = 1
    def __init__(
        self,
        inplanes,
        planes,
        input_resolution,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias_attr=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        attn_type="isa_local",
        ffn_type="conv_mlp",
    ):
        super().__init__()
        self.dim = inplanes
        self.out_dim = planes
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.attn_type = attn_type
        self.ffn_type = ffn_type
        self.mlp_ratio = mlp_ratio

        if self.attn_type in ["conv"]:
            """modified basic block with seperable 3x3 convolution"""
            self.sep_conv1 = nn.Sequential(
                nn.Conv2D(
                    inplanes,
                    planes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=inplanes,
                    bias_attr=None,
                ),
                nn.BatchNorm2D(planes),
                nn.Conv2D(planes, planes, kernel_size=1, stride=1, bias_attr=None),
                nn.BatchNorm2D(planes),
                nn.ReLU(),
            )
            self.sep_conv2 = nn.Sequential(
                nn.Conv2D(
                    planes,
                    planes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=planes,
                    bias_attr=None,
                ),
                nn.BatchNorm2D(planes),
                nn.Conv2D(planes, planes, kernel_size=1, stride=1, bias_attr=None),
                nn.BatchNorm2D(planes),
            )
            self.relu = nn.ReLU()
        elif self.attn_type in ["isa_local"]:
            self.attn = MultiheadISAAttention(
                self.dim,
                num_heads=num_heads,
                window_size=window_size,
                attn_type=attn_type,
                rpe=True,
                dropout=attn_drop,
            )
            self.norm1 = norm_layer(self.dim)
            self.norm2 = norm_layer(self.out_dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            mlp_hidden_dim = int(self.dim * mlp_ratio)

            if self.ffn_type in ["conv_mlp"]:
                self.mlp = MlpDWBN(
                    in_features=self.dim,
                    hidden_features=mlp_hidden_dim,
                    out_features=self.out_dim,
                    act_layer=act_layer,
                    drop=drop,
                )
            elif self.ffn_type in ["identity"]:
                self.mlp = nn.Identity()
            else:
                raise RuntimeError("Unsupported ffn type: {}".format(self.ffn_type))

        else:
            raise RuntimeError("Unsupported attention type: {}".format(self.attn_type))

    def forward(self, x):
        if self.attn_type in ["conv"]:
            residual = x
            out = self.sep_conv1(x)
            out = self.sep_conv2(out)
            out += residual
            out = self.relu(out)
            return out
        elif self.attn_type in ["isa_local"]:
            B, C, H, W = x.shape
            # reshape
            x = x.reshape((B, C, -1)).transpose((0, 2, 1))
            # Attention
            x = x + self.drop_path(self.attn(self.norm1(x), H, W))
            # FFN
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
            # reshape
            x = x.transpose((0, 2, 1)).reshape((B, C, H, W))
            return x
        else:
            B, C, H, W = x.shape
            # reshape
            x = x.reshape((B, C, -1)).transpose((0, 2, 1))
            # Attention
            x = x + self.drop_path(self.attn(self.norm1(x)))
            # FFN
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
            # reshape
            x = x.transpose((0, 2, 1)).reshape((B, C, H, W))
            return x

# region drop
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob, dtype='float32')
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor.floor_()  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
# endregion

# region
class Mlp(nn.Layer):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MlpDWBN(nn.Layer):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        dw_act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2D(in_features, hidden_features, kernel_size=1)
        self.act1 = act_layer()
        self.norm1 = nn.BatchNorm2D(hidden_features)
        self.dw3x3 = nn.Conv2D(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            groups=hidden_features,
            padding=1,
        )
        self.act2 = dw_act_layer()
        self.norm2 = nn.BatchNorm2D(hidden_features)
        self.fc2 = nn.Conv2D(hidden_features, out_features, kernel_size=1)
        self.act3 = act_layer()
        self.norm3 = nn.BatchNorm2D(out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        if len(x.shape) == 3:
            B, N, C = x.shape
            if N == (H * W + 1):
                cls_tokens = x[:, 0, :]
                x_ = x[:, 1:, :].transpose((0, 2, 1)).reshape((B, C, H, W))
            else:
                x_ = x.transpose((0, 2, 1)).reshape((B, C, H, W))

            x_ = self.fc1(x_)
            x_ = self.norm1(x_)
            x_ = self.act1(x_)
            x_ = self.dw3x3(x_)
            x_ = self.norm2(x_)
            x_ = self.act2(x_)
            x_ = self.drop(x_)
            x_ = self.fc2(x_)
            x_ = self.norm3(x_)
            x_ = self.act3(x_)
            x_ = self.drop(x_)
            x_ = x_.reshape((B, C, -1)).transpose((0, 2, 1))
            if N == (H * W + 1):
                x = torch.cat((cls_tokens.unsqueeze(1), x_), axis=1)
            else:
                x = x_
            return x

        elif len(x.shape) == 4:
            x = self.fc1(x)
            x = self.norm1(x)
            x = self.act1(x)
            x = self.dw3x3(x)
            x = self.norm2(x)
            x = self.act2(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.norm3(x)
            x = self.act3(x)
            x = self.drop(x)
            return x

        else:
            raise RuntimeError("Unsupported input shape: {}".format(x.shape))
# endregion

# endregion


# region multihead_isa_attention

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)
class PadBlock(object):
    """ "Make the size of feature map divisible by local group size."""

    def __init__(self, local_group_size=7):
        self.lgs = local_group_size
        if not isinstance(self.lgs, (tuple, list)):
            self.lgs = to_2tuple(self.lgs)
        assert len(self.lgs) == 2

    def pad_if_needed(self, x, size):
        n, h, w, c = size
        pad_h = math.ceil(h / self.lgs[0]) * self.lgs[0] - h
        pad_w = math.ceil(w / self.lgs[1]) * self.lgs[1] - w
        if pad_h > 0 or pad_w > 0:  # center-pad the feature on H and W axes
            return F.pad(
                x,
                # (0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            )
        return x

    def depad_if_needed(self, x, size):
        n, h, w, c = size
        pad_h = math.ceil(h / self.lgs[0]) * self.lgs[0] - h
        pad_w = math.ceil(w / self.lgs[1]) * self.lgs[1] - w
        if pad_h > 0 or pad_w > 0:  # remove the center-padding on feature
            return x[:, pad_h // 2 : pad_h // 2 + h, pad_w // 2 : pad_w // 2 + w, :]
        return x


class LocalPermuteModule(object):
    """ "Permute the feature map to gather pixels in local groups, and the reverse permutation"""

    def __init__(self, local_group_size=7):
        self.lgs = local_group_size
        if not isinstance(self.lgs, (tuple, list)):
            self.lgs = to_2tuple(self.lgs)
        assert len(self.lgs) == 2

    def permute(self, x, size):
        n, h, w, c = size
        qh=h // self.lgs[0]
        ph=self.lgs[0]
        qw=w // self.lgs[0]
        pw=self.lgs[0]
        x = x.reshape((n, qh, ph, qw, pw, c))
        x = x.transpose((2, 4, 0, 1, 3, 5))
        x = x.reshape((ph*pw, n*qh*qw, c))
        return x


    def rev_permute(self, x, size):
        n, h, w, c = size
        qh=h // self.lgs[0]
        ph=self.lgs[0]
        qw=w // self.lgs[0]
        pw=self.lgs[0]
        x = x.reshape((ph, pw, n, qh,  qw,  c))
        x = x.transpose((2, 3, 0, 4, 1, 5))
        x = x.reshape((n, qh*ph, qw*pw, c))
        return x



class MultiheadISAAttention(nn.Layer):
    r"""interlaced sparse multi-head self attention (ISA) module with relative position bias.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): Window size.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        window_size=7,
        attn_type="isa_local",
        rpe=True,
        **kwargs,
    ):
        super(MultiheadISAAttention, self).__init__()

        self.dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.attn_type = attn_type
        self.with_rpe = rpe

        self.attn = MultiheadAttentionRPE(
            embed_dim, num_heads, rpe=rpe, window_size=window_size, **kwargs
        )
        self.pad_helper = PadBlock(window_size)
        assert attn_type in ["isa_local"]
        if attn_type == "isa_local":
            self.permute_helper = LocalPermuteModule(window_size)
        else:
            raise NotImplementedError("We only support ['isa_local'] Now.")

    def forward(self, x, H, W, **kwargs):
        # H, W = self.input_resolution
        B, N, C = x.shape
        x = x.reshape((B, H, W, C))
        # attention
        if self.attn_type in ["isa_local"]:
            # pad
            x_pad = self.pad_helper.pad_if_needed(x, x.shape)
            # permute
            x_permute = self.permute_helper.permute(x_pad, x_pad.shape)
            # attention
            out, _, _ = self.attn(
                x_permute, x_permute, x_permute, rpe=self.with_rpe, **kwargs
            )
            # reverse permutation
            out = self.permute_helper.rev_permute(out, x_pad.shape)
        else:
            raise NotImplementedError("We only support ['isa_local'] Now.")
        # de-pad, pooling with `ceil_mode=True` will do implicit padding, so we need to remove it, too
        out = self.pad_helper.depad_if_needed(out, x.shape)
        return out.reshape((B, N, C))

    def extra_repr(self) -> str:
        return f"axis={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops
# endregion


# region multihead_attention.py
import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle.nn.functional import linear, pad, softmax, dropout

class MultiheadAttention(nn.Layer):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias_attr=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
    ):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias_attr=bias_attr)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias_attr=bias_attr)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias_attr=bias_attr)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.in_proj_bias = None
        self.in_proj_weight = None
        self.bias_k = self.bias_v = None
        self.q_proj_weight = None
        self.k_proj_weight = None
        self.v_proj_weight = None
        self.add_zero_attn = add_zero_attn

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=False,
        attn_mask=None,
        residual_attn=None,
    ):
        if not self._qkv_same_embed_dim:
            return self.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                out_axis=self.vdim,
                residual_attn=residual_attn,
            )
        else:
            return self.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                out_axis=self.vdim,
                residual_attn=residual_attn,
            )

    def multi_head_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Tensor,
        bias_k: Optional[Tensor],
        bias_v: Optional[Tensor],
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Tensor,
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight: Optional[Tensor] = None,
        k_proj_weight: Optional[Tensor] = None,
        v_proj_weight: Optional[Tensor] = None,
        static_k: Optional[Tensor] = None,
        static_v: Optional[Tensor] = None,
        out_dim: Optional[Tensor] = None,
        residual_attn: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # if not torch.jit.is_scripting():
        #     tens_ops = (
        #         query,
        #         key,
        #         value,
        #         in_proj_weight,
        #         in_proj_bias,
        #         bias_k,
        #         bias_v,
        #         out_proj_weight,
        #         out_proj_bias,
        #     )
        #     if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(
        #         tens_ops
        #     ):
        #         return handle_torch_function(
        #             multi_head_attention_forward,
        #             tens_ops,
        #             query,
        #             key,
        #             value,
        #             embed_dim_to_check,
        #             num_heads,
        #             in_proj_weight,
        #             in_proj_bias,
        #             bias_k,
        #             bias_v,
        #             add_zero_attn,
        #             dropout_p,
        #             out_proj_weight,
        #             out_proj_bias,
        #             training=training,
        #             key_padding_mask=key_padding_mask,
        #             need_weights=need_weights,
        #             attn_mask=attn_mask,
        #             use_separate_proj_weight=use_separate_proj_weight,
        #             q_proj_weight=q_proj_weight,
        #             k_proj_weight=k_proj_weight,
        #             v_proj_weight=v_proj_weight,
        #             static_k=static_k,
        #             static_v=static_v,
        #         )
        tgt_len, bsz, embed_dim = query.shape
        key = query if key is None else key
        value = query if value is None else value

        assert embed_dim == embed_dim_to_check
        # allow MHA to have different sizes for the feature dimension
        assert key.shape[0] == value.shape[0] and key.shape[1] == value.shape[1]

        head_dim = embed_dim // num_heads
        v_head_dim = out_dim // num_heads
        assert (
            head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        q = self.q_proj(query) * scaling
        k = self.k_proj(key)
        v = self.v_proj(value)

        if attn_mask is not None:
            assert (
                attn_mask.dtype == torch.float32
                or attn_mask.dtype == torch.float64
                or attn_mask.dtype == torch.float16
                or attn_mask.dtype == torch.uint8
                or attn_mask.dtype == torch.bool
            ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(
                attn_mask.dtype
            )
            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
                )
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.shape) != [1, query.shape[0], key.shape[0]]:
                    raise RuntimeError("The size of the 2D attn_mask is not correct.")
            elif attn_mask.dim() == 3:
                if list(attn_mask.shape) != [
                    bsz * num_heads,
                    query.shape[0],
                    key.shape[0],
                ]:
                    raise RuntimeError("The size of the 3D attn_mask is not correct.")
            else:
                raise RuntimeError(
                    "attn_mask's dimension {} is not supported".format(attn_mask.dim())
                )

        # convert ByteTensor key_padding_mask to bool
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
            )
            key_padding_mask = key_padding_mask.to(torch.bool)

        q = q.reshape((tgt_len, bsz * num_heads, head_dim)).transpose((1, 0, 2))
        if k is not None:
            k = k.reshape((-1, bsz * num_heads, head_dim)).transpose((1, 0, 2))
        if v is not None:
            v = v.reshape((-1, bsz * num_heads, v_head_dim)).transpose((1, 0, 2))

        src_len = k.shape[1]

        if key_padding_mask is not None:
            assert key_padding_mask.shape[0] == bsz
            assert key_padding_mask.shape[1] == src_len

        if add_zero_attn:
            src_len += 1
            k = torch.cat(
                [
                    k,
                    torch.zeros(
                        (k.shape[0], 1) + k.shape[2:], dtype=k.dtype, device=k.device
                    ),
                ],
                axis=1,
            )
            v = torch.cat(
                [
                    v,
                    torch.zeros(
                        (v.shape[0], 1) + v.shape[2:], dtype=v.dtype, device=v.device
                    ),
                ],
                axis=1,
            )
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))

        attn_output_weights = paddle.bmm(q, k.transpose((0, 2, 1)))
        assert list(attn_output_weights.shape) == [bsz * num_heads, tgt_len, src_len]

        """
        Attention weight for the invalid region is -inf
        """
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.reshape((
                bsz, num_heads, tgt_len, src_len
            ))
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.reshape((
                bsz * num_heads, tgt_len, src_len
            ))

        if residual_attn is not None:
            attn_output_weights = attn_output_weights.reshape((
                bsz, num_heads, tgt_len, src_len
            ))
            attn_output_weights += residual_attn.unsqueeze(0)
            attn_output_weights = attn_output_weights.reshape((
                bsz * num_heads, tgt_len, src_len
            ))

        """
        Reweight the attention map before softmax().
        attn_output_weights: (b*n_head, n, hw)
        """
        attn_output_weights = softmax(attn_output_weights, axis=-1)
        attn_output_weights = dropout(
            attn_output_weights, p=dropout_p, training=training
        )

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.shape) == [bsz * num_heads, tgt_len, v_head_dim]
        attn_output = (
            attn_output.transpose((1, 0, 2)).reshape((tgt_len, bsz, out_dim))
        )
        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.reshape((
                bsz, num_heads, tgt_len, src_len
            ))
            return attn_output, attn_output_weights.sum(axis=1) / num_heads
        else:
            return attn_output


class MultiheadAttentionRPE(MultiheadAttention):
    """ "Multihead Attention with extra flags on the q/k/v and out projections."""

    def __init__(self, *args, rpe=False, window_size=7, **kwargs):
        super(MultiheadAttentionRPE, self).__init__(*args, **kwargs)

        self.rpe = rpe
        if rpe:
            self.window_size = [window_size] * 2
            # define a parameter table of relative position bias
            self.relative_position_bias_table = paddle.create_parameter(
                shape=[(2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads],
                # default_initializer=nn.initializer.Constant(value=0.0),
                default_initializer=nn.initializer.TruncatedNormal(mean=0.0, std=0.02),
                dtype='float32',
                )# 2*Wh-1 * 2*Ww-1, nH
            # get pair-wise relative position index for each token inside the window
            coords_h = paddle.arange(0, self.window_size[0])
            coords_w = paddle.arange(0, self.window_size[1])
            coords = paddle.stack(paddle.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = (
                coords_flatten[:, :, None] - coords_flatten[:, None, :]
            )  # 2, Wh*Ww, Wh*Ww
            # relative_coords = relative_coords.permute(
            #     1, 2, 0
            # )  # Wh*Ww, Wh*Ww, 2
            relative_coords = relative_coords.transpose((1, 2, 0))
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)
            # trunc_normal_(self.relative_position_bias_table, std=0.02)
            # https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/initializer/TruncatedNormal_cn.html#truncatednormal

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=False,
        attn_mask=None,
        do_qkv_proj=True,
        do_out_proj=True,
        rpe=True,
    ):
        if not self._qkv_same_embed_dim:
            return self.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                out_dim=self.vdim,
                do_qkv_proj=do_qkv_proj,
                do_out_proj=do_out_proj,
                rpe=rpe,
            )
        else:
            return self.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                out_dim=self.vdim,
                do_qkv_proj=do_qkv_proj,
                do_out_proj=do_out_proj,
                rpe=rpe,
            )

    def multi_head_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Tensor,
        bias_k: Optional[Tensor],
        bias_v: Optional[Tensor],
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Tensor,
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight: Optional[Tensor] = None,
        k_proj_weight: Optional[Tensor] = None,
        v_proj_weight: Optional[Tensor] = None,
        static_k: Optional[Tensor] = None,
        static_v: Optional[Tensor] = None,
        out_dim: Optional[Tensor] = None,
        do_qkv_proj: bool = True,
        do_out_proj: bool = True,
        rpe=True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # if not torch.jit.is_scripting():
        #     tens_ops = (
        #         query,
        #         key,
        #         value,
        #         in_proj_weight,
        #         in_proj_bias,
        #         bias_k,
        #         bias_v,
        #         out_proj_weight,
        #         out_proj_bias,
        #     )
        #     if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(
        #         tens_ops
        #     ):
        #         return handle_torch_function(
        #             multi_head_attention_forward,
        #             tens_ops,
        #             query,
        #             key,
        #             value,
        #             embed_dim_to_check,
        #             num_heads,
        #             in_proj_weight,
        #             in_proj_bias,
        #             bias_k,
        #             bias_v,
        #             add_zero_attn,
        #             dropout_p,
        #             out_proj_weight,
        #             out_proj_bias,
        #             training=training,
        #             key_padding_mask=key_padding_mask,
        #             need_weights=need_weights,
        #             attn_mask=attn_mask,
        #             use_separate_proj_weight=use_separate_proj_weight,
        #             q_proj_weight=q_proj_weight,
        #             k_proj_weight=k_proj_weight,
        #             v_proj_weight=v_proj_weight,
        #             static_k=static_k,
        #             static_v=static_v,
        #         )

        tgt_len, bsz, embed_dim = query.shape
        key = query if key is None else key
        value = query if value is None else value

        assert embed_dim == embed_dim_to_check
        # allow MHA to have different sizes for the feature dimension
        assert key.shape[0] == value.shape[0] and key.shape[1] == value.shape[1]

        head_dim = embed_dim // num_heads
        v_head_dim = out_dim // num_heads
        assert (
            head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        # whether or not use the original query/key/value
        q = self.q_proj(query) * scaling if do_qkv_proj else query
        k = self.k_proj(key) if do_qkv_proj else key
        v = self.v_proj(value) if do_qkv_proj else value

        if attn_mask is not None:
            assert (
                attn_mask.dtype == torch.float32
                or attn_mask.dtype == torch.float64
                or attn_mask.dtype == torch.float16
                or attn_mask.dtype == torch.uint8
                or attn_mask.dtype == torch.bool
            ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(
                attn_mask.dtype
            )
            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
                )
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.shape) != [1, query.shape[0], key.shape[0]]:
                    raise RuntimeError("The size of the 2D attn_mask is not correct.")
            elif attn_mask.dim() == 3:
                if list(attn_mask.shape) != [
                    bsz * num_heads,
                    query.shape[0],
                    key.shape[0],
                ]:
                    raise RuntimeError("The size of the 3D attn_mask is not correct.")
            else:
                raise RuntimeError(
                    "attn_mask's dimension {} is not supported".format(attn_mask.dim())
                )

        # convert ByteTensor key_padding_mask to bool
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
            )
            key_padding_mask = key_padding_mask.to(torch.bool)

        q = q.reshape((tgt_len, bsz * num_heads, head_dim)).transpose((1, 0, 2))
        if k is not None:
            k = k.reshape((-1, bsz * num_heads, head_dim)).transpose((1, 0, 2))
        if v is not None:
            v = v.reshape((-1, bsz * num_heads, v_head_dim)).transpose((1, 0, 2))

        src_len = k.shape[1]

        if key_padding_mask is not None:
            assert key_padding_mask.shape[0] == bsz
            assert key_padding_mask.shape[1] == src_len

        if add_zero_attn:
            src_len += 1
            k = torch.cat(
                [
                    k,
                    torch.zeros(
                        (k.shape[0], 1) + k.shape[2:], dtype=k.dtype, device=k.device
                    ),
                ],
                axis=1,
            )
            v = torch.cat(
                [
                    v,
                    torch.zeros(
                        (v.shape[0], 1) + v.shape[2:], dtype=v.dtype, device=v.device
                    ),
                ],
                axis=1,
            )
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))

        attn_output_weights = paddle.bmm(q, k.transpose((0, 2, 1)))
        assert list(attn_output_weights.shape) == [bsz * num_heads, tgt_len, src_len]

        """
        Add relative position embedding
        """
        if self.rpe and rpe:
            # NOTE: for simplicity, we assume the src_len == tgt_len == window_size**2 here
            assert (
                src_len == self.window_size[0] * self.window_size[1]
                and tgt_len == self.window_size[0] * self.window_size[1]
            ), f"src{src_len}, tgt{tgt_len}, window{self.window_size[0]}"
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.reshape([-1])
            ].reshape((
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            ))  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.transpose((
                2, 0, 1
            ))  # nH, Wh*Ww, Wh*Ww
            attn_output_weights = attn_output_weights.reshape((
                bsz, num_heads, tgt_len, src_len
            )) + relative_position_bias.unsqueeze(0)
            attn_output_weights = attn_output_weights.reshape((
                bsz * num_heads, tgt_len, src_len
            ))

        """
        Attention weight for the invalid region is -inf
        """
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.reshape((
                bsz, num_heads, tgt_len, src_len
            ))
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.reshape((
                bsz * num_heads, tgt_len, src_len
            ))

        """
        Reweight the attention map before softmax().
        attn_output_weights: (b*n_head, n, hw)
        """
        attn_output_weights = softmax(attn_output_weights, axis=-1)
        attn_output_weights = dropout(
            attn_output_weights, p=dropout_p, training=training
        )

        attn_output = paddle.bmm(attn_output_weights, v)
        assert list(attn_output.shape) == [bsz * num_heads, tgt_len, v_head_dim]
        attn_output = (
            attn_output.transpose((1, 0, 2)).reshape((tgt_len, bsz, out_dim))
        )
        if do_out_proj:
            attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.reshape((
                bsz, num_heads, tgt_len, src_len
            ))
            return attn_output, q, k, attn_output_weights.sum(axis=1) / num_heads
        else:
            return attn_output, q, k  # additionaly return the query and key


# endregion

