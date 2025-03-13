"""
Point Transformer - V3 Mode1
Pointcept detached version

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import sys
from functools import partial
from addict import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
import torch_scatter
from timm.models.layers import DropPath
from collections import OrderedDict
from easydict import EasyDict

import copy

try:
    import flash_attn
except ImportError:
    flash_attn = None

from .serialization import encode

class RotaryPositionEncoding3D(nn.Module):

    def __init__(self, feature_dim, pe_type="Rotary3D"):
        super().__init__()
        self.feature_dim = feature_dim
        self.pe_type = pe_type

    @torch.no_grad()
    def forward(self, XYZ):
        """
        @param XYZ: [B,N,3]
        @return:
        """
        npoint, _ = XYZ.shape
        x_position, y_position, z_position = XYZ[..., 0:1], XYZ[..., 1:2], XYZ[..., 2:3]
        div_term = torch.exp(
            torch.arange(
                0, self.feature_dim // 3, 2, dtype=torch.float, device=XYZ.device
            )
            * (-math.log(10000.0) / (self.feature_dim // 3))
        )
        div_term = div_term.view(1, 1, -1)  # [1, 1, d//6]

        sinx = torch.sin(x_position * div_term)  # [B, N, d//6]
        cosx = torch.cos(x_position * div_term)
        siny = torch.sin(y_position * div_term)
        cosy = torch.cos(y_position * div_term)
        sinz = torch.sin(z_position * div_term)
        cosz = torch.cos(z_position * div_term)

        sinx, cosx, siny, cosy, sinz, cosz = map(
            lambda feat: torch.stack([feat, feat], -1).view(npoint, -1),
            [sinx, cosx, siny, cosy, sinz, cosz],
        )

        cos_pos = torch.cat([cosx, cosy, cosz], dim=-1).detach()
        sin_pos = torch.cat([sinx, siny, sinz], dim=-1).detach()
        
        return cos_pos.unsqueeze(1) , sin_pos.unsqueeze(1) 


def embed_rotary(x, cos, sin):
    x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()
    x = x * cos + x2 * sin
    return x


@torch.inference_mode()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.inference_mode()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.inference_mode()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


class Point(Dict):
    """
    Point Structure of Pointcept

    A Point (point cloud) in Pointcept is a dictionary that contains various properties of
    a batched point cloud. The property with the following names have a specific definition
    as follows:

    - "coord": original coordinate of point cloud;
    - "grid_coord": grid coordinate for specific grid size (related to GridSampling);
    Point also support the following optional attributes:
    - "offset": if not exist, initialized as batch size is 1;
    - "batch": if not exist, initialized as batch size is 1;
    - "feat": feature of point cloud, default input of model;
    - "grid_size": Grid size of point cloud (related to GridSampling);
    (related to Serialization)
    - "serialized_depth": depth of serialization, 2 ** depth * grid_size describe the maximum of point cloud range;
    - "serialized_code": a list of serialization codes;
    - "serialized_order": a list of serialization order determined by code;
    - "serialized_inverse": a list of inverse mapping determined by code;
    (related to Sparsify: SpConv)
    - "sparse_shape": Sparse shape for Sparse Conv Tensor;
    - "sparse_conv_feat": SparseConvTensor init with information provide by Point;
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If one of "offset" or "batch" do not exist, generate by the existing one
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        """
        Point Cloud Serialization

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]
        """
        assert "batch" in self.keys()
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust `grid_size` to what your want)
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        if depth is None:
            # Adaptive measure the depth of serialization cube (length = 2 ^ depth)
            depth = int(self.grid_coord.max()).bit_length()
        self["serialized_depth"] = depth
        # Maximum bit length for serialization code is 63 (int64)
        assert depth * 3 + len(self.offset).bit_length() <= 63
        # Here we follow OCNN and set the depth limitation to 16 (48bit) for the point position.
        # Although depth is limited to less than 16, we can encode a 655.36^3 (2^16 * 0.01) meter^3
        # cube with a grid size of 0.01 meter. We consider it is enough for the current stage.
        # We can unlock the limitation by optimizing the z-order encoding function if necessary.
        assert depth <= 16

        # The serialization codes are arranged as following structures:
        # [Order1 ([n]),
        #  Order2 ([n]),
        #   ...
        #  OrderN ([n])] (k, n)
        code = [
            encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order
        ]
        code = torch.stack(code)
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        self["serialized_code"] = code
        self["serialized_order"] = order
        self["serialized_inverse"] = inverse

    def sparsify(self, pad=96):
        """
        Point Cloud Serialization

        Point cloud is sparse, here we use "sparsify" to specifically refer to
        preparing "spconv.SparseConvTensor" for SpConv.

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]

        pad: padding sparse for sparse shape.
        """
        assert {"feat", "batch"}.issubset(self.keys())
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust `grid_size` to what your want)
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()
        if "sparse_shape" in self.keys():
            sparse_shape = self.sparse_shape
        else:
            sparse_shape = torch.add(
                torch.max(self.grid_coord, dim=0).values, pad
            ).tolist()
        sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=torch.cat(
                [self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=self.batch[-1].tolist() + 1,
        )
        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = sparse_conv_feat


class PointModule(nn.Module):
    r"""PointModule
    placeholder, all module subclass from this will take Point in PointSequential.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PointSequential(PointModule):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for k, module in self._modules.items():
            # Point module
            if isinstance(module, PointModule):
                input = module(input)
            # Spconv module
            elif spconv.modules.is_spconv_module(module):
                if isinstance(input, Point):
                    input.sparse_conv_feat = module(input.sparse_conv_feat)
                    input.feat = input.sparse_conv_feat.features
                else:
                    input = module(input)
            # PyTorch module
            else:
                if isinstance(input, Point):
                    input.feat = module(input.feat)
                    if "sparse_conv_feat" in input.keys():
                        input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(
                            input.feat
                        )
                elif isinstance(input, spconv.SparseConvTensor):
                    if input.indices.shape[0] != 0:
                        input = input.replace_feature(module(input.features))
                else:
                    input = module(input)
        return input


class PDNorm(PointModule):
    def __init__(
        self,
        num_features,
        norm_layer,
        context_channels=256,
        conditions=("ScanNet", "S3DIS", "Structured3D"),
        decouple=True,
        adaptive=False,
    ):
        super().__init__()
        self.conditions = conditions
        self.decouple = decouple
        self.adaptive = adaptive
        if self.decouple:
            self.norm = nn.ModuleList([norm_layer(num_features) for _ in conditions])
        else:
            self.norm = norm_layer(num_features)
        if self.adaptive:
            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(context_channels, 2 * num_features, bias=True)
            )

    def forward(self, point):
        assert {"feat"}.issubset(point.keys())
        
        if self.decouple:
            assert {"condition"}.issubset(point.keys())
            if isinstance(point.condition, str):
                condition = point.condition
            else:
                condition = point.condition[0]
            assert condition in self.conditions
            norm = self.norm[self.conditions.index(condition)]
        else:
            norm = self.norm

        point.feat = norm(point.feat)
        if self.adaptive:
            assert "context" in point.keys()
            shift_scale = self.modulation(point.context)
            # context is per point cloud
            # shift_scale = torch.repeat_interleave(
            #     shift_scale, offset2bincount(point.offset), dim=0
            # ) # for diffusion noise, this is no longer needed
            shift, scale = shift_scale.chunk(2, dim=1)
            # print(shift.shape, scale.shape)
            point.feat = point.feat * (1.0 + scale) + shift
        return point


class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        qk_norm=False,
        scaled_cosine_attn=False,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        add_coords_in_attn='none',
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        # self.enable_flash = enable_flash = False
        self.qk_norm = qk_norm
        self.scaled_cosine_attn = scaled_cosine_attn
        if self.scaled_cosine_attn:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        if enable_flash:
            assert (
                enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

        # TODO: eps should be 1 / 65530 if using fp16 (eps=1e-6)
        self.q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-6) if self.qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-6) if self.qk_norm else nn.Identity()

        if add_coords_in_attn != 'none':
            self.coords_proj = torch.nn.Linear(3, channels, bias=False)
        self.add_coords_in_attn = add_coords_in_attn           

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        if self.add_coords_in_attn == 'qkv':
            point.feat = point.feat + self.coords_proj(point.coord)
            qkv = self.qkv(point.feat)
            qkv = qkv[order]
        elif self.add_coords_in_attn == 'qk':
            qkv = self.qkv(point.feat)
            qk_coords = self.coords_proj(point.coord).repeat(1, 2)
            qk_coords = torch.cat(
                [qk_coords, torch.zeros(qkv.size(0), C, dtype=qkv.dtype).to(qkv.device)], dim=-1
            )
            qkv = qkv + qk_coords
            qkv = qkv[order]
        else:
            qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            q = self.q_norm(q)
            k = self.k_norm(k)

            if self.scaled_cosine_attn:
                attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
                logit_scale = torch.clamp(
                    self.logit_scale, max=torch.log(torch.tensor(1. / 0.01)).to(self.logit_scale.device)
                ).exp()
                attn = attn * logit_scale
            else:
                attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)

            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            # print(attn.size(), attn.max(), torch.norm(q), torch.norm(k))
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)

        else:
            qkv = qkv.reshape(-1, 3, H, C // H)  # (N'*K, 3, H, C')
            q, k, v = qkv.unbind(dim=1)     # (N'*K, H, C')
            q = self.q_norm(q)              # (N'*K, H, C')
            k = self.k_norm(k)              # (N'*K, H, C')
            if self.scaled_cosine_attn:
                # TODO: not sure if the implementation is correct
                logit_scale = torch.clamp(
                    self.logit_scale, max=torch.log(torch.tensor(1. / 0.01)).to(self.logit_scale.device)
                ).exp()
                q = F.normalize(q, dim=-1) * logit_scale
                k = F.normalize(k, dim=-1)
            qkv = torch.stack([q, k, v], dim=1)     # (N'*K, 3, H, C')

            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half(), #.reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale if not self.scaled_cosine_attn else 1,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        qk_norm=False,
        scaled_cosine_attn=False,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        add_coords_in_attn='none',
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            qk_norm=qk_norm,
            scaled_cosine_attn=scaled_cosine_attn,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            add_coords_in_attn=add_coords_in_attn
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        # point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        # TODO: add support to grid pool (any stride)
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # collect information
        # print(indices.max(), indices.min())
        # print(point.feat.size(), point.coord.size())
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context
        if "context_offset" in point.keys():
            point_dict["context_offset"] = point.context_offset
        if "grid_size" in point.keys():
            point_dict["grid_size"] = point.grid_size * self.stride

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point


class SerializedUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]

        if self.traceable:
            parent["unpooling_parent"] = point
        return parent


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # TODO: check remove spconv
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point


class PointTransformerV3(PointModule):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        qk_norm=False,
        scaled_cosine_attn=False,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_context_channels=256,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        pdnorm_only_decoder=False,
        add_coords_in_attn='none',
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
                context_channels=pdnorm_context_channels,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        vanilla_bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
                context_channels=pdnorm_context_channels,
            )
        else:
            ln_layer = nn.LayerNorm
        vanilla_ln_layer = nn.LayerNorm

        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=vanilla_bn_layer if pdnorm_only_decoder else bn_layer,
            # norm_layer=vanilla_ln_layer if pdnorm_only_decoder else ln_layer,
            act_layer=act_layer,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=vanilla_bn_layer if pdnorm_only_decoder else bn_layer,
                        # norm_layer=vanilla_ln_layer if pdnorm_only_decoder else ln_layer,
                        act_layer=act_layer,
                        shuffle_orders=self.shuffle_orders,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        qk_norm=qk_norm,
                        scaled_cosine_attn=scaled_cosine_attn,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=vanilla_ln_layer if (pdnorm_only_decoder and s < (self.num_stages-1)) else ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                        add_coords_in_attn=add_coords_in_attn
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        # norm_layer=ln_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            qk_norm=qk_norm,
                            scaled_cosine_attn=scaled_cosine_attn,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                            add_coords_in_attn=add_coords_in_attn
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

    def _pack_point_dict(self, point):
        return EasyDict({
            'feat': point.feat,
            'coord': point.coord,
            'offset': point.offset,
        })

    def forward(self, data_dict, return_dec_layers=False):
        """
        A data_dict is a dictionary containing properties of a batched point cloud.
        It should contain the following properties for PTv3:
        1. "feat": feature of point cloud
        2. "grid_coord": discrete coordinate after grid sampling (voxelization) or "coord" + "grid_size"
        3. "offset" or "batch": https://github.com/Pointcept/Pointcept?tab=readme-ov-file#offset
        """
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        # print('before', offset2bincount(point.offset))

        point = self.embedding(point)
        point = self.enc(point)
        # print('after', offset2bincount(point.offset))

        layer_outputs = [self._pack_point_dict(point)]

        if not self.cls_mode:
            if return_dec_layers:
                for i in range(len(self.dec)):
                    for dec_block in self.dec[i]:
                        point = dec_block(point)
                        if type(dec_block) == Block:
                            layer_outputs.append(self._pack_point_dict(point))
                return layer_outputs
            else:
                point = self.dec(point)
        return point


class CrossAttention(PointModule):
    def __init__(
        self, channels, num_heads, kv_channels=None, attn_drop=0, proj_drop=0, 
        qk_norm=False, enable_flash=True
    ):
        super().__init__()
        if kv_channels is None:
            kv_channels = channels
        self.q = nn.Linear(channels, channels, bias=True)
        self.kv = nn.Linear(kv_channels, channels * 2, bias=True)
        self.attn_drop = attn_drop

        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qk_norm = qk_norm
        self.enable_flash = enable_flash

        # TODO: eps should be 1 / 65530 if using fp16 (eps=1e-6)
        self.q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-6) if self.qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-6) if self.qk_norm else nn.Identity()

    def forward(self, point: Point):
        device = point.feat.device

        q = self.q(point.feat).view(-1, self.num_heads, self.head_dim)
        kv = self.kv(point.context).view(-1, 2, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(kv[:, 0])
        kv = torch.stack([k, kv[:, 1]], dim=1)

        if self.enable_flash:
            cu_seqlens_q = torch.cat([torch.zeros(1).int().to(device), point.offset.int()], dim=0)
            cu_seqlens_k = torch.cat([torch.zeros(1).int().to(device), point.context_offset.int()], dim=0)
            max_seqlen_q = offset2bincount(point.offset).max()
            max_seqlen_k = offset2bincount(point.context_offset).max()

            feat = flash_attn.flash_attn_varlen_kvpacked_func(
                q.half(), kv.half(), cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale
            ).reshape(-1, self.channels)
            feat = feat.to(q.dtype)
        else:
            # q: (#all points, #heads, #dim)
            # kv: (#all words, k/v, #heads, #dim)
            # print(q.size(), kv.size())
            npoints_in_batch = offset2bincount(point.offset).data.cpu().numpy().tolist()
            nwords_in_batch = offset2bincount(point.context_offset).data.cpu().numpy().tolist()
            word_padded_masks = torch.from_numpy(
                gen_seq_masks(nwords_in_batch)
            ).to(q.device).logical_not()
            # print(word_padded_masks)

            q_pad = pad_tensors_wgrad(
                torch.split(q, npoints_in_batch, dim=0), npoints_in_batch
            )
            kv_pad = pad_tensors_wgrad(
                torch.split(kv, nwords_in_batch), nwords_in_batch
            )
            # q_pad: (batch_size, #points, #heads, #dim)
            # kv_pad: (batch_size, #words, k/v, #heads, #dim)
            # print(q_pad.size(), kv_pad.size())
            logits = torch.einsum('bphd,bwhd->bpwh', q_pad, kv_pad[:, :, 0]) * self.scale
            logits.masked_fill_(word_padded_masks.unsqueeze(1).unsqueeze(-1), -1e4)
            attn_probs = torch.softmax(logits, dim=2)
            # print(attn_probs.size())
            feat = torch.einsum('bpwh,bwhd->bphd', attn_probs, kv_pad[:, :, 1])
            feat = torch.cat([ft[:npoints_in_batch[i]] for i, ft in enumerate(feat)], 0)
            feat = feat.reshape(-1, self.channels).float()
            # print(feat.size())

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point
    

class CABlock(PointModule):
    def __init__(
        self, channels, num_heads, kv_channels=None, attn_drop=0.0, proj_drop=0.0,
        mlp_ratio=4.0, norm_layer=nn.LayerNorm, act_layer=nn.GELU, pre_norm=True,
        qk_norm=False, enable_flash=True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = CrossAttention(
            channels=channels,
            num_heads=num_heads,
            kv_channels=kv_channels,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            qk_norm=qk_norm,
            enable_flash=enable_flash,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )

    def forward(self, point: Point):
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.attn(point)
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.mlp(point)
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point

class CCBlock(PointModule):
    '''
    Cross-Attention and Convolution Block
    '''
    def __init__(
        self, channels, num_heads, kv_channels=None, attn_drop=0.0, proj_drop=0.0,
        mlp_ratio=4.0, norm_layer=nn.LayerNorm, act_layer=nn.GELU, pre_norm=True,
        qk_norm=False, enable_flash=True, conv_indice_key=None, conv_out_channels=None,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = CrossAttention(
            channels=channels,
            num_heads=num_heads,
            kv_channels=kv_channels,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            qk_norm=qk_norm,
            enable_flash=enable_flash,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.conv = PointSequential(
            spconv.SubMConv3d(
                channels,
                conv_out_channels,
                kernel_size=3,
                bias=True,
                indice_key=conv_indice_key,
            ),
            nn.Linear(conv_out_channels, conv_out_channels),
            norm_layer(conv_out_channels),
        )

    def forward(self, point: Point):
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.attn(point)
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.mlp(point)
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)

        # before = point

        # point = self.conv(point)

        conv_result = self.conv(Point(point))
        return point, conv_result


class PointTransformerV3CA(PointTransformerV3):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        ctx_channels=256,
        qkv_bias=True,
        qk_scale=None,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_context_channels=256,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        pdnorm_only_decoder=False,
        add_coords_in_attn=False,
        scaled_cosine_attn=False, # TODO
    ):
        PointModule.__init__(self)
        # assert enable_flash, 'only implemented flash attention'

        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
                context_channels=pdnorm_context_channels,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
                context_channels=pdnorm_context_channels,
            )
        else:
            ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                        shuffle_orders=self.shuffle_orders,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                        add_coords_in_attn=add_coords_in_attn,
                        qk_norm=qk_norm,
                    ),
                    name=f"block{i}",
                )
                if (not pdnorm_only_decoder) or (s == self.num_stages - 1):
                    enc.add(
                        CABlock(
                            channels=enc_channels[s],
                            num_heads=enc_num_head[s],
                            kv_channels=ctx_channels,
                            mlp_ratio=mlp_ratio,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            qk_norm=qk_norm,
                            enable_flash=enable_flash,
                        ),
                        name=f"ca_block{i}",
                    )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                            add_coords_in_attn=add_coords_in_attn,
                            qk_norm=qk_norm,
                        ),
                        name=f"block{i}",
                    )
                    dec.add(
                        CABlock(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            kv_channels=ctx_channels,
                            mlp_ratio=mlp_ratio,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            qk_norm=qk_norm,
                            enable_flash=enable_flash,
                        ),
                        name=f"ca_block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict, return_dec_layers=False):
        """
        A data_dict is a dictionary containing properties of a batched point cloud.
        It should contain the following properties for PTv3:
        1. "feat": feature of point cloud
        2. "grid_coord": discrete coordinate after grid sampling (voxelization) or "coord" + "grid_size"
        3. "offset" or "batch": https://github.com/Pointcept/Pointcept?tab=readme-ov-file#offset
        """
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        # print('before', offset2bincount(point.offset))

        point = self.embedding(point)
        point = self.enc(point)
        # print('after', offset2bincount(point.offset))

        layer_outputs = [self._pack_point_dict(point)]

        if not self.cls_mode:
            if return_dec_layers:
                for i in range(len(self.dec)):
                    for dec_block in self.dec[i]:
                        point = dec_block(point)
                        if type(dec_block) == CABlock:
                            layer_outputs.append(self._pack_point_dict(point))
                return layer_outputs
            else:
                point = self.dec(point)
        return point

class QuerySupportAttention(PointModule):
    def __init__(
        self, channels, num_heads, kv_channels=None, attn_drop=0, proj_drop=0, 
        qk_norm=False, enable_flash=True
    ):
        super().__init__()
        if kv_channels is None:
            kv_channels = channels
        
        # print("channels", channels)
        # print("kv_channels", kv_channels)
        self.q = nn.Linear(channels, channels, bias=True)
        self.kv = nn.Linear(kv_channels, channels * 2, bias=True)
        self.attn_drop = attn_drop

        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)



        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        # print("channels", channels)
        # print("num_heads", num_heads)
        # print("head_dim", self.head_dim)

        self.scale = self.head_dim ** -0.5
        self.qk_norm = qk_norm
        self.enable_flash = enable_flash

        self.roper = RotaryPositionEncoding3D(self.head_dim)

        # TODO: eps should be 1 / 65530 if using fp16 (eps=1e-6)
        self.q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-6) if self.qk_norm else nn.Identity() # TODO: why not use LayerNorm
        self.k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-6) if self.qk_norm else nn.Identity()

    def forward(self, anchor: Point, point: Point):
        device = point.feat.device

        # calc 3d-rope

        q = self.q(anchor.feat).view(-1, self.num_heads, self.head_dim)
        kv = self.kv(point.feat).view(-1, 2, self.num_heads, self.head_dim)

        # apply rope at q and k, not v.
        q_cos, q_sin = self.roper(anchor.coord)
        k_cos, k_sin = self.roper(point.coord)

        # print("q_cos shape", q_cos.shape)
        # print("q_sin shape", q_sin.shape)
        # print("k_cos shape", k_cos.shape)
        # print("k_sin shape", k_sin.shape)
        # print("q shape", q.shape)
        # print("kv shape", kv.shape)

        # print("qkv shape", q.shape, kv.shape)
        q = embed_rotary(q, q_cos, q_sin)
        q = self.q_norm(q)
        # apply Film here
        k = embed_rotary(kv[:, 0], k_cos, k_sin)
        k = self.k_norm(kv[:, 0])
        kv = torch.stack([k, kv[:, 1]], dim=1)

        if self.enable_flash:
            cu_seqlens_q = torch.cat([torch.zeros(1).int().to(device), anchor.offset.int()], dim=0)
            cu_seqlens_k = torch.cat([torch.zeros(1).int().to(device), point.offset.int()], dim=0)

            max_seqlen_q = offset2bincount(anchor.offset).max() # TODO: known
            max_seqlen_k = offset2bincount(point.offset).max()

            feat = flash_attn.flash_attn_varlen_kvpacked_func(
                q.half(), kv.half(), cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale
            ).reshape(-1, self.channels)
            feat = feat.to(q.dtype)
        else:
            raise NotImplementedError
            # # q: (#all points, #heads, #dim)
            # # kv: (#all words, k/v, #heads, #dim)
            # # print(q.size(), kv.size())
            # npoints_in_batch = offset2bincount(point.offset).data.cpu().numpy().tolist()
            # nwords_in_batch = offset2bincount(point.context_offset).data.cpu().numpy().tolist()
            # word_padded_masks = torch.from_numpy(
            #     gen_seq_masks(nwords_in_batch)
            # ).to(q.device).logical_not()
            # # print(word_padded_masks)

            # q_pad = pad_tensors_wgrad(
            #     torch.split(q, npoints_in_batch, dim=0), npoints_in_batch
            # )
            # kv_pad = pad_tensors_wgrad(
            #     torch.split(kv, nwords_in_batch), nwords_in_batch
            # )
            # # q_pad: (batch_size, #points, #heads, #dim)
            # # kv_pad: (batch_size, #words, k/v, #heads, #dim)
            # # print(q_pad.size(), kv_pad.size())
            # logits = torch.einsum('bphd,bwhd->bpwh', q_pad, kv_pad[:, :, 0]) * self.scale
            # logits.masked_fill_(word_padded_masks.unsqueeze(1).unsqueeze(-1), -1e4)
            # attn_probs = torch.softmax(logits, dim=2)
            # # print(attn_probs.size())
            # feat = torch.einsum('bpwh,bwhd->bphd', attn_probs, kv_pad[:, :, 1])
            # feat = torch.cat([ft[:npoints_in_batch[i]] for i, ft in enumerate(feat)], 0)
            # feat = feat.reshape(-1, self.channels).float()
            # # print(feat.size())

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        anchor.feat = feat
        return anchor



class NeckBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, num_heads, kv_channels=None, attn_drop=0.0, proj_drop=0.0,
        mlp_ratio=4.0, norm_layer=nn.LayerNorm, act_layer=nn.GELU, pre_norm=True,
        qk_norm=False, enable_flash=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pre_norm = pre_norm

        self.norm1 = PointSequential(norm_layer(in_channels))
        # print("kv_channels:", kv_channels)
        # print("in_channels:", in_channels)
        self.attn = QuerySupportAttention(
            channels=in_channels,
            num_heads=num_heads,
            kv_channels=kv_channels,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            qk_norm=qk_norm,
            enable_flash=enable_flash,
        )
        self.norm2 = PointSequential(norm_layer(in_channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=in_channels,
                hidden_channels=int(in_channels * mlp_ratio),
                out_channels=out_channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )

    def forward(self, anchor: Point, point: Point):

        shortcut = anchor.feat
        # print(anchor.feat.size(), anchor.context.size())
        if self.pre_norm:
            anchor = self.norm1(anchor)
        # print(anchor.feat.size(), anchor.context.size())
        anchor = self.attn(anchor, point)
        # print(anchor.feat.size(), anchor.context.size())
        anchor.feat = shortcut + anchor.feat
        if not self.pre_norm:
            anchor = self.norm1(anchor)

        shortcut = anchor.feat
        # print(anchor.feat.size(), anchor.context.size())
        if self.pre_norm:
            anchor = self.norm2(anchor)
        anchor = self.mlp(anchor)
        anchor.feat = shortcut + anchor.feat
        if not self.pre_norm:
            anchor = self.norm2(anchor)
        # anchor.sparse_conv_feat = anchor.sparse_conv_feat.replace_feature(anchor.feat) 
        # TODO: this can not be done because never init, Maybe not necessary
        return anchor

class PTv3withNeck(PointTransformerV3):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        ctx_channels=256,
        qkv_bias=True,
        qk_scale=None,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_context_channels=256,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        pdnorm_only_decoder=False,
        add_coords_in_attn=False,
        scaled_cosine_attn=False, # TODO
    ):
        PointModule.__init__(self)
        # assert enable_flash, 'only implemented flash attention'

        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders
        self.layer_cache = []
        self.conv_cache = []

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
                context_channels=pdnorm_context_channels,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
                context_channels=pdnorm_context_channels,
            )
        else:
            ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        anchor_ln_layer = partial(
            PDNorm,
            norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
            conditions=pdnorm_conditions,
            decouple=pdnorm_decouple,
            adaptive=True,
            context_channels=pdnorm_context_channels,
        )

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        self.anchor_proj = PointSequential(MLP(
            in_channels=3,
            hidden_channels=3 * mlp_ratio,
            out_channels=dec_channels[0],act_layer=act_layer,)
            )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                        shuffle_orders=self.shuffle_orders,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                        add_coords_in_attn=add_coords_in_attn,
                        qk_norm=qk_norm,
                    ),
                    name=f"block{i}",
                )
                if (not pdnorm_only_decoder) or (s == self.num_stages - 1):
                    enc.add(
                        CABlock(
                            channels=enc_channels[s],
                            num_heads=enc_num_head[s],
                            kv_channels=ctx_channels,
                            mlp_ratio=mlp_ratio,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            qk_norm=qk_norm,
                            enable_flash=enable_flash,
                        ),
                        name=f"ca_block{i}",
                    )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            self.nec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                            add_coords_in_attn=add_coords_in_attn,
                            qk_norm=qk_norm,
                        ),
                        name=f"block{i}",
                    )
                    dec.add(
                        CCBlock(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            kv_channels=ctx_channels,
                            mlp_ratio=mlp_ratio,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            qk_norm=qk_norm,
                            enable_flash=enable_flash,
                            conv_indice_key=f"stage{s}",
                            conv_out_channels=dec_channels[0],
                        ),
                        name=f"cc_block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")
                self.nec.add(module=NeckBlock(
                            in_channels=dec_channels[0],
                            out_channels=dec_channels[0],
                            num_heads=dec_num_head[0],
                            kv_channels=dec_channels[s + 1],
                            mlp_ratio=mlp_ratio,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            norm_layer=anchor_ln_layer, 
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            qk_norm=qk_norm,
                            enable_flash=enable_flash,
                        ), name=f"neck{s}")

    def forward(self, data_dict, return_dec_layers=False):
        """
        A data_dict is a dictionary containing properties of a batched point cloud.
        It should contain the following properties for PTv3:
        1. "feat": feature of point cloud
        2. "grid_coord": discrete coordinate after grid sampling (voxelization) or "coord" + "grid_size"
        3. "offset" or "batch": https://github.com/Pointcept/Pointcept?tab=readme-ov-file#offset
        """
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        # print('before', offset2bincount(point.offset))

        point = self.embedding(point)
        point = self.enc(point)
        # print('after', offset2bincount(point.offset))

        layer_outputs = [self._pack_point_dict(point)]
        self.layer_cache.clear()
        self.conv_cache.clear()
        self.layer_cache.append(Point(point))


        if not self.cls_mode:
            if return_dec_layers:
                for i in range(len(self.dec)):
                    for dec_block in self.dec[i]:
                        if type(dec_block) == CCBlock:
                            point, conv = dec_block(point)
                            self.conv_cache.append(conv)
                            layer_outputs.append(self._pack_point_dict(Point(point)))
                            self.layer_cache.append(Point(point))
                        else:
                            point = dec_block(point)

                return layer_outputs
            else:
                for i in range(len(self.dec)):
                    for dec_block in self.dec[i]:
                        if type(dec_block) == CCBlock:
                            point, conv = dec_block(point)
                            self.conv_cache.append(conv)
                            self.layer_cache.append(Point(point))   
                        else:                     
                            point = dec_block(point)

        return point
    
    def forward_only_neck(self, data_dict, return_dec_layers=False):
        anchor = Point(data_dict)
        anchor = self.anchor_proj(anchor)


        # print("layer_cache size", len(self.layer_cache))
        # print("conv_cache size", len(self.conv_cache))
        # for i in self.layer_cache:
        #     print(i.grid_coord)
        #     print(i.grid_size)
        #     print(i.feat.shape)

        # for i in self.conv_cache:
        #     print(i.grid_coord)
        #     print(i.grid_size)
        #     print(i.feat.shape)
        
        # align query and support index

        for i in range(len(self.nec)):
            # print("anchor size", anchor.feat.shape)
            # print("point size", self.layer_cache[i].feat.shape)
            # print("neck block", i)
            anchor = self.nec[i](anchor, self.layer_cache[i])
            # find local feautures
            
            # add local features
        return anchor
    
def compute_query_index(noise_anchor: Point, point: Point, length):
    pass

def compute_support_conv(noise_anchor: Point, point: Point, length):
    pass

def extract_aligned_features(noise_anchor: Point, point: Point):
    """
    Efficiently extracts features from B at voxel locations of A.
    Zeros out unmatched locations.
    """
    device = point.feat.device
    feat_dim = point.feat.shape[1]

    # Prepare indices (batch, voxel coord) for hashing
    A_indices = torch.cat([noise_anchor.batch[:, None], noise_anchor.grid_coord], dim=1)
    B_indices = torch.cat([point.batch[:, None], point.grid_coord], dim=1)

    # Hash scales (unique mapping to integers)
    spatial_shape = point.sparse_shape
    hash_scale = torch.tensor(
        [spatial_shape[0] * spatial_shape[1] * spatial_shape[2],
         spatial_shape[1] * spatial_shape[2],
         spatial_shape[2],
         1],
        device=point.feat.device
    )

    # Compute hashes
    A_hash = (A_indices * hash_scale).sum(dim=1)
    B_hash = (B_indices * hash_scale).sum(dim=1)

    # Efficient intersection using torch functions
    B_hash_sorted, indices_B = torch.sort(B_hash)
    idx_in_B = torch.searchsorted(B_hash_sorted, A_hash)

    # Clamp indices to be within range
    idx_in_B_clamped = torch.clamp(idx_in_B, 0, B_hash_sorted.size(0) - 1)
    matched = B_hash_sorted[idx_in_B_clamped] == A_hash
    matched_B_indices = indices_B[idx_in_B[matched]]

    # Initialize zero features for noise_anchor
    noise_feat = torch.zeros((len(noise_anchor.coord), point.feat.size(1)), device=point.feat.device)

    # Assign matched features from B
    noise_feat[matched_mask] = point.feat[indices_B[idx_in_B[matched_mask]]]

    noise_anchor.feat = noise_feat

    return noise_anchor
