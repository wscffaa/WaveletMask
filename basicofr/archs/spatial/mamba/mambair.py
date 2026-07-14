"""MambaIR: 基于状态空间模型的图像修复网络

实现 MambaIR 论文中的视觉状态空间块 (VSSB) 和残差状态空间组 (RSSG)。

Ref:
    MambaIR: A Simple Baseline for Image Restoration with State Space Model.
"""

import math
import warnings
from functools import lru_cache, partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat

from .dynamic_conv import DynamicConvolution

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ModuleNotFoundError as e:  # pragma: no cover - optional dependency check happens at runtime
    if (e.name or "").split(".")[0] != "mamba_ssm":
        raise

    def selective_scan_fn(*args, **kwargs):  # type: ignore[override]
        raise ImportError(
            "Missing optional dependency `mamba_ssm`. "
            "Install it to enable SS2D selective scan."
        )


@lru_cache(maxsize=1)
def _load_mair_nss_ops():
    try:
        from basicofr.archs._projects.p004_mair.components.nss_scan import (
            nss_ids_generate,
            nss_shift_ids_generate,
            nss_ids_scan,
            nss_ids_inverse,
        )
    except ImportError:
        return None
    return nss_ids_generate, nss_shift_ids_generate, nss_ids_scan, nss_ids_inverse


def _require_mair_nss_ops():
    nss_ops = _load_mair_nss_ops()
    if nss_ops is None:
        raise ImportError(
            "`scan_mode='nss'` 依赖 MaIR 项目的 `nss_scan` 组件。"
        )
    return nss_ops


@lru_cache(maxsize=1)
def _load_mair_shuffle_attn():
    try:
        from basicofr.archs._projects.p004_mair.components.shuffle_attn import ShuffleAttn
    except ImportError:
        return None
    return ShuffleAttn


def _require_mair_shuffle_attn():
    shuffle_attn_cls = _load_mair_shuffle_attn()
    if shuffle_attn_cls is None:
        raise ImportError(
            "`use_shuffle_attn=True` 依赖 MaIR 项目的 `shuffle_attn` 组件。"
        )
    return shuffle_attn_cls


@lru_cache(maxsize=1)
def _load_scsegamba_sass_scan_orders():
    try:
        from basicofr.archs.ideas._archive.scsegamba.components.sass import sass_scan_orders
    except ImportError:
        warnings.warn(
            "`scan_mode='sass'` 不可用，回退到 baseline `standard` 扫描。",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return sass_scan_orders


@lru_cache(maxsize=1)
def _load_scsegamba_gbc():
    try:
        from basicofr.archs.ideas._archive.scsegamba.components.gbc import GBC
    except ImportError:
        warnings.warn(
            "`ffn_type='gbc'` 不可用，回退到 baseline `cab` FFN。",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return GBC


class ChannelAttention(nn.Module):
    """通道注意力模块 (RCAN 风格)

    Args:
        num_feat: 通道数
        squeeze_factor: 压缩因子，默认 16
    """

    def __init__(self, num_feat: int, squeeze_factor: int = 16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attention(x)


class CAB(nn.Module):
    """通道注意力块

    使用动态卷积和通道注意力进行特征增强。

    Args:
        num_feat: 通道数
        is_light_sr: 是否为轻量级 SR 模式
        compress_ratio: 压缩比
        squeeze_factor: 通道注意力压缩因子
    """

    def __init__(self, num_feat: int, is_light_sr: bool = False,
                 compress_ratio: int = 3, squeeze_factor: int = 30):
        super().__init__()
        if is_light_sr:
            compress_ratio = 6
        self.cab = nn.Sequential(
            DynamicConvolution(8, 2, in_channels=num_feat, out_channels=num_feat // compress_ratio,
                             kernel_size=3, padding=1, temperature=1, bias=True),
            nn.GELU(),
            DynamicConvolution(8, 2, in_channels=num_feat // compress_ratio, out_channels=num_feat,
                             kernel_size=3, padding=1, temperature=1, bias=True),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cab(x)


class SS2D(nn.Module):
    """2D 选择性状态空间模块

    实现 Mamba 的 2D 扫描，支持四方向扫描。

    Args:
        d_model: 模型维度
        d_state: 状态维度
        d_conv: 卷积核大小
        expand: 扩展因子
        dt_rank: 时间步投影维度
        dt_min/dt_max: 时间步初始化范围
        dropout: dropout 率
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: float = 2.0,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        dropout: float = 0.0,
        conv_bias: bool = True,
        bias: bool = False,
        device=None,
        dtype=None,
        scan_mode: str = "standard",
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.nss_scan_len = int(kwargs.pop("scan_len", 8))
        self.nss_shift_len = int(kwargs.pop("shift_len", self.nss_scan_len // 2))
        self.use_shuffle_attn = bool(kwargs.pop("use_shuffle_attn", False))
        self.nss_use_shift = bool(kwargs.pop("nss_use_shift", False))
        assert scan_mode in ("standard", "sass", "nss"), f"scan_mode must be 'standard', 'sass' or 'nss', got {scan_mode}"
        self.scan_mode = scan_mode

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # 四方向 x 投影
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        # 四方向 dt 投影
        self.dt_projs = (
            self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self._A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self._D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self._nss_ids_generate = None
        self._nss_shift_ids_generate = None
        self._nss_ids_scan = None
        self._nss_ids_inverse = None
        if self.scan_mode == "nss":
            (
                self._nss_ids_generate,
                self._nss_shift_ids_generate,
                self._nss_ids_scan,
                self._nss_ids_inverse,
            ) = _require_mair_nss_ops()
            if self.use_shuffle_attn:
                shuffle_attn_cls = _require_mair_shuffle_attn()
                self.shuffle_attn = shuffle_attn_cls(
                    self.d_inner * 4,
                    self.d_inner * 4,
                    group=self.d_inner,
                )
            else:
                self.shuffle_attn = None
        else:
            self.shuffle_attn = None
        self.register_buffer("_nss_scan_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("_nss_inverse_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("_nss_shift_scan_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("_nss_shift_inverse_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self._nss_H: Optional[int] = None
        self._nss_W: Optional[int] = None

    @staticmethod
    def _dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1,
                 dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def _A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def _D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        scan_ids = None
        inverse_ids = None
        effective_scan_mode = self.scan_mode

        if effective_scan_mode == "sass":
            sass_scan_orders = _load_scsegamba_sass_scan_orders()
            if sass_scan_orders is None:
                effective_scan_mode = "standard"
            else:
                orders, inv_orders, _ = sass_scan_orders(H, W, device=x.device)
        if effective_scan_mode == "sass":
            x_flat = x.view(B, -1, L)
            xs = torch.stack([x_flat[:, :, order] for order in orders], dim=1).contiguous()
        elif effective_scan_mode == "nss":
            if self.nss_use_shift:
                scan_ids = self._nss_shift_scan_ids
                inverse_ids = self._nss_shift_inverse_ids
            else:
                scan_ids = self._nss_scan_ids
                inverse_ids = self._nss_inverse_ids
            xs = self._nss_ids_scan(x, scan_ids)
        else:
            # 标准四向扫描
            x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
            xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias, delta_softplus=True, return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        if effective_scan_mode == "sass":
            ys = [
                out_y[:, idx, :, inv_orders[idx]].contiguous()
                for idx in range(K)
            ]
            return ys[0], ys[1], ys[2], ys[3]
        elif effective_scan_mode == "nss":
            y = self._nss_ids_inverse(out_y, inverse_ids, shape=(B, -1, H, W))
            if self.use_shuffle_attn:
                y = y * self.shuffle_attn(y)
            y1, y2, y3, y4 = y.chunk(4, dim=1)
            return y1 + y2 + y3 + y4

        # 标准模式还原
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, H, W, C = x.shape
        if self.scan_mode == "nss":
            if (
                self._nss_scan_ids.numel() == 0
                or self._nss_H != H
                or self._nss_W != W
                or self._nss_scan_ids.device != x.device
            ):
                device = x.device
                scan_ids, inv_ids = self._nss_ids_generate((1, 1, H, W), scan_len=self.nss_scan_len)
                shift_scan_ids, shift_inv_ids = self._nss_shift_ids_generate(
                    (1, 1, H, W),
                    scan_len=self.nss_scan_len,
                    shift_len=self.nss_shift_len,
                )
                self._nss_scan_ids = scan_ids.to(device=device, dtype=torch.long)
                self._nss_inverse_ids = inv_ids.to(device=device, dtype=torch.long)
                self._nss_shift_scan_ids = shift_scan_ids.to(device=device, dtype=torch.long)
                self._nss_shift_inverse_ids = shift_inv_ids.to(device=device, dtype=torch.long)
                self._nss_H = H
                self._nss_W = W
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        if self.scan_mode == "nss":
            y = self.forward_core(x)
            assert y.dtype == torch.float32
            y = y.permute(0, 2, 3, 1).contiguous()
        else:
            y1, y2, y3, y4 = self.forward_core(x)
            assert y1.dtype == torch.float32
            y = y1 + y2 + y3 + y4
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    """视觉状态空间块 (VSSB)

    Args:
        hidden_dim: 隐藏维度
        drop_path: DropPath 率
        d_state: 状态维度
        expand: 扩展因子
        is_light_sr: 是否为轻量级 SR
        ffn_type: FFN 类型 ('cab' | 'gbc' | 'ssa')
    """

    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        expand: float = 2.0,
        is_light_sr: bool = False,
        ffn_type: str = "cab",
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state, expand=expand, dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))

        # SSA 的序列融合在 SS2D(scan_mode='nss') 内处理，这里仍复用卷积 FFN。
        self.conv_blk = CAB(hidden_dim, is_light_sr)
        if ffn_type == "gbc":
            gbc_cls = _load_scsegamba_gbc()
            if gbc_cls is not None:
                self.conv_blk = gbc_cls(hidden_dim, norm_type='GN')

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input: torch.Tensor, x_size: tuple) -> torch.Tensor:
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()
        x = self.ln_1(input)
        x = input * self.skip_scale + self.drop_path(self.self_attention(x))
        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x


class BasicLayer(nn.Module):
    """基础 MambaIR 层

    Args:
        dim: 输入通道数
        input_resolution: 输入分辨率
        depth: 块数量
        d_state: 状态维度
        mlp_ratio: MLP 扩展比
        drop_path: DropPath 率
        use_checkpoint: 是否使用梯度检查点
        is_light_sr: 是否为轻量级 SR
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple,
        depth: int,
        drop_path: float = 0.0,
        d_state: int = 16,
        mlp_ratio: float = 2.0,
        norm_layer: nn.Module = nn.LayerNorm,
        downsample: nn.Module = None,
        use_checkpoint: bool = False,
        is_light_sr: bool = False,
        scan_mode: str = "standard",
        ffn_type: str = "cab",
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=d_state,
                expand=self.mlp_ratio,
                input_resolution=input_resolution,
                is_light_sr=is_light_sr,
                scan_mode=scan_mode,
                ffn_type=ffn_type,
                **kwargs,
            ))

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor, x_size: tuple) -> torch.Tensor:
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size, use_reentrant=False)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    """将 2D 特征图转换为 1D 令牌序列"""

    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3,
                 embed_dim: int = 96, norm_layer: nn.Module = None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    """将 1D 令牌序列还原为 2D 特征图"""

    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3,
                 embed_dim: int = 96, norm_layer: nn.Module = None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, x_size: tuple) -> torch.Tensor:
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])
        return x


class ResidualGroup(nn.Module):
    """残差状态空间组 (RSSG)

    Args:
        dim: 输入通道数
        input_resolution: 输入分辨率
        depth: 块数量
        d_state: 状态维度
        mlp_ratio: MLP 扩展比
        drop_path: DropPath 率
        use_checkpoint: 是否使用梯度检查点
        img_size: 图像大小
        patch_size: Patch 大小
        resi_connection: 残差连接类型
        is_light_sr: 是否为轻量级 SR
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple,
        depth: int,
        d_state: int = 16,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        downsample: nn.Module = None,
        use_checkpoint: bool = False,
        img_size: int = None,
        patch_size: int = None,
        resi_connection: str = '1conv',
        is_light_sr: bool = False,
        scan_mode: str = "standard",
        ffn_type: str = "cab",
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            d_state=d_state,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            is_light_sr=is_light_sr,
            scan_mode=scan_mode,
            ffn_type=ffn_type,
            **kwargs,
        )

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1)
            )

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x: torch.Tensor, x_size: tuple) -> torch.Tensor:
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x


class MambaIR(nn.Module):
    """MambaIR: 基于状态空间模型的图像修复网络

    Args:
        img_size: 输入图像大小，默认 64
        patch_size: Patch 大小，默认 1
        in_chans: 输入通道数，默认 3
        embed_dim: 嵌入维度，默认 96
        depths: 各层深度
        d_state: 状态维度，默认 16
        mlp_ratio: MLP 扩展比，默认 2.0
        drop_rate: Dropout 率
        drop_path_rate: DropPath 率
        use_checkpoint: 是否使用梯度检查点
        upscale: 上采样因子
        img_range: 图像值范围
        upsampler: 上采样器类型
        resi_connection: 残差连接类型
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 1,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: tuple = (6, 6, 6, 6),
        drop_rate: float = 0.0,
        d_state: int = 16,
        mlp_ratio: float = 2.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        upscale: int = 2,
        img_range: float = 1.0,
        upsampler: str = '',
        resi_connection: str = '1conv',
        scan_mode: str = "standard",
        ffn_type: str = "cab",
        **kwargs,
    ):
        super().__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.mlp_ratio = mlp_ratio
        self.scan_mode = scan_mode
        self.ffn_type = ffn_type

        # ========== 浅层特征提取 ==========
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ========== 深层特征提取 ==========
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim,
            embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None
        )
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim,
            embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None
        )

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.is_light_sr = True if self.upsampler == 'pixelshuffledirect' else False

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # ========== 构建残差状态空间组 ==========
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ResidualGroup(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                d_state=d_state,
                mlp_ratio=self.mlp_ratio,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                is_light_sr=self.is_light_sr,
                scan_mode=scan_mode,
                ffn_type=ffn_type,
                **kwargs,
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1)
            )

        # ========== 重建模块 ==========
        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)
        else:
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def forward_deep_residual(self, x: torch.Tensor) -> torch.Tensor:
        """返回步骤②的深特征残差，跳过conv_last和全局残差。

        Returns:
            deep_res: (B, 64, H, W)
        """
        x_first = self.conv_first(x)
        return self.conv_after_body(self.forward_features(x_first)) + x_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsampler == 'pixelshuffle':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        else:
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)
        return x


class UpsampleOneStep(nn.Sequential):
    """单步上采样（轻量级 SR 使用）"""

    def __init__(self, scale: int, num_feat: int, num_out_ch: int):
        self.num_feat = num_feat
        m = [
            nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1),
            nn.PixelShuffle(scale)
        ]
        super().__init__(*m)


class Upsample(nn.Sequential):
    """上采样模块，支持 2^n 和 3 倍上采样"""

    def __init__(self, scale: int, num_feat: int):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported: 2^n and 3.')
        super().__init__(*m)
