"""
BraTS-PED 2025 — MedSwinNet (scaled)

Unified model (~40 M params) combining:
  • MedNeXt-style blocks  — depthwise k=3/5 conv → IN → GRN → 4× expand → GELU → project
    (MedNeXt: Woo et al. 2023 — BraTS 2023 winner)
  • 4× stacked Swin transformer blocks at bottleneck (window-MHSA + FFN, Flash Attention)
  • SPADE decoder conditioning on an auxiliary WT head (Park et al. 2019)
  • Classwise modality attention output heads — each of [ET,NET,CC,ED] learns
    its own softmax attention over [t1c,t1n,t2f,t2w] instead of sharing one
    conv's weights across all classes
  • Stochastic depth (drop path) linearly scheduled 0 → 0.2 across all blocks
  • Deep supervision at full + half resolution

Architecture for 128³ input:
  stem  stride=1  →  128³ ×  64 ch   (2 blocks, k=3)
  enc1  stride=2  →   64³ × 128 ch   (3 blocks, k=3)
  enc2  stride=2  →   32³ × 256 ch   (4 blocks, k=3)
  enc3  stride=2  →   16³ × 320 ch   (4 blocks, k=5)  ← auxiliary WT head
  enc4  stride=2  →    8³ × 512 ch   (4 blocks, k=5)  ← 4× SwinBlock
  ─ decoder mirrors encoder; dec3+dec2 use SPADE conditioning ─
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt

PATCH_SIZE   = (128, 128, 128)
IN_CHANNELS  = 4    # t1c, t1n, t2f, t2w
OUT_CHANNELS = 4    # [ET, NET, CC, ED] — each sub-region predicted directly

_FILTERS  = [64, 128, 256, 320, 512]
_N_BLOCKS = [2,   3,   4,   4,   4]   # MedNeXt blocks per encoder level
_KERNELS  = [3,   3,   3,   5,   5]   # depthwise kernel per level
_N_SWIN   = 4                          # stacked SwinBlocks at bottleneck


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class DropPath(nn.Module):
    """Stochastic depth: drop the entire residual branch with probability drop_prob."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device) < keep_prob
        return x * mask / keep_prob


class GRN(nn.Module):
    """Global Response Normalization (MedNeXt, Woo et al. 2023).

    L2-norm per channel normalised by mean norm across channels acts as
    learned inter-channel feature competition. Initialised to identity.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3, 4), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class MedNeXtBlock(nn.Module):
    """ConvNeXt-style 3D residual block (MedNeXt) with optional stochastic depth.

    DWConv(k×k×k) → InstanceNorm → GRN → 1×1 expand(4×) → GELU → 1×1 project (+residual)
    """

    def __init__(self, channels: int, kernel_size: int = 3, expansion: int = 4,
                 drop_path_rate: float = 0.0, use_checkpoint: bool = False) -> None:
        super().__init__()
        expanded = channels * expansion
        self.dw             = nn.Conv3d(channels, channels, kernel_size,
                                        padding=kernel_size // 2, groups=channels, bias=False)
        self.norm           = nn.InstanceNorm3d(channels, affine=True)
        self.grn            = GRN(channels)
        self.pw_up          = nn.Conv3d(channels, expanded, 1, bias=False)
        self.act            = nn.GELU()
        self.pw_dn          = nn.Conv3d(expanded, channels, 1, bias=False)
        self.drop_path      = DropPath(drop_path_rate)
        self.use_checkpoint = use_checkpoint

    def _inner(self, x: torch.Tensor) -> torch.Tensor:
        h = self.grn(self.norm(self.dw(x)))
        return x + self.drop_path(self.pw_dn(self.act(self.pw_up(h))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return grad_ckpt(self._inner, x, use_reentrant=False)
        return self._inner(x)


def _make_blocks(channels: int, n: int, kernel_size: int,
                 drop_path_rates: Optional[List[float]] = None,
                 use_checkpoint: bool = False) -> nn.Sequential:
    if drop_path_rates is None:
        drop_path_rates = [0.0] * n
    return nn.Sequential(*[
        MedNeXtBlock(channels, kernel_size, drop_path_rate=dpr, use_checkpoint=use_checkpoint)
        for dpr in drop_path_rates
    ])


class EncBlock(nn.Module):
    """Encoder stage: stride-2 projection → n MedNeXtBlocks."""

    def __init__(self, in_ch: int, out_ch: int, n_blocks: int, kernel_size: int,
                 drop_path_rates: Optional[List[float]] = None,
                 use_checkpoint: bool = False) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 2, stride=2, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.blocks = _make_blocks(out_ch, n_blocks, kernel_size, drop_path_rates, use_checkpoint)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.down(x))


class DecBlock(nn.Module):
    """Decoder stage: ConvTranspose upsample → concat skip → project → n blocks → SPADE."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, n_blocks: int,
                 kernel_size: int, use_spade: bool = False,
                 drop_path_rates: Optional[List[float]] = None,
                 use_checkpoint: bool = False) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Conv3d(out_ch + skip_ch, out_ch, 1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.blocks = _make_blocks(out_ch, n_blocks, kernel_size, drop_path_rates, use_checkpoint)
        self.spade  = SPADE(out_ch) if use_spade else None

    def forward(self, x: torch.Tensor, skip: torch.Tensor,
                wt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = self.blocks(self.proj(torch.cat([x, skip], dim=1)))
        if self.spade is not None and wt_mask is not None:
            x = self.spade(x, wt_mask)
        return x


class WindowAttention3D(nn.Module):
    """Local window multi-head self-attention (Swin UNETR).

    Partitions the volume into (ws × ws × ws) windows and applies MHSA within
    each. Uses F.scaled_dot_product_attention for automatic Flash Attention
    dispatch on Ada/Ampere GPUs (PyTorch ≥ 2.0, CUDA, BF16 or FP16).
    """

    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 4) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.ws        = window_size
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.norm      = nn.LayerNorm(dim)
        self.qkv       = nn.Linear(dim, 3 * dim, bias=False)
        self.proj      = nn.Linear(dim, dim)

    def _partition(self, x: torch.Tensor, ws: int) -> torch.Tensor:
        B, C, H, W, D = x.shape
        x = x.view(B, C, H // ws, ws, W // ws, ws, D // ws, ws)
        return x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous().view(-1, ws ** 3, C)

    def _unpartition(self, x: torch.Tensor, ws: int, B: int, C: int,
                     H: int, W: int, D: int) -> torch.Tensor:
        nH, nW, nD = H // ws, W // ws, D // ws
        x = x.view(B, nH, nW, nD, ws, ws, ws, C)
        return x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous().view(B, C, H, W, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W, D = x.shape
        ws = self.ws
        pH = (ws - H % ws) % ws
        pW = (ws - W % ws) % ws
        pD = (ws - D % ws) % ws
        if pH or pW or pD:
            x = F.pad(x, (0, pD, 0, pW, 0, pH))
        _, _, Hp, Wp, Dp = x.shape

        tokens = self._partition(x, ws)           # (B*nW, ws^3, C)
        TW, N, _ = tokens.shape
        residual = tokens

        normed = self.norm(tokens)
        qkv = self.qkv(normed).view(TW, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        attn_out = F.scaled_dot_product_attention(q, k, v)    # Flash Attention on CUDA BF16/FP16
        attn_out = attn_out.transpose(1, 2).reshape(TW, N, C)

        tokens = residual + self.proj(attn_out)   # residual over original (un-normed) tokens
        x = self._unpartition(tokens, ws, B, C, Hp, Wp, Dp)
        return x[:, :, :H, :W, :D]


class SwinBlock(nn.Module):
    """Swin transformer block: window-MHSA + channel-MLP FFN + stochastic depth.

    Stacks multiple of these at the bottleneck to give the model several passes
    of global context reasoning before decoding.
    """

    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 4,
                 mlp_ratio: float = 4.0, drop_path_rate: float = 0.0,
                 use_checkpoint: bool = False) -> None:
        super().__init__()
        self.attn           = WindowAttention3D(dim, num_heads, window_size)
        self.norm_ffn       = nn.LayerNorm(dim)
        mlp_dim             = int(dim * mlp_ratio)
        self.ffn            = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim),
        )
        self.drop_path      = DropPath(drop_path_rate)
        self.use_checkpoint = use_checkpoint

    def _inner(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        B, C, H, W, D = x.shape
        y = x.permute(0, 2, 3, 4, 1).contiguous()
        y = y + self.drop_path(self.ffn(self.norm_ffn(y)))
        return y.permute(0, 4, 1, 2, 3).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return grad_ckpt(self._inner, x, use_reentrant=False)
        return self._inner(x)


class ClasswiseModalityAttention(nn.Module):
    """Spatial, per-class soft attention over the input MRI modalities, added
    as a learned residual to each class's segmentation logit.

    A plain 1x1-conv head shares its weights across all sub-regions, so a
    modality cue useful for one class but not another has to be encoded
    somewhere upstream in the shared decoder features. Here each class instead
    gets its own softmax attention over [t1c, t1n, t2f, t2w], computed per
    voxel from the decoder features, so e.g. ED (edema) can learn to lean on
    T2f/T2w while ET leans on post-contrast T1c — learned per class rather
    than hard-restricting the whole model to one modality subset, which would
    starve classes that need the others.
    """

    def __init__(self, feat_ch: int, num_modalities: int, num_classes: int,
                 hidden: int = 32, use_checkpoint: bool = False) -> None:
        super().__init__()
        self.num_modalities = num_modalities
        self.num_classes    = num_classes
        self.class_conv = nn.Conv3d(feat_ch, num_classes, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Conv3d(feat_ch, hidden, kernel_size=1),
            nn.InstanceNorm3d(hidden, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(hidden, num_classes * num_modalities, kernel_size=1),
        )
        # Per-class residual strength, initialised to 0 so training starts
        # identical to a plain 1x1-conv head and only grows if it helps.
        self.mod_scale = nn.Parameter(torch.zeros(num_classes))
        self.use_checkpoint = use_checkpoint

    def _inner(self, feat: torch.Tensor, modalities: torch.Tensor) -> torch.Tensor:
        B, _, H, W, D = feat.shape
        base = self.class_conv(feat)

        if modalities.shape[2:] != feat.shape[2:]:
            modalities = F.interpolate(modalities, size=(H, W, D),
                                        mode="trilinear", align_corners=False)

        attn = self.gate(feat).view(B, self.num_classes, self.num_modalities, H, W, D)
        attn = torch.softmax(attn, dim=2)
        mod_maps = (attn * modalities.unsqueeze(1)).sum(dim=2)   # (B, num_classes, H, W, D)

        scale = self.mod_scale.view(1, -1, 1, 1, 1)
        return base + scale * mod_maps

    def forward(self, feat: torch.Tensor, modalities: torch.Tensor) -> torch.Tensor:
        # The (B, classes, modalities, H, W, D) attention/product tensors this
        # computes at full resolution are cheap to recompute but expensive to
        # keep resident for backward — checkpoint like the other blocks here.
        if self.use_checkpoint and self.training:
            return grad_ckpt(self._inner, feat, modalities, use_reentrant=False)
        return self._inner(feat, modalities)


class SPADE(nn.Module):
    """Spatially Adaptive Denormalization (Park et al. 2019).

    Conditions decoder feature normalisation on an auxiliary WT probability mask
    predicted from the encoder. Interpolated to decoder feature resolution.
    """

    def __init__(self, norm_nc: int, label_nc: int = 1, hidden: int = 64) -> None:
        super().__init__()
        self.norm   = nn.InstanceNorm3d(norm_nc, affine=False)
        self.shared = nn.Sequential(
            nn.Conv3d(label_nc, hidden, 3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.gamma  = nn.Conv3d(hidden, norm_nc, 3, padding=1)
        self.beta   = nn.Conv3d(hidden, norm_nc, 3, padding=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.shape[2:] != x.shape[2:]:
            mask = F.interpolate(mask.float(), size=x.shape[2:], mode="nearest")
        h = self.shared(mask)
        return self.norm(x) * (1.0 + self.gamma(h)) + self.beta(h)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class MedSwinNet(nn.Module):
    """BraTS-PED segmentation model (~40 M parameters).

    Scaled-up MedNeXt encoder-decoder:
    1. Filters [64,128,256,320,512], blocks [2,3,4,4,4] — 2× capacity vs original
    2. 4× stacked SwinBlocks at the 8³ bottleneck — multiple global-context passes
    3. SPADE in the first two decoder stages for spatially-adaptive normalisation
    4. Linear stochastic depth schedule (0 → drop_path_rate) across all blocks
    5. Deep supervision at full + half resolution
    """

    def __init__(
        self,
        in_channels:      int   = IN_CHANNELS,
        out_channels:     int   = OUT_CHANNELS,
        deep_supervision: bool  = True,
        drop_path_rate:   float = 0.2,
        use_checkpoint:   bool  = False,
    ) -> None:
        super().__init__()
        self.deep_supervision = deep_supervision
        F, NB, KS = _FILTERS, _N_BLOCKS, _KERNELS
        ck = use_checkpoint

        # Linear stochastic depth schedule across all blocks
        enc_blocks = sum(NB)                 # encoder MedNeXt blocks: 2+3+4+4+4 = 17
        dec_blocks = sum(NB[:-1])            # decoder MedNeXt blocks: 2+3+4+4  = 13
        total      = enc_blocks + _N_SWIN + dec_blocks   # 34 total
        dprs       = torch.linspace(0, drop_path_rate, total).tolist()
        ptr        = 0

        def _next(n: int) -> List[float]:
            nonlocal ptr
            out = dprs[ptr:ptr + n]; ptr += n; return out

        # ── Stem (full resolution) ────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, F[0], 3, padding=1, bias=False),
            nn.InstanceNorm3d(F[0], affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            *[MedNeXtBlock(F[0], KS[0], drop_path_rate=dpr, use_checkpoint=ck)
              for dpr in _next(NB[0])],
        )                                                   # 128³ × 64 ch

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = EncBlock(F[0], F[1], NB[1], KS[1], _next(NB[1]), ck)  #  64³ × 128 ch
        self.enc2 = EncBlock(F[1], F[2], NB[2], KS[2], _next(NB[2]), ck)  #  32³ × 256 ch
        self.enc3 = EncBlock(F[2], F[3], NB[3], KS[3], _next(NB[3]), ck)  #  16³ × 320 ch
        self.enc4 = EncBlock(F[3], F[4], NB[4], KS[4], _next(NB[4]), ck)  #   8³ × 512 ch

        # ── Bottleneck: 4 stacked Swin transformer blocks ────────────────────
        self.bottleneck = nn.Sequential(*[
            SwinBlock(F[4], num_heads=8, window_size=4, drop_path_rate=dpr, use_checkpoint=ck)
            for dpr in _next(_N_SWIN)
        ])

        # ── Auxiliary WT head at enc3 (for SPADE conditioning) ───────────────
        self.aux_wt = nn.Conv3d(F[3], 1, kernel_size=1)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec3 = DecBlock(F[4], F[3], F[3], NB[3], KS[3], use_spade=True,  drop_path_rates=_next(NB[3]), use_checkpoint=ck)
        self.dec2 = DecBlock(F[3], F[2], F[2], NB[2], KS[2], use_spade=True,  drop_path_rates=_next(NB[2]), use_checkpoint=ck)
        self.dec1 = DecBlock(F[2], F[1], F[1], NB[1], KS[1],                  drop_path_rates=_next(NB[1]), use_checkpoint=ck)
        self.dec0 = DecBlock(F[1], F[0], F[0], NB[0], KS[0],                  drop_path_rates=_next(NB[0]), use_checkpoint=ck)

        # ── Output heads ─────────────────────────────────────────────────────
        # Classwise modality attention: each output class learns its own
        # softmax attention over the 4 input MRI modalities (see class
        # docstring) instead of sharing one conv's weights across all classes.
        self.head_full = ClasswiseModalityAttention(F[0], in_channels, out_channels, use_checkpoint=ck)
        if deep_supervision:
            self.head_half = ClasswiseModalityAttention(F[1], in_channels, out_channels, use_checkpoint=ck)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], torch.Tensor]:
        """
        Returns:
            seg_outputs:   [full, half] when deep_supervision=True, else single tensor
            wt_aux_logit:  (B, 1, H/8, W/8, D/8) for auxiliary BCE loss
        """
        s0 = self.stem(x)       # 128³, 64 ch
        e1 = self.enc1(s0)      #  64³, 128 ch
        e2 = self.enc2(e1)      #  32³, 256 ch
        e3 = self.enc3(e2)      #  16³, 320 ch
        e4 = self.enc4(e3)      #   8³, 512 ch

        bn = self.bottleneck(e4)

        wt_aux_logit = self.aux_wt(e3)
        wt_mask      = torch.sigmoid(wt_aux_logit)

        d3 = self.dec3(bn, e3, wt_mask)
        d2 = self.dec2(d3, e2, wt_mask)
        d1 = self.dec1(d2, e1)
        d0 = self.dec0(d1, s0)

        logits = self.head_full(d0, x)

        if self.deep_supervision:
            return [logits, self.head_half(d1, x)], wt_aux_logit
        return logits, wt_aux_logit

    def forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        """Full-resolution logit only — for sliding_window_inference."""
        out, _ = self.forward(x)
        return out[0] if isinstance(out, list) else out


# ─────────────────────────────────────────────────────────────────────────────
# Loss utilities
# ─────────────────────────────────────────────────────────────────────────────

class ETFocalLoss(nn.Module):
    """BCE focal loss with per-channel class weights for [ET, NET, CC, ED].

    Weights are set relative to each class's *subject-level* prevalence in
    the BraTS-PEDs training set (ET ~67%, NET ~99%, CC ~35%, ED ~22%): ED is
    actually the rarest sub-region, present in only ~1 in 5 subjects, so it
    is weighted higher than ET. Focal exponent concentrates loss on uncertain
    boundary voxels.
    """

    def __init__(self, et_weight: float = 2.5, ed_weight: float = 3.0,
                 gamma: float = 2.0) -> None:
        super().__init__()
        self.et_weight = et_weight
        self.ed_weight = ed_weight
        self.gamma     = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p     = torch.sigmoid(logits)
        bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt    = torch.where(targets > 0.5, p, 1.0 - p)
        focal = (1.0 - pt) ** self.gamma * bce
        w     = torch.ones(logits.shape[1], device=logits.device)
        w[0]  = self.et_weight    # ET — small & hard
        w[3]  = self.ed_weight    # ED — diffuse, easily dominated by core loss
        return (focal * w[None, :, None, None, None]).mean()


def deep_supervision_loss(
    outputs: List[torch.Tensor],
    criterion: nn.Module,
    targets: torch.Tensor,
    weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """Weighted DiceCE across deep-supervision scales."""
    if weights is None:
        weights = [1.0 / (2 ** i) for i in range(len(outputs))]
    total = sum(weights)
    loss  = torch.tensor(0.0, device=outputs[0].device)
    for out, w in zip(outputs, weights):
        tgt = (
            F.interpolate(targets.float(), size=out.shape[2:], mode="nearest")
            if out.shape[2:] != targets.shape[2:]
            else targets
        )
        loss = loss + (w / total) * criterion(out, tgt)
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing
# ─────────────────────────────────────────────────────────────────────────────

def post_process_et(
    pred: torch.Tensor,
    threshold: float = 0.5,
    min_et_voxels: int = 10,
) -> torch.Tensor:
    """Threshold (all channels) and remove ET connected components smaller
    than min_et_voxels.

    NOTE: this binarizes every channel at a single `threshold`, which is the
    right behaviour for train.py's validate_epoch (it needs a hard 0/1
    prediction to score against DiceMetric). It is NOT suitable for inference
    pipelines that apply different thresholds per sub-region (e.g. a lower
    recall-favouring cutoff for the diffuse ED channel) — use
    remove_small_components() for that instead, on already-thresholded
    per-channel masks, so the real per-channel thresholds see raw
    probabilities rather than a pre-binarized 0.5 cutoff.
    """
    binary = (pred >= threshold).cpu().numpy()
    for b in range(binary.shape[0]):
        et = binary[b, 0]
        labeled, n = ndimage.label(et)
        for k in range(1, n + 1):
            if (labeled == k).sum() < min_et_voxels:
                et[labeled == k] = 0
        binary[b, 0] = et
    return torch.from_numpy(binary.astype(np.float32)).to(pred.device)


def remove_small_components(mask: np.ndarray, min_voxels: int = 10) -> np.ndarray:
    """Zero out connected components of a binary mask smaller than min_voxels.

    Operates on a single (H, W, D) boolean/binary array — use this at
    inference on an already per-channel-thresholded mask (e.g. ET) instead of
    post_process_et(), which forces every channel to the same threshold.
    """
    labeled, n = ndimage.label(mask)
    out = mask.copy()
    for k in range(1, n + 1):
        if (labeled == k).sum() < min_voxels:
            out[labeled == k] = 0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    x     = torch.randn(1, IN_CHANNELS, *PATCH_SIZE, device=device)
    model = MedSwinNet(deep_supervision=True).to(device)

    outs, aux = model(x)
    print(f"Output scales : {[tuple(o.shape) for o in outs]}")
    print(f"Aux WT logit  : {tuple(aux.shape)}")
    print(f"Params        : {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")
    print("Sanity check passed.")
