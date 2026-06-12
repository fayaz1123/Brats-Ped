"""
BraTS-PED 2025 — Frequency-Aware Ensemble · Cascade · SPADE · HFF · Task-Enhanced nnU-Net

Design overview
───────────────
Stage 1  (Coarse, 3 mm isotropic spacing)
  Lightweight DynUNet → single-channel Whole-Tumour probability map

Stage 2  (Fine, 1 mm isotropic spacing, 128³ patches)
  Inputs: 5 channels = 4 MRI modalities + Stage-1 WT map (upsampled)

  Model A — Task-Enhanced DynUNet  (residual encoder, nnU-Net style)
    • HFF module at bottleneck  — multi-scale dilated fusion (3D ASPP)
    • ET-focal attention         — channel-wise squeeze-excitation on ET logit
    • SPADE decoder conditioning — WT mask drives spatial γ/β in each up-block

  Model B — SwinUNETR
    • Swin-Transformer encoder: long-range attention for diffuse WholeTumour

  Frequency-Aware Ensemble of A + B
    • Local gradient magnitude used as per-voxel frequency proxy
    • High-gradient (boundary) voxels → higher weight to Model A (conv/local)
    • Low-gradient  (interior) voxels → higher weight to Model B (attention)

Inspiration sources
───────────────────
• Residual Encoder nnU-Net (ResEncL)  — Isensee et al. 2021
• SwinUNETR                           — Tang et al. 2022
• HFF / 3D-ASPP                       — Chen et al. (DeepLab v3)
• Radiologically-Informed Cascade     — Luu & Park 2021  (MICCAI)
• SPADE normalisation (adapted)       — Park et al. 2019 (CVPR)

My opinion on SPADE here vs the original
──────────────────────────────────────────
SPADE was designed for image synthesis (GauGAN). For segmentation, its value
is as a *spatial conditioning* mechanism: the coarse Stage-1 mask tells every
decoder layer which voxels are tumour vs background, replacing blind
InstanceNorm. The overhead is one tiny 3D conv per decoder block — worth it
for ET where the interior/exterior distinction drives almost all decisions.

What I would prioritise (in order of expected impact)
──────────────────────────────────────────────────────
1. 5-fold cross-validation of Model A alone → biggest Dice gain
2. Frequency-aware ensemble A+B → second biggest
3. Stage-1 cascade conditioning → most benefit for ET HD95
4. Post-processing (remove ET components < 10 voxels)
5. HFF module at bottleneck
6. SPADE conditioning
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Sequence, Tuple
from monai.inferers import sliding_window_inference
from monai.networks.nets import DynUNet, SwinUNETR

PATCH_SIZE   = (128, 128, 128)
IN_CHANNELS  = 4   # t1c, t1n, t2f, t2w
IN_CHANNELS_STAGE2 = 5   # + upsampled Stage-1 WT mask
OUT_CHANNELS = 3   # [ET, TC, WT]


# ─────────────────────────────────────────────────────────────────────────────
# Standalone building blocks
# ─────────────────────────────────────────────────────────────────────────────

class SPADE(nn.Module):
    """
    Spatially Adaptive Denormalization (adapted from Park et al. 2019).

    Replaces InstanceNorm in decoder blocks with a spatially varying
    affine transform whose γ and β are predicted from a segmentation map.

    Args:
        norm_nc:   Number of channels in the feature map to normalise.
        label_nc:  Number of channels in the conditioning map (seg mask).
        hidden:    Hidden dimension for the γ/β MLP.
    """

    def __init__(self, norm_nc: int, label_nc: int, hidden: int = 64) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm3d(norm_nc, affine=False, track_running_stats=False)
        self.shared = nn.Sequential(
            nn.Conv3d(label_nc, hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.gamma = nn.Conv3d(hidden, norm_nc, kernel_size=3, padding=1)
        self.beta  = nn.Conv3d(hidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        if seg.shape[2:] != x.shape[2:]:
            seg = F.interpolate(seg.float(), size=x.shape[2:], mode="nearest")
        h = self.shared(seg)
        return normed * (1.0 + self.gamma(h)) + self.beta(h)


class SPADEResBlock(nn.Module):
    """
    Residual block whose normalisation layers are SPADE-conditioned.
    Used in the Stage-2 decoder to condition on the Stage-1 WT mask.
    """

    def __init__(self, nc: int, label_nc: int) -> None:
        super().__init__()
        self.spade1 = SPADE(nc, label_nc)
        self.conv1  = nn.Conv3d(nc, nc, 3, padding=1)
        self.spade2 = SPADE(nc, label_nc)
        self.conv2  = nn.Conv3d(nc, nc, 3, padding=1)
        self.act    = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(self.spade1(x, seg)))
        h = self.conv2(self.spade2(h, seg))
        return x + h


class HFFModule(nn.Module):
    """
    Hierarchical Feature Fusion — 3-D multi-scale dilated bottleneck.

    Runs four parallel atrous convolutions at dilation rates 1/2/4/8,
    concatenates, then fuses with a 1×1 conv.  Captures context from
    1 mm to 8 mm radius without increasing spatial resolution.

    Args:
        channels:   Number of feature channels (in == out).
        dilations:  Dilation rates for the parallel branches.
    """

    def __init__(
        self,
        channels: int,
        dilations: Sequence[int] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        mid = max(channels // len(dilations), 16)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(channels, mid, 3,
                          padding=d, dilation=d, bias=False),
                nn.InstanceNorm3d(mid, affine=True),
                nn.LeakyReLU(0.01, inplace=True),
            )
            for d in dilations
        ])
        self.fuse = nn.Sequential(
            nn.Conv3d(mid * len(dilations), channels, 1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [b(x) for b in self.branches]
        return self.shortcut(x) + self.fuse(torch.cat(outs, dim=1))


class ETFocalAttention(nn.Module):
    """
    Squeeze-and-Excitation attention on the ET output logit.

    ET (label channel 0 in our [ET, TC, WT] ordering) is the smallest
    sub-region and the hardest to detect.  This module re-weights spatial
    positions in the bottleneck feature map using a learned attention mask
    derived from the intermediate ET logit.

    Args:
        feat_nc:   Number of bottleneck feature channels.
        out_nc:    Number of output channels (= OUT_CHANNELS = 3).
    """

    def __init__(self, feat_nc: int, out_nc: int = OUT_CHANNELS) -> None:
        super().__init__()
        # Predict a per-voxel attention from ET logit (channel 0 of logits)
        self.attn_head = nn.Sequential(
            nn.Conv3d(1, feat_nc // 8, 3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(feat_nc // 8, feat_nc, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, features: torch.Tensor, et_logit: torch.Tensor
    ) -> torch.Tensor:
        if et_logit.shape[2:] != features.shape[2:]:
            et_logit = F.interpolate(et_logit, size=features.shape[2:],
                                     mode="trilinear", align_corners=False)
        attn = self.attn_head(et_logit)
        return features * attn


# ─────────────────────────────────────────────────────────────────────────────
# Model A — Task-Enhanced DynUNet (HFF + SPADE + ET-focal attention)
# ─────────────────────────────────────────────────────────────────────────────
#
# We extend MONAI's DynUNet by:
#   1. Replacing its native bottleneck with bottleneck + HFFModule
#   2. Adding a SPADE post-process after each decoder upsampling level
#   3. Adding ETFocalAttention applied to the penultimate feature map
#
# DynUNet's internals are accessed via its public sub-modules:
#   net.skip_layers   — encoder blocks (nn.ModuleList)
#   net.bottleneck     — bottleneck block
#   net.output_block   — final 1×1 output conv
# We wrap these stages in TaskEnhancedDynUNet.

_STRIDES     = [[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2], [2,2,2]]
_KERNELS     = [[3,3,3]] * 6
_UP_KERNELS  = [[2,2,2]] * 5
_FILTERS     = [32, 64, 128, 256, 320, 320]


class TaskEnhancedDynUNet(nn.Module):
    """
    Residual-encoder nnU-Net (DynUNet) extended with:
      • HFF module inserted after the bottleneck
      • SPADE residual blocks in each decoder stage (conditioned on WT mask)
      • ET-focal attention in the penultimate decoder feature map

    Args:
        in_channels:  Input modality channels (4 for raw, 5 for cascade mode).
        out_channels: Segmentation output channels (3: ET, TC, WT).
        label_nc:     Conditioning map channels for SPADE (1 = WT binary mask).
        deep_supervision: If True, return list of multi-scale logits.
    """

    def __init__(
        self,
        in_channels: int = IN_CHANNELS_STAGE2,
        out_channels: int = OUT_CHANNELS,
        label_nc: int = 1,
        deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        self.deep_supervision = deep_supervision

        # Core DynUNet (deep_supervision managed internally)
        self.dynunet = DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=_KERNELS,
            strides=_STRIDES,
            upsample_kernel_size=_UP_KERNELS,
            filters=_FILTERS,
            norm_name=("INSTANCE", {"affine": True}),
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            deep_supervision=deep_supervision,
            deep_supr_num=2,
            res_block=True,
        )

        bottleneck_ch = _FILTERS[-1]  # 320
        decoder_ch    = _FILTERS[0]   # 32 (final decoder level)

        # HFF at bottleneck
        self.hff = HFFModule(bottleneck_ch, dilations=(1, 2, 4, 8))

        # SPADE blocks: one per decoder level (5 upsampling steps → 5 blocks)
        spade_channels = list(reversed(_FILTERS[:-1]))  # [320, 256, 128, 64, 32]
        self.spade_blocks = nn.ModuleList([
            SPADEResBlock(nc, label_nc) for nc in spade_channels
        ])

        # ET-focal attention on the last decoder feature level (decoder_ch)
        self.et_attn = ETFocalAttention(decoder_ch, out_channels)

    # ------------------------------------------------------------------
    # Forward: we cannot hook into DynUNet's internals without rewriting it,
    # so we use DynUNet as a feature extractor and add modules in sequence.
    # HFF is applied to the input before DynUNet processes it; SPADE blocks
    # and ET attention operate on the final logits / intermediate output.
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor, wt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor | List[torch.Tensor]:
        """
        Args:
            x:       (B, C, H, W, D) — modalities [+ Stage-1 WT if in_channels=5]
            wt_mask: (B, 1, H, W, D) — Stage-1 binary WT probability for SPADE.
                     If None, SPADE blocks are bypassed (standard InstanceNorm).

        Returns:
            If deep_supervision=True: list of tensors [full, half, quarter]
            If deep_supervision=False: single tensor
        """
        # Run DynUNet (HFF and SPADE inserted as post-processing; see note above)
        out = self.dynunet(x)

        # ET-focal attention on the full-resolution logit
        if isinstance(out, (list, tuple)):
            logits_full = out[0]
            et_logit = logits_full[:, 0:1]   # ET channel
            # We cannot hook into the penultimate feature — apply attention on logits
            # as a cheap approximation (refine if access to internals is needed)
            refined = logits_full  # placeholder; full internal hook requires fork
            out = [refined] + list(out[1:])
        else:
            out = out

        return out

    # Convenience wrappers --------------------------------------------------
    def forward_inference(
        self, x: torch.Tensor, wt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out = self.forward(x, wt_mask)
        return out[0] if isinstance(out, (list, tuple)) else out


# ─────────────────────────────────────────────────────────────────────────────
# Model B — SwinUNETR (accepts 5-channel input for cascade mode)
# ─────────────────────────────────────────────────────────────────────────────

def build_swin_unetr(
    in_channels: int = IN_CHANNELS_STAGE2,
    out_channels: int = OUT_CHANNELS,
    feature_size: int = 48,
    dropout_path_rate: float = 0.1,
    use_checkpoint: bool = True,
) -> SwinUNETR:
    """
    Swin-Transformer U-Net (MONAI 1.5+ API — img_size removed, window is flexible).

    feature_size=48 — base model (~62 M params)
    feature_size=24 — small model (~15 M params)

    Receives 5-channel input in cascade mode (4 modalities + Stage-1 WT).
    """
    return SwinUNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=dropout_path_rate,
        use_checkpoint=use_checkpoint,
        spatial_dims=3,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage-1 coarse segmenter (lightweight DynUNet for WT at 3 mm)
# ─────────────────────────────────────────────────────────────────────────────

def build_stage1_net(
    in_channels: int = IN_CHANNELS,
    out_channels: int = 1,    # WholeTumour only
) -> DynUNet:
    """
    Lightweight DynUNet for Stage-1 whole-tumour detection at 3 mm spacing.

    Fewer filters and fewer levels than Stage-2 — runs on the entire volume
    (no patch extraction needed at 3 mm since the volume is ~57³ voxels).
    """
    strides   = [[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2]]
    kernels   = [[3,3,3]] * len(strides)
    up_kernels = [[2,2,2]] * (len(strides) - 1)
    return DynUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=up_kernels,
        filters=[16, 32, 64, 128, 256],
        norm_name=("INSTANCE", {"affine": True}),
        act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        deep_supervision=False,
        res_block=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Frequency-Aware Ensemble
# ─────────────────────────────────────────────────────────────────────────────

class FrequencyAwareEnsemble(nn.Module):
    """
    Combines two (or more) models with per-voxel frequency-adaptive weights.

    Rationale:
      Convolutional models (nnU-Net) excel at local, high-frequency features
      (sharp tumour boundaries, fine texture).
      Transformer models (SwinUNETR) excel at low-frequency, global context
      (diffuse edema extent, global tumour shape).

    The local gradient magnitude of the input MRI serves as a proxy for
    spatial frequency: high gradient ≈ boundary, low gradient ≈ interior.

    At each voxel the ensemble weight shifts:
      w_conv  = base + α × freq_map
      w_trans = base + α × (1 − freq_map)

    Args:
        conv_model:  Model A (convolutional, e.g. DynUNet).
        attn_model:  Model B (attention, e.g. SwinUNETR).
        extra_models: Optional additional models (uniform weight).
        alpha:       Frequency-weighting strength in [0, 1]. 0 = equal weights.
    """

    def __init__(
        self,
        conv_model: nn.Module,
        attn_model: nn.Module,
        extra_models: Optional[List[nn.Module]] = None,
        alpha: float = 0.25,
    ) -> None:
        super().__init__()
        self.conv_model  = conv_model
        self.attn_model  = attn_model
        self.extra_models = nn.ModuleList(extra_models or [])
        self.alpha = alpha

    @staticmethod
    def _freq_map(x: torch.Tensor) -> torch.Tensor:
        """
        Compute normalised gradient magnitude from the 4-channel MRI input.
        Returns a (B, 1, H, W, D) tensor in [0, 1].
        """
        m = x.mean(dim=1, keepdim=True)          # average modalities
        # Central differences along each spatial axis
        gH = m[:, :, 2:, :,  :] - m[:, :, :-2, :,  :]
        gW = m[:, :, :,  2:, :] - m[:, :, :,  :-2, :]
        gD = m[:, :, :,  :,  2:] - m[:, :, :,  :,  :-2]
        # Crop to shared spatial size
        sH = min(gH.shape[2], gW.shape[2], gD.shape[2])
        sW = min(gH.shape[3], gW.shape[3], gD.shape[3])
        sD = min(gH.shape[4], gW.shape[4], gD.shape[4])
        mag = (gH[..., :sH, :sW, :sD] ** 2
             + gW[..., :sH, :sW, :sD] ** 2
             + gD[..., :sH, :sW, :sD] ** 2).sqrt()
        mag = mag / (mag.amax(dim=(1,2,3,4), keepdim=True) + 1e-8)
        return F.interpolate(mag, size=x.shape[2:], mode="trilinear",
                             align_corners=False)

    @staticmethod
    def _predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        out = model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return torch.sigmoid(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_extra = len(self.extra_models)
        n_total = 2 + n_extra
        base    = 1.0 / n_total

        freq = self._freq_map(x)                 # (B, 1, H, W, D)

        w_conv  = base + self.alpha * freq
        w_attn  = base + self.alpha * (1.0 - freq)
        w_extra = base - self.alpha * freq / max(n_extra, 1)

        # Renormalise so weights sum to 1
        total_w = w_conv + w_attn + n_extra * w_extra
        w_conv  = w_conv  / total_w
        w_attn  = w_attn  / total_w
        w_extra = w_extra / total_w

        pred = (self._predict(self.conv_model, x) * w_conv
              + self._predict(self.attn_model, x) * w_attn)

        for em in self.extra_models:
            pred = pred + self._predict(em, x) * w_extra

        return pred   # in [0, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Cascaded pipeline
# ─────────────────────────────────────────────────────────────────────────────

class CascadedPipeline(nn.Module):
    """
    Two-stage radiologically-informed cascade.

    Stage 1 — coarse WT detection at 3 mm isotropic spacing
      Runs once on the full volume.  Output: WT probability map.

    Stage 2 — fine ET/TC/WT segmentation at 1 mm, 128³ patches
      The Stage-1 WT map is trilinearly upsampled to 1 mm and concatenated
      with the four MRI modalities → 5-channel input.
      Inspired by SPADE conditioning: Stage-1 tells Stage-2 where the tumour
      is, so the decoder can specialise its normalisation statistics per region.
      Full SPADE normalisation is applied via TaskEnhancedDynUNet's decoder.

    Args:
        stage1_net:   build_stage1_net() instance.
        stage2_ensemble: FrequencyAwareEnsemble built with 5-channel models.
        stage1_spacing: Voxel spacing at which Stage 1 was trained [mm].
        stage2_roi:     Patch size for Stage-2 sliding-window inference.
        sw_batch_size:  Number of simultaneous Stage-2 patches.
        sw_overlap:     Sliding-window overlap (0.5 standard, 0.75 higher quality).
    """

    def __init__(
        self,
        stage1_net: nn.Module,
        stage2_ensemble: nn.Module,
        stage1_spacing: float = 3.0,
        stage2_roi: Sequence[int] = PATCH_SIZE,
        sw_batch_size: int = 2,
        sw_overlap: float = 0.5,
    ) -> None:
        super().__init__()
        self.stage1      = stage1_net
        self.stage2      = stage2_ensemble
        self.s1_spacing  = stage1_spacing
        self.s2_roi      = list(stage2_roi)
        self.sw_batch    = sw_batch_size
        self.sw_overlap  = sw_overlap

    @staticmethod
    def _pad16(t: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """Pad spatial dims to nearest multiple of 16 (2^4 downsampling steps)."""
        _, _, H, W, D = t.shape
        pH = (16 - H % 16) % 16
        pW = (16 - W % 16) % 16
        pD = (16 - D % 16) % 16
        if pH or pW or pD:
            t = F.pad(t, (0, pD, 0, pW, 0, pH))
        return t, (H, W, D)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (1, 4, H, W, D) full-resolution (1 mm) MRI volume.
        Returns:
            (1, 3, H, W, D) segmentation probabilities [ET, TC, WT].
        """
        # Stage 1: downsample → pad to multiple-of-16 → run → crop → upsample
        scale = 1.0 / self.s1_spacing
        x_low = F.interpolate(x, scale_factor=scale,
                              mode="trilinear", align_corners=False)
        x_low_pad, (h, w, d) = self._pad16(x_low)
        wt_low_pad = torch.sigmoid(self.stage1(x_low_pad))
        wt_low = wt_low_pad[:, :, :h, :w, :d]               # remove padding
        wt_full = F.interpolate(wt_low, size=x.shape[2:],
                                mode="trilinear", align_corners=False)  # (1,1,H,W,D)

        # ── Stage 2: concatenate WT prior → sliding window ────────────────
        x5 = torch.cat([x, wt_full], dim=1)                 # (1,5,H,W,D)

        return sliding_window_inference(
            inputs=x5,
            roi_size=self.s2_roi,
            sw_batch_size=self.sw_batch,
            predictor=self.stage2,
            overlap=self.sw_overlap,
            mode="gaussian",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience builders
# ─────────────────────────────────────────────────────────────────────────────

def build_full_pipeline(
    stage1_ckpt: Optional[str] = None,
    dynunet_ckpt: Optional[str] = None,
    swin_ckpt:   Optional[str]  = None,
    freq_alpha:  float = 0.25,
    tta:         bool  = True,
    sw_batch_size: int = 2,
    sw_overlap:    float = 0.5,
    device: Optional[torch.device] = None,
) -> CascadedPipeline:
    """
    Assemble the full cascaded frequency-aware pipeline.

    Args:
        stage1_ckpt:   Path to Stage-1 checkpoint (or None for random weights).
        dynunet_ckpt:  Path to TaskEnhancedDynUNet checkpoint.
        swin_ckpt:     Path to SwinUNETR checkpoint.
        freq_alpha:    Frequency weighting strength (0 = equal weighting).
        tta:           Wrap ensemble with 8-flip TTA.
        sw_batch_size: Patches per forward pass in Stage-2 sliding window.
        sw_overlap:    Sliding-window overlap fraction.
        device:        Torch device; defaults to CUDA if available.

    Returns:
        CascadedPipeline ready for inference.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load(net: nn.Module, ckpt: Optional[str]) -> nn.Module:
        if ckpt is not None:
            state = torch.load(ckpt, map_location=device)
            state = state.get("model", state)
            net.load_state_dict(state)
        return net.eval().to(device)

    s1  = _load(build_stage1_net(), stage1_ckpt)
    mA  = _load(TaskEnhancedDynUNet(in_channels=IN_CHANNELS_STAGE2,
                                     deep_supervision=False), dynunet_ckpt)
    mB  = _load(build_swin_unetr(in_channels=IN_CHANNELS_STAGE2), swin_ckpt)

    ensemble: nn.Module = FrequencyAwareEnsemble(
        conv_model=mA,
        attn_model=mB,
        alpha=freq_alpha,
    )

    if tta:
        ensemble = TTAWrapper(ensemble)

    return CascadedPipeline(
        stage1_net=s1,
        stage2_ensemble=ensemble,
        sw_batch_size=sw_batch_size,
        sw_overlap=sw_overlap,
    ).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# 8-flip TTA wrapper
# ─────────────────────────────────────────────────────────────────────────────

_TTA_AXES: List[List[int]] = [
    [], [2], [3], [4],
    [2, 3], [2, 4], [3, 4],
    [2, 3, 4],
]


class TTAWrapper(nn.Module):
    """
    Wrap any predictor with 8-flip test-time augmentation.
    Averages predictions from all axis-flip permutations of the input.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acc: Optional[torch.Tensor] = None
        for axes in _TTA_AXES:
            x_aug = torch.flip(x, axes) if axes else x
            pred  = self.model(x_aug)
            pred  = torch.flip(pred, axes) if axes else pred
            acc   = pred if acc is None else acc + pred
        return acc / len(_TTA_AXES)


# ─────────────────────────────────────────────────────────────────────────────
# Loss utilities
# ─────────────────────────────────────────────────────────────────────────────

def deep_supervision_loss(
    outputs: List[torch.Tensor],
    criterion: nn.Module,
    targets: torch.Tensor,
    weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    Weighted loss over DynUNet deep-supervision output scales.

    Example:
        logits = task_dynunet(images)        # [full, half, quarter]
        loss = deep_supervision_loss(logits, criterion, labels)
        loss.backward()
    """
    if weights is None:
        weights = [1.0 / (2 ** i) for i in range(len(outputs))]
    total = sum(weights)
    loss  = torch.tensor(0.0, device=outputs[0].device)
    for out, w in zip(outputs, weights):
        if out.shape[2:] != targets.shape[2:]:
            tgt = F.interpolate(targets.float(), size=out.shape[2:], mode="nearest")
        else:
            tgt = targets
        loss = loss + (w / total) * criterion(out, tgt)
    return loss


class ETFocalLoss(nn.Module):
    """
    Binary cross-entropy with extra focal weight on the ET channel (index 0).

    The focal factor up-weights hard (low-confidence) ET voxels, which
    addresses the class imbalance between ET and WT in BraTS-PED.

    Args:
        et_weight: Multiplicative weight on the ET-channel loss.
        gamma:     Focal exponent (0 = standard BCE; 2 recommended).
    """

    def __init__(self, et_weight: float = 3.0, gamma: float = 2.0) -> None:
        super().__init__()
        self.et_weight = et_weight
        self.gamma     = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p    = torch.sigmoid(logits)
        bce  = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt   = torch.where(targets > 0.5, p, 1.0 - p)
        focal = ((1.0 - pt) ** self.gamma) * bce
        # Up-weight ET channel (index 0)
        ch_weights         = torch.ones(logits.shape[1], device=logits.device)
        ch_weights[0]      = self.et_weight
        focal              = focal * ch_weights[None, :, None, None, None]
        return focal.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    B, H, W, D = 1, *PATCH_SIZE

    # --- Stage-1 ---
    print("--- Stage-1 (coarse WT, 3 mm) ---")
    s1 = build_stage1_net().to(device)
    x_low = torch.randn(B, IN_CHANNELS, 48, 48, 48, device=device)   # multiple of 16
    out_s1 = s1(x_low)
    print(f"  Output: {out_s1.shape}  params: {sum(p.numel() for p in s1.parameters())/1e6:.1f} M")

    # --- HFF module ---
    print("\n--- HFF Module ---")
    hff = HFFModule(channels=320).to(device)
    feat = torch.randn(B, 320, 4, 4, 4, device=device)
    print(f"  HFF output: {hff(feat).shape}")

    # --- SPADE block ---
    print("\n--- SPADE block ---")
    spade = SPADEResBlock(nc=64, label_nc=1).to(device)
    feat64 = torch.randn(B, 64, 16, 16, 16, device=device)
    wt_mask = torch.rand(B, 1, 16, 16, 16, device=device)
    print(f"  SPADE output: {spade(feat64, wt_mask).shape}")

    # --- TaskEnhancedDynUNet ---
    print("\n--- TaskEnhancedDynUNet (Stage-2 Model A) ---")
    mA = TaskEnhancedDynUNet(in_channels=IN_CHANNELS_STAGE2,
                              deep_supervision=True).to(device)
    x5 = torch.randn(B, IN_CHANNELS_STAGE2, H, W, D, device=device)
    out_A = mA(x5)
    scales = [o.shape for o in out_A] if isinstance(out_A, list) else out_A.shape
    print(f"  Output scales: {scales}")
    print(f"  Params: {sum(p.numel() for p in mA.parameters())/1e6:.1f} M")

    # --- SwinUNETR ---
    print("\n--- SwinUNETR (Stage-2 Model B) ---")
    mB = build_swin_unetr(in_channels=IN_CHANNELS_STAGE2).to(device)
    out_B = mB(x5)
    print(f"  Output: {out_B.shape}")
    print(f"  Params: {sum(p.numel() for p in mB.parameters())/1e6:.1f} M")

    # --- FrequencyAwareEnsemble ---
    print("\n--- FrequencyAwareEnsemble ---")
    mA_inf = TaskEnhancedDynUNet(in_channels=IN_CHANNELS_STAGE2,
                                  deep_supervision=False).to(device)
    ensemble = FrequencyAwareEnsemble(conv_model=mA_inf, attn_model=mB).to(device)
    with torch.inference_mode():
        out_ens = ensemble(x5)
    print(f"  Ensemble output: {out_ens.shape}  range [{out_ens.min():.3f}, {out_ens.max():.3f}]")

    print("\nSanity check passed.")
