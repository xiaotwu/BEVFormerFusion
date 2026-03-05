# Decoder-Side LiDAR Fusion

## Overview

Decoder-side fusion injects LiDAR information into the BEV representation **after** the encoder and **before** the decoder. Unlike encoder-side fusion (which blends features within each encoder layer via learnable weighted SCA), decoder-side fusion performs a single concat-and-project operation, giving the decoder access to a combined camera+LiDAR BEV through a complementary fusion mechanism.

When `fusion_mode='encoder_decoder'`, both fusion stages are active — the encoder already blends LiDAR into the BEV at every layer, and the decoder-side fusion adds a second injection point with its own independent projection.

## Pipeline

### 1. Input: Encoder Output + LiDAR BEV

At this point, `bev_embed` from the encoder already contains encoder-side fused features (camera + LiDAR blended at each layer). The raw LiDAR BEV `bev_lidar` (B, 64, 512, 512) is available separately for a second round of fusion.

```
bev_embed (B, 10000, 256)          bev_lidar (B, 64, 512, 512)
   (encoder output,                    (raw PointPillars output)
    already has encoder-side
    LiDAR fusion)
```

### 2. Camera-Only BEV Snapshot

Before decoder-side fusion modifies `bev_embed`, a clone is saved to preserve the temporal motion signal:

```python
bev_embed_cam = bev_embed.clone()  # (B, 10000, 256)
```

This clone is used exclusively by the velocity head, which cross-attends decoder queries to camera-only BEV features to predict (vx, vy). See [Encoder-side_Fusion.md](Encoder-side_Fusion.md#velocity-problem-and-solution) for the full velocity problem analysis.

### 3. LiDAR Projection and Normalization

The raw LiDAR BEV is projected, spatially aligned, and magnitude-normalized to match the camera BEV features.

```
bev_lidar (B, 64, 512, 512)
     |
  F.interpolate (bilinear) --> (B, 64, 100, 100)    # match BEV grid
     |
  lidar_proj (Conv2d 64->256, 1x1) --> (B, 256, 100, 100)
     |
  flatten + permute --> (B, 10000, 256)   # bev_lidar_tok
     |
  F.layer_norm --> normalize to unit variance
     |
  * alpha (=5.0) --> scale to match camera BEV magnitudes
     |
  bev_lidar_tok (B, 10000, 256)
```

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `lidar_proj` | `Conv2d(64, 256, 1)`, Xavier-init | Project LiDAR channels to embed_dims |
| Interpolation | Bilinear, 512&rarr;100 | Spatial alignment with BEV grid |
| LayerNorm | Over embed_dims (256) | Normalize feature variance |
| Alpha | 5.0 (constant) | Scale to camera BEV magnitude range |

The alpha scaling brings LiDAR features into the same order of magnitude as camera BEV features (~16.0), ensuring meaningful gradient flow through the downstream fusion linear layer.

### 4. Concat + Linear Fusion

The camera BEV and projected LiDAR tokens are concatenated along the feature dimension and compressed back to the original dimensionality.

```
bev_embed (B, HW, C=256)    bev_lidar_tok (B, HW, C=256)
            \                     /
         torch.cat(dim=-1)
              |
        (B, HW, 2C=512)
              |
     lidar_fuse_linear (Linear 512->256)
              |
     lidar_fuse_norm (LayerNorm 256)
              |
     bev_embed (B, HW, C=256)   <-- now contains decoder-side fusion
```

**Initialization strategy** (critical for training stability):

```python
# In init_weights():
lidar_fuse_linear.weight.zero_()           # start with all zeros
lidar_fuse_linear.weight[:, :C] = eye(C)   # camera half = identity
# lidar half remains zero
```

**Design rationale**: At initialization, the fusion linear layer **passes camera features through unchanged** and ignores LiDAR entirely. This identity initialization prevents the sudden injection of noisy LiDAR features from destabilizing early training. The network gradually learns to incorporate LiDAR information as training progresses, allowing the linear layer to discover what LiDAR information is complementary to the encoder's already-fused BEV representation.

### 5. Decoder Receives Fused BEV

The fused `bev_embed` is permuted and used as the value (memory) for the transformer decoder's cross-attention.

```python
value = bev_embed.permute(1, 0, 2)  # (HW, B, C)

inter_states, inter_references = self.decoder(
    query=query,          # (num_query, B, C) — 450 object queries
    value=value,          # (HW, B, C) — fused BEV as decoder memory
    reference_points=..., # (B, num_query, 3) — 3D reference points
    ...
)
```

Each of the 6 decoder layers performs:
1. **Self-attention** among the 450 object queries
2. **Cross-attention**: queries attend to the fused BEV, which contains both camera and LiDAR information from both fusion stages

## Comparison: Encoder-Side vs Decoder-Side Fusion

| Aspect | Encoder-Side | Decoder-Side |
|--------|-------------|--------------|
| **Location** | Inside each encoder layer (4x) | Once, between encoder and decoder |
| **Mechanism** | Dual SCA with learnable blend | Concat + Linear + LayerNorm |
| **LiDAR projection** | `lidar_encoder_proj` (Conv2d 64&rarr;256) | `lidar_proj` (Conv2d 64&rarr;256) |
| **Blending** | `cam * w + lidar * (1-w)`, w = sigmoid(logit) | Concat then learned linear projection |
| **Initialization** | Standard Xavier | Identity passthrough (camera=I, lidar=0) |
| **Effect on temporal** | Dilutes TSA signal at each layer | Single dilution after encoder |
| **Granularity** | Fine-grained, per-layer refinement | Coarse, single-shot injection |
| **Parameters** | `cross_model_weights_logit` (1 scalar per layer) | `lidar_fuse_linear` (256x512 + bias) |

**Finding**: When both fusion stages are active (`encoder_decoder` mode), LiDAR information enters the BEV through two complementary paths:
1. **Encoder**: fine-grained, per-layer blending via deformable attention — allows gradual refinement of BEV features with LiDAR geometry at each abstraction level
2. **Decoder**: coarse, single-shot injection via concatenation — provides a direct LiDAR signal to the decoder, bypassing any information loss from iterative encoder blending

## Modules

All decoder-side fusion modules are defined in `transformer.py` within the `PerceptionTransformer` class.

| Module | Type | Shape | Purpose |
|--------|------|-------|---------|
| `lidar_proj` | `Conv2d(64, 256, 1)` | (64, 256, 1, 1) | Project LiDAR channels to embed_dims |
| `lidar_fuse_linear` | `Linear(512, 256)` | (256, 512) | Compress concat(cam, lidar) back to C |
| `lidar_fuse_norm` | `LayerNorm(256)` | (256,) | Normalize after fusion |

## File Reference

| File | Location | Relevant Code |
|------|----------|---------------|
| `transformer.py` | Lines 54-62 | `fusion_mode` parameter and validation |
| `transformer.py` | Lines 98-103 | Module definitions (`lidar_proj`, `lidar_fuse_linear`, `lidar_fuse_norm`) |
| `transformer.py` | Lines 161-186 | `init_weights()` — identity init for camera passthrough |
| `transformer.py` | Lines 376-377 | `bev_embed_cam` clone (before decoder-side fusion) |
| `transformer.py` | Lines 379-469 | Forward pass — projection, normalization, concat, linear fusion |
| `bevformer_project.py` | Line 148 | `fusion_mode='encoder_decoder'` config |
