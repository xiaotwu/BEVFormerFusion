# Encoder-Side LiDAR Fusion

## Overview

Encoder-side fusion integrates LiDAR information directly into the BEV encoder layers, allowing camera and LiDAR features to be jointly refined at each encoder level before reaching the decoder. Each of the 4 encoder layers runs dual spatial cross-attention (SCA) — one attending to camera image features, the other to LiDAR BEV tokens — and blends the results with a learnable weight, enabling the model to progressively balance camera and LiDAR contributions.

## Pipeline

### 1. LiDAR Feature Extraction

LiDAR point clouds are processed by PointPillars into a dense BEV feature map.

```
Raw points (N, 4)  -->  Voxelize  -->  PillarFeatureNet  -->  PointPillarsScatter
                                                               --> (B, 64, 512, 512)
```

**Key detail**: Under FP16 mixed precision training, the `hard_voxelize` CUDA kernel does not support half-precision inputs. Point clouds are explicitly cast to FP32 before voxelization in `extract_pts_bev_feat()`, while downstream operations run in FP16.

| Module | Location | Description |
|--------|----------|-------------|
| Voxelization | `bevformer.py:extract_pts_bev_feat()` | Hard voxelize with FP32 cast |
| PillarFeatureNet | `bevformer.py` | Pillar-level feature aggregation |
| PointPillarsScatter | `bevformer.py` | Scatter pillars to (B, 64, 512, 512) BEV grid |

### 2. Projection to BEV Tokens

The LiDAR BEV map is projected to match BEV query dimensions before entering the encoder.

```
bev_lidar (B, 64, 512, 512)
     |
  lidar_encoder_proj (Conv2d 64->256, 1x1)   # Xavier-initialized
     |
  F.interpolate (bilinear) --> (B, 256, 100, 100)   # match BEV grid
     |
  flatten + permute --> (B, 10000, 256)
     |
  F.layer_norm --> normalize to unit variance
     |
  * alpha (=5.0) --> scale to match camera BEV magnitudes
     |
  lidar_bev_tokens (B, 10000, 256)
```

**Finding**: Raw LiDAR tokens have significantly smaller norms (~0.4) compared to camera BEV features (~16.0). LayerNorm followed by constant scaling (alpha=5.0) bridges this magnitude gap, producing tokens with norms of ~35 — in the same order of magnitude as camera features. This normalization is critical for stable gradient flow through the learnable blend weights.

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `lidar_encoder_proj` | `Conv2d(64, 256, 1)` | Channel projection to embed_dims |
| Interpolation | Bilinear, 512&rarr;100 | Spatial alignment with BEV grid |
| LayerNorm | Over embed_dims (256) | Normalize feature variance |
| Alpha | 5.0 (constant) | Scale to camera BEV magnitude range |

### 3. Fusion Inside Each Encoder Layer

Each of the 4 encoder layers runs two parallel SCA operations and blends the results with a learnable weight.

```
                    BEV query
                   /         \
    Camera SCA (deformable)   LiDAR SCA (deformable)
    attend to image features   attend to lidar_bev_tokens
           |                         |
        new_query1                new_query2
                   \         /
            Learnable weighted blend
     query = new_query1 * w + new_query2 * (1 - w)
```

| Component | Module | Details |
|-----------|--------|---------|
| Camera SCA | `MSDeformableAttention3D` | 4 sampling points, 4 feature levels |
| LiDAR SCA | `CustomMSDeformableAttention` | 4 sampling points, 1 level (BEV plane) |
| Blend weights | `cross_model_weights` (`nn.Parameter`) | Softmax-normalized, initialized to 0.5/0.5 |

**Design rationale**: The softmax normalization ensures the two weights always sum to 1, constraining the blend to a convex combination. Initializing at 0.5/0.5 gives both modalities equal influence at the start of training, allowing the network to discover the optimal balance through gradient descent. Because this blend is applied at every encoder layer, the model can learn layer-specific weighting — earlier layers may prefer different camera/LiDAR ratios than later layers.

### 4. Camera-Only BEV Preservation

After the encoder produces the fused `bev_embed`, a clone is saved **before** decoder-side fusion modifies it:

```python
bev_embed_cam = bev_embed.clone()  # (B, 10000, 256) — preserves temporal motion signal
```

This clone is used exclusively by the velocity head (see below). It preserves the temporal motion signal from TemporalSelfAttention (TSA) that would otherwise be further diluted by static LiDAR features during decoder-side fusion.

## Velocity Problem and Solution

### The Problem

The encoder's TemporalSelfAttention (TSA) compares current vs previous BEV frames to infer object motion — this is the **primary velocity signal** in BEVFormer. When LiDAR features (which are single-frame, static geometry with no temporal information) are blended in at each encoder layer, they dilute this temporal signal.

The velocity regression (vx, vy at bbox indices 8-9) already has a weak training signal:

| Factor | Value | Impact |
|--------|-------|--------|
| Code weight | 0.5 | Lower than position/size dimensions |
| L1 loss weight | 0.25 | Smallest among regression losses |
| Effective contribution | Small | Easily dominated by other losses |

**Observation**: In the base BEVFormer (camera-only), velocity is predicted from the same BEV features used for position and size, relying entirely on TSA's temporal signal. Adding LiDAR fusion improves geometry (position, size) but degrades velocity, because the LiDAR contribution is spatially rich but temporally uninformative.

### The Solution: Dedicated Velocity Head

A separate velocity prediction pathway bypasses the fusion-diluted features by cross-attending to the camera-only BEV snapshot.

```
Decoder queries (hs[lvl])  ----+
                                |
bev_embed_cam (camera-only) ---+--> MultiheadAttention (cross-attn)
                                |
                              vel_context
                                |
                            MLP (256 --> 256 --> 2)
                                |
                            vel_pred (vx, vy)
```

| Component | Details |
|-----------|---------|
| Cross-attention | `nn.MultiheadAttention(256, num_heads=8, batch_first=True)`, one per decoder layer |
| MLP | `Linear(256, 256)` &rarr; `ReLU` &rarr; `Linear(256, 2)`, one per decoder layer |
| Training loss | SmoothL1 against GT velocity targets, weight=0.25, scaled by VEL_W=1.0 |
| Inference | `vel_preds` override `bbox_preds[..., 8:10]` in the bbox coder |

### Why This Works

1. **Preserved temporal signal**: The camera-only BEV retains full temporal information from TSA (prev_bev comparison), uncontaminated by static LiDAR features
2. **Selective attention**: Cross-attention lets each object query selectively extract motion-relevant features from the camera BEV, rather than using a spatially pooled representation
3. **Decoupled gradients**: The separate MLP is not forced to compromise between geometry (helped by LiDAR) and motion (hurt by LiDAR)
4. **Independent loss path**: The velocity loss has its own gradient path, unaffected by competing bbox regression priorities

## Configuration

| Parameter | Value |
|-----------|-------|
| Encoder layers | 4 |
| Fusion mode | `encoder_decoder` |
| LiDAR SCA sampling points | 4 |
| Camera SCA sampling points | 4 |
| Camera SCA feature levels | 4 |
| Blend weight initialization | 0.5 / 0.5 (softmax) |
| Velocity loss | SmoothL1, weight=0.25 |
| Velocity code_weight | 0.5 (increased from default 0.2) |
| Precision | FP16 mixed |

## File Reference

| File | Location | Relevant Code |
|------|----------|---------------|
| `bevformer.py` | `extract_pts_bev_feat()` | PointPillars pipeline, FP32 voxelization cast |
| `transformer.py` | Lines 54-62 | `fusion_mode` parameter and validation |
| `transformer.py` | Lines 98-103 | `lidar_encoder_proj` module definition |
| `transformer.py` | Lines 376-377 | `bev_embed_cam` clone before decoder-side fusion |
| `encoder.py` | `MM_BEVFormerLayer` | `lidar_cross_attn_layer`, `cross_model_weights`, dual SCA blend |
| `bevformer_head.py` | `_init_layers()` | `vel_cross_attn`, `vel_branches` definitions |
| `bevformer_head.py` | `forward()` | Velocity cross-attention and prediction per decoder level |
| `bevformer_head.py` | `loss_single()` | `loss_vel` computation |
| `nms_free_coder.py` | `decode_single()` | `vel_preds` override of `bbox_preds[..., 8:10]` |
