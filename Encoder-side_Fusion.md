# Encoder-Side LiDAR Fusion

## Overview

Encoder-side fusion integrates LiDAR information directly into the BEV encoder layers, allowing camera and LiDAR features to be jointly refined at each encoder level before reaching the decoder.

## Pipeline

### 1. LiDAR Feature Extraction

LiDAR point clouds are processed by PointPillars in `bevformer.py:extract_pts_bev_feat()`:

```
Raw points (N, 4)  -->  Voxelize  -->  PillarFeatureNet  -->  PointPillarsScatter
                                                               --> (B, 64, 512, 512)
```

### 2. Projection to BEV Tokens

In `transformer.py` — `PerceptionTransformer.forward()`:

The LiDAR BEV map is projected to match BEV query dimensions:

```
(B, 64, 512, 512) --> Conv2d + Interpolate --> (B, 256, 100, 100) --> flatten --> (B, 10000, 256)
                      lidar_encoder_proj         lidar_bev_tokens
```

After projection, LayerNorm + learnable scaling (alpha=5.0) normalizes the LiDAR tokens to match camera BEV feature magnitudes.

### 3. Fusion Inside Each Encoder Layer

In `encoder.py` — `MM_BEVFormerLayer.forward()`:

Each of the 4 encoder layers runs two parallel Spatial Cross-Attention (SCA) operations:

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

- `cross_model_weights`: `nn.Parameter` with softmax, initialized to 0.5/0.5
- Camera SCA: `MSDeformableAttention3D` with 4 sampling points across 4 feature levels
- LiDAR SCA: `CustomMSDeformableAttention` with 4 sampling points on 1 level (BEV plane)

This happens at every encoder layer, so the BEV representation is progressively refined with both camera and LiDAR information.

### 4. Camera-Only BEV Preservation

After the encoder produces the fused `bev_embed`, a clone is saved **before** decoder-side fusion:

```python
bev_embed_cam = bev_embed.clone()  # (B, 10000, 256) — camera-only temporal BEV
```

This clone is used exclusively by the velocity head, preserving the temporal motion signal from TemporalSelfAttention (TSA) that would otherwise be diluted by static LiDAR features.

## Velocity Problem and Solution

### The Problem

The encoder's TemporalSelfAttention (TSA) compares current vs previous BEV frames to infer object motion — this is the **primary velocity signal**. When LiDAR features (which are single-frame, static geometry) are blended in at 50% weight, they dilute this temporal signal.

The velocity regression (vx, vy at bbox indices 8-9) already has a weak training signal:
- Code weight: 0.5 (lower than position/size)
- L1 loss weight: 0.25
- Effective contribution: small relative to other losses

### The Solution: Velocity Head

A dedicated velocity prediction pathway bypasses the fusion-diluted features:

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

- **Per decoder level**: Each of the 6 decoder layers has its own `vel_cross_attn` + `vel_branch`
- **Training**: SmoothL1 loss against GT velocity targets, weight=0.25, scaled by VEL_W=1.0
- **Inference**: `vel_preds` override `bbox_preds[..., 8:10]` in the bbox coder

### Why This Works

1. The camera-only BEV retains full temporal information from TSA (prev_bev comparison)
2. Cross-attention lets each object query selectively extract motion-relevant features
3. The separate MLP is not forced to compromise between geometry (helped by LiDAR) and motion (hurt by LiDAR)
4. The velocity loss has its own gradient path, unaffected by bbox regression competing priorities

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Encoder layers | 4 |
| Fusion mode | encoder_decoder |
| LiDAR SCA points | 4 |
| Camera SCA points | 4 |
| Blend weights | Learnable (softmax, init 0.5) |
| Velocity loss | SmoothL1, weight 0.25 |
| Velocity code_weight | 0.5 (increased from 0.2) |
| Precision | FP16 mixed |
| Iterations | 200K |

## File Reference

| File | Relevant Code |
|------|---------------|
| `bevformer.py` | `extract_pts_bev_feat()` — PointPillars pipeline |
| `transformer.py` | `lidar_encoder_proj`, `bev_embed_cam` clone, decoder-side fusion |
| `encoder.py` | `lidar_cross_attn_layer`, `cross_model_weights`, dual SCA blend |
| `bevformer_head.py` | `vel_cross_attn`, `vel_branches`, `loss_vel` |
| `nms_free_coder.py` | `vel_preds` override in `decode_single()` |
