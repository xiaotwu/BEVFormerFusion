# BEVFormerFusion

Multi-modal 3D object detection fusing camera and LiDAR inputs via BEVFormer with dual encoder-decoder side fusion.

## Architecture Overview

BEVFormerFusion extends the BEVFormer transformer architecture with a two-stage LiDAR fusion strategy. Camera images are encoded by ResNet50 + FPN into multi-scale features, while LiDAR point clouds are processed by PointPillars into a dense BEV feature map. The two modalities are fused at both the encoder and decoder stages, each with distinct mechanisms and complementary roles.

```
Camera (6 views)                         LiDAR point cloud
     |                                          |
  ResNet50 + FPN                          PointPillars
     |                                    (B,64,512,512)
  Multi-scale features                          |
     |                                   Conv2d + interpolate
     |                                   LayerNorm + alpha scaling
     v                                          v
  img_feats                              lidar_bev_tokens
  (4 levels)                              (B, HW, 256)
     |                                          |
     +------------------------------------------+
     |                                          |
     v            ENCODER-SIDE FUSION           v
+-------------------------------------------------------------------+
|                   BEV Encoder (4 layers)                          |
|                                                                   |
|  For each layer:                                                  |
|    1. TSA: temporal self-attention (current BEV vs prev_bev)      |
|    2. Camera SCA: deformable cross-attn to multi-scale img_feats  |
|    3. LiDAR SCA: deformable cross-attn to lidar_bev_tokens       |
|    4. Blend: query = cam_out * w + lidar_out * (1-w)              |
|              (w = learnable softmax weight, init 0.5)             |
|    5. FFN                                                         |
+-------------------------------------------------------------------+
     |                              |
     v                              v
  bev_embed                    bev_embed_cam = bev_embed.clone()
  (encoder output,              (camera-only BEV snapshot,
   includes LiDAR)               preserves temporal motion signal)
     |                              |
     v    DECODER-SIDE FUSION       |
+-------------------------------+   |
|  LiDAR BEV re-projected      |   |
|  concat([bev_embed,          |   |
|          lidar_tokens])       |   |
|  --> Linear(2C, C) + LN      |   |
+-------------------------------+   |
     |                              |
     v                              v
+-------------------------------+   |
|  Transformer Decoder          |   |
|  (6 layers, 450 queries)      |   |
|  Self-attn + cross-attn       |   |
|  to fused BEV                 |   |
+-------------------------------+   |
     |                              |
     v                              v
  hs[lvl]                      bev_embed_cam
  (decoder queries)             (camera-only BEV)
     |                              |
     +----------+------------------+
     |          |                  |
     v          v                  v
  cls_branch  reg_branch      vel_cross_attn(hs, bev_embed_cam)
  (10 cls)    (bbox 10-dim)        |
                               vel_branch --> (vx, vy)
     |          |
     v          v
  yaw_bin    yaw_res
  (24 bins)  (sin, cos residual)
```

### Encoder-Side Fusion

Each of the 4 encoder layers performs dual spatial cross-attention (SCA), allowing the BEV representation to be progressively refined with both camera and LiDAR information at every layer.

| Component | Mechanism | Details |
|-----------|-----------|---------|
| Camera SCA | `MSDeformableAttention3D` | 4 sampling points, 4 feature levels |
| LiDAR SCA | `CustomMSDeformableAttention` | 4 sampling points, 1 level (BEV plane) |
| Blending | Learnable weighted sum | `cam * w + lidar * (1-w)`, softmax-normalized `nn.Parameter`, init 0.5 |

For detailed pipeline and design rationale, see [Encoder-side_Fusion.md](Encoder-side_Fusion.md).

### Decoder-Side Fusion

After the encoder, the BEV is further enriched through a single-shot concat-and-project operation before entering the decoder.

| Component | Mechanism | Details |
|-----------|-----------|---------|
| Projection | `Conv2d(64, 256, 1)` + LayerNorm | Xavier-initialized, alpha=5.0 scaling |
| Fusion | `Linear(512, 256)` + LayerNorm | Identity-initialized (camera passthrough at start) |
| Decoder | 6-layer transformer | Cross-attends to doubly-fused BEV |

For detailed pipeline and design rationale, see [Decoder-side_Fusion.md](Decoder-side_Fusion.md).

### Velocity Head (Camera-Only)

LiDAR features are single-frame snapshots with no temporal information. Blending them into BEV dilutes the motion signal from TemporalSelfAttention (TSA). The velocity head addresses this with a dedicated prediction pathway.

| Component | Mechanism | Details |
|-----------|-----------|---------|
| BEV snapshot | `bev_embed_cam = bev_embed.clone()` | Cloned **before** decoder-side fusion |
| Cross-attention | `nn.MultiheadAttention` per decoder layer | Queries attend to camera-only BEV |
| Prediction | MLP (256 &rarr; 256 &rarr; 2) | Outputs (vx, vy) per object query |
| Inference | Override `bbox_preds[..., 8:10]` | Replaces regression-based velocity |

For detailed analysis of the velocity problem and solution, see [Encoder-side_Fusion.md](Encoder-side_Fusion.md#velocity-problem-and-solution).

## FP16 Mixed Precision Training

Training uses `Fp16OptimizerHook` with dynamic loss scaling for approximately 2x speedup. The forward pass runs in FP16 (convolutions, attention, matmuls), while loss computation, optimizer state, and master weights remain in FP32.

```
Fp16OptimizerHook
  |
  +-- Forward pass: FP16 (convolutions, attention, matmuls)
  +-- Loss computation: FP32
  +-- Optimizer state: FP32 (master weights)
  +-- Gradient scaling: dynamic (auto loss_scale)
```

**Special handling for LiDAR**: The `hard_voxelize` CUDA kernel does not support FP16 inputs. Point clouds are explicitly cast to FP32 before voxelization in `extract_pts_bev_feat()`, while the rest of the LiDAR pipeline runs in FP16.

| Metric | FP32 | FP16 |
|--------|------|------|
| Speed | 1.73 s/iter | 0.85 s/iter |
| GPU memory | 8.9 GB | 8.0 GB |
| 200K iterations | ~4 days | ~2 days |

## Quick Start

```bash
conda activate bev
export PYTHONPATH=.
```

### Train (~2 days on RTX 5070 Ti 16GB)

```bash
python tools/train.py projects/configs/bevformer/bevformer_project.py
```

### Validate (with BEV visualization)

```bash
python tools/test.py \
    projects/configs/bevformer/bevformer_project.py \
    work_dirs/bevformer_project/iter_200000.pth \
    --eval bbox \
    --viz-bev --viz-num 20 --viz-score-thr 0.2 \
    --viz-outdir work_dirs/bevformer_project/bev_viz_val
```

### Test (save predictions + visualize)

```bash
python tools/test.py \
    projects/configs/bevformer/bevformer_project.py \
    work_dirs/bevformer_project/iter_200000.pth \
    --eval bbox \
    --out work_dirs/bevformer_project/results.pkl \
    --viz-bev --viz-num 30 --viz-score-thr 0.1 \
    --viz-outdir work_dirs/bevformer_project/bev_viz_test
```

### Monitor

```bash
tensorboard --logdir work_dirs/bevformer_project/tf_logs
tail -f work_dirs/bevformer_project/*.log
```

### Resume

```bash
python tools/train.py \
    projects/configs/bevformer/bevformer_project.py \
    --resume-from work_dirs/bevformer_project/iter_100000.pth
```

### Extract Results

```bash
bash tools/extract_results.sh work_dirs/bevformer_project
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Iterations | 200,000 |
| Precision | FP16 mixed (`Fp16OptimizerHook`, dynamic loss scale) |
| Optimizer | AdamW, lr=2e-4, weight_decay=0.01 |
| LR schedule | Cosine annealing, warmup=10K iters, min_lr_ratio=1e-3 |
| Backbone | ResNet50 (frozen stage 1, lr_mult=0.1) |
| BEV size | 100 x 100 |
| Embed dims | 256 |
| Encoder layers | 4 |
| Decoder layers | 6 |
| Object queries | 450 |
| Temporal queue | 4 frames |
| Fusion mode | `encoder_decoder` (both stages active) |
| Gradient clipping | max_norm=1.0 |

## Loss Functions

| Loss | Type | Weight | Description |
|------|------|--------|-------------|
| `loss_cls` | Focal | 2.0 | 10-class classification |
| `loss_bbox` | L1 | 0.25 | 3D bounding box regression (10-dim) |
| `loss_vel` | SmoothL1 | 0.25 | Velocity from camera-only BEV head |
| `loss_yaw_bin` | CrossEntropy | 0.2 | Yaw bin classification (24 bins) |
| `loss_yaw_res` | SmoothL1 | 0.2 | Yaw residual regression (sin, cos) |

## Files Modified from Base BEVFormer

| File | Changes |
|------|---------|
| `bevformer.py` | LiDAR branch (PointPillars), FP32 voxelization cast for FP16 compatibility |
| `transformer.py` | Encoder/decoder-side fusion modules, `bev_embed_cam` clone for velocity head |
| `encoder.py` | LiDAR SCA cross-attention, learnable blend weights (`cross_model_weights`) |
| `bevformer_head.py` | Velocity head (`vel_cross_attn` + MLP), `loss_vel`, yaw bin/res heads |
| `nms_free_coder.py` | Velocity override from velocity head, yaw bin/res decoding |
| `train.py` | Disabled `detect_anomaly` for FP16 compatibility |

## Dataset

nuScenes with temporal annotations. 10 classes: car, truck, bus, trailer, construction_vehicle, pedestrian, motorcycle, bicycle, traffic_cone, barrier.

## Requirements

- PyTorch 2.x with CUDA
- mmcv-full 1.7.x
- mmdet 2.28.x
- mmdet3d 1.0.0rc6
- nuscenes-devkit
