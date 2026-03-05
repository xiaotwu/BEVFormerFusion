# BEVFormerFusion

Multi-modal 3D object detection fusing camera and LiDAR inputs via BEVFormer with encoder-decoder side fusion.

## Architecture

```
Camera (6 views)                    LiDAR point cloud
     |                                     |
  ResNet50 + FPN                    PointPillars
     |                               (B,64,512,512)
  Multi-scale features                     |
     |                              Conv2d projection
     v                                     v
+----------------------------------------------------+
|              BEV Encoder (4 layers)                |
|  TSA (temporal) --> SCA (camera) + SCA (LiDAR)     |
|  Learnable blend: cam * w + lidar * (1-w)          |
+----------------------------------------------------+
     |                    |
     v                    v
  bev_embed          bev_embed_cam
  (fused BEV)        (camera-only, pre-fusion clone)
     |                    |
     v                    |
+---------------------------+
| Decoder-side fusion       |
| concat + linear + norm    |
+---------------------------+
     |                    |
     v                    v
+---------------------------+
| Transformer Decoder       |
| (6 layers, 450 queries)   |
+---------------------------+
     |                    |
     v                    v
  bbox/cls/yaw heads    vel_cross_attn --> vel_branch
  (from fused BEV)      (queries attend to camera-only BEV)
                         --> vx, vy predictions
```

### Key Design Choices

- **Encoder-side fusion**: LiDAR BEV tokens are blended with camera SCA output at each encoder layer via learnable weights
- **Decoder-side fusion**: Camera and LiDAR BEV are concatenated and linearly projected before the decoder
- **Velocity head**: Separate cross-attention + MLP that uses **camera-only BEV** (pre-fusion) to predict vx/vy, avoiding LiDAR dilution of temporal motion signals
- **Yaw prediction**: Discretized bin classification + residual regression (24 bins)

## Quick Start

```bash
conda activate bev
export PYTHONPATH=.
```

### Train (FP16, ~47 hours on RTX 5070 Ti)

```bash
python tools/train.py projects/configs/bevformer/bevformer_project.py
```

- 200K iterations, FP16 mixed precision
- ~0.85 s/iter, ~8 GB VRAM
- Checkpoints saved every 20K iters

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

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Iterations | 200,000 |
| Precision | FP16 (mixed) |
| Optimizer | AdamW, lr=2e-4, weight_decay=0.01 |
| LR schedule | Cosine annealing, warmup=10K iters |
| Backbone | ResNet50 (frozen stage 1, lr_mult=0.1) |
| BEV size | 100 x 100 |
| Embed dims | 256 |
| Encoder layers | 4 |
| Decoder layers | 6 |
| Queries | 450 |
| Temporal queue | 4 frames |
| Fusion mode | encoder_decoder |

## Losses

| Loss | Description | Weight |
|------|-------------|--------|
| `loss_cls` | Focal loss (classification) | 2.0 |
| `loss_bbox` | L1 loss (3D bbox regression) | 0.25 |
| `loss_vel` | SmoothL1 (velocity from camera-only BEV) | 0.25 |
| `loss_yaw_bin` | Cross-entropy (yaw bin classification) | 0.2 |
| `loss_yaw_res` | SmoothL1 (yaw residual regression) | 0.2 |

## Files Modified from Base BEVFormer

| File | Changes |
|------|---------|
| `bevformer.py` | LiDAR branch (PointPillars), FP32 voxelization cast |
| `transformer.py` | Encoder/decoder-side fusion, `bev_embed_cam` output |
| `encoder.py` | LiDAR SCA cross-attention, learnable blend weights |
| `bevformer_head.py` | Velocity head (cross-attn + MLP), `loss_vel`, yaw bin/res heads |
| `nms_free_coder.py` | Velocity override from vel_head, yaw bin/res decoding |

## Dataset

nuScenes with temporal annotations. 10 classes: car, truck, bus, trailer, construction_vehicle, pedestrian, motorcycle, bicycle, traffic_cone, barrier.

## Requirements

- PyTorch 2.x with CUDA
- mmcv-full 1.7.x
- mmdet 2.28.x
- mmdet3d 1.0.0rc6
- nuscenes-devkit
