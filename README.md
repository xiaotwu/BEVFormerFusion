# BEVFormerFusion

Multi-modal 3D object detection fusing camera and LiDAR inputs via BEVFormer with dual encoder-decoder side fusion.

## Architecture

BEVFormerFusion extends BEVFormer with three innovations:

- **Encoder-Side Fusion** -- LiDAR BEV features are injected into every encoder layer via parallel deformable cross-attention, blended with camera cross-attention through a learnable weight
- **Decoder-Side Fusion** -- After the encoder, LiDAR features are concatenated and projected with the BEV embedding before entering the decoder
- **Velocity Head** -- A dedicated cross-attention head that attends to camera-only BEV features (before LiDAR fusion) to predict object velocity, solving the temporal signal dilution problem

```mermaid
graph TD
    %% ── Inputs ──
    IMG["Multi-View Images<br/>(6 cameras)"]
    PTS["Point Cloud"]

    %% ── Backbones ──
    RES["ResNet-50 + FPN"]
    PP["PointPillars"]

    IMG --> RES
    PTS --> PP

    RES --> |"Multi-Scale Features"| CAM_SCA
    PP  --> |"LiDAR BEV Features"| LID_PROJ["1x1 Conv Projection"]

    %% ── Encoder (x4 layers) ──
    subgraph ENC["Encoder Layer  x 4"]
        direction TB
        TSA["Temporal Self-Attention<br/>with History BEV"]
        CAM_SCA["Camera Spatial<br/>Cross-Attention"]
        LID_SCA["LiDAR Deformable<br/>Cross-Attention"]
        BLEND["Learnable Blend<br/>w * Camera + (1-w) * LiDAR"]
        FFN_E["Feed-Forward Network"]

        TSA --> CAM_SCA
        TSA --> LID_SCA
        CAM_SCA --> BLEND
        LID_SCA --> BLEND
        BLEND --> FFN_E
    end

    LID_PROJ --> |"(a) Encoder Fusion"| LID_SCA
    PREV["History BEV<br/>B(t-1)"] -.-> TSA

    FFN_E --> BEV_CAM["Camera-Only BEV"]
    FFN_E --> CONCAT

    %% ── Decoder Fusion ──
    LID_PROJ --> |"(b) Decoder Fusion"| CONCAT["Concat + Linear<br/>Projection"]
    CONCAT --> BEV_FUSED["Fused BEV"]

    %% ── Decoder (x6 layers) ──
    subgraph DEC["Decoder Layer  x 6"]
        direction TB
        SELF_A["Query Self-Attention"]
        CROSS_A["Deformable Cross-Attention<br/>to Fused BEV"]
        FFN_D["Feed-Forward Network"]
        REFINE["Reference Point Refinement"]
        SELF_A --> CROSS_A --> FFN_D --> REFINE
    end

    BEV_FUSED --> CROSS_A
    QUERIES["Object Queries<br/>(450)"] --> SELF_A

    %% ── Heads ──
    REFINE --> CLS["Classification<br/>Head"]
    REFINE --> BOX["BBox Regression<br/>Head"]
    REFINE --> YAW["Yaw Bin/Residual<br/>Head"]
    REFINE --> VEL_XA["Velocity Cross-Attention<br/>→ Camera-Only BEV"]
    BEV_CAM -.-> |"temporal signal<br/>preserved"| VEL_XA
    VEL_XA --> VEL["Velocity Head<br/>(vx, vy)"]

    %% ── Styles ──
    classDef input fill:#e8f4fd,stroke:#4a90d9
    classDef cam fill:#d4edda,stroke:#28a745
    classDef lid fill:#fff3cd,stroke:#ffc107
    classDef fuse fill:#f8d7da,stroke:#dc3545
    classDef head fill:#e2d9f3,stroke:#6f42c1
    classDef dec fill:#d1ecf1,stroke:#17a2b8

    class IMG,PTS,PREV,QUERIES input
    class RES,CAM_SCA,TSA,BEV_CAM cam
    class PP,LID_PROJ,LID_SCA lid
    class BLEND,CONCAT,BEV_FUSED,VEL_XA fuse
    class CLS,BOX,YAW,VEL head
    class SELF_A,CROSS_A,FFN_D,REFINE dec
```

## Documentation

Comprehensive technical documentation is organized as a book in the [`doc/`](doc/) folder:

| Chapter | Title | Content |
|---------|-------|---------|
| [00](doc/00-overview.md) | System Overview | Architecture diagrams, design philosophy, chapter guide |
| [01](doc/01-data-pipeline.md) | Data Pipeline | nuScenes dataset, temporal queue, CAN bus, transforms |
| [02](doc/02-camera-branch.md) | Camera Branch | ResNet50 + FPN feature extraction |
| [03](doc/03-lidar-branch.md) | LiDAR Branch | PointPillars: voxelization, pillar features, BEV scatter |
| [04](doc/04-encoder-fusion.md) | Encoder-Side Fusion | TSA, dual SCA, learnable blend weights |
| [05](doc/05-decoder-fusion.md) | Decoder-Side Fusion | Concat+linear fusion, identity initialization |
| [06](doc/06-transformer-decoder.md) | Transformer Decoder | 6-layer decoder, reference point refinement |
| [07](doc/07-detection-heads.md) | Detection Heads | Classification, bbox, yaw bin/res, velocity head |
| [08](doc/08-loss-and-training.md) | Loss & Training | 5 loss functions, gradient isolation, training config |
| [09](doc/09-inference.md) | Inference & Decoding | NMS-free decoding, temporal test-time processing |
| [A](doc/appendix-tensor-shapes.md) | Tensor Shapes | Complete tensor shape reference |
| [B](doc/appendix-file-map.md) | File Map | Key files, class hierarchy |

## Quick Start

```bash
conda activate bev
export PYTHONPATH=.
```

### Train

```bash
python tools/train.py projects/configs/bevformer/bevformer_project.py
```

### Evaluate

```bash
python tools/test.py \
    projects/configs/bevformer/bevformer_project.py \
    work_dirs/bevformer_project/iter_200000.pth \
    --eval bbox
```

### Visualize BEV

```bash
python tools/test.py \
    projects/configs/bevformer/bevformer_project.py \
    work_dirs/bevformer_project/iter_200000.pth \
    --eval bbox \
    --viz-bev --viz-num 20 --viz-score-thr 0.2 \
    --viz-outdir work_dirs/bevformer_project/bev_viz
```

### Resume Training

```bash
python tools/train.py \
    projects/configs/bevformer/bevformer_project.py \
    --resume-from work_dirs/bevformer_project/iter_100000.pth
```

### Monitor

```bash
tensorboard --logdir work_dirs/bevformer_project/tf_logs
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Iterations | 200,000 |
| Precision | FP32 |
| Optimizer | AdamW, lr=2e-4, weight_decay=0.01 |
| LR schedule | Cosine annealing, warmup=10K iters |
| Backbone | ResNet50 (frozen stage 1, lr_mult=0.1) |
| BEV size | 100 x 100 |
| Encoder / Decoder layers | 4 / 6 |
| Object queries | 450 |
| Fusion mode | `encoder_decoder` |

See [Chapter 8](doc/08-loss-and-training.md) for full training details.

## Dataset

nuScenes with temporal annotations. 10 classes: car, truck, bus, trailer, construction_vehicle, pedestrian, motorcycle, bicycle, traffic_cone, barrier.

## Requirements

- PyTorch 2.x with CUDA
- mmcv-full 1.7.x
- mmdet 2.28.x
- mmdet3d 1.0.0rc6
- nuscenes-devkit
