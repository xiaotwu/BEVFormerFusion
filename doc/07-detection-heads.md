# Chapter 7: Detection Heads

[00 Overview](00-overview.md) | [01 Data Pipeline](01-data-pipeline.md) | [02 Camera Branch](02-camera-branch.md) | [03 LiDAR Branch](03-lidar-branch.md) | [04 Encoder Fusion](04-encoder-fusion.md) | [05 Decoder Fusion](05-decoder-fusion.md) | [06 Decoder](06-transformer-decoder.md) | **07 Detection Heads** | [08 Loss & Training](08-loss-and-training.md) | [09 Inference](09-inference.md) | [Appendix A: Tensors](appendix-tensor-shapes.md) | [Appendix B: Files](appendix-file-map.md)

---

## Overview

Each decoder layer produces a set of predictions through 5 parallel heads. Three of these are novel additions to BEVFormer: the yaw bin/residual heads and the velocity head with camera-only cross-attention.

---

## Head Architecture

```mermaid
graph TD
    DEC["Decoder Output<br/>hs[layer]"]

    DEC --> CLS["Classification Head<br/>MLP → 10 classes"]
    DEC --> REG["Regression Head<br/>MLP → 10-dim bbox"]
    DEC --> YAW_B["Yaw Bin Head<br/>MLP → 24 bins"]
    DEC --> YAW_R["Yaw Residual Head<br/>MLP → (sin, cos)"]

    DEC --> VEL_CA["Velocity Cross-Attention"]
    CAM_BEV["Camera-Only BEV<br/>(no LiDAR fusion)"] --> VEL_CA
    VEL_CA --> VEL["Velocity Head<br/>MLP → (vx, vy)"]

    CLS --> OUT["Final Predictions"]
    REG --> OUT
    YAW_B --> OUT
    YAW_R --> OUT
    VEL --> OUT

    classDef standard fill:#e8f4fd,stroke:#4a90d9
    classDef novel fill:#fff3e0,stroke:#f5a623
    classDef velocity fill:#d4edda,stroke:#28a745

    class CLS,REG standard
    class YAW_B,YAW_R novel
    class VEL_CA,VEL,CAM_BEV velocity
```

### Head Details

| Head | Architecture | Output | Notes |
|------|-------------|--------|-------|
| Classification | Linear+LN+ReLU (x N) -> Linear(256, 10) | Class scores | Focal loss supervision |
| Regression | Linear+ReLU (x N) -> Linear(256, 10) | 10-dim bbox code | Yaw & velocity indices zeroed in loss |
| Yaw Bin | Linear(256,256)+ReLU -> Linear(256, 24) | 24 bin logits | CrossEntropy loss |
| Yaw Residual | Linear(256,256)+ReLU -> Linear(256, 2) | (sin_res, cos_res) | SmoothL1 loss |
| Velocity | CrossAttn + Linear(256,256)+ReLU -> Linear(256, 2) | (vx, vy) | Camera-only BEV input |

All heads are cloned per decoder layer (6 independent copies each), enabling auxiliary supervision.

---

## BBox Code Layout

The regression head outputs a 10-dimensional vector with this fixed layout:

| Index | Symbol | Encoding | Supervised By |
|-------|--------|----------|---------------|
| 0 | cx | sigmoid + pc_range scaling | `reg_branches` |
| 1 | cy | sigmoid + pc_range scaling | `reg_branches` |
| 2 | log(w) | log-space width | `reg_branches` |
| 3 | log(l) | log-space length | `reg_branches` |
| 4 | cz | sigmoid + pc_range scaling | `reg_branches` |
| 5 | log(h) | log-space height | `reg_branches` |
| 6 | sin(yaw) | unit-circle normalized | `yaw_bin/res_branches` only |
| 7 | cos(yaw) | unit-circle normalized | `yaw_bin/res_branches` only |
| 8 | vx | direct | `vel_branches` only |
| 9 | vy | direct | `vel_branches` only |

Indices 6-9 are **overwritten at inference** by the dedicated yaw and velocity heads. During training, their loss weights in `bbox_weights` are set to zero to prevent conflicting gradients.

---

## Yaw Bin/Residual Head

### The Problem with Direct Yaw Regression

Yaw angles have a wrapping discontinuity at +/-pi. A small angular change near the boundary causes a huge regression target jump, destabilizing gradients.

### The Bin + Residual Solution

```mermaid
graph TD
    subgraph "Encoding (Training)"
        YAW["Ground Truth Yaw"] --> BIN_IDX["Bin Index<br/>floor((yaw + pi) / bin_size)<br/>24 bins, 15 deg each"]
        YAW --> CENTER["Bin Center<br/>(bin_idx + 0.5) * bin_size - pi"]
        YAW --> RES["Residual = yaw - center<br/>Small angle within +/-7.5 deg"]
        RES --> SINCOS["Encode as (sin, cos)<br/>Avoids wrapping issues"]
    end

    subgraph "Decoding (Inference)"
        PRED_BIN["Predicted Bin Logits"] --> ARGMAX["argmax → winning bin"]
        ARGMAX --> PRED_CENTER["Bin Center"]
        PRED_RES["Predicted (sin, cos)"] --> ATAN["atan2(sin, cos) → residual angle"]
        PRED_CENTER --> FINAL["Final Yaw = center + residual"]
        ATAN --> FINAL
    end

    style SINCOS fill:#d4edda,stroke:#28a745
    style FINAL fill:#d4edda,stroke:#28a745
```

**Why this works**: The bin classification handles coarse orientation (no wrapping issues with cross-entropy), while the residual regression handles fine-grained correction within a narrow +/-7.5 degree range using sin/cos encoding.

### Sin/Cos Normalization in Regression Head

The regression branch also outputs sin/cos at indices 6-7. These are stabilized during training:

```python
sc = reg_output[..., 6:8].tanh()                        # clamp to (-1, 1)
sc = sc / (sc.norm(dim=-1, keepdim=True) + 1e-7)        # project onto unit circle
```

This prevents the sin/cos pair from drifting off the unit circle during early training.

---

## Velocity Head

The velocity head is the key innovation for maintaining accurate motion prediction despite LiDAR fusion.

### The Temporal Dilution Problem

```mermaid
graph LR
    subgraph "Temporal Self-Attention"
        PREV["Previous BEV<br/>(frame t-1)"] --> TSA["TSA compares<br/>current vs previous"]
        CUR["Current BEV<br/>(frame t)"] --> TSA
        TSA --> MOTION["Motion Signal<br/>encoded in BEV"]
    end

    subgraph "The Problem"
        MOTION --> BLEND["Blended with<br/>static LiDAR features"]
        LIDAR["LiDAR BEV<br/>(single frame,<br/>no temporal info)"] --> BLEND
        BLEND --> DILUTED["Diluted<br/>Motion Signal"]
    end

    style MOTION fill:#d4edda,stroke:#28a745
    style DILUTED fill:#f8d7da,stroke:#dc3545
    style LIDAR fill:#fff3e0,stroke:#f5a623
```

### The Solution: Camera-Only Cross-Attention

```mermaid
graph TD
    subgraph "Velocity Head"
        DQ["Decoder Query<br/>(object representation)"]
        CAM["Camera-Only BEV<br/>(cloned BEFORE<br/>LiDAR fusion)"]

        DQ -->|"Q"| MHA["Multi-Head Attention<br/>8 heads"]
        CAM -->|"K, V"| MHA

        MHA --> CTX["Velocity Context<br/>(motion-aware features)"]
        CTX --> MLP["MLP: 256 → 256 → 2"]
        MLP --> VEL["Velocity (vx, vy)"]
    end

    subgraph "Why Camera-Only BEV?"
        P1["Preserves full temporal signal from TSA"]
        P2["No dilution from static LiDAR geometry"]
        P3["Each query selectively attends to<br/>motion-relevant BEV regions"]
    end

    style CAM fill:#d4edda,stroke:#28a745
    style VEL fill:#d4edda,stroke:#28a745
```

**Key design choices**:

1. **Camera-only BEV**: `bev_embed_cam` is cloned before decoder-side LiDAR fusion, preserving the full temporal motion signal from TSA
2. **Full attention** (not deformable): `nn.MultiheadAttention` allows each query to attend over the entire BEV grid, important for capturing motion patterns that may span large regions
3. **Per-layer**: One cross-attention + MLP per decoder layer, receiving the progressively refined query

---

## Gradient Isolation

A critical design pattern ensures each specialized head receives exclusive supervision:

```mermaid
graph TD
    subgraph "Regression Head Output (10-dim)"
        R0["cx"] --- R1["cy"]
        R1 --- R2["log_w"]
        R2 --- R3["log_l"]
        R3 --- R4["cz"]
        R4 --- R5["log_h"]
        R5 --- R6["sin_yaw"]
        R6 --- R7["cos_yaw"]
        R7 --- R8["vx"]
        R8 --- R9["vy"]
    end

    R0 --> L1["loss_bbox<br/>(L1 loss)"]
    R1 --> L1
    R2 --> L1
    R3 --> L1
    R4 --> L1
    R5 --> L1

    R6 -.->|"weight = 0"| L1
    R7 -.->|"weight = 0"| L1
    R8 -.->|"weight = 0"| L1
    R9 -.->|"weight = 0"| L1

    YB["Yaw Bin Head"] --> L2["loss_yaw_bin<br/>(CrossEntropy)"]
    YR["Yaw Res Head"] --> L3["loss_yaw_res<br/>(SmoothL1)"]
    VH["Velocity Head"] --> L4["loss_vel<br/>(SmoothL1)"]

    classDef active fill:#d4edda,stroke:#28a745
    classDef zeroed fill:#f8d7da,stroke:#dc3545

    class R0,R1,R2,R3,R4,R5 active
    class R6,R7,R8,R9 zeroed
```

By zeroing `bbox_weights` at indices 6-9, the regression head receives no gradient for yaw or velocity. All yaw learning flows through the bin + residual heads, and all velocity learning flows through the camera-only cross-attention head.

---

## Key Files

| File | Path | Role |
|------|------|------|
| `bevformer_head.py` | `bevformer/dense_heads/bevformer_head.py` | All prediction heads, velocity cross-attention, loss computation |
| `util.py` | `core/bbox/util.py` | `normalize_bbox`, `denormalize_bbox`, yaw bin encode/decode |

---

[Next: Chapter 8 - Loss Functions & Training](08-loss-and-training.md)
