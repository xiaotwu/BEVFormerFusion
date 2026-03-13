# Chapter 9: Inference & Decoding

[00 Overview](00-overview.md) | [01 Data Pipeline](01-data-pipeline.md) | [02 Camera Branch](02-camera-branch.md) | [03 LiDAR Branch](03-lidar-branch.md) | [04 Encoder Fusion](04-encoder-fusion.md) | [05 Decoder Fusion](05-decoder-fusion.md) | [06 Decoder](06-transformer-decoder.md) | [07 Detection Heads](07-detection-heads.md) | [07a Velocity Head](07a-velocity-head.md) | [08 Loss & Training](08-loss-and-training.md) | **09 Inference** | [Appendix A: Tensors](appendix-tensor-shapes.md) | [Appendix B: Files](appendix-file-map.md)

---

## Overview

Inference uses NMS-free decoding: the top-K scoring queries are selected, their yaw and velocity slots are overwritten by the dedicated heads, bounding boxes are denormalized, and spatial filtering is applied. No Non-Maximum Suppression is needed because the DETR-style Hungarian matching already ensures one-to-one query-to-object assignment during training.

---

## Decoding Pipeline

```mermaid
graph TD
    subgraph "Score Selection"
        CLS["Class Scores<br/>(450 queries x 10 classes)"]
        CLS --> SIG["Sigmoid Activation"]
        SIG --> FLAT["Flatten to 4500 scores"]
        FLAT --> TOPK["Top-K Selection<br/>(K = 300)"]
        TOPK --> IDX["Extract:<br/>label = index % 10<br/>query_id = index // 10"]
    end

    subgraph "Head Overrides"
        IDX --> GATHER["Gather matched<br/>bbox, yaw, velocity predictions"]
        GATHER --> YAW_OVR["Yaw Override<br/>Bin argmax + residual atan2<br/>→ overwrite sin/cos slots"]
        YAW_OVR --> VEL_OVR["Velocity Override<br/>vel_preds → overwrite vx/vy slots"]
    end

    subgraph "Denormalization"
        VEL_OVR --> DENORM["Denormalize BBox:<br/>exp() for w, l, h<br/>atan2 for yaw<br/>cx, cy, cz, vx, vy direct"]
    end

    subgraph "Post-Processing"
        DENORM --> RANGE["Spatial Range Filter<br/>Keep boxes within<br/>[-61.2, -61.2, -10] to [61.2, 61.2, 10]"]
        RANGE --> SCORE["Score Threshold<br/>(default 0.25)"]
        SCORE --> SIZE["Optional Size Clamp<br/>[0.2, 8.0]"]
        SIZE --> OUT["Final Detections<br/>(bboxes, scores, labels)"]
    end

    style TOPK fill:#e8f4fd,stroke:#4a90d9
    style YAW_OVR fill:#fff3e0,stroke:#f5a623
    style VEL_OVR fill:#d4edda,stroke:#28a745
    style OUT fill:#d4edda,stroke:#28a745
```

---

## Yaw Decoding Detail

The yaw angle is reconstructed from the bin classification and residual regression:

```mermaid
graph LR
    BIN["Bin Logits<br/>(24 values)"] --> ARGMAX["argmax<br/>→ winning bin index"]
    ARGMAX --> CENTER["Bin Center =<br/>-pi + (idx + 0.5) * 15deg"]

    RES["Residual (sin, cos)"] --> NORM["Normalize to<br/>unit circle"]
    NORM --> ATAN["atan2(sin, cos)<br/>→ residual angle"]

    CENTER --> ADD["Final Yaw =<br/>center + residual"]
    ATAN --> ADD

    ADD --> WRITE["Write to bbox:<br/>slot 6 = sin(yaw)<br/>slot 7 = cos(yaw)"]

    style ADD fill:#d4edda,stroke:#28a745
```

## Velocity Override

Simple replacement: the velocity head's output directly overwrites the regression branch's velocity slots:

```
bbox_preds[..., 8:10] = vel_preds    # (vx, vy) from camera-only BEV head
```

---

## Test-Time Temporal Processing

During inference, the system maintains temporal context across frames within a scene:

```mermaid
graph TD
    subgraph "Frame-by-Frame Inference"
        NEW_FRAME["New Frame Arrives"]
        NEW_FRAME --> CHECK["Same scene<br/>as previous?"]

        CHECK -->|"No (new scene)"| RESET["Reset prev_bev = None<br/>Reset prev_pos, prev_angle"]
        CHECK -->|"Yes (same scene)"| EGO["Compute Ego Motion<br/>delta_pos = cur_pos - prev_pos<br/>delta_angle = cur_angle - prev_angle"]

        RESET --> FWD["Forward Pass"]
        EGO --> ROTATE["Rotate prev_bev<br/>by ego yaw delta"]
        ROTATE --> FWD

        FWD --> DET["Detections"]
        FWD --> SAVE["Save bev_embed<br/>as next prev_bev"]
        FWD --> UPDATE["Update prev_pos,<br/>prev_angle"]
    end

    style DET fill:#d4edda,stroke:#28a745
```

**Key details**:
- Scene token tracking detects scene boundaries (new sequence = reset temporal state)
- `prev_bev` is rotated to compensate for ego-vehicle yaw change using `torchvision.transforms.functional.rotate`
- CAN bus provides the relative ego-motion delta (translation and yaw)
- When `prev_bev` is None (first frame or new scene), TSA is skipped and the system operates in single-frame mode

---

## Inference Configuration

| Parameter | Value |
|-----------|-------|
| Score threshold | 0.25 |
| Max detections | 300 |
| Post-center range | [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0] |
| NMS | None (NMS-free) |
| Video test mode | True (carry prev_bev across frames) |

---

## Why NMS-Free?

The DETR-style decoder with Hungarian matching trains each query to specialize in detecting one object. Unlike anchor-based detectors that produce many overlapping proposals for each object, the one-to-one matching ensures queries learn non-overlapping responsibilities. At inference, simple top-K selection is sufficient.

---

## Key Files

| File | Path | Role |
|------|------|------|
| `nms_free_coder.py` | `core/bbox/coders/nms_free_coder.py` | `decode_single()`: top-K, yaw/vel override, denormalize |
| `util.py` | `core/bbox/util.py` | `denormalize_bbox()`, `overwrite_sincos_from_bins()` |
| `bevformer.py` | `bevformer/detectors/bevformer.py` | `forward_test()`, `simple_test()`: temporal state management |

---

[Next: Appendix A - Tensor Shape Reference](appendix-tensor-shapes.md)
