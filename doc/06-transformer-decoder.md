# Chapter 6: Transformer Decoder

[00 Overview](00-overview.md) | [01 Data Pipeline](01-data-pipeline.md) | [02 Camera Branch](02-camera-branch.md) | [03 LiDAR Branch](03-lidar-branch.md) | [04 Encoder Fusion](04-encoder-fusion.md) | [05 Decoder Fusion](05-decoder-fusion.md) | **06 Decoder** | [07 Detection Heads](07-detection-heads.md) | [08 Loss & Training](08-loss-and-training.md) | [09 Inference](09-inference.md) | [Appendix A: Tensors](appendix-tensor-shapes.md) | [Appendix B: Files](appendix-file-map.md)

---

## Overview

The transformer decoder transforms 450 learned object queries into structured 3D detections. Each query represents a candidate object and progressively refines its understanding by cross-attending to the fused BEV representation. The decoder uses 6 iterative layers with reference point refinement -- each layer's output improves the spatial precision of the next.

---

## Decoder Architecture

```mermaid
graph TD
    subgraph Inputs
        OQ["450 Learned Object Queries"]
        PE["Positional Embeddings"]
        BEV["Fused BEV from Decoder-Side Fusion"]
    end

    OQ --> SPLIT["Split query_embed into<br/>query + query_pos"]
    PE --> SPLIT
    SPLIT --> REF["Predict Initial Reference Points<br/>Linear + sigmoid → (x, y, z)"]

    REF --> L1

    subgraph Layer["Decoder Layer (x6)"]
        L1["Self-Attention<br/>Queries interact with each other"]
        L1 --> N1["LayerNorm"]
        N1 --> L2["Cross-Attention to BEV<br/>Deformable attention at<br/>learned sampling offsets"]
        L2 --> N2["LayerNorm"]
        N2 --> L3["Feed-Forward Network"]
        L3 --> N3["LayerNorm"]
        N3 --> REFINE["Reference Point Refinement<br/>Update (x, y, z) using<br/>regression branch output"]
    end

    BEV --> L2

    REFINE --> OUT["Collect intermediate outputs<br/>for auxiliary losses"]

    style Inputs fill:#e8f4fd,stroke:#4a90d9
    style Layer fill:#fff3e0,stroke:#f5a623
```

---

## How One Decoder Layer Works

### 1. Self-Attention

The 450 object queries attend to each other via standard multi-head attention (8 heads, dropout 0.1). This allows queries to reason about relationships between candidate objects -- for example, two nearby queries can coordinate to avoid predicting the same object.

### 2. Cross-Attention to Fused BEV

Each query attends to the BEV feature map using `CustomMSDeformableAttention`:

```mermaid
graph LR
    Q["Object Query"] --> OFFSET["Predict Sampling<br/>Offsets"]
    Q --> WEIGHT["Predict Attention<br/>Weights"]
    RP["Reference Point<br/>(x, y) on BEV"] --> SAMPLE["Sampling Locations =<br/>Reference + Offsets"]
    OFFSET --> SAMPLE
    SAMPLE --> DA["Sample BEV Features<br/>at 4 Deformable Points"]
    WEIGHT --> DA
    DA --> OUT["Weighted Sum →<br/>Updated Query"]

    style Q fill:#e8f4fd,stroke:#4a90d9
    style RP fill:#fff3e0,stroke:#f5a623
```

Key properties:
- **1 spatial level**: the BEV grid (100 x 100)
- **4 deformable sampling points** per query per head
- **8 attention heads**
- Reference points are in normalized BEV coordinates (0 to 1)

### 3. Feed-Forward Network

Standard two-layer MLP: Linear(256, 512) -> ReLU -> Linear(512, 256) with residual connection.

### 4. Reference Point Refinement

After each layer, the regression branch predicts coordinate offsets that refine the reference points:

```mermaid
graph TD
    OLD["Current Reference Point<br/>(x, y, z) in sigmoid space"]
    REG["Regression Branch Output<br/>(10-dim bbox code)"]

    OLD --> INV["inverse_sigmoid(x, y, z)"]
    REG --> EXTRACT["Extract offsets:<br/>dx, dy from indices 0:2<br/>dz from index 4"]

    INV --> ADD["Add: logit + offset"]
    EXTRACT --> ADD

    ADD --> SIG["sigmoid → new (x, y, z)"]
    SIG --> DET["detach<br/>(stop gradient)"]
    DET --> NEXT["Refined Reference Point<br/>for next layer"]

    style OLD fill:#e8f4fd,stroke:#4a90d9
    style NEXT fill:#d4edda,stroke:#28a745
```

The `detach()` is critical -- gradients do not flow backward through refined references. Each layer learns to predict accurate outputs given its own reference points, rather than learning to produce offsets that are useful for the next layer.

> **Note**: The z-offset is read from index 4 of the regression output (the `cz` slot), not index 2. This matches the bbox code layout: `[cx, cy, log_w, log_l, cz, log_h, ...]`.

---

## Iterative Refinement Across 6 Layers

```mermaid
graph LR
    subgraph "Layer 0"
        Q0["Query"] --> D0["Decode"]
        R0["Initial Ref"] --> D0
        D0 --> Q1["Updated Query"]
        D0 --> R1["Refined Ref"]
    end

    subgraph "Layer 1"
        Q1 --> D1["Decode"]
        R1 --> D1
        D1 --> Q2["Updated Query"]
        D1 --> R2["Refined Ref"]
    end

    subgraph "..."
        Q2 --> D2["..."]
        R2 --> D2
    end

    subgraph "Layer 5"
        D2 --> Q5["Final Query"]
        D2 --> R5["Final Ref"]
    end

    Q0 -.-> H0["Heads → Predictions"]
    Q1 -.-> H1["Heads → Predictions"]
    Q5 -.-> H5["Heads → Predictions<br/>(primary output)"]

    style H5 fill:#d4edda,stroke:#28a745
```

All 6 layers produce predictions (via the detection heads in Chapter 7). The last layer provides the primary output; earlier layers provide auxiliary supervision that stabilizes training.

---

## Output

The decoder returns two tensors:

| Output | Shape | Description |
|--------|-------|-------------|
| `inter_states` | (6, B, 450, 256) | Decoder query features at each layer |
| `inter_references` | (6, B, 450, 3) | Refined reference points at each layer (sigmoid space) |

Both are used by the detection heads: `inter_states` feeds all prediction branches, and `inter_references` provides the spatial anchor for bbox coordinate decoding.

---

## Key Files

| File | Path | Role |
|------|------|------|
| `decoder.py` | `bevformer/modules/decoder.py` | `DetectionTransformerDecoder` and `CustomMSDeformableAttention` |
| `transformer.py` | `bevformer/modules/transformer.py` | Decoder invocation within `PerceptionTransformer.forward` |

---

[Next: Chapter 7 - Detection Heads](07-detection-heads.md)
