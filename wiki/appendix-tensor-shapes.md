# Appendix A: Tensor Shape Reference

[00 Overview](00-overview.md) | [01 Data Pipeline](01-data-pipeline.md) | [02 Camera Branch](02-camera-branch.md) | [03 LiDAR Branch](03-lidar-branch.md) | [04 Encoder Fusion](04-encoder-fusion.md) | [05 Decoder Fusion](05-decoder-fusion.md) | [06 Decoder](06-transformer-decoder.md) | [07 Detection Heads](07-detection-heads.md) | [08 Loss & Training](08-loss-and-training.md) | [09 Inference](09-inference.md) | **Appendix A: Tensors** | [Appendix B: Files](appendix-file-map.md)

---

Complete tensor shape reference through the full pipeline. Shapes assume `B=1, bev_h=bev_w=100, C=256, Ncam=6, num_query=450`.

---

## Input Stage

| Tensor | Shape | Description |
|--------|-------|-------------|
| `img` (train) | `(B, T=4, 6, 3, H, W)` | Stacked temporal multi-view images |
| `img` (test) | `(B, 6, 3, H, W)` | Single-frame multi-view images |
| `points` | `list[(N_i, 4)]` | Raw LiDAR point clouds per sample |
| `can_bus` | `(18,)` | Ego-motion vector per frame |

---

## Camera Branch

| Tensor | Shape | Description |
|--------|-------|-------------|
| ResNet stage outputs | `(B, 6, C_l, H_l, W_l)` | 4 levels: C = 256, 512, 1024, 2048 |
| FPN outputs (`mlvl_feats[l]`) | `(B, 6, 256, H_l, W_l)` | 4 levels, all 256 channels |
| `feat_flatten` | `(6, sum_HW, B, 256)` | Flattened + annotated with camera/level embeddings |
| `spatial_shapes` | `(4, 2)` | H, W per FPN level |
| `level_start_index` | `(4,)` | Cumulative start indices per level |

---

## LiDAR Branch

| Tensor | Shape | Description |
|--------|-------|-------------|
| `voxels` | `(total_pillars, 20, 4)` | Voxelized point clouds |
| `coors` | `(total_pillars, 4)` | Pillar coordinates [batch, z, y, x] |
| `num_points` | `(total_pillars,)` | Valid points per pillar |
| Pillar features | `(total_pillars, 64)` | After PillarFeatureNet |
| `bev_lidar` | `(B, 64, 512, 512)` | After PointPillarsScatter |

---

## Encoder

| Tensor | Shape | Description |
|--------|-------|-------------|
| `bev_queries` | `(B, 10000, 256)` | Learned BEV query embeddings |
| `bev_pos` | `(B, 10000, 256)` | BEV positional encoding |
| `prev_bev` | `(B, 10000, 256)` or `None` | Previous frame BEV (frozen) |
| `lidar_bev_tokens` (encoder) | `(B, 10000, 256)` | Projected LiDAR tokens for encoder-side fusion |
| `ref_3d` | `(B, 4, 10000, 3)` | 3D reference points (4 pillar heights) |
| `ref_2d` | `(B, 10000, 1, 2)` | 2D BEV grid reference points |
| `reference_points_cam` | `(6, B, 10000, 4, 2)` | Per-camera 2D projections |
| `bev_mask` | `(6, B, 10000, 4)` | Per-camera visibility mask |
| TSA `q_cat` | `(B, 10000, 512)` | Concatenated [prev, cur] for offset prediction |
| TSA `value` | `(B*2, 10000, 256)` | Stacked temporal frames |
| SCA `queries_rebatch` | `(B, 6, max_vis, 256)` | Sparse per-camera queries |
| Camera SCA output | `(B, 10000, 256)` | After scatter-add + average |
| LiDAR SCA output | `(B, 10000, 256)` | After deformable attention |
| Encoder output (`bev_embed`) | `(B, 10000, 256)` | Final BEV embedding |

---

## Decoder-Side Fusion

| Tensor | Shape | Description |
|--------|-------|-------------|
| `bev_embed_cam` | `(B, 10000, 256)` | Cloned before LiDAR fusion (for velocity head) |
| `bev_lidar_tok` (decoder) | `(B, 10000, 256)` | Projected + normalized LiDAR tokens |
| Concatenated | `(B, 10000, 512)` | [camera BEV, LiDAR tokens] |
| Fused `bev_embed` | `(B, 10000, 256)` | After Linear(512, 256) + LayerNorm |

---

## Decoder

| Tensor | Shape | Description |
|--------|-------|-------------|
| Object queries | `(B, 450, 256)` | Learned query embeddings |
| Query positional | `(B, 450, 256)` | Positional part of query_embed |
| Initial reference points | `(B, 450, 3)` | Predicted via Linear + sigmoid |
| Decoder memory (value) | `(10000, B, 256)` | Fused BEV (permuted) |
| `inter_states` | `(6, B, 450, 256)` | Query features at each layer |
| `inter_references` | `(6, B, 450, 3)` | Refined reference points at each layer |

---

## Detection Heads

| Tensor | Shape | Description |
|--------|-------|-------------|
| `all_cls_scores` | `(6, B, 450, 10)` | Classification logits per layer |
| `all_bbox_preds` | `(6, B, 450, 10)` | 10-dim bbox code per layer |
| `all_yaw_bin_logits` | `(6, B, 450, 24)` | Yaw bin logits per layer |
| `all_yaw_res_preds` | `(6, B, 450, 2)` | Yaw (sin, cos) residual per layer |
| `all_vel_preds` | `(6, B, 450, 2)` | Velocity (vx, vy) per layer |
| Velocity context | `(B, 450, 256)` | After cross-attention to camera-only BEV |

---

## Inference Output

| Tensor | Shape | Description |
|--------|-------|-------------|
| Selected scores | `(K,)` | Top-K sigmoid scores (K <= 300) |
| Selected labels | `(K,)` | Class labels |
| Selected bboxes | `(K, 9)` | Denormalized: cx, cy, cz, w, l, h, yaw, vx, vy |

---

[Next: Appendix B - File Map & Class Hierarchy](appendix-file-map.md)
