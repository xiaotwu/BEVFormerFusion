# API Reference

This reference maps the documented method to the repository files that implement it. It is intentionally scoped to the active BEVFormerFusion path and excludes legacy or experimental files that are not treated as part of the public method.

## Code-to-documentation mapping

| Module | File path | Role in method | Description |
| --- | --- | --- | --- |
| Active config | `projects/configs/bevformer/bevformer_project.py` | Configuration root | Declares the published model, dataset wiring, loss configuration, and optimization schedule. |
| Detector | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` | Train and test orchestration | Builds image and LiDAR features, manages temporal caches, and dispatches to the BEVFormer head. |
| Transformer | `projects/mmdet3d_plugin/bevformer/modules/transformer.py` | BEV construction and decoder fusion | Computes BEV tokens, projects LiDAR BEV maps, snapshots `bev_embed_cam`, and applies decoder-side fusion. |
| Encoder sequence | `projects/mmdet3d_plugin/bevformer/modules/encoder.py` | Multi-modal BEV refinement | Implements `BEVFormerEncoder`, `BEVFormerLayer`, and `MM_BEVFormerLayer`, including the dual-attention fusion path. |
| Camera cross-attention | `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py` | Camera-to-BEV lifting | Projects BEV reference points into camera frames and performs image-side deformable attention. |
| Temporal attention | `projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py` | Temporal BEV update | Mixes current and previous BEV states through deformable attention over the BEV grid. |
| Decoder | `projects/mmdet3d_plugin/bevformer/modules/decoder.py` | Object-query decoding | Runs decoder self-attention and cross-attention over the fused BEV tensor. |
| Detection head | `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py` | Multi-branch prediction | Produces class scores, boxes, yaw bins, yaw residuals, and dedicated velocity predictions. |
| BBox coder | `projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py` | Final decoding | Converts decoder outputs into final boxes and injects velocity-head predictions into the output boxes. |
| Dataset | `projects/mmdet3d_plugin/datasets/nuscenes_dataset.py` | Temporal sample assembly | Builds temporal queues, enriches `img_metas`, and forwards geometry metadata required by the transformer. |

## Public command surface

| Entry point | Path | Purpose |
| --- | --- | --- |
| Training CLI | `tools/train.py` | Launches model training from a config file. |
| Evaluation CLI | `tools/test.py` | Runs evaluation and optional BEV visualization. |
| Benchmark CLI | `tools/analysis_tools/benchmark.py` | Measures inference throughput when the runtime stack is available. |
| Dataset conversion | `tools/create_data.py` | Creates nuScenes temporal info files and related annotations. |

## Documentation roles

| Documentation page | Primary code anchors |
| --- | --- |
| `architecture.md` | Detector, transformer, encoder, head, dataset |
| `bevformer_comparison.md` | Official BEVFormer config plus active config, transformer, encoder, head |
| `experiments.md` | Result workbooks, active config, canonical metrics JSON |
| `usage.md` | Active config, `tools/train.py`, `tools/test.py`, `tools/create_data.py`, `tools/analysis_tools/benchmark.py` |

## Legacy and excluded files

The following paths are intentionally excluded from the public method narrative:

| Path | Reason |
| --- | --- |
| `projects/configs/bevformerv2/` | Separate experiment family, not the active BEVFormerFusion path. |
| `projects/mmdet3d_plugin/bevformer/modules/transformer copy.py` | Stale duplicate of the active transformer implementation. |
| `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head_old.py` | Retired head variant. |
| `tools/tests/test_petr3d_bevformer.py` | Imports missing `petr3d_embedding.py` and therefore is not a valid regression test for the published method. |

## Interface notes

- No new public Python API is introduced by this documentation overhaul.
- The authoritative public outputs are the README, the MkDocs site rooted at `/`, and the canonical metrics artifact at `docs/assets/data/metrics.json`.
- The generated `site/` directory remains a build artifact and is not the editable source of truth.
