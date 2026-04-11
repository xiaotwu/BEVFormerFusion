# BEVFormerFusion

BEVFormerFusion is a BEVFormer-derived multi-modal 3D detector for nuScenes. The published implementation keeps the upstream camera-to-BEV pipeline and adds a PointPillars LiDAR branch, encoder-side LiDAR cross-attention, decoder-side BEV fusion, and a dedicated velocity head. All metrics reported below are traced to the checked-in experiment workbooks or to repository code; no estimated runtime values are published.

## Introduction

The active method scope is defined by `projects/configs/bevformer/bevformer_project.py`, `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py`, `projects/mmdet3d_plugin/bevformer/modules/transformer.py`, `projects/mmdet3d_plugin/bevformer/modules/encoder.py`, `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py`, and `projects/mmdet3d_plugin/datasets/nuscenes_dataset.py`. The official BEVFormer repository is used as the baseline reference for all architectural comparisons in this repository.

Project documentation:

- GitHub Pages: <https://xiaotwu.github.io/BEVFormerFusion/>
- Architecture: [docs/architecture.md](docs/architecture.md)
- BEVFormer comparison: [docs/bevformer-comparison.md](docs/bevformer-comparison.md)
- Experiments: [docs/experiments.md](docs/experiments.md)
- Usage: [docs/usage.md](docs/usage.md)
- API reference: [docs/api-reference.md](docs/api-reference.md)

## Key Contributions

- Adds a PointPillars LiDAR BEV branch alongside the BEVFormer camera path.
- Injects LiDAR evidence inside the encoder through a second deformable cross-attention path blended with camera attention.
- Fuses LiDAR BEV features again before the decoder so detection queries operate on a joint representation.
- Preserves camera-only temporal cues for velocity estimation through a dedicated motion head.

## Method Overview

The image branch follows the BEVFormer backbone and transformer scaffold: multi-view image features are encoded into BEV queries with temporal self-attention and deformable camera cross-attention. BEVFormerFusion augments this path with projected LiDAR BEV features from PointPillars. Each encoder layer applies parallel camera and LiDAR cross-attention, then blends the two responses before the feed-forward block. After the encoder, the LiDAR BEV tensor is concatenated with the camera-derived BEV embedding and projected before the decoder. The decoder retains the BEVFormer query refinement path, while the detection head adds explicit yaw-bin/residual supervision and a velocity head that attends to the camera-only BEV branch.

## Installation

The repository does not currently ship a lockfile. Install the BEVFormer/MMDetection3D stack in a compatible CUDA environment, then export the repository root on `PYTHONPATH`.

Validated dependency targets from the repository configuration:

- Python 3.9
- PyTorch 2.x with CUDA
- `mmcv-full` 1.7.x
- `mmdet` 2.28.x
- `mmdet3d` 1.0.0rc6
- `nuscenes-devkit`

Environment setup:

```bash
export PYTHONPATH=.
```

## Quick Start

Train:

```bash
python3 tools/train.py projects/configs/bevformer/bevformer_project.py
```

Evaluate:

```bash
python3 tools/test.py \
  projects/configs/bevformer/bevformer_project.py \
  work_dirs/bevformer_project/iter_100000.pth \
  --eval bbox
```

Visualize BEV predictions:

```bash
python3 tools/test.py \
  projects/configs/bevformer/bevformer_project.py \
  work_dirs/bevformer_project/iter_100000.pth \
  --eval bbox \
  --viz-bev --viz-num 20 --viz-score-thr 0.2 \
  --viz-outdir work_dirs/bevformer_project/bev_viz
```

Benchmark throughput when the runtime stack is available:

```bash
python3 tools/analysis_tools/benchmark.py \
  projects/configs/bevformer/bevformer_project.py \
  --checkpoint work_dirs/bevformer_project/iter_100000.pth \
  --samples 200
```

## Dataset

The repository targets nuScenes with temporal metadata. The data conversion path is implemented in `tools/create_data.py` and writes temporal info files such as `nuscenes_infos_temporal_train.pkl` and `nuscenes_infos_temporal_val.pkl`.

Example conversion command:

```bash
python3 tools/create_data.py nuscenes \
  --root-path ./data/nuscenes \
  --canbus ./data \
  --version v1.0 \
  --max-sweeps 10
```

The published model predicts the standard nuScenes 10-class set: car, truck, construction_vehicle, bus, trailer, barrier, motorcycle, bicycle, pedestrian, and traffic_cone.

## Results

The baseline metrics were normalized from `results/Baseline_Results_Summary.xlsx`. The fused-model metrics were normalized from `results/Enc_Dec_Results_Summary.xlsx`; `results/EncoderFusion_Results_SHS.xlsx` was byte-identical and is treated as a duplicate artifact rather than a second run.

| Model | Best Checkpoint | mAP | NDS | FPS | Memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| BEVFormer baseline | 100000 | 0.2011 | 0.2192 | Pending | Pending |
| BEVFormerFusion | 100000 | 0.2507 | 0.2546 | Pending | Pending |

At the best recorded checkpoint, the fused model improves mAP by `+0.0496` and NDS by `+0.0354` over the baseline workbook curve. A fresh 12GB GPU runtime profile is still pending because no validated profiling artifact is currently tracked in this repository.

## Documentation

The canonical documentation source now lives in `docs/` and is published through MkDocs/Material:

- Landing page: [docs/index.md](docs/index.md)
- Architecture summary: [docs/architecture.md](docs/architecture.md)
- BEVFormer comparison: [docs/bevformer-comparison.md](docs/bevformer-comparison.md)
- Experiment tables and provenance: [docs/experiments.md](docs/experiments.md)
- Command reference: [docs/usage.md](docs/usage.md)
- Module map: [docs/api-reference.md](docs/api-reference.md)
