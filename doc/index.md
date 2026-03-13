# BEVFormerFusion

Multi-modal 3D object detection fusing camera and LiDAR inputs via BEVFormer with dual encoder-decoder side fusion.

## Three Key Innovations

- **Encoder-Side Fusion** -- LiDAR BEV features are injected into every encoder layer via parallel deformable cross-attention, blended with camera cross-attention through a learnable weight
- **Decoder-Side Fusion** -- After the encoder, LiDAR features are concatenated and projected with the BEV embedding before entering the decoder
- **Velocity Head** -- A dedicated cross-attention head that attends to camera-only BEV features (before LiDAR fusion) to predict object velocity

## Chapters

| # | Chapter | Topic |
|---|---------|-------|
| 0 | [Architecture Overview](00-overview.md) | End-to-end architecture, design philosophy |
| 1 | [Data Pipeline](01-data-pipeline.md) | nuScenes dataset, temporal queue, CAN bus |
| 2 | [Camera Branch](02-camera-branch.md) | ResNet-50 + FPN feature extraction |
| 3 | [LiDAR Branch](03-lidar-branch.md) | PointPillars: voxelization, pillar features, BEV scatter |
| 4 | [Encoder Fusion](04-encoder-fusion.md) | TSA, dual SCA, learnable blend weights |
| 5 | [Decoder Fusion](05-decoder-fusion.md) | Concat + linear fusion, identity initialization |
| 6 | [Transformer Decoder](06-transformer-decoder.md) | 6-layer decoder, reference point refinement |
| 7 | [Detection Heads](07-detection-heads.md) | Classification, bbox, yaw bin/res, velocity head |
| 8 | [Loss & Training](08-loss-and-training.md) | 5 loss functions, gradient isolation, training config |
| 9 | [Inference](09-inference.md) | NMS-free decoding, temporal test-time processing |

### Appendices

- [Tensor Shapes Reference](appendix-tensor-shapes.md)
- [File Map](appendix-file-map.md)
