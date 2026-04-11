# BEVFormerFusion

<div class="landing-hero">
  <span class="hero-kicker">Research landing page</span>
  <h1>Camera-LiDAR BEV fusion built on the BEVFormer detector path</h1>
  <p class="hero-subtitle">
    BEVFormerFusion extends the active BEVFormer implementation in this repository with PointPillars-based LiDAR injection at the encoder and decoder stages, plus a dedicated velocity head that reads from the pre-fusion BEV state.
  </p>
  <ul class="hero-bullets">
    <li>Encoder-side LiDAR deformable attention added to every BEV encoder layer.</li>
    <li>Decoder-side concatenate-and-project fusion initialized as a camera-preserving shortcut.</li>
    <li>Dedicated velocity cross-attention head trained independently from box regression.</li>
    <li>Accuracy tables derived only from repo-tracked result workbooks and code-validated comparisons.</li>
  </ul>
  <div class="hero-actions">
    <a class="primary" href="architecture/">Read the architecture</a>
    <a href="bevformer-comparison/">Compare against BEVFormer</a>
    <a href="experiments/">Inspect experiments</a>
    <a href="https://github.com/xiaotwu/BEVFormerFusion">GitHub repository</a>
  </div>
</div>

<div id="hero-metrics" class="metric-strip"></div>

<div class="pending-note">
  <strong>Runtime evidence status.</strong> <span id="hero-runtime-note">Loading metrics...</span>
</div>

<div class="section-shell">
  <h2>Overview and motivation</h2>
  <p>
    The upstream BEVFormer design builds BEV tokens from multi-view images through temporal self-attention and camera spatial cross-attention. The active BEVFormerFusion path keeps that detector scaffold but adds a LiDAR BEV branch produced by PointPillars. The resulting design injects geometric evidence twice: first inside the encoder by parallel cross-attention, then again before decoder cross-attention through a concatenate-and-project block.
  </p>
  <p>
    The repository also changes how orientation and motion are supervised. Yaw is factorized into discrete-bin and residual heads, while velocity is moved out of the box regression branch and predicted by a dedicated query-to-BEV attention module. This isolates gradients for motion estimation from the heavily LiDAR-enriched detection path.
  </p>
  <div class="comparison-callout">
    <p><strong>Active implementation scope.</strong> The published documentation tracks `projects/configs/bevformer/bevformer_project.py`, the `BEVFormer` detector, `PerceptionTransformer`, `MM_BEVFormerLayer`, `BEVFormerHead`, and `CustomNuScenesDataset`.</p>
    <p><strong>Excluded from the main narrative.</strong> `projects/configs/bevformerv2/`, `transformer copy.py`, `*_old.py`, and the PETR3D test path are treated as legacy or experimental code and are not presented as the canonical method.</p>
  </div>
</div>

## Method

### Original BEVFormer

The official BEVFormer baseline used for this documentation is the public `fundamentalvision/BEVFormer` repository. Its base configuration is camera-only, uses `BEVFormerLayer` inside the encoder, and predicts box geometry, yaw, and velocity directly from the standard detection head without a separate LiDAR branch or dedicated motion head.

### Proposed modifications

BEVFormerFusion adds three code-backed changes to that baseline:

1. `MM_BEVFormerLayer` adds a LiDAR deformable-attention branch to each encoder layer and blends it with the camera cross-attention output through a learned sigmoid gate.
2. `PerceptionTransformer` snapshots the camera-path BEV before decoder-side fusion, then concatenates projected LiDAR BEV tokens with the encoder output and compresses them back to the model dimension through an identity-initialized linear layer.
3. `BEVFormerHead` replaces direct yaw and velocity supervision with yaw-bin / yaw-residual heads and a dedicated velocity cross-attention head that reads from the pre-fusion BEV snapshot.

### Comparison table

| Component | BEVFormer | Ours | Change | Benefit |
| --- | --- | --- | --- | --- |
| Sensor inputs | Multi-view camera only | Multi-view camera plus PointPillars LiDAR BEV branch | Architectural addition | Adds an explicit geometric feature path in BEV coordinates. |
| Encoder layer | `BEVFormerLayer` with camera spatial cross-attention only | `MM_BEVFormerLayer` with camera and LiDAR deformable attention branches | Architectural modification | Allows each BEV layer to blend image evidence with LiDAR BEV evidence. |
| Decoder input | Encoder BEV output only | Encoder BEV plus projected LiDAR BEV through concat + linear fusion | Architectural addition | Preserves a second, direct LiDAR path for object-query decoding. |
| Motion supervision | Velocity channels trained inside the box head | Dedicated velocity cross-attention head with separate loss | Architectural modification | Keeps motion estimation tied to the pre-fusion BEV state instead of the LiDAR-heavy decoder input. |
| Orientation supervision | Direct box regression channels | Yaw-bin and yaw-residual branches | Architectural modification | Separates coarse orientation classification from residual refinement. |
| Token budget | `bev_h = bev_w = 200`, `num_query = 900`, encoder depth 6, ResNet-101 backbone | `bev_h = bev_w = 100`, `num_query = 450`, encoder depth 4, ResNet-50 backbone | Training and config change | Reduces BEV/query budget and backbone depth relative to the local base config. |
| Temporal handling | Standard `prev_bev` path in official code | Scene-keyed BEV cache plus no-grad history passes in `obtain_history_bev` | Efficiency and memory change | Avoids backpropagating through history frames while keeping temporal context. |

<div class="research-table-wrap">
  <table class="research-table">
    <thead>
      <tr>
        <th>Model</th>
        <th>Best iter</th>
        <th>mAP</th>
        <th>NDS</th>
        <th>FPS</th>
        <th>Memory</th>
      </tr>
    </thead>
    <tbody id="main-results-body">
      <tr><td colspan="6">Loading metrics...</td></tr>
    </tbody>
  </table>
</div>

<div class="provenance-note">
  <strong>Published metric provenance.</strong> The tables on this page are driven by `docs/assets/data/metrics.json`, which was normalized from `results/Baseline_Results_Summary.xlsx` and one canonical fused-run workbook. Runtime metrics remain pending because the repository does not include a profiling artifact for FPS or memory.
</div>

## Results

<div id="insight-grid" class="insight-grid"></div>

<p id="fusion-delta-note"></p>
<p id="best-checkpoint-note"></p>

<div class="curve-frame">
  <div id="nds-curve"></div>
</div>

### 12GB GPU table

<div class="research-table-wrap">
  <table class="research-table">
    <thead>
      <tr>
        <th>Setting</th>
        <th>Memory</th>
        <th>FPS</th>
        <th>Accuracy</th>
        <th>Notes</th>
      </tr>
    </thead>
    <tbody id="profile-results-body">
      <tr><td colspan="5">Loading profile status...</td></tr>
    </tbody>
  </table>
</div>

The repository does not currently contain a 12GB runtime log, benchmark export, or checkpoint-specific memory trace. The table is therefore wired into the canonical metrics artifact but intentionally left pending until a fresh profiling run is recorded.

## Experiments

The local experiment evidence currently tracks checkpoint summaries rather than a full benchmark harness. The public experiment pages therefore focus on:

- validation mAP and NDS across checkpoint milestones,
- the best local baseline checkpoint at 100k iterations,
- the best fused checkpoint at 100k iterations,
- provenance retained after workbook deletion through checksums and normalized JSON.

The training configuration for the active method is drawn from `projects/configs/bevformer/bevformer_project.py`: AdamW with `lr = 2e-4`, BEV resolution `100 x 100`, object-query count `450`, temporal queue length `4`, and LiDAR fusion mode `encoder_decoder`.

## Documentation

<div class="doc-grid">
  <div class="doc-card">
    <a href="architecture/">Architecture</a>
    <p>System summary, module roles, data flow, and training/inference paths.</p>
  </div>
  <div class="doc-card">
    <a href="bevformer-comparison/">BEVFormer comparison</a>
    <p>Code-backed differences between the official baseline and the active fusion path.</p>
  </div>
  <div class="doc-card">
    <a href="experiments/">Experiments</a>
    <p>Checkpoint metrics, analysis, and provenance for published tables.</p>
  </div>
  <div class="doc-card">
    <a href="usage/">Usage</a>
    <p>Installation assumptions, dataset expectations, and common train/eval commands.</p>
  </div>
  <div class="doc-card">
    <a href="api-reference/">API reference</a>
    <p>Code-to-documentation mapping for the active implementation path.</p>
  </div>
</div>
