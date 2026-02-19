# tools/train_sanity_petr3d.py
# tools/train_sanity_petr3d.py
import os
import sys
import argparse
from pprint import pprint

import torch
import mmcv
from mmcv import Config
from mmcv.runner import set_random_seed
from mmcv.parallel import scatter


# ---- Robust optimizer builder (handles different versions) ----
build_optimizer_mmcv = None
build_optimizer_mmdet_core = None
try:
    from mmcv.runner import build_optimizer as build_optimizer_mmcv  # MMDet 2.x compatible
except Exception:
    pass
try:
    from mmdet.core import build_optimizer as build_optimizer_mmdet_core  # Some installs keep it here
except Exception:
    pass

def build_optimizer_fallback(model, opt_cfg):
    """Very small fallback if neither builder is available."""
    import torch.optim as optim
    lr = opt_cfg.get('lr', 1e-4)
    wd = opt_cfg.get('weight_decay', 0.01)
    # No paramwise here; this is just a smoke fallback
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

def build_optimizer_any(model, opt_cfg):
    if build_optimizer_mmcv is not None:
        return build_optimizer_mmcv(model, opt_cfg)
    if build_optimizer_mmdet_core is not None:
        return build_optimizer_mmdet_core(model, opt_cfg)
    return build_optimizer_fallback(model, opt_cfg)

# ---- Builders for dataset/model (MMDet3D v1.x) ----
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model as build_detector  # alias in mmdet3d v1.x

def to_device(data, device):
    """Recursively move batch to device."""
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    if isinstance(data, (list, tuple)):
        return type(data)(to_device(x, device) for x in data)
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    return data

def main():
    parser = argparse.ArgumentParser("BEVFormer+PETR3D training sanity")
    parser.add_argument("--config", required=True, help="Path to your BEVFormer config (.py)")
    parser.add_argument("--iters", type=int, default=2, help="Number of mini iters to run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    # Make repo importable if running from arbitrary cwd
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    cfg = Config.fromfile(args.config)
    print("Loaded config:", args.config)

    # Basic overrides for a quick smoke
    cfg.seed = args.seed
    set_random_seed(cfg.seed, deterministic=False)
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Make the dataloader tiny & deterministic
    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = 0

    # ---- Build dataset & dataloader (train split) ----
    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )
    print(f"Dataset: {type(dataset).__name__}, len={len(dataset)}")

    # ---- Build model (train mode) ----
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get("train_cfg"),
        test_cfg=cfg.get("test_cfg"),
    )
    model.init_weights()
    model = model.to(device)
    model.train()

    # ---- Optimizer ----
    optimizer = build_optimizer_any(model, cfg.optimizer)

    # ---- Tiny training loop ----
    iters = max(1, int(args.iters))
    print(f"Running {iters} iteration(s)...\n")
    data_iter = iter(data_loader)

    for step in range(iters):
        try:
            data_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            data_batch = next(data_iter)

        # Properly move batch to device respecting DataContainer
        if device != "cpu" and torch.cuda.is_available():
            gpu_id = torch.cuda.current_device()  # typically 0
            data_batch = scatter(data_batch, [gpu_id])[0]
        # else: keep on CPU

        # Standard MMDet-style step: returns dict(loss, log_vars, num_samples)
        out = model.train_step(data_batch, optimizer)
        # Backprop is usually done inside train_step; we only log here
        log_vars = out.get("log_vars", {})
        num_samples = out.get("num_samples", 0)

        # Pretty print losses
        print(f"[iter {step+1}/{iters}] num_samples={num_samples}")
        for k, v in log_vars.items():
            try:
                val = float(v)
            except Exception:
                val = v
            print(f"  {k:>25s}: {val}")
        print("")

    print("Sanity run complete ✅")

if __name__ == "__main__":
    main()
