# tools/quick_eval_probe.py
# Quick probe for BEVFormer(+PETR3D) inference outputs on a tiny val slice.
# Prints per-layer max sigmoid, number of boxes after postproc, and one box.
#
# Usage (Windows):
#   cd C:\Users\sangh\myproject\BEVFormer_PETR
#   set PYTHONPATH=%CD%;%PYTHONPATH%
#   python tools\quick_eval_probe.py projects\configs\bevformer\bevformer_blob_iter2.py 0.001
#
# Tip: try 0.05 and 0.001 thresholds to check calibration.

import os, sys, math
import numpy as np
import torch

# --- Make repo root importable so 'projects' works ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.environ['PYTHONPATH'] = REPO_ROOT + os.pathsep + os.environ.get('PYTHONPATH', '')

from copy import deepcopy
from mmcv import Config, ConfigDict
from mmcv.parallel import scatter
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model

def clean_dataset_cfg(d):
    d = deepcopy(d)
    for k in ['samples_per_gpu','workers_per_gpu','persistent_workers',
              'drop_last','pin_memory','shuffle']:
        d.pop(k, None)
    return d

def make_permissive_test_cfg(cfg):
    # ensure model.test_cfg.pts exists and is a ConfigDict
    tc = ConfigDict(cfg.model.get('test_cfg', {}) or {})
    pts = ConfigDict(tc.get('pts', {}) or {})
    pts['score_thr'] = 0.0
    pts['max_per_img'] = 5000
    # If your head supports it, uncomment to disable NMS entirely
    # pts['nms_type'] = 'none'
    tc['pts'] = pts
    cfg.model['test_cfg'] = tc

    # widen post_center_range if present
    try:
        cfg.model['pts_bbox_head']['bbox_coder']['post_center_range'] = [-1e9, -1e9, -1e9, 1e9, 1e9, 1e9]
    except Exception:
        pass

def patch_get_bboxes_class(model):
    """Patch get_bboxes at the class level (not instance) to bypass fp16 wrappers."""
    head = getattr(model, 'pts_bbox_head', None)
    if head is None:
        print("[WARN] model has no pts_bbox_head; skipping patch")
        return

    # find unbound original method on the class dict
    try:
        orig_unbound = head.__class__.__dict__['get_bboxes']
    except KeyError:
        print("[WARN] could not locate get_bboxes on class; skipping patch")
        return

    def dbg_get_bboxes_unbound(self, *args, **kwargs):
        # Normalize signature to avoid duplicate 'rescale'
        if len(args) < 3:
            return orig_unbound(self, *args, **kwargs)
        all_cls_scores, all_bbox_preds, img_metas = args[:3]
        rescale_pos = args[3] if len(args) >= 4 else None
        rescale_kw  = kwargs.pop('rescale', None)
        if rescale_pos is not None and rescale_kw is not None:
            rescale = bool(rescale_pos); args = args[:3]
        elif rescale_pos is not None:
            rescale = bool(rescale_pos); args = args[:3]
        elif rescale_kw is not None:
            rescale = bool(rescale_kw)
        else:
            rescale = False

        # Debug: per-layer max sigmoid (ignore non-tensors)
        try:
            max_per_layer = []
            for t in all_cls_scores:  # list over decoder layers
                if torch.is_tensor(t):
                    max_per_layer.append(float(torch.sigmoid(t).max().detach().cpu()))
            if max_per_layer:
                print("[DBG] max sigmoid score per layer:", ["{:.4f}".format(x) for x in max_per_layer])
        except Exception as e:
            print(f"[DBG] (skip score debug) {e}")

        out = orig_unbound(self, all_cls_scores, all_bbox_preds, img_metas, rescale=rescale, **kwargs)

        # Debug: count after postproc + example box (best-effort)
        try:
            first = out[0]
            boxes, scores = None, None
            if isinstance(first, dict) and 'boxes_3d' in first:
                boxes = first['boxes_3d'].tensor
                scores = first['scores_3d']
            elif isinstance(first, (list, tuple)) and len(first) > 0:
                tup = first[0] if isinstance(first[0], (list, tuple)) else first
                boxes = tup[0].tensor if hasattr(tup[0], 'tensor') else tup[0]
                scores = tup[1]
            if boxes is not None:
                N = int(boxes.shape[0])
                smax = float(scores.max().cpu()) if (scores is not None and N>0) else 'NA'
                print(f"[DBG] postproc preds N={N}, score_max={smax}")
                if N>0 and boxes.shape[-1] >= 7:
                    b0 = boxes[0].detach().cpu().numpy()
                    x,y,z,w,l,h,yaw = b0[:7]
                    print(f"[DBG] postproc ex: x={x:.2f} y={y:.2f} z={z:.2f} w={w:.2f} l={l:.2f} h={h:.2f} yaw={yaw:.2f}")
        except Exception as e:
            print(f"[DBG] (skip postproc debug) {e}")

        return out

    head.__class__.get_bboxes = dbg_get_bboxes_unbound

def main(cfg_path, thr=0.05, max_batches=5, device_id=0):
    cfg = Config.fromfile(cfg_path)

    # --- permissive eval settings ---
    make_permissive_test_cfg(cfg)

    # --- build dataset/datloader (clean val cfg) ---
    val_cfg = clean_dataset_cfg(cfg.data.val)
    val_cfg.test_mode = True
    dataset = build_dataset(val_cfg)

    # --- show where format_results is coming from, and wrap it for debug ---
    import inspect, types

    print("[DBG] dataset class:", dataset.__class__)
    print("[DBG] MRO:", dataset.__class__.mro())

    src = dataset.format_results.__func__.__code__.co_filename if hasattr(dataset.format_results, "__func__") else dataset.format_results.__code__.co_filename
    print("[DBG] format_results defined in:", src)

    # keep original method
    _orig_format_results = dataset.format_results

    def _dbg_format_results(self, results, jsonfile_prefix=None, **kwargs):
        out = _orig_format_results(results, jsonfile_prefix=jsonfile_prefix, **kwargs)
        try:
            # Print first few packed entries straight from the JSON that was just written
            # MMDet3D usually returns (result_files, tmp_dir). Handle both returns.
            if isinstance(out, tuple):
                result_files, tmp_dir = out
            else:
                result_files, tmp_dir = out, None

            # Prefer pts bbox json path
            import mmcv, os
            json_paths = []
            if isinstance(result_files, dict):
                # common key in mmdet3d: 'pts_bbox'
                if 'pts_bbox' in result_files and isinstance(result_files['pts_bbox'], str) and result_files['pts_bbox'].endswith('.json'):
                    json_paths.append(result_files['pts_bbox'])
                # fallback: any json in mapping
                for v in result_files.values():
                    if isinstance(v, str) and v.endswith('.json'):
                        json_paths.append(v)
            elif isinstance(result_files, str) and result_files.endswith('.json'):
                json_paths.append(result_files)

            json_paths = list(dict.fromkeys(json_paths))[:1]  # keep unique, first one
            if json_paths:
                data = mmcv.load(json_paths[0])
                # nuScenes schema: {'results': {sample_token: [detections...]}, 'meta': {...}}
                res = data.get('results', {})
                printed = 0
                for tok, dets in res.items():
                    print(f"[PKG] sample_token={tok}  #preds={len(dets)}")
                    for j, d in enumerate(dets[:3]):
                        print("   ", j,
                            d.get('detection_name', '?'),
                            "trans=", [round(x,2) for x in d.get('translation', [])],
                            "size=",  [round(x,2) for x in d.get('size', [])],
                            "score=", round(float(d.get('detection_score', 0.0)),3))
                    printed += 1
                    if printed >= 3:
                        break
            else:
                print("[PKG] Could not locate JSON path in result_files:", result_files)
        except Exception as e:
            print("[PKG] debug print failed:", e)
        return out

    # install wrapper
    dataset.format_results = types.MethodType(_dbg_format_results, dataset)


    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False)

    # --- build model ---
    model = build_model(cfg.model)
    if hasattr(model, 'video_test_mode'):
        model.video_test_mode = False  # avoid queue dependency in probe
    model.cuda(device_id)
    model.eval()

    # --- patch get_bboxes for debug ---
    patch_get_bboxes_class(model)

    # --- iterate a few batches ---
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= max_batches:
                break

            # unwrap DataContainers & move tensors to GPU
            data = scatter(data, [device_id])[0]

            # print image file order once for sanity (after scatter)
            if i == 0 and 'img_metas' in data:
                try:
                    metas0 = data['img_metas'][0][0]  # [batch][sample]
                    # metas0 typically has 'filename' -> list of 6 cam paths
                    if isinstance(metas0, dict):
                        if 'filename' in metas0:
                            print("[DBG] file order:", metas0['filename'])
                        elif 'filenames' in metas0:
                            print("[DBG] file order:", metas0['filenames'])
                        else:
                            print("[DBG] img_metas keys:", list(metas0.keys()))
                    else:
                        print("[DBG] unexpected img_metas[0][0] type:", type(metas0))
                except Exception as e:
                    print(f"[DBG] could not print filenames: {e}")

            result = model(return_loss=False, rescale=True, **data)
            out = result[0]['pts_bbox']


            # extract arrays safely
            boxes3d = out['boxes_3d'].tensor.detach().cpu().numpy()  # (N, >=7)
            scores3d = out['scores_3d'].detach().cpu().numpy()
            labels3d = out['labels_3d'].detach().cpu().numpy()
            N = boxes3d.shape[0]
            print(f"[DBG] preds N={N}, score_max={scores3d.max() if N>0 else 'NA'}")

            if N > 0 and boxes3d.shape[1] >= 7:
                x,y,z,w,l,h,yaw = boxes3d[0][:7]
                print(f"ex box: x={x:.2f} y={y:.2f} z={z:.2f} w={w:.2f} l={l:.2f} h={h:.2f} yaw={yaw:.2f}")
                xyz = boxes3d[:, :3]
                print(f"[DBG] x[{xyz[:,0].min():.1f},{xyz[:,0].max():.1f}] "
                      f"y[{xyz[:,1].min():.1f},{xyz[:,1].max():.1f}] "
                      f"z[{xyz[:,2].min():.1f},{xyz[:,2].max():.1f}]")

            # simple per-class counts above external thr (for visibility)
            keep = scores3d >= thr
            counts = np.bincount(labels3d[keep], minlength=len(dataset.CLASSES)) if N>0 else np.zeros(len(dataset.CLASSES), int)
            total = int(keep.sum())
            print(f"[DBG] thr={thr} -> kept {total} boxes")
            if total > 0:
                for ci, name in enumerate(dataset.CLASSES):
                    if counts[ci] > 0:
                        print(f"  {ci:02d} {name:20s}: {counts[ci]}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Quick eval probe for BEVFormer(+PETR3D)")
    ap.add_argument("config", type=str, help="Path to config.py")
    ap.add_argument("thr", type=float, nargs='?', default=0.05, help="Score threshold for counting (external)")
    ap.add_argument("--batches", type=int, default=5, help="Num batches to probe")
    ap.add_argument("--gpu", type=int, default=0, help="GPU id")
    args = ap.parse_args()
    main(args.config, args.thr, args.batches, args.gpu)

