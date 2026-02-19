# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import argparse
import mmcv
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os, os.path as osp, time
from pathlib import Path
import matplotlib.pyplot as plt


import math
import numpy as np
import matplotlib.pyplot as plt

import numpy as np, torch

"""
def _apply_post_aug_to_lidar2img_inplace(data):
    
    metas = data['img_metas'][0].data[0]
    for m in metas:
        A = m.get('img_aug_matrix', None)      # 4x4 per-image
        Ls = m.get('lidar2img', None)          # list[4x4]
        if A is None or Ls is None:
            continue
        # ensure numpy
        if hasattr(A, 'cpu'):
            A = A.cpu().numpy()
        new_Ls = []
        for L in Ls:
            if hasattr(L, 'cpu'):
                L = L.cpu().numpy()
            new_Ls.append(A @ L)
        m['lidar2img'] = new_Ls
"""
# --- put these near the top of tools/test.py ---
import csv, numpy as np, torch
import csv
import numpy as np

def _write_rownorm_confusion(conf, dataset, out_csv):
    """
    Write a row-normalized confusion matrix:
      - Each row i is GT class i
      - Columns are predicted classes j
      - Values are percentages of matched GT[i] that were predicted as j
      - Also writes row_sum (matches) and row_acc = conf[i,i]/row_sum
    """
    C = conf.shape[0]
    names = getattr(dataset, 'CLASSES', None)

    # row sums and safe normalization
    row_sum = conf.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        row_pct = np.where(row_sum[:, None] > 0,
                           conf / row_sum[:, None],
                           0.0)

    # row-wise accuracy given detection
    diag = np.diag(conf).astype(float)
    row_acc = np.where(row_sum > 0, diag / row_sum, 0.0)

    # write CSV
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        # header
        header = ['gt_class_id', 'gt_class_name', 'row_sum', 'row_acc'] \
                 + [f'pred_{j}' for j in range(C)]
        w.writerow(header)

        for i in range(C):
            cname = names[i] if names and i < len(names) else ''
            # percentage (0..100), round for readability
            pct_vals = (row_pct[i] * 100.0).tolist()
            pct_vals = [round(float(x), 1) for x in pct_vals]
            w.writerow([i, cname, int(row_sum[i]), round(float(row_acc[i]), 4)] + pct_vals)

    print(f"[CONF] wrote row-normalized confusion to {out_csv}")

def _radius_dedup_by_class(P, scores, labels, C, radius=1.0, per_class_topk=0):
    """Return indices of predictions kept after simple per-class radius-NMS on BEV centers."""
    keep_idx = []
    centers = P.gravity_center[:, :2]     # (N,2) torch
    for c in range(C):
        sel = (labels == c).nonzero(as_tuple=False).squeeze(1)
        if sel.numel() == 0:
            continue
        sc = scores[sel]
        ctr = centers[sel]
        order = torch.argsort(sc, descending=True)
        sel = sel[order]
        ctr = ctr[order]
        kept = []
        kept_ctr = []
        for i in range(sel.numel()):
            p = sel[i].item()
            xy = ctr[i:i+1]                      # (1,2)
            if len(kept_ctr) == 0:
                kept.append(p); kept_ctr.append(xy)
            else:
                KC = torch.cat(kept_ctr, dim=0)  # (K,2)
                if float(torch.cdist(xy, KC).min()) >= radius:
                    kept.append(p); kept_ctr.append(xy)
            if per_class_topk > 0 and len(kept) >= per_class_topk:
                break
        keep_idx += kept
    if len(keep_idx) == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    return torch.tensor(keep_idx, dtype=torch.long, device=scores.device)

def _compute_eval_confusion(outputs, dataset, score_thr=0.05, dist_thr=2.0, out_csv='_confmat_eval.csv',
                            dedup=False, nms_radius=1.0, per_class_topk=0):
    """One-to-one (greedy) GT↔Pred matching with optional de-dup; BEV center distance."""
    C = len(getattr(dataset, 'CLASSES', [])) or 10
    conf = np.zeros((C, C), dtype=np.int64)    # rows: GT class, cols: Pred class
    gt_total = np.zeros(C, dtype=np.int64)
    pred_total = np.zeros(C, dtype=np.int64)
    fp_per_class = np.zeros(C, dtype=np.int64)
    miss_per_class = np.zeros(C, dtype=np.int64)

    N = len(outputs)
    for idx in range(N):
        # ---- predictions (score filter) ----
        boxes3d, scores, labels = _unpack_pred_triplet(outputs, idx)
        keep = (scores >= score_thr)
        if keep.sum() == 0:
            P = boxes3d.new_box(torch.empty((0, 7), device=scores.device)) if hasattr(boxes3d, 'new_box') else boxes3d.__class__(torch.empty((0, 7), device=scores.device))
            Lp_full = np.empty((0,), dtype=np.int64)
        else:
            P = boxes3d[keep]
            Lp_full = labels[keep].detach().cpu().numpy().astype(np.int64).reshape(-1)

        # optional de-dup per class
        if P.tensor.numel() > 0 and dedup:
            kept_idx = _radius_dedup_by_class(P, scores[keep], labels[keep], C, radius=nms_radius,
                                              per_class_topk=per_class_topk)
            P = P[kept_idx]
            Lp_full = Lp_full[kept_idx.cpu().numpy()]

        # count valid pred labels for precision denominator
        valid_pred_mask = (Lp_full >= 0) & (Lp_full < C)
        if valid_pred_mask.any():
            pred_total += np.bincount(Lp_full[valid_pred_mask], minlength=C)

        # ---- ground-truth (filter ignored labels) ----
        ann = dataset.get_ann_info(idx)
        gt_boxes = ann['gt_bboxes_3d']
        Lg = np.asarray(ann.get('gt_labels_3d', ann.get('gt_labels', [])), dtype=np.int64).reshape(-1)
        valid_gt_mask = (Lg >= 0) & (Lg < C)
        Lg_valid = Lg[valid_gt_mask]
        if Lg_valid.size > 0:
            gt_total += np.bincount(Lg_valid, minlength=C)

        if Lg_valid.size == 0 and (P.tensor.numel() == 0 or Lp_full.size == 0):
            continue
        if Lg_valid.size == 0:
            # all preds are FP
            if valid_pred_mask.any():
                fp_per_class += np.bincount(Lp_full[valid_pred_mask], minlength=C)
            continue
        if P.tensor.numel() == 0 or Lp_full.size == 0:
            # all GTs are missed
            miss_per_class += np.bincount(Lg_valid, minlength=C)
            continue

        # ---- one-to-one greedy matching by distance ----
        G_all = gt_boxes.gravity_center[:, :2]
        G = G_all[torch.from_numpy(valid_gt_mask).to(G_all.device)]   # (Ngv,2)
        R = P.gravity_center[:, :2]                                   # (Np,2)
        D = torch.cdist(G, R).cpu().numpy()                            # (Ngv,Np)

        # consider only pairs under threshold
        gi, pj = np.where(D <= dist_thr)
        if gi.size > 0:
            order = np.argsort(D[gi, pj])  # smallest distance first
            gi = gi[order]; pj = pj[order]
        assigned_g = set(); assigned_p = set()

        for g, p in zip(gi, pj):
            if g in assigned_g or p in assigned_p:
                continue
            assigned_g.add(g); assigned_p.add(p)
            gt_cls = int(Lg_valid[g])
            pred_cls = int(Lp_full[p])
            if 0 <= pred_cls < C:
                conf[gt_cls, pred_cls] += 1
            else:
                # invalid pred label: ignore for conf, it will be FP later
                pass

        # misses: valid GT not assigned
        if len(assigned_g) < G.shape[0]:
            unassigned_g = [g for g in range(G.shape[0]) if g not in assigned_g]
            if len(unassigned_g) > 0:
                miss_per_class += np.bincount(Lg_valid[unassigned_g], minlength=C)

        # FPs: valid preds not assigned
        if len(assigned_p) < R.shape[0]:
            unassigned_p = [p for p in range(R.shape[0]) if p not in assigned_p]
            valid_fp = [int(Lp_full[p]) for p in unassigned_p if 0 <= int(Lp_full[p]) < C]
            if len(valid_fp) > 0:
                fp_per_class += np.bincount(np.array(valid_fp, dtype=np.int64), minlength=C)

    # ---- metrics ----
    tp = np.diag(conf)
    precision = tp / np.maximum(tp + fp_per_class, 1)
    recall    = tp / np.maximum(gt_total, 1)
    f1        = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    overall_acc = float(tp.sum()) / max(int(gt_total.sum()), 1)

    # ---- write CSV ----
    names = getattr(dataset, 'CLASSES', None)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        header = ['class_id','class_name','gt','pred_total','tp','fp','miss','precision','recall','f1'] \
                 + [f'pred_{k}' for k in range(C)]
        w.writerow(header)
        for c in range(C):
            cname = names[c] if names and c < len(names) else ''
            w.writerow([c, cname, int(gt_total[c]), int(pred_total[c]),
                        int(tp[c]), int(fp_per_class[c]), int(miss_per_class[c]),
                        round(float(precision[c]),4), round(float(recall[c]),4), round(float(f1[c]),4)]
                       + conf[c,:].tolist())
        w.writerow([])
        w.writerow(['overall_acc', round(overall_acc,4)])
    print(f"[CONF] wrote {out_csv}")
    if out_csv is not None:
        # derive a default name if the user didn't pass one via CLI
        # (or use args.confmat_rowcsv — see call site below)
        pass

def _as_np4x4(M):
    if hasattr(M, 'cpu'): M = M.cpu().numpy()
    M = np.asarray(M)
    if M.shape == (4, 4):
        return M
    if M.shape == (3, 3):
        H = np.eye(4, dtype=np.float32)
        H[:3, :3] = M
        return H
    raise ValueError(f"Unexpected matrix shape {M.shape}")

def _left_mult(A4, mats):
    out = []
    for M in mats:
        if hasattr(M, 'cpu'): M = M.cpu().numpy()
        M = np.asarray(M)
        if M.shape == (4, 4):
            out.append(A4 @ M)
        elif M.shape == (3, 3):
            H = np.eye(4, dtype=np.float32); H[:3,:3] = M
            out.append((A4 @ H)[:3,:3])
        else:
            out.append(M)
    return out

def _apply_post_aug_to_metas_inplace(m):
    """Apply image post-augmentation to ALL relevant camera matrices."""
    A = m.get('img_aug_matrix', None)
    if A is None:
        return
    A4 = _as_np4x4(A)

    # lidar->img (list of 4x4)
    if m.get('lidar2img', None) is not None:
        m['lidar2img'] = _left_mult(A4, m['lidar2img'])

    # intrinsics / cam2img (various forks)
    if m.get('cam2img', None) is not None:
        m['cam2img'] = _left_mult(A4, m['cam2img'])
        m['camera2img'] = m['cam2img']
    elif m.get('camera2img', None) is not None:
        m['camera2img'] = _left_mult(A4, m['camera2img'])
        m['cam2img'] = m['camera2img']

    if m.get('intrinsics', None) is not None:
        m['intrinsics'] = _left_mult(A4, m['intrinsics'])

def _apply_post_aug_to_batch_inplace(data):
    metas = data['img_metas'][0].data[0]
    for m in metas:
        _apply_post_aug_to_metas_inplace(m)

from mmcv.parallel import scatter
import torch

@torch.no_grad()
def custom_single_gpu_test_with_postaug(model, data_loader, show=False, out_dir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    for i, data in enumerate(data_loader):
        _apply_post_aug_to_batch_inplace(data)  # <-- important
        # scatter to current device (same as single_gpu_test)
        data = scatter(data, [torch.cuda.current_device()])[0]
        result = model(return_loss=False, rescale=True, **data)
        # MMDet3D test apis sometimes return list; unify to list
        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)
    return results

def _fit_rigid_2d(src_xy, dst_xy):
    """Return rotation(deg), tx, ty, rmse for best-fit rigid transform src->dst."""
    # center
    mu_s = src_xy.mean(0, keepdims=True)
    mu_d = dst_xy.mean(0, keepdims=True)
    X = src_xy - mu_s
    Y = dst_xy - mu_d
    # SVD
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1,:] *= -1
        R = Vt.T @ U.T
    theta = np.degrees(np.arctan2(R[1,0], R[0,0]))
    t = (mu_d - mu_s @ R).ravel()
    rmse = np.sqrt(((src_xy @ R + t - dst_xy)**2).sum(axis=1).mean())
    return theta, float(t[0]), float(t[1]), float(rmse)

def _print_best_rigid(dataset, outputs, idx, match_dist=3.0, score_thr=0.05):
    # unpack preds
    boxes3d, scores, labels = _unpack_pred_triplet(outputs, idx)
    keep = (scores >= score_thr)
    if keep.sum() == 0:
        print(f"[FIT] idx={idx} no preds ≥{score_thr}")
        return
    P = boxes3d.gravity_center[keep, :2].cpu()  # (Np,2)

    # GT
    ann = dataset.get_ann_info(idx)
    gt_boxes = ann['gt_bboxes_3d']
    if gt_boxes.tensor.numel() == 0:
        print(f"[FIT] idx={idx} no GT")
        return
    G = gt_boxes.gravity_center[:, :2].cpu()    # (Ng,2)

    # pair each GT to nearest pred
    d = torch.cdist(G, P)
    dmin, j = d.min(dim=1)
    mask = dmin <= match_dist
    if mask.sum() < 3:
        print(f"[FIT] idx={idx} not enough matches within {match_dist} m (got {int(mask.sum())})")
        return
    S = P[j[mask], :].numpy()
    D = G[mask, :].numpy()

    theta, tx, ty, rmse = _fit_rigid_2d(S, D)
    print(f"[FIT] idx={idx} best rigid: rotate {theta:+.1f}°  translate ({tx:+.2f},{ty:+.2f}) m  rmse={rmse:.2f}")


def _boxes3d_to_bev_xy_corners(boxes3d):
    """
    LiDARInstance3DBoxes -> (N, 4, 2) array of oriented box corners in BEV.
    Assumes dims order (w, l, h) and yaw in radians.
    """
    # centers (x, y), dims (w, l), yaw
    centers = boxes3d.gravity_center[:, :2].cpu().numpy()
    dims    = boxes3d.dims[:, :2].cpu().numpy()  # w, l
    yaws    = boxes3d.yaw.cpu().numpy() if hasattr(boxes3d, 'yaw') else boxes3d.tensor[:, 6].cpu().numpy()
    N = centers.shape[0]
    corners = np.zeros((N, 4, 2), dtype=np.float32)
    for i in range(N):
        w, l = dims[i]
        c, s = math.cos(yaws[i]), math.sin(yaws[i])
        # local corners (w forward along x half, l along y half)
        local = np.array([[ w/2,  l/2],
                          [ w/2, -l/2],
                          [-w/2, -l/2],
                          [-w/2,  l/2]], dtype=np.float32)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        corners[i] = centers[i] + local @ R.T
    return corners  # (N,4,2)

def _unpack_pred_triplet(outputs, idx):
    """
    Return (boxes3d, scores, labels) for frame idx from various MMDet3D output formats.
    Supports:
      - list[ (boxes3d, scores, labels) ]
      - list[ {'boxes_3d','scores_3d','labels_3d'} ]
      - list[ {'pts_bbox': {'boxes_3d','scores_3d','labels_3d'}} ]
      - dict with key 'bbox_results' or 'results' -> list[...]
    """
    o = outputs
    # container dict?
    if isinstance(o, dict):
        if 'bbox_results' in o:
            o = o['bbox_results']
        elif 'results' in o:
            o = o['results']

    item = o[idx]

    # A) already a triplet
    if isinstance(item, (list, tuple)):
        if len(item) == 3:
            return item[0], item[1], item[2]
        if len(item) == 1:
            item = item[0]  # unwrap and continue

    # B) dict forms
    if isinstance(item, dict):
        inner = item
        if 'pts_bbox' in inner and isinstance(inner['pts_bbox'], dict):
            inner = inner['pts_bbox']

        # explicit key checks (no `or` on tensors!)
        boxes3d = None
        if 'boxes_3d' in inner:
            boxes3d = inner['boxes_3d']
        elif 'bbox3d' in inner:
            boxes3d = inner['bbox3d']
        elif 'box3d' in inner:
            boxes3d = inner['box3d']

        scores = None
        if 'scores_3d' in inner:
            scores = inner['scores_3d']
        elif 'scores' in inner:
            scores = inner['scores']

        labels = None
        if 'labels_3d' in inner:
            labels = inner['labels_3d']
        elif 'labels' in inner:
            labels = inner['labels']

        if boxes3d is not None and scores is not None and labels is not None:
            return boxes3d, scores, labels

    # Unknown format → print a hint and fail clearly
    typ = type(item)
    keys = list(item.keys()) if isinstance(item, dict) else None
    raise RuntimeError(f"Unrecognized prediction format at idx={idx}: type={typ}, keys={keys}")

def save_bev_overlay(dataset, outputs, idx, out_path, pc_range, score_thr=0.1, class_names=None):
    """Draw GT (green) and predictions (red; per-class colors optional) on BEV."""
    boxes3d, scores, labels = _unpack_pred_triplet(outputs, idx)
    keep = scores >= score_thr
    P = boxes3d[keep]
    Lp = labels[keep].cpu().numpy()
    Sp = scores[keep].cpu().numpy()
    # GT
    ann = dataset.get_ann_info(idx)
    gt_boxes = ann['gt_bboxes_3d']
    Lg = ann.get('gt_labels_3d', ann.get('gt_labels', None))
    Lg = np.array(Lg) if Lg is not None else np.zeros(len(gt_boxes), dtype=int)

    # Convert to corners
    pred_corners = _boxes3d_to_bev_xy_corners(P) if P.tensor.numel() > 0 else np.zeros((0,4,2), np.float32)
    gt_corners   = _boxes3d_to_bev_xy_corners(gt_boxes) if gt_boxes.tensor.numel() > 0 else np.zeros((0,4,2), np.float32)

    # Plot
    xmin, ymin, zmin, xmax, ymax, zmax = pc_range
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim([xmin, xmax]); ax.set_ylim([ymin, ymax]); ax.set_aspect('equal')
    ax.set_title(f"Frame {idx} — GT (green) vs Pred (red)  thr={score_thr}")

    # Draw GT
    for c in gt_corners:
        ax.plot([c[0,0], c[1,0], c[2,0], c[3,0], c[0,0]],
                [c[0,1], c[1,1], c[2,1], c[3,1], c[0,1]], linewidth=1.5, color='g', alpha=0.9)

    # Draw Predictions
    for c in pred_corners:
        ax.plot([c[0,0], c[1,0], c[2,0], c[3,0], c[0,0]],
                [c[0,1], c[1,1], c[2,1], c[3,1], c[0,1]], linewidth=1.0, color='r', alpha=0.9)

    # Optional legend
    ax.legend(['GT', 'Pred'], loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    # argparse section
    parser.add_argument('--viz-bev', action='store_true', help='Save BEV overlays for a few frames')
    parser.add_argument('--viz-num', type=int, default=10, help='How many frames to plot')
    parser.add_argument('--viz-score-thr', type=float, default=0.1, help='Min pred score to draw')
    parser.add_argument('--viz-outdir', type=str, default='_bev_viz', help='Output dir for BEV images')
    parser.add_argument('--confmat', action='store_true',
                        help='Compute confusion matrix on eval outputs')
    parser.add_argument('--confmat-thr', type=float, default=0.05,
                        help='Score threshold for predictions used in the confusion pass')
    parser.add_argument('--confmat-dist', type=float, default=2.0,
                        help='BEV center distance (m) to match GT and pred')
    parser.add_argument('--confmat-out', type=str, default='_confmat_eval.csv',
                        help='Where to save the confusion matrix CSV')
    parser.add_argument('--confmat-dedup', action='store_true',
                        help='Per-class BEV radius de-dup before matching')
    parser.add_argument('--confmat-nms-radius', type=float, default=1.0,
                        help='Radius (m) for de-dup when --confmat-dedup is on')
    parser.add_argument('--confmat-perclass-topk', type=int, default=0,
                        help='Keep at most K predictions per class per frame (0=off)')
    parser.add_argument(
        '--confmat-rowcsv',
        type=str,
        default='_confmat_rowpct.csv',
        help='Path to write a row-normalized confusion matrix (percent per GT class).'
    )


    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args

def _get_pc_range(model, cfg):
    """Find [xmin,ymin,zmin,xmax,ymax,zmax] from the unwrapped model or cfg."""
    m = model.module if hasattr(model, "module") else model

    # 1) direct attributes
    for obj in [m, getattr(m, "bbox_head", None), getattr(getattr(m, "bbox_head", None), "bbox_coder", None)]:
        if obj is None: 
            continue
        pr = getattr(obj, "pc_range", None)
        if pr is not None:
            return list(pr)

    # 2) common places in cfg (covers many forks)
    #    - cfg.model.bbox_head.bbox_coder.pc_range
    #    - cfg.model.pts_bbox_head.bbox_coder.pc_range
    #    - cfg.point_cloud_range (sometimes 6 numbers)
    #    - cfg.plugin and nested detector variants
    try:
        return list(cfg.model["bbox_head"]["bbox_coder"]["pc_range"])
    except Exception:
        pass
    try:
        return list(cfg.model["pts_bbox_head"]["bbox_coder"]["pc_range"])
    except Exception:
        pass
    pr = cfg.get("point_cloud_range", None)
    if pr is not None and len(pr) == 6:
        return list(pr)

    # 3) dataset sometimes exposes it (rare)
    # (only if you pass dataset in; otherwise skip)

    return None

def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')
    
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # set tf32
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        # On Windows use gloo; otherwise NCCL. Pull from cfg if present.
        dist_params = cfg.get('dist_params', {})
        if os.name == 'nt' and dist_params.get('backend', '').lower() == '':
            dist_params['backend'] = 'gloo'
        init_dist(args.launcher, **dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        # single-GPU test
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = custom_single_gpu_test_with_postaug(model, data_loader, args.show, args.show_dir)

    rank, _ = get_dist_info()
    if rank == 0:
        # 1) optionally dump raw outputs (pickle) if user passed --out
        if args.out:
            osp_dir = osp.dirname(args.out)
            if osp_dir:
                os.makedirs(osp_dir, exist_ok=True)
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)   # ← remove the assert; just dump

        # 2) build a json prefix for nuScenes formatter/evaluator
        time_str = time.strftime('%Y%m%d_%H%M%S')
        out_prefix = osp.join('test', Path(args.config).stem, time_str)
        os.makedirs(osp.dirname(out_prefix), exist_ok=True)

        kwargs = {} if args.eval_options is None else args.eval_options
        kwargs['jsonfile_prefix'] = out_prefix

        # 3) format-only (writes JSONs) or evaluate
        if args.format_only:
            dataset.format_results(outputs, **kwargs)

        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # strip hook-only keys
            for key in ['interval','tmpdir','start','gpu_collect','save_best','rule']:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))

        # By ChatGPT

        if args.viz_bev and rank == 0:
            # get pc_range from your model/head/coder (same as training)
            pc_range = _get_pc_range(model, cfg)

            # CLI override if present (won't crash if the arg wasn't added)
            viz_pc_range_arg = getattr(args, 'viz_pc_range', '')
            if (not pc_range) and viz_pc_range_arg:
                pc_range = [float(x) for x in viz_pc_range_arg.split(',')]

            if not pc_range:
                raise AssertionError(
                    "pc_range not found; either add --viz-pc-range or pull from cfg/model."
                )
            print(f"[BEV] using pc_range = {pc_range}")# get pc_range for BEV viz

            K = min(args.viz_num, len(outputs))
            for i in range(K):
                out_path = os.path.join(args.viz_outdir, f"bev_{i:05d}.png")
                save_bev_overlay(dataset, outputs, i, out_path, pc_range, score_thr=args.viz_score_thr,
                                class_names=getattr(dataset, 'CLASSES', None))
                _print_best_rigid(dataset, outputs, i, match_dist=3.0, score_thr=args.viz_score_thr)
            print(f"[BEV] saved {K} BEV overlays to {args.viz_outdir}")

        if args.confmat and rank == 0:
            _compute_eval_confusion(
                outputs, dataset,
                score_thr=args.confmat_thr,
                dist_thr=args.confmat_dist,
                out_csv=args.confmat_out,
                dedup=getattr(args, 'confmat_dedup', False),
                nms_radius=getattr(args, 'confmat_nms_radius', 1.0),
                per_class_topk=getattr(args, 'confmat_perclass_topk', 0),
            )

if __name__ == '__main__':
    main()
