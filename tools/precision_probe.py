# tools/precision_probe.py
# Compute per-class Precision@K from a results_nusc.json against nuScenes GT.
# Requirements:
#   pip install nuscenes-devkit mmcv
#
# Example:
#   python tools/precision_probe.py \
#     --json test\...\pts_bbox\results_nusc.json \
#     --dataroot E:\datasets_all\nuscenes --version v1.0-mini --K 100 --thr 2.0

import argparse
import math
from collections import defaultdict, Counter

import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.utils import category_to_detection_name

NUSC_10 = [
    'car','truck','bus','trailer','construction_vehicle',
    'pedestrian','motorcycle','bicycle','traffic_cone','barrier'
]

def l2(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.linalg.norm(a - b))

def load_gt_by_token(nusc: NuScenes, split='val'):
    """Return dict: sample_token -> list of GT {name, center} in detection-10 space."""
    # Get sample tokens belonging to the chosen split
    # (nuScenes devkit uses a split file; here we filter by scene split)
    from nuscenes.utils.splits import create_splits_scenes
    scenes = set(create_splits_scenes()[split])
    scene_tokens = set([s['token'] for s in nusc.scene if s['name'] in scenes])

    # Map sample->scene filter
    valid_sample_tokens = []
    for s in nusc.sample:
        if s['scene_token'] in scene_tokens:
            valid_sample_tokens.append(s['token'])

    gt = defaultdict(list)
    for ann in nusc.sample_annotation:
        st = ann['sample_token']
        if st not in valid_sample_tokens:
            continue
        det_name = category_to_detection_name(ann['category_name'])
        if det_name is None or det_name not in NUSC_10:
            continue
        gt[st].append({
            'name': det_name,
            'center': ann['translation'][:3]  # global XYZ
        })
    return gt, set(valid_sample_tokens)

def precision_at_k_for_class(preds, gts, K, thr):
    """
    preds: list of dicts with keys {translation, detection_name, detection_score}, sorted desc by score
    gts:   list of dicts with keys {name, center}
    Return (P@K, TP, FP, GT_count)
    """
    # Keep only class-matching GTs
    # We’ll match greedily, removing a GT once claimed.
    gt_centers = [g['center'] for g in gts]
    gt_used = [False] * len(gt_centers)

    TP = 0
    FP = 0
    considered = 0

    for p in preds[:K]:
        considered += 1
        pc = p['translation']
        # find nearest unmatched GT within thr
        best_i = -1
        best_d = 1e9
        for i, (c, used) in enumerate(zip(gt_centers, gt_used)):
            if used:
                continue
            d = l2(pc, c)
            if d < best_d:
                best_d, best_i = d, i
        if best_i >= 0 and best_d <= thr:
            TP += 1
            gt_used[best_i] = True
        else:
            FP += 1

    denom = max(1, min(K, len(preds)))  # avoid div by zero
    precision = TP / denom
    return precision, TP, FP, len(gt_centers)

def main():
    ap = argparse.ArgumentParser("Per-class Precision@K from nuScenes results_nusc.json")
    ap.add_argument("--json", required=True, help="Path to results_nusc.json")
    ap.add_argument("--dataroot", required=True, help="nuScenes dataroot (folder with samples/, sweeps/, v1.0-*)")
    ap.add_argument("--version", default="v1.0-mini", help="nuScenes version (e.g., v1.0-mini, v1.0-trainval)")
    ap.add_argument("--split", default="val", help="Split to evaluate (val or mini_val for v1.0-mini)")
    ap.add_argument("--K", type=int, default=100, help="Top-K per class per sample")
    ap.add_argument("--thr", type=float, default=2.0, help="Distance threshold (meters) for TP")
    args = ap.parse_args()

    data = mmcv.load(args.json)
    res = data.get('results', {})
    print(f"[INFO] Loaded {len(res)} sample_tokens from {args.json}")

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    gt_by_tok, valid_tokens = load_gt_by_token(nusc, split=args.split)
    # Restrict to tokens present in both JSON & split
    common_tokens = [t for t in res.keys() if t in valid_tokens]
    print(f"[INFO] Tokens in split & json: {len(common_tokens)}")

    # Pre-index predictions per token & class, sorted by score desc
    preds_by_tok_class = {}
    for tok in common_tokens:
        dets = res.get(tok, [])
        per_cls = {c: [] for c in NUSC_10}
        for d in dets:
            name = d.get('detection_name')
            if name in per_cls:
                per_cls[name].append(d)
        for c in NUSC_10:
            per_cls[c].sort(key=lambda x: float(x.get('detection_score', 0.0)), reverse=True)
        preds_by_tok_class[tok] = per_cls

    # Aggregate Precision@K per class across tokens
    agg = {c: {'TP':0, 'FP':0, 'GT':0, 'Ntok':0} for c in NUSC_10}
    for tok in common_tokens:
        gts = gt_by_tok.get(tok, [])
        if not gts:
            continue
        # group GT by detection class
        gt_by_cls = defaultdict(list)
        for g in gts:
            if g['name'] in NUSC_10:
                gt_by_cls[g['name']].append(g)

        for c in NUSC_10:
            cls_gts = gt_by_cls.get(c, [])
            cls_preds = preds_by_tok_class[tok][c]
            if len(cls_preds) == 0 and len(cls_gts) == 0:
                continue
            Pk, TP, FP, GTc = precision_at_k_for_class(cls_preds, cls_gts, args.K, args.thr)
            agg[c]['TP']  += TP
            agg[c]['FP']  += FP
            agg[c]['GT']  += GTc
            agg[c]['Ntok']+= 1

    # Print table
    print("\nPer-class Precision@K (distance-thr = {:.2f} m, K = {})".format(args.thr, args.K))
    print("{:<22s} {:>8s} {:>8s} {:>8s} {:>8s}".format("Class","P@K","TP","FP","GT"))
    total_TP=total_FP=total_GT=0
    for c in NUSC_10:
        TP = agg[c]['TP']; FP = agg[c]['FP']; GT = agg[c]['GT']
        denom = max(1, (TP+FP))
        Pk = TP/denom
        print("{:<22s} {:>8.3f} {:>8d} {:>8d} {:>8d}".format(c, Pk, TP, FP, GT))
        total_TP += TP; total_FP += FP; total_GT += GT
    denom = max(1, (total_TP+total_FP))
    overall = total_TP/denom
    print("{:<22s} {:>8.3f} {:>8d} {:>8d} {:>8d}".format("OVERALL", overall, total_TP, total_FP, total_GT))

    print("\nNotes:")
    print("- This is a quick precision probe (NOT nuScenes mAP).")
    print("- It matches by class and center-distance ≤ thr, greedy per token.")
    print("- Use it to tune score_thr / max_per_img / NMS before full mAP.")

if __name__ == "__main__":
    main()
