# tools/precision_probe_p50_p100.py
# Per-class Precision@50 and Precision@100 from a nuScenes results_nusc.json.
#
# Example (Windows):
#   set PYTHONPATH=%CD%;%PYTHONPATH%
#   python tools\precision_probe_p50_p100.py ^
#     --json test\bevformer_blob_iter2\20251030_221910\pts_bbox\results_nusc.json ^
#     --dataroot E:\datasets_all\nuscenes --version v1.0-mini --split val --thr 2.0

import argparse
from collections import defaultdict
import numpy as np
import mmcv

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
    """Return: (gt_by_token, valid_tokens_set). Each GT entry: {'name','center'} in GLOBAL coords."""
    from nuscenes.utils.splits import create_splits_scenes
    split_name = split
    scenes = set(create_splits_scenes()[split_name])
    scene_tokens = {s['token'] for s in nusc.scene if s['name'] in scenes}

    valid_sample_tokens = [s['token'] for s in nusc.sample if s['scene_token'] in scene_tokens]

    gt = defaultdict(list)
    for ann in nusc.sample_annotation:
        st = ann['sample_token']
        if st not in valid_sample_tokens:
            continue
        det_name = category_to_detection_name(ann['category_name'])
        if det_name is None or det_name not in NUSC_10:
            continue
        gt[st].append({'name': det_name, 'center': ann['translation'][:3]})
    return gt, set(valid_sample_tokens)

def precision_at_k_for_class(preds, gts, K, thr):
    """Greedy match by center-distance ≤ thr (meters), class already filtered."""
    gt_centers = [g['center'] for g in gts]
    gt_used = [False] * len(gt_centers)

    TP = FP = 0
    for p in preds[:K]:
        pc = p['translation']
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
    denom = max(1, min(K, len(preds)))
    return TP / denom, TP, FP, len(gt_centers)

def main():
    ap = argparse.ArgumentParser("Per-class Precision@50 & Precision@100 (distance-threshold matching)")
    ap.add_argument("--json", required=True, help="Path to results_nusc.json")
    ap.add_argument("--dataroot", required=True, help="nuScenes dataroot")
    ap.add_argument("--version", default="v1.0-mini", help="nuScenes version (e.g., v1.0-mini, v1.0-trainval)")
    ap.add_argument("--split", default="val", help="Split name for version (e.g., val, mini_val)")
    ap.add_argument("--thr", type=float, default=2.0, help="Distance threshold (meters) for TP")
    args = ap.parse_args()

    data = mmcv.load(args.json)
    res = data.get('results', {})
    print(f"[INFO] Loaded {len(res)} sample_tokens from JSON")

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    gt_by_tok, valid_tokens = load_gt_by_token(nusc, split=args.split)
    tokens = [t for t in res.keys() if t in valid_tokens]
    print(f"[INFO] Tokens in split & json: {len(tokens)}")

    # Build per-token, per-class predictions sorted by score desc
    preds_by_tok_cls = {}
    for tok in tokens:
        dets = res.get(tok, [])
        per_cls = {c: [] for c in NUSC_10}
        for d in dets:
            name = d.get('detection_name')
            if name in per_cls:
                per_cls[name].append(d)
        for c in NUSC_10:
            per_cls[c].sort(key=lambda x: float(x.get('detection_score', 0.0)), reverse=True)
        preds_by_tok_cls[tok] = per_cls

    # Aggregate P@50 and P@100
    Ks = [50, 100]
    agg = {c: {k: {'TP':0,'FP':0,'GT':0,'Ntok':0} for k in Ks} for c in NUSC_10}
    for tok in tokens:
        gts = gt_by_tok.get(tok, [])
        if not gts:
            continue
        gt_by_cls = defaultdict(list)
        for g in gts:
            if g['name'] in NUSC_10:
                gt_by_cls[g['name']].append(g)

        for c in NUSC_10:
            cls_gts = gt_by_cls.get(c, [])
            cls_preds = preds_by_tok_cls[tok][c]
            if len(cls_preds) == 0 and len(cls_gts) == 0:
                continue
            for K in Ks:
                Pk, TP, FP, GTc = precision_at_k_for_class(cls_preds, cls_gts, K, args.thr)
                agg[c][K]['TP']  += TP
                agg[c][K]['FP']  += FP
                agg[c][K]['GT']  += GTc
                agg[c][K]['Ntok']+= 1

    # Print table
    print("\nPer-class Precision (thr = {:.2f} m)".format(args.thr))
    header = "{:<22s} {:>10s} {:>10s}   {:>8s} {:>8s} | {:>8s} {:>8s}".format(
        "Class", "P@50", "P@100", "TP@50", "FP@50", "TP@100", "FP@100"
    )
    print(header)
    print("-"*len(header))

    totals = {50:{'TP':0,'FP':0}, 100:{'TP':0,'FP':0}}
    total_GT = 0

    for c in NUSC_10:
        TP50 = agg[c][50]['TP']; FP50 = agg[c][50]['FP']; GTc = agg[c][50]['GT']
        TP100= agg[c][100]['TP']; FP100= agg[c][100]['FP']
        P50  = TP50 / max(1, TP50+FP50)
        P100 = TP100 / max(1, TP100+FP100)
        print("{:<22s} {:>10.3f} {:>10.3f}   {:>8d} {:>8d} | {:>8d} {:>8d}".format(
            c, P50, P100, TP50, FP50, TP100, FP100
        ))
        totals[50]['TP']  += TP50;  totals[50]['FP']  += FP50
        totals[100]['TP'] += TP100; totals[100]['FP'] += FP100
        total_GT          += GTc

    P50_overall  = totals[50]['TP']  / max(1, totals[50]['TP']  + totals[50]['FP'])
    P100_overall = totals[100]['TP'] / max(1, totals[100]['TP'] + totals[100]['FP'])
    print("-"*len(header))
    print("{:<22s} {:>10.3f} {:>10.3f}   {:>8d} {:>8d} | {:>8d} {:>8d}".format(
        "OVERALL", P50_overall, P100_overall,
        totals[50]['TP'], totals[50]['FP'],
        totals[100]['TP'], totals[100]['FP']
    ))

    print("\nNotes:")
    print("- This is a quick precision probe (NOT nuScenes mAP).")
    print("- Matching by class + center-distance ≤ thr (greedy per token).")
    print("- Use it to tune score_thr / max_per_img / NMS; aim to raise P@50/P@100.")

if __name__ == "__main__":
    main()
