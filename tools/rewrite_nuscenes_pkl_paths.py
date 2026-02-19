# rewrite_nuscenes_pkl_paths.py
import os, pickle, sys

old_root = "/mnt/d/datasets/nuscenes"
new_root = "E:/datasets_all/nuscenes"  # use forward slashes; Python handles them on Windows

def fix_path(p):
    if isinstance(p, str) and p.startswith(old_root):
        return new_root + p[len(old_root):]
    return p

def patch_file(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    # BEVFormer/mmdet3d infos are typically list[dict]
    changed = 0
    for item in data:
        # common keys the pipelines read
        for k in ("cam_paths", "cams", "img_paths", "filename", "lidar_path"):
            if k in item:
                if isinstance(item[k], dict):
                    for kk, vv in item[k].items():
                        # nested dicts often have 'data_path' or 'img_path'
                        if isinstance(vv, dict):
                            for sk, sv in vv.items():
                                nv = fix_path(sv)
                                if nv != sv:
                                    vv[sk] = nv; changed += 1
                        else:
                            nv = fix_path(vv)
                            if nv != vv:
                                item[k][kk] = nv; changed += 1
                elif isinstance(item[k], list):
                    item[k] = [fix_path(x) for x in item[k]]
                    changed += sum(1 for x in item[k] if isinstance(x, str) and x.startswith(new_root))
                elif isinstance(item[k], str):
                    nv = fix_path(item[k])
                    if nv != item[k]:
                        item[k] = nv; changed += 1
        # some infos store per-camera dict like item['cams'][cam]['data_path']
        if "cams" in item and isinstance(item["cams"], dict):
            for cam, meta in item["cams"].items():
                if isinstance(meta, dict) and "data_path" in meta:
                    nv = fix_path(meta["data_path"])
                    if nv != meta["data_path"]:
                        meta["data_path"] = nv; changed += 1

    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Patched {pkl_path}, changed {changed} paths.")

if __name__ == "__main__":
    # pass all your info PKLs here
    for p in sys.argv[1:]:
        patch_file(p)
