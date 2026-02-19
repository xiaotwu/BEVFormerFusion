# projects/mmdet3d_plugin/train.py
def _resolve_trainer():
    """
    Pick a training entrypoint that exists in this environment.
    Supports legacy MMDet3D and classic MMDet (no MMEngine).
    """
    # 1) Preferred: mmdet3d.apis.train_model
    try:
        from mmdet3d.apis import train_model as fn
        return fn
    except Exception:
        pass
    # 2) Legacy alt: mmdet3d.apis.train_detector
    try:
        from mmdet3d.apis import train_detector as fn
        return fn
    except Exception:
        pass
    # 3) Fallbacks via MMDet 2D (some forks use these)
    try:
        from mmdet.apis import train_model as fn
        return fn
    except Exception:
        pass
    from mmdet.apis import train_detector as fn
    return fn

_TRAIN = _resolve_trainer()

def custom_train_model(*args, **kwargs):
    """Thin wrapper so tools/train.py can call plugin-specific trainer if present."""
    return _TRAIN(*args, **kwargs)
