
# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform
import random
from functools import partial

import numpy as np
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg
from torch.utils.data import DataLoader

from mmdet.datasets.samplers import GroupSampler
from projects.mmdet3d_plugin.datasets.samplers.group_sampler import DistributedGroupSampler
from projects.mmdet3d_plugin.datasets.samplers.distributed_sampler import DistributedSampler
from projects.mmdet3d_plugin.datasets.samplers.sampler import build_sampler

# projects/mmdet3d_plugin/datasets/builder.py

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from mmcv.runner import get_dist_info
from functools import partial
from mmcv.parallel import collate as mmcv_collate
from torch.utils.data import DataLoader

# --- add near top of builder.py ---
import pickle

def _find_unpicklable(obj, prefix=""):
    try:
        pickle.dumps(obj)
        return None
    except Exception as e:
        # scan dict-like __dict__
        if hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                bad = _find_unpicklable(v, prefix + f".{k}")
                if bad: return bad
        # scan mappings
        if isinstance(obj, dict):
            for k, v in obj.items():
                bad = _find_unpicklable(v, prefix + f"[{k!r}]")
                if bad: return bad
        # scan sequences
        if isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                bad = _find_unpicklable(v, prefix + f"[{i}]")
                if bad: return bad
        # last resort: report this node
        return f"{prefix} ({type(obj).__name__}): {e}"


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     seed=None,
                     shuffler_sampler=None,
                     nonshuffler_sampler=None,
                     **loader_kwargs):

    rank, world_size = get_dist_info()
    is_train = getattr(dataset, 'test_mode', False) is False

    # Choose which sampler config to use
    sampler_cfg = shuffler_sampler if is_train else nonshuffler_sampler
    sampler = None

    if dist:
        if sampler_cfg is None or sampler_cfg.get('type', '') == 'DistributedSampler':
            # Plain DistributedSampler (no unsupported kwargs)
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=is_train,
                drop_last=is_train,   # drop last only for train
                # seed is supported on newer torch; if yours errors, remove it
                **({} if seed is None else {'seed': seed})
            )
        elif sampler_cfg.get('type', '') in ('DistributedGroupSampler', 'GroupSampler'):
            # If you have a custom group sampler, import and construct it here,
            # but DO NOT forward samples_per_gpu into DistributedSampler.
            from .samplers.group_sampler import DistributedGroupSampler
            # Remove unsupported keys from cfg copy
            sc = dict(sampler_cfg)
            sc.pop('samples_per_gpu', None)
            sampler = DistributedGroupSampler(
                dataset=dataset,
                num_replicas=world_size,
                rank=rank,
                samples_per_gpu=samples_per_gpu,  # if your custom sampler needs it
                seed=seed if seed is not None else 0
            )
        else:
            # Fallback: plain DistributedSampler
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=is_train,
                drop_last=is_train,
                **({} if seed is None else {'seed': seed})
            )
    else:
        sampler = None  # single-process

    # Build the PyTorch DataLoader. Pass samples_per_gpu here (NOT to sampler).
    # Remove kwargs that DataLoader doesn’t accept twice (avoid multiple values).
    for k in ('sampler', 'shuffle', 'drop_last', 'persistent_workers'):
        loader_kwargs.pop(k, None)

    def _materialize_views(x):
        if isinstance(x, type({}.keys())) or isinstance(x, type({}.values())) or isinstance(x, type({}.items())):
            return list(x)
        if isinstance(x, dict):
            return {k: _materialize_views(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            t = [_materialize_views(v) for v in x]
            return type(x)(t)
        return x

    loader_kwargs = _materialize_views(loader_kwargs)

    bad = _find_unpicklable(dataset)
    if bad:
        print(f"[PICKLE] Dataset has an unpicklable field at: {bad}", flush=True)
        # optional: raise to stop early
        # raise RuntimeError(bad)
    cfg = getattr(dataset, 'eval_detection_configs', None)
    if cfg is not None and hasattr(cfg, 'class_names') and not isinstance(cfg.class_names, list):
        try:
            cfg.class_names = list(cfg.class_names)
        except TypeError:
            pass


    data_loader = DataLoader(
        dataset,
        batch_size=samples_per_gpu,
        num_workers=workers_per_gpu,
        sampler=sampler,
        shuffle=(sampler is None and is_train),
        drop_last=is_train,
        persistent_workers=False,
        pin_memory=True,
        collate_fn=partial(mmcv_collate, samples_per_gpu=samples_per_gpu)  # <<< add this
    )
    return data_loader



def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Copyright (c) OpenMMLab. All rights reserved.
import platform
from mmcv.utils import Registry, build_from_cfg

from mmdet.datasets import DATASETS
from mmdet.datasets.builder import _concat_dataset

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

OBJECTSAMPLERS = Registry('Object sampler')


def custom_build_dataset(cfg, default_args=None):
    from mmdet3d.datasets.dataset_wrappers import CBGSDataset
    from mmdet.datasets.dataset_wrappers import (ClassBalancedDataset,
                                                 ConcatDataset, RepeatDataset)
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([custom_build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [custom_build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            custom_build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            custom_build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif cfg['type'] == 'CBGSDataset':
        dataset = CBGSDataset(custom_build_dataset(cfg['dataset'], default_args))
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset
