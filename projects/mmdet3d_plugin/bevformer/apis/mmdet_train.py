# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import random
import warnings
from mmdet3d.datasets import build_dataset as build_dataset_3d

import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner, get_dist_info)
from mmcv.utils import build_from_cfg

from mmdet.core import EvalHook

from mmdet.datasets import (build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_root_logger
import time
import os.path as osp
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.core.evaluation.eval_hooks import CustomDistEvalHook

#from projects.mmdet3d_plugin.datasets import custom_build_dataset
def custom_train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   eval_model=None,
                   meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders (distributed + explicit samplers, Windows-safe)
    from mmcv.runner import get_dist_info
    from mmdet.datasets.samplers import DistributedGroupSampler, DistributedSampler
    from mmdet3d.datasets import build_dataset

    # support both styles (passed-in dataset or build from cfg)
    if not isinstance(dataset, (list, tuple)):
        dataset = [dataset]
    ds0 = dataset[0] if dataset and dataset[0] is not None else build_dataset(cfg.data.train)

    rank, world_size = get_dist_info()

    # Use GroupSampler if dataset has .flag
    if getattr(ds0, 'flag', None) is not None:
        sampler = DistributedGroupSampler(
            dataset=ds0,
            samples_per_gpu=cfg.data.samples_per_gpu,
            num_replicas=world_size,      # <— was world_size=...
            rank=rank,
            seed=getattr(cfg, 'seed', 0)
        )

    # prepare data loaders (distributed, Windows-safe, no explicit sampler kw)
    if not isinstance(dataset, (list, tuple)):
        dataset = [dataset]

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            len(cfg.gpu_ids),              # num_gpus; ignored when distributed=True
            dist=distributed,
            seed=cfg.seed,
            shuffle=False,                  # sampler will handle shuffling
            drop_last=True,                 # avoid leftover shard on rank0
            persistent_workers=False,       # Windows stability
            # IMPORTANT: keep using your repo's sampler configs
            shuffler_sampler=cfg.data.shuffler_sampler,       # e.g., {'type': 'DistributedGroupSampler'}
            nonshuffler_sampler=cfg.data.nonshuffler_sampler  # e.g., {'type': 'DistributedSampler'}
        ) for ds in dataset
    ]


    if distributed:
        import os, torch
        local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', 0)))
        try:
            if getattr(cfg, 'gpu_ids', None):
                torch.cuda.set_device(cfg.gpu_ids[local_rank])
            else:
                torch.cuda.set_device(local_rank)
        except Exception:
            torch.cuda.set_device(local_rank)

        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters= True #find_unused_parameters
        )
    else:
        model = MMDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)


    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs
            
    from mmcv.runner.hooks import IterTimerHook

    if eval_model is not None:
        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                eval_model=eval_model,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta))
    else:
        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta))

    # --- ensure IterTimerHook is present so 'time' and 'data_time' exist ---
    has_timer = any(isinstance(h, IterTimerHook) for h in runner._hooks)
    if not has_timer:
        # VERY_HIGH so it runs before logging hooks
        runner.register_hook(IterTimerHook(), priority='VERY_HIGH')
    # ----------------------------------------------------------------------

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp


    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    
    # register profiler hook
    #trace_config = dict(type='tb_trace', dir_name='work_dir')
    #profiler_config = dict(on_trace_ready=trace_config)
    #runner.register_profiler_hook(profiler_config)
    
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    if validate:
        # ---- define val_samples_per_gpu safely ----
        from mmdet3d.datasets import build_dataset as build_dataset_3d
        from mmdet.datasets.samplers import DistributedSampler
        from mmcv.runner import get_dist_info

        val_cfg = cfg.data.val.copy()
        val_samples_per_gpu = val_cfg.pop('samples_per_gpu', 1)
        val_cfg['test_mode'] = True
        val_dataset = build_dataset_3d(val_cfg)

        rank, world_size = get_dist_info()

        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            drop_last=False,
            persistent_workers=False,
            shuffler_sampler=cfg.data.shuffler_sampler,         # your repo’s pathway
            nonshuffler_sampler=cfg.data.nonshuffler_sampler
        )

        # ---- eval hook ----
        eval_cfg = cfg.get('evaluation', {}).copy()
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_cfg.setdefault(
            'jsonfile_prefix',
            osp.join('val', cfg.work_dir, time.ctime().replace(' ', '_').replace(':', '_'))
        )
        eval_hook_cls = CustomDistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook_cls(val_dataloader, **eval_cfg))


    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)

