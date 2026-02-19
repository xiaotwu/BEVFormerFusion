# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import time
import copy
import numpy as np
import mmdet3d
from projects.mmdet3d_plugin.models.utils.bricks import run_time
import os
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models.builder import (
    build_voxel_encoder, build_middle_encoder, build_backbone, build_neck
)
from mmcv.ops import Voxelization

@DETECTORS.register_module()
class BEVFormer(MVXTwoStageDetector):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False
                 ):

        super(BEVFormer,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

        self.use_prev_bev = getattr(self, 'use_prev_bev', True)
        self._bev_cache = {}      # key -> tensor (H*W, C) in BF layout
        self._last_scene = {}     # idx -> scene_token seen last time

        # --- temporal BEV memory ---
        self._use_prev_bev = bool(
            getattr(getattr(self.pts_bbox_head, "transformer", None), "encoder", None)
            and getattr(self.pts_bbox_head.transformer.encoder, "use_prev_bev", False)
        )
        # Cache BEV per scene; works with DistributedDataParallel since it’s per-process.
        self._prev_bev_cache = {}   # { scene_token: tensor of shape (B, H*W, C) or (B, C, H, W) depending on your head }

        self.prev_bev_buffer = None
        self._prev_scene_token = None

        # ---- PointPillars LiDAR BEV branch ----
        self.voxel_layer = None
        if pts_voxel_layer is not None:
            self.voxel_layer = Voxelization(**pts_voxel_layer)

        # Build / override PointPillars components
        if pts_voxel_encoder is not None:
            self.pts_voxel_encoder = build_voxel_encoder(pts_voxel_encoder)
        if pts_middle_encoder is not None:
            self.pts_middle_encoder = build_middle_encoder(pts_middle_encoder)
        if pts_backbone is not None:
            self.pts_backbone = build_backbone(pts_backbone)
        if pts_neck is not None:
            self.pts_neck = build_neck(pts_neck)

    def voxelize(self, points):
        """points: list[Tensor], each (Ni, C)"""
        assert self.voxel_layer is not None, "voxel_layer is None; check pts_voxel_layer in config"
        voxels, coors, num_points = [], [], []
        for i, res in enumerate(points):
            v, c, n = self.voxel_layer(res)
            voxels.append(v)
            num_points.append(n)
            # prepend batch index
            c = torch.cat([c.new_full((c.size(0), 1), i), c], dim=1)
            coors.append(c)
        return torch.cat(voxels, 0), torch.cat(coors, 0), torch.cat(num_points, 0)            

    def extract_pts_bev_feat(self, points):
        """Return LiDAR BEV feature map (B, C, H, W)."""
        if points is None:
            return None
        if self.voxel_layer is None or self.pts_voxel_encoder is None or self.pts_middle_encoder is None:
            return None

        # ---- normalize points into list[Tensor (Ni, C)] ----
        # Common cases:
        #  - points is a list of length B, each is Tensor
        #  - points is a list of length B, each is BasePoints (has .tensor)
        #  - points is double-nested (e.g., [ [Tensor], [Tensor] ] ) due to test-time wrappers
        if isinstance(points, (list, tuple)):
            # unwrap one level if needed
            if len(points) > 0 and isinstance(points[0], (list, tuple)):
                # If it looks like [[tensor]] per sample, take the first element
                points = [p[0] if len(p) > 0 else p for p in points]

            pts_list = []
            for p in points:
                if p is None:
                    pts_list.append(None)
                    continue
                # BasePoints or similar
                if hasattr(p, "tensor"):
                    p = p.tensor
                # sometimes still nested
                if isinstance(p, (list, tuple)) and len(p) > 0:
                    p = p[0]
                    if hasattr(p, "tensor"):
                        p = p.tensor
                if not torch.is_tensor(p):
                    raise TypeError(f"Unsupported points element type: {type(p)}")
                pts_list.append(p)
            points = pts_list
        else:
            # single Tensor -> wrap
            if hasattr(points, "tensor"):
                points = points.tensor
            if torch.is_tensor(points):
                points = [points]
            else:
                raise TypeError(f"Unsupported points container type: {type(points)}")

        # filter out None samples (shouldn't happen with normal pipelines)
        if any(p is None for p in points):
            return None

        voxels, coors, num_points = self.voxelize(points)
        batch_size = len(points)

        x = self.pts_voxel_encoder(voxels, num_points, coors)
        x = self.pts_middle_encoder(x, coors, batch_size)  # (B, C, H, W)
        return x

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats


    def forward_pts_train(self,
                        pts_feats,
                        gt_bboxes_3d,
                        gt_labels_3d,
                        img_metas,
                        gt_bboxes_ignore=None,
                        prev_bev=None,
                        points=None):
        """
        Train forward for the pts branch (BEVFormer head), with optional LiDAR BEV.
        Args:
            pts_feats: image features from extract_feat (list of multi-level feats)
            gt_bboxes_3d, gt_labels_3d: GT
            img_metas: list[dict]
            prev_bev: previous BEV embedding for temporal
            points: list[Tensor] or None, each (Ni, C) for PointPillars
        """
        import inspect, os
        if not hasattr(self, "_DBG_FILE_ONCE"):
            print("[DET/DBG] detector class file =", inspect.getfile(self.__class__))
            print("[DET/DBG] detector cwd =", os.getcwd())
            self._DBG_FILE_ONCE = True

        # 2.1 Pull prev_bev from our cache unless the caller explicitly passed one
        if prev_bev is None:
            prev_bev = self._pop_prev_bev(img_metas)
        self._dbg_print_prev("got", prev_bev)  # optional

        if not hasattr(self, "_dbg_prev_once"):
            print(f"[DET/CHECK] prev_bev is None? {prev_bev is None}")
            if isinstance(prev_bev, torch.Tensor):
                print(f"[DET/CHECK] prev_bev shape={tuple(prev_bev.shape)} dtype={prev_bev.dtype}")
            self._dbg_prev_once = True

        # 2.2 Build LiDAR BEV (PointPillars scatter output), if points are provided
        bev_lidar = None
        try:
            bev_lidar = self.extract_pts_bev_feat(points)
        except Exception as e:
            print("[LIDAR/ERR] extract_pts_bev_feat failed:", repr(e))
            raise   # <-- for now, crash so we don't silently disable LiDAR

        # [DBG] Detector: confirm LiDAR BEV existence before passing to head (print once)
        if not hasattr(self, "_dbg_lidar_det_once"):
            print("[DET/DBG] points is None?", points is None)
            print("[DET/DBG] bev_lidar computed is None?", bev_lidar is None)
            if bev_lidar is not None:
                print("[DET/DBG] bev_lidar shape:", tuple(bev_lidar.shape), bev_lidar.dtype)
            self._dbg_lidar_det_once = True

        # Optional: one-time debug for bev_lidar
        if not hasattr(self, "_dbg_lidar_once"):
            if bev_lidar is None:
                print("[DET/CHECK] bev_lidar is None (camera-only path).")
            else:
                print(f"[DET/CHECK] bev_lidar shape={tuple(bev_lidar.shape)} dtype={bev_lidar.dtype}")
            self._dbg_lidar_once = True

        # 2.3 Run the head; pass prev_bev through
        # Some heads won't accept bev_lidar kwarg -> fallback gracefully.
        #bev_lidar = None

        #print("[DET/DBG] will pass bev_lidar?", bev_lidar is not None)

        if not hasattr(self, "_dbg_lidar_flow_once"):
            print("[DET/DBG] bev_lidar computed is None?", bev_lidar is None)
            if bev_lidar is not None:
                print("[DET/DBG] bev_lidar shape:", tuple(bev_lidar.shape), bev_lidar.dtype)
            self._dbg_lidar_flow_once = True

        #bev_lidar = None

        try:
            outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev, bev_lidar=bev_lidar)
        except TypeError:
            # Old signature: (x, img_metas, prev_bev)
            outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev)

        # 2.4 Update the cache with the BEV from this forward for the next frame
        # Your head/transformer returns outs that may include 'bev_embed'
        if isinstance(outs, dict) and 'bev_embed' in outs:
            self._push_prev_bev(img_metas, outs['bev_embed'])
            self._dbg_print_prev("push", outs['bev_embed'])  # optional

        # 2.5 Compute losses (unchanged)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses


    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated."""
        # Run the whole model in eval mode for history passes
        was_training = self.training
        self.eval()

        # --- BEGIN: cuDNN/AMP safe context for tiny-batch BN ---
        import torch
        from contextlib import ExitStack

        with torch.no_grad(), ExitStack() as stack:
            # Make cuDNN deterministic and avoid odd kernels for BN/Conv at bs=1
            stack.enter_context(torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True))
            # Disable autocast to force FP32 in the backbone during history building
            stack.enter_context(torch.cuda.amp.autocast(enabled=False))

            # Sanitize input layout/dtype for the backbone (expects NCHW float32)
            assert imgs_queue.dim() == 6, f"imgs_queue shape should be (B, T, Ncam, C, H, W), got {tuple(imgs_queue.shape)}"
            imgs_queue = imgs_queue.contiguous().to(dtype=torch.float32)

            # Optional: zero any NaN/Inf just in case
            if not torch.isfinite(imgs_queue).all():
                imgs_queue = torch.nan_to_num(imgs_queue, nan=0.0, posinf=0.0, neginf=0.0)

            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            # (B, T, N, C, H, W) -> (B*T, N, C, H, W)
            imgs_queue = imgs_queue.view(bs * len_queue, num_cams, C, H, W).contiguous()

            # Backbone+neck extraction in FP32 and deterministic cuDNN
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)

            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0].get('prev_bev_exists', True):
                    prev_bev = None

                # Per-scale features for the i-th timestep
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]

                # Only BEV (encoder only) – still inside FP32/no-grad
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True
                )


        # Restore mode
        if was_training:
            self.train()
        return prev_bev

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                    points=None,
                    img_metas=None,
                    gt_bboxes_3d=None,
                    gt_labels_3d=None,
                    gt_labels=None,
                    gt_bboxes=None,
                    img=None,
                    proposals=None,
                    gt_bboxes_ignore=None,
                    img_depth=None,
                    img_mask=None,
                    **kwargs):
        """
        Override forward_train so we can pass `points` into forward_pts_train()
        and keep BEVFormer temporal behavior when img is a queue.
        """

        assert img is not None, "img must be provided"
        assert img_metas is not None, "img_metas must be provided"

        # ---- BEVFormer temporal queue case: img is (B, T, Ncam, C, H, W) ----
        if img.dim() == 6:
            len_queue = img.size(1)

            prev_img = img[:, :-1, ...]   # history frames
            cur_img  = img[:, -1, ...]    # current frame

            prev_img_metas = copy.deepcopy(img_metas)
            prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

            # use metas of current frame only
            cur_img_metas = [each[len_queue - 1] for each in img_metas]
            if not cur_img_metas[0].get('prev_bev_exists', True):
                prev_bev = None

            img_feats = self.extract_feat(img=cur_img, img_metas=cur_img_metas)

            losses_pts = self.forward_pts_train(
                img_feats,
                gt_bboxes_3d,
                gt_labels_3d,
                cur_img_metas,
                gt_bboxes_ignore,
                prev_bev,
                points=points
            )
            return losses_pts

        # ---- Single-frame case ----
        else:
            prev_bev = None
            img_feats = self.extract_feat(img=img, img_metas=img_metas)

            losses_pts = self.forward_pts_train(
                img_feats,
                gt_bboxes_3d,
                gt_labels_3d,
                img_metas,
                gt_bboxes_ignore,
                prev_bev,
                points=points
            )
            return losses_pts

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        return bbox_results

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False, points=None):
        """Test function (pts branch) with optional LiDAR points."""

        if not hasattr(self, "_dbg_test_once"):
            print("[TEST] points is None?", points is None)
            self._dbg_test_once = True

        bev_lidar = None
        try:
            bev_lidar = self.extract_pts_bev_feat(points)
        except Exception as e:
            print(f"[WARN] extract_pts_bev_feat failed in test, running camera-only. Reason: {repr(e)}")
            bev_lidar = None

        #bev_lidar = None
        # Head may or may not accept bev_lidar; fallback safely.
        try:
            outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev, bev_lidar=bev_lidar)
        except TypeError:
            outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        
        #Evaluation debug
        bboxes, scores, labels = bbox_list[0]   # from get_bboxes
        bt = bboxes.tensor  # (N, box_dim)
        print("[PRED] N =", bt.size(0),
            "score min/mean/max =", float(scores.min()), float(scores.mean()), float(scores.max()))
        print("[PRED] finite ratio =", torch.isfinite(bt).all(-1).float().mean().item())

        # centers
        cent = bboxes.gravity_center
        print("[PRED] center xyz mean =", cent.mean(0).tolist(),
            "min =", cent.min(0).values.tolist(),
            "max =", cent.max(0).values.tolist())

        # compare to pc_range
        pc = self.pts_bbox_head.pc_range  # or from coder
        inside = (cent[:,0] > pc[0]) & (cent[:,0] < pc[3]) & (cent[:,1] > pc[1]) & (cent[:,1] < pc[4])
        print("[PRED] inside pc_range ratio =", inside.float().mean().item())
        # Evaluation bug test end

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs['bev_embed'], bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False, points=None, **kwargs):
        """Test function without augmentation."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for _ in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev=prev_bev, rescale=rescale, points=points
        )
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list

    def _scene_key(self, img_metas):
        # NuScenes-style: many repos store these; use what your pipeline provides.
        # Fallback to a single key if missing.
        m0 = img_metas[0]
        return m0.get('scene_token') or m0.get('sequence_name') or "default_scene"

    def _is_sequence_reset(self, img_metas):
        m0 = img_metas[0]
        # Common flags seen in BEVFormer/NuScenes loaders; keep all three as fallbacks.
        return bool(
            m0.get('prev_bev_is_none', False) or
            m0.get('frame_id', 0) in (0, '0') or
            m0.get('scene_change', False)
        )

    def _pop_prev_bev(self, img_metas):
        if not self._use_prev_bev:
            return None
        key = self._scene_key(img_metas)
        if self._is_sequence_reset(img_metas):
            # reset at beginning of a new clip/scene
            self._prev_bev_cache.pop(key, None)
            return None
        return self._prev_bev_cache.get(key, None)

    @torch.no_grad()
    def _push_prev_bev(self, img_metas, bev_embed):
        if not self._use_prev_bev:
            return
        key = self._scene_key(img_metas)
        # Store a detached copy to avoid holding the whole graph.
        self._prev_bev_cache[key] = bev_embed.detach()

    def _dbg_print_prev(self, tag, prev):
        if getattr(self, "_dbg_prev_once", True):
            print(f"[DET/PREV] {tag}:",
                "None" if prev is None else f"shape={tuple(prev.shape)} dtype={prev.dtype}")
            self._dbg_prev_once = False
