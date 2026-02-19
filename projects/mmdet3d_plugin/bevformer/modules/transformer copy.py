# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate
from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import MSDeformableAttention3D
from .decoder import CustomMSDeformableAttention
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from mmcv.runner import force_fp32, auto_fp16
import math
import inspect
import torch.nn.functional as F
from mmcv.runner import auto_fp16

@TRANSFORMER.register_module()
class PerceptionTransformer(BaseModule):
    """Implements the Detr3D/BEVFormer transformer with optional PETR3D fusion.
    PETR3D now *augments* BEV queries (positional + FiLM), while BEV references remain 2D from BEV grid.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=3,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 use_prev_bev: bool = False,
                 rotate_center=[50, 50],
                 pc_range: list = None,
                 lidar_in_channels=64,      # ✅ add
                 lidar_gate_init=-2.0,      # ✅ add
                 **kwargs):
        super(PerceptionTransformer, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.num_feature_levels = num_feature_levels
        # right after self.encoder/self.decoder init in __init__:
        self.num_feature_levels = num_feature_levels

        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds
        self.use_prev_bev = use_prev_bev
        if not hasattr(self, "_dbg_prev_flag_once"):
            print(f"[TRANS] use_prev_bev flag = {self.use_prev_bev}")
            self._dbg_prev_flag_once = True


        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center
        self.pc_range = pc_range

    def init_layers(self):
        """Initialize layers of the transformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

        # ---- LiDAR fusion v3 (concat + linear) ----
        self.lidar_gate = nn.Parameter(torch.tensor(-2.0))   # sigmoid ~ 0.12 initially
        self.lidar_proj = nn.Conv2d(64, self.embed_dims, kernel_size=1)  # 64 is PillarFeatureNet out (your case)
        self.lidar_fuse_linear = nn.Linear(self.embed_dims * 2, self.embed_dims)
        self.lidar_fuse_norm = nn.LayerNorm(self.embed_dims)

    def _ensure_level_embeds(self, num_lvls: int):
        """Make self.level_embeds match the incoming number of FPN levels."""
        if num_lvls == self.num_feature_levels:
            return
        old = self.level_embeds
        C = old.size(1)
        if num_lvls > self.num_feature_levels:
            # grow: keep old rows, initialize new rows
            extra = old.new_empty((num_lvls - self.num_feature_levels, C))
            nn.init.normal_(extra)                       # same init you use elsewhere
            new_param = torch.nn.Parameter(torch.cat([old, extra], dim=0))
        else:
            # shrink
            new_param = torch.nn.Parameter(old[:num_lvls].clone())
        self.level_embeds = new_param
        self.num_feature_levels = num_lvls

    # --- NEW: helpers for PETR→BEV pooling ---
    @staticmethod
    def _make_bev_grid_xy(bev_h, bev_w, device):
        # normalized [0,1] xy grid centers; order=(x,y)
        ys = torch.linspace(0.5 / bev_h, 1 - 0.5 / bev_h, bev_h, device=device)
        xs = torch.linspace(0.5 / bev_w, 1 - 0.5 / bev_w, bev_w, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        xy = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)  # (BHW, 2)
        return xy

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        #xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)
        # can_bus_mlp: init each Linear inside it (safer than xavier_init on Sequential)
        for m in self.can_bus_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

        # 4) ---- LiDAR fusion v3: force intended init AFTER the global loop ----
        if hasattr(self, "lidar_gate"):
            nn.init.constant_(self.lidar_gate, -2.0)   # sigmoid ~ 0.119 initially

        if hasattr(self, "lidar_proj"):
            nn.init.xavier_uniform_(self.lidar_proj.weight)
            if self.lidar_proj.bias is not None:
                nn.init.constant_(self.lidar_proj.bias, 0.)

        if hasattr(self, "lidar_fuse_linear"):
            nn.init.xavier_uniform_(self.lidar_fuse_linear.weight)
            if self.lidar_fuse_linear.bias is not None:
                nn.init.constant_(self.lidar_fuse_linear.bias, 0.)

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def get_bev_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[1.02, 1.02],
            bev_pos=None,
            prev_bev=None,
            **kwargs):

        # --- make level embeds match the runtime FPN outputs ---
        if isinstance(mlvl_feats, (list, tuple)):
            num_lvls_runtime = len(mlvl_feats)
        else:
            num_lvls_runtime = 1
            mlvl_feats = [mlvl_feats]  # normalize to list for downstream code

        self._ensure_level_embeds(num_lvls_runtime)

        # Guard: encoder deformable attention must agree with FPN num levels
        try:
            for m in self.modules():
                if isinstance(m, MSDeformableAttention3D):  # encoder-side
                    assert m.num_levels == self.num_feature_levels, \
                        f"Encoder deformable num_levels={m.num_levels} != FPN levels={self.num_feature_levels}"
        except AssertionError as e:
            raise RuntimeError(str(e) + " — set MSDeformableAttention3D.num_levels to match FPN num_outs.")
        

        # DEBUG: what did we get from the detector?
        if not hasattr(self, "_dbg_bev_feat_prev_once"):
            print("[PERCEP] get_bev_features: incoming prev_bev =",
                  "None" if prev_bev is None else tuple(prev_bev.shape))
            print("[PERCEP] use_prev_bev flag:", getattr(self, "use_prev_bev", None))
            self._dbg_bev_feat_prev_once = True

        # --- standard BEVFormer prep ---
        if not getattr(self, 'use_prev_bev', False):
            print("[PERCEP] use_prev_bev=False -> dropping prev_bev before encoder")
            prev_bev = None
        else:
            if not hasattr(self, "_dbg_bev_feat_keep_once"):
                print("[PERCEP] use_prev_bev=True -> keeping prev_bev for encoder")
                self._dbg_bev_feat_keep_once = True

        bs = mlvl_feats[0].size(0)

        # keep a pristine copy *before* flattening
        bev_pos_in   = bev_pos                           # (B, C, H, W)
        bev_pos_used = bev_pos_in.flatten(2).permute(2, 0, 1)   # (BHW, B, C)

        # ego motion -> shift
        delta_x = np.array([each['can_bus'][0] for each in kwargs['img_metas']])
        delta_y = np.array([each['can_bus'][1] for each in kwargs['img_metas']])
        ego_angle = np.array([each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift_np = np.stack([shift_x, shift_y], axis=-1).astype(np.float32)  # (bs, 2)
        shift = torch.from_numpy(shift_np).to(device=bev_pos_used.device, dtype=bev_pos_used.dtype)

        # rotate previous BEV if provided
        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    if torch.is_tensor(rotation_angle):
                        rotation_angle = float(rotation_angle.detach().cpu().item())
                    tmp_prev_bev = prev_bev[:, i].reshape(bev_h, bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle, center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add CAN bus
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)   # (BHW, B, C)
        can_bus = bev_queries.new_tensor([each['can_bus'] for each in kwargs['img_metas']])
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]           # (1, B, C)
        bev_queries = bev_queries + can_bus * self.use_can_bus

        # flatten FPN levels
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)  # (num_cam, B, HW, C)
            if self.use_cams_embeds:
                B0, N0, _, _, _ = mlvl_feats[0].shape
                cam_emb = self.cams_embeds
                if cam_emb.size(0) != N0:
                    if not hasattr(self, "_warned_num_cams"):
                        print(f"[WARN] cams_embeds rows={cam_emb.size(0)} but batch has N={N0}; slicing/tiling at runtime.")
                        self._warned_num_cams = True
                    cam_emb = cam_emb[:N0] if cam_emb.size(0) >= N0 else torch.cat(
                        [cam_emb, cam_emb[-1:].expand(N0 - cam_emb.size(0), -1)], dim=0)
                feat = feat + cam_emb[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)  # (num_cam, sum(HW), B, C)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_pos_used.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W_all, B, C)

        # --- encoder ---
        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos_used,                # <-- PETR pos (replaced) or original
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs
        )
        return bev_embed

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                grid_length=[1.02, 1.02],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                bev_lidar=None,   # ✅ comes from head/detector
                **kwargs):

        if not hasattr(self, "_dbg_lidar_tr_once"):
            print("[TR/DBG] bev_lidar is None?", bev_lidar is None)
            if bev_lidar is not None:
                print("[TR/DBG] bev_lidar shape:", tuple(bev_lidar.shape), bev_lidar.dtype)
            self._dbg_lidar_tr_once = True

        if not hasattr(self, "_dbg_test_tr_once"):
            print("[TEST/TR] bev_lidar is None?", bev_lidar is None)
            self._dbg_test_tr_once = True

        # 1) camera BEV from encoder (standard BEVFormer)
        bev_embed = self.get_bev_features(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs
        )  # (B, HW, C)

        # 2) V3: concat + linear fusion (single memory level)
        if bev_lidar is not None:
            # bev_embed: (B, HW, C)
            B, HW, C = bev_embed.shape

            # gate (define FIRST so logging never references undefined alpha)
            #alpha = torch.sigmoid(self.lidar_gate)  # scalar in (0,1)
            alpha = bev_embed.new_tensor(0.5)   # set fixed alpha

            # --- safe gate logging (DDP-safe-ish, won't crash if attribute missing) ---
            if self.training:
                # create counter on first use
                if not hasattr(self, "_gate_iter"):
                    self._gate_iter = 0
                self._gate_iter += 1

                if self._gate_iter % 500 == 0:
                    a = float(alpha.detach().cpu())
                    g = float(self.lidar_gate.detach().cpu())
                    #print(f"[LiDAR-GATE] iter={self._gate_iter} sigmoid={a:.4f} raw={g:.4f}")
                    print(f"[LiDAR-GATE] iter={self._gate_iter} alpha={a:.4f} raw={g:.4f}")

            if self.training and (self._gate_iter % 500 == 0):
                #print(f"[LiDAR-GATE] sigmoid={float(alpha.detach().cpu()):.4f} raw={float(self.lidar_gate.detach().cpu()):.4f}")
                print(f"[LiDAR-GATE] sigmoid={float(alpha.detach().cpu()):.4f} raw={float(self.lidar_gate.detach().cpu()):.4f}")

            # ---- safe rank-0 only printing helper (works even without dist) ----
            def _is_rank0():
                try:
                    import torch.distributed as dist
                    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
                except Exception:
                    return True

            # Print once when fusion is actually used
            if not hasattr(self, "_gate_once") and _is_rank0():
                #print(f"[LiDAR-GATE] first seen, sigmoid(gate)={float(alpha.detach().cpu()):.4f}")
                print(f"[LiDAR-GATE] first seen, alpha={alpha}")
                self._gate_once = True

            # resize lidar bev to (bev_h, bev_w)
            bev_lidar = F.interpolate(
                bev_lidar, size=(bev_h, bev_w),
                mode='bilinear', align_corners=False
            )  # (B, C_lidar, H, W)

            # project lidar channels -> C
            bev_lidar = self.lidar_proj(bev_lidar)            # (B, C, H, W)
            bev_lidar = bev_lidar.flatten(2).permute(0, 2, 1) # (B, HW, C)

            # apply gate
            bev_lidar = bev_lidar * alpha

            # fuse: concat then linear -> C
            bev_fused = torch.cat([bev_embed, bev_lidar], dim=-1)          # (B, HW, 2C)
            bev_embed = self.lidar_fuse_norm(self.lidar_fuse_linear(bev_fused))  # (B, HW, C)

            # periodic gate logging (rank0 only)
            if self.training and _is_rank0():
                if not hasattr(self, "_gate_iter"):
                    self._gate_iter = 0
                self._gate_iter += 1
                if self._gate_iter % 500 == 0:
                    print(f"[LiDAR-GATE] sigmoid(gate) = {float(alpha.detach().cpu()):.4f} (iter={self._gate_iter})")


        # 3) standard BEVFormer decoder prep
        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)

        reference_points = self.reference_points(query_pos).sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)        # (num_query, B, C)
        query_pos = query_pos.permute(1, 0, 2)

        # decoder value expected as (num_value, B, C) in your repo
        value = bev_embed.permute(1, 0, 2)    # (HW, B, C)

        spatial_shapes = torch.tensor([[bev_h, bev_w]], device=value.device, dtype=torch.long)
        level_start_index = torch.tensor([0], device=value.device, dtype=torch.long)

 
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=value,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            **kwargs
        )

        return bev_embed, inter_states, init_reference_out, inter_references

