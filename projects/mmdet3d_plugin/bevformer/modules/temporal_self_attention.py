# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from projects.mmdet3d_plugin.models.utils.bricks import run_time
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION
import math
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
                        to_2tuple)

from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@ATTENTION.register_module()
class TemporalSelfAttention(BaseModule):
    """An attention module used in BEVFormer based on Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        num_bev_queue (int): In this version, we only use one history BEV and one currenct BEV.
         the length of BEV queue is 2.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 num_bev_queue=2,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):

        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.sampling_offsets = nn.Linear(
            embed_dims*self.num_bev_queue, num_bev_queue*num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims*self.num_bev_queue,
                                           num_bev_queue*num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels*self.num_bev_queue, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                flag='decoder',
                **kwargs):

        # ------ Expectations ------
        #self.batch_first = True
        assert self.batch_first, "[TSA] set self.batch_first=True"
        H = kwargs.get('bev_h', None)
        W = kwargs.get('bev_w', None)
        assert (H is not None) and (W is not None), "[TSA] needs bev_h/bev_w in kwargs"

        # ------ Periodic stats (prints every N calls) ------
        if not hasattr(self, "_tsastats"):
            # change period to 0/None to silence
            self._tsastats = {"calls": 0, "uses": 0, "skips": 0, "period": 50}
        self._tsastats["calls"] += 1
        used_this_call = False

        # ------ Normalize query to [B, Nq, C] ------
        if query.dim() == 3 and query.shape[1] < query.shape[0]:   # likely [Nq,B,C]
            query = query.permute(1, 0, 2).contiguous()
        B, Nq, C = query.shape
        assert C == self.embed_dims, f"[TSA] embed_dims mismatch: query C={C} vs self.embed_dims={self.embed_dims}"
        assert getattr(self, "num_bev_queue", 2) == 2, "[TSA] assumes num_bev_queue=2"
        assert Nq == H * W, f"[TSA] len_bev S={Nq} mismatches bev grid {H}x{W}={H*W}"

        # ------ Positional add (not concat) ------
        cur = query if (query_pos is None) else (query + query_pos)   # [B, Nq, C]

        # ------ Pull prev_bev and canonicalize to [B, Nq, C] ------
        prev = kwargs.get('prev_bev', None)

        if not hasattr(self, "_dbg_prev_once"):
            print("[TSA/DBG] first call: prev_bev is",
                "None" if prev is None else f"Tensor with shape {tuple(prev.shape)}")
            self._dbg_prev_once = True


        if prev is not None:
            if prev.dim() == 3 and prev.shape[1] < prev.shape[0]:  # [Nq,B,C] -> [B,Nq,C]
                prev = prev.permute(1, 0, 2).contiguous()
            if prev.size(0) == 1 and prev.size(1) == Nq and prev.size(2) == C:
                prev = prev.expand(B, -1, -1).contiguous()
            can_use_prev = (prev.shape == (B, Nq, C))
        else:
            can_use_prev = False

        # ---------- DEBUG BLOCK: confirm prev_bev vs cur ----------
        if prev is not None and can_use_prev and not hasattr(self, "_dbg_prev_curr_once"):
            # Check pointer equality: are they literally the same storage?
            prev_ptr = prev.data_ptr()
            cur_ptr = cur.data_ptr()
            max_diff = (prev - cur).abs().max().item()
            print("[DBG/TSA] first use of prev_bev:")
            print("  prev_bev data_ptr:", prev_ptr)
            print("  cur      data_ptr:", cur_ptr)
            print("  max |prev_bev - cur| before fusion:", max_diff)

            if prev_ptr == cur_ptr:
                print("[WARN/TSA] prev_bev and cur share the same storage! "
                    "This suggests BEV(t) is being reused as prev_bev(t) "
                    "(i.e., 1/2*(BEV(t)+BEV(t))). Check the code where "
                    "prev_bev is stored (should use detach()/clone()).")
            self._dbg_prev_curr_once = True
        # ----------------------------------------------------------        

        if not can_use_prev:
            # clean pass-through when no history exists
            self._tsastats["skips"] += 1
            if identity is None:
                identity = query
            # periodic print
            #P = self._tsastats["period"]
            #if P and (self._tsastats["calls"] % P == 0):
            #    s = self._tsastats
            #    rate = s["uses"] / max(1, s["calls"])
            #    print(f"[TSA] {s['calls']} calls | uses={s['uses']} | skips={s['skips']} | use-rate={rate:.1%}", flush=True)
            return identity

        used_this_call = True
        self._tsastats["uses"] += 1

        # ------ Build temporal value for deformable attn ------
        # value: [B*2, Nq, C]  with order {prev, curr}
        value = torch.stack([prev, cur], dim=1).reshape(B * 2, Nq, C)

        # ------ Build q_cat for offsets/weights: in_features must be 2*C ------
        q_cat = torch.cat([prev, cur], dim=-1)  # [B, Nq, 2*C]
        # hard assertions so you catch wrong Linear input sizes early
        assert getattr(self.sampling_offsets, "in_features", 2*C) == 2 * C, \
            f"[TSA] sampling_offsets.in_features={getattr(self.sampling_offsets,'in_features',None)} but expected {2*C}"
        assert getattr(self.attention_weights, "in_features", 2*C) == 2 * C, \
            f"[TSA] attention_weights.in_features={getattr(self.attention_weights,'in_features',None)} but expected {2*C}"

        # ------ Spatial shapes for BEV deformable attention (single level HxW) ------
        need_rebuild = (
            spatial_shapes is None or
            level_start_index is None or
            int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item()) != Nq
        )
        if need_rebuild:
            spatial_shapes = torch.as_tensor([[H, W]], device=query.device, dtype=torch.long)
            level_start_index = torch.as_tensor([0], device=query.device, dtype=torch.long)

        # Optional key padding (rare for BEV tokens)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)

        # ------ Projections & params ------
        v_proj = self.value_proj(value)                           # [B*2, Nq, C]
        Nh = self.num_heads
        L  = getattr(self, "num_levels", 1)                       # usually 1 for TSA
        Pp = getattr(self, "num_points", 4)                       # sampling points
        v_proj = v_proj.view(B * 2, Nq, Nh, -1)                   # [B*2, Nq, Nh, Ch]

        so = self.sampling_offsets(q_cat)                         # [B,Nq,Nh*2*L*Pp*2]
        aw = self.attention_weights(q_cat)                        # [B,Nq,Nh*2*L*Pp]
        so = so.view(B, Nq, Nh, self.num_bev_queue, L, Pp, 2)
        aw = aw.view(B, Nq, Nh, self.num_bev_queue, L, Pp).softmax(-1)

        # reshape to MS-Deformable expected layout
        aw = aw.permute(0, 3, 1, 2, 4, 5).reshape(B * 2, Nq, Nh, L, Pp).contiguous()
        so = so.permute(0, 3, 1, 2, 4, 5, 6).reshape(B * 2, Nq, Nh, L, Pp, 2).contiguous()

        # ------ Reference points (2D BEV centers in [0,1]) ------
        if reference_points is None:
            yy, xx = torch.meshgrid(
                torch.linspace(0.5 / H, 1.0 - 0.5 / H, H, device=query.device),
                torch.linspace(0.5 / W, 1.0 - 0.5 / W, W, device=query.device),
                indexing='ij'
            )
            reference_points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1) \
                                .unsqueeze(0).expand(B, -1, -1)  # [B,Nq,2]
        else:
            # ensure [B,Nq,2]
            if reference_points.dim() == 3 and reference_points.shape[1] < reference_points.shape[0]:  # [Nq,B,2]
                reference_points = reference_points.permute(1, 0, 2).contiguous()
            assert reference_points.shape[:2] == (B, Nq) and reference_points.size(-1) == 2, \
                f"[TSA] reference_points must be [B,Nq,2], got {tuple(reference_points.shape)}"

        # sampling locations around BEV refs (normalized by H/W)
        ref2d = reference_points[:, :, None, None, None, :]                 # [B,Nq,1,1,1,2]
        ref2d = ref2d.repeat(2, 1, 1, L, Pp, 1)                             # [B*2,Nq,1,L,Pp,2]
        ref2d = ref2d.expand(B * 2, Nq, Nh, L, Pp, 2)                       # [B*2,Nq,Nh,L,Pp,2]
        norm = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1)  # [L,2]=[W,H]
        norm = norm[None, None, None, :, None, :].to(so.dtype).to(so.device)          # [1,1,1,L,1,2]
        sampling_locations = ref2d + so / norm

        # ------ Deformable attention ------
        use_cuda = torch.cuda.is_available() and v_proj.is_cuda
        if use_cuda:
            try:
                # alias present in some MMCV builds
                from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction_fp32 as MSDA
            except ImportError:
                try:
                    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction as MSDA
                except ImportError:
                    MSDA = None

            if MSDA is not None:
                out = MSDA.apply(
                    v_proj,                   # [B*2, Nq, Nh, Ch]
                    spatial_shapes,           # [[H, W]]
                    level_start_index,        # [0]
                    sampling_locations,       # [B*2, Nq, Nh, L, Pp, 2]
                    aw,                       # [B*2, Nq, Nh, L, Pp]
                    self.im2col_step
                )
            else:
                from mmcv.ops.multi_scale_deform_attn import multi_scale_deform_attn_pytorch
                out = multi_scale_deform_attn_pytorch(
                    v_proj, spatial_shapes, sampling_locations, aw
                )
        else:
            from mmcv.ops.multi_scale_deform_attn import multi_scale_deform_attn_pytorch
            out = multi_scale_deform_attn_pytorch(
                v_proj, spatial_shapes, sampling_locations, aw
            )

        # out: [B*2, Nq, C] (since Nh*Ch == C)

        # ------ fuse {prev,curr} time queue (mean) ------
        out = out.permute(1, 2, 0)           # [Nq, C, B*2]
        out = out.view(Nq, C, B, 2).mean(-1) # [Nq, C, B]
        out = out.permute(2, 0, 1)           # [B, Nq, C]
        out = self.output_proj(out)          # [B, Nq, C]

        if identity is None:
            identity = query

        # periodic print
        #P = self._tsastats["period"]
        #if P and (self._tsastats["calls"] % P == 0):
        #    s = self._tsastats
        #    rate = s["uses"] / max(1, s["calls"])
        #    print(f"[TSA] {s['calls']} calls | uses={s['uses']} | skips={s['skips']} | use-rate={rate:.1%}", flush=True)

        return self.dropout(out) + identity
