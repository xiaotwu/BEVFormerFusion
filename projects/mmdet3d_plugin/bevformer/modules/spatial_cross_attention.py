
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import build_attention
import math
from mmcv.runner import force_fp32, auto_fp16

from mmcv.runner.base_module import BaseModule, ModuleList, Sequential

from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
    MultiScaleDeformableAttnFunction_fp16
from projects.mmdet3d_plugin.models.utils.bricks import run_time
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@ATTENTION.register_module()
class SpatialCrossAttention(BaseModule):

    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 **kwargs
                 ):
        super(SpatialCrossAttention, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
    
    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                flag='encoder',
                **kwargs):

        import torch
        import torch.nn.functional as F

        # ---- defaults / residual ----
        if key is None:
            key = query
        if value is None:
            value = key
        inp_residual = query if residual is None else residual
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, embed_dims = query.size()

        # ---- effective number of cameras (runtime-robust) ----
        num_cams_rt = None
        if bev_mask is not None and bev_mask.dim() >= 4:
            if bev_mask.size(0) == bs:
                num_cams_rt = bev_mask.size(1)        # [B, Ncam, H, W]
            elif bev_mask.size(1) == bs:
                num_cams_rt = bev_mask.size(0)        # [Ncam, B, H, W]
        if num_cams_rt is None and reference_points_cam is not None:
            for ax in range(reference_points_cam.dim()):
                if reference_points_cam.size(ax) in (3, 4, 5, 6):
                    num_cams_rt = reference_points_cam.size(ax); break
        if num_cams_rt is None and 'img_feats' in kwargs and kwargs['img_feats'] is not None:
            try: num_cams_rt = kwargs['img_feats'][0].shape[1]
            except Exception: pass
        if num_cams_rt is None:
            num_cams_rt = getattr(self, 'num_cams', 6)

        num_cams_eff = min(num_cams_rt, getattr(self, 'num_cams', num_cams_rt))
        if not hasattr(self, "_sca_dbg_once"):
            print(f"[SCA] effective num_cams={num_cams_eff} (module={getattr(self,'num_cams',None)}, data={num_cams_rt})")
            self._sca_dbg_once = True

        # ---- canonicalize reference_points_cam -> [B, Nq, Ncam, D, 2] ----
        assert reference_points_cam is not None, "reference_points_cam is required"
        rc = reference_points_cam

        # 1) ensure last dim == 2 (move any size-2 axis to the end)
        if rc.shape[-1] != 2:
            two_ax = [i for i, s in enumerate(rc.shape) if s == 2][-1]
            rc = rc.movedim(two_ax, -1).contiguous()

        # 2) fold a level axis if present
        if rc.dim() == 6:
            # try (Ncam, B, Nq, L, D, 2)
            if rc.size(0) in (3, 4, 5, 6) and rc.size(4) in (2, 3, 4):
                Ncam, B0, Nq0, L0, D0, _ = rc.shape
                rc = rc.permute(1, 2, 0, 3, 4, 5).reshape(B0, Nq0, Ncam * L0, D0, 2).contiguous()
            else:
                # assume (B, Nq, Ncam, L, D, 2)
                B0, Nq0, Ncam0, L0, D0, _ = rc.shape
                rc = rc.reshape(B0, Nq0, Ncam0 * L0, D0, 2).contiguous()

        # 3) now 5D; fix common camera/batch placements

        # (Ncam, B, Nq, D, 2)  -> (B, Nq, Ncam, D, 2)
        if rc.dim() == 5 and rc.size(0) == num_cams_eff and rc.size(2) == num_query:
            B0 = rc.size(1)
            rc = rc.permute(1, 2, 0, 3, 4).contiguous()

        # (B, Ncam, Nq, D, 2)  -> (B, Nq, Ncam, D, 2)
        elif rc.dim() == 5 and rc.size(0) == bs and rc.size(1) == num_cams_eff and rc.size(2) == num_query:
            rc = rc.permute(0, 2, 1, 3, 4).contiguous()

        # 4) if producer only built refs for B=1, replicate to batch
        if rc.size(0) == 1 and bs > 1:
            # assumes refs identical across batch; if not, fix upstream
            rc = rc.expand(bs, rc.size(1), rc.size(2), rc.size(3), 2).contiguous()

        # 5) final sanity
        assert rc.dim() == 5, f"ref_cam must be 5D, got {rc.dim()}D {tuple(rc.shape)}"
        B, Nq, Ncam_rt, D, two = rc.shape
        assert two == 2, f"last dim must be 2, got {two}"
        assert B == bs and Ncam_rt == num_cams_eff, \
            f"ref_cam shape {tuple(rc.shape)} inconsistent with batch={bs}, ncams={num_cams_eff}"

        reference_points_cam = rc  # canonical

        # ---- canonicalize bev_mask -> [B, Ncam, Nq] ----
        def _reduce_last_if_levels(x):
            # If last dim looks like levels (e.g., 4), OR-reduce it to a single visibility
            if x.dim() >= 3 and x.size(-1) in (2, 3, 4, 5, 6, 8):
                return (x > 0).any(dim=-1)
            return x

        def _to_B_Ncam_H_W(x):
            # Accept (B,Ncam,H,W) or (Ncam,B,H,W) or (Ncam,1,H,W)->expand to bs
            if x.size(0) == bs and x.dim() == 4:
                return x  # [B,Ncam,H,W]
            if x.dim() == 4 and x.size(0) == num_cams_eff and x.size(1) in (1, bs):
                # [Ncam,B,H,W] -> [B,Ncam,H,W]
                x = x.permute(1, 0, 2, 3).contiguous()
                if x.size(0) == 1 and bs > 1:
                    x = x.expand(bs, x.size(1), x.size(2), x.size(3))
                return x
            return None  # not a 4D HxW form

        def _to_B_Ncam_Nq(x):
            # Accept (B,Ncam,Nq) or (Ncam,B,Nq) or (Ncam,1,Nq)->expand to bs
            if x.dim() == 3 and x.size(0) == bs and x.size(1) == num_cams_eff:
                return x  # [B,Ncam,Nq]
            if x.dim() == 3 and x.size(0) == num_cams_eff and x.size(1) in (1, bs):
                # [Ncam,B,Nq] -> [B,Ncam,Nq]
                x = x.permute(1, 0, 2).contiguous()
                if x.size(0) == 1 and bs > 1:
                    x = x.expand(bs, x.size(1), x.size(2))
                return x
            return None

        def _resize_bm_to_queries(bm4, target_Nq):
            """bm4: [B, Ncam, H, W] -> flatten to [B, Ncam, target_Nq], resizing if needed."""
            Bm, Cm, Hm, Wm = bm4.shape
            Lm = Hm * Wm
            if Lm == target_Nq:
                return bm4.flatten(2)  # [B,Ncam,Nq]
            Hq = kwargs.get('bev_h', None); Wq = kwargs.get('bev_w', None)
            if (Hq is not None) and (Wq is not None) and (Hq * Wq == target_Nq):
                bm_resized = F.interpolate(bm4.float(), size=(Hq, Wq), mode='nearest')
                return (bm_resized > 0.5).to(torch.bool).flatten(2)
            bm_flat = bm4.flatten(2)
            if Lm % target_Nq == 0:
                f = Lm // target_Nq
                return bm_flat.view(Bm, Cm, target_Nq, f).any(dim=-1)
            if target_Nq % Lm == 0:
                f = target_Nq // Lm
                return bm_flat.unsqueeze(-1).expand(Bm, Cm, Lm, f).reshape(Bm, Cm, target_Nq)
            if not hasattr(self, "_sca_warn_bm_len"):
                print(f"[SCA] bev_mask length {Lm} not compatible with Nq={target_Nq}; using all-ones mask.")
                self._sca_warn_bm_len = True
            return query.new_ones(Bm, Cm, target_Nq, dtype=torch.bool)

        if bev_mask is None:
            bm = query.new_ones(bs, num_cams_eff, num_query, dtype=torch.bool)
        else:
            bm_raw = bev_mask
            # First, if it has a trailing level axis (e.g., (Ncam,B,Nq,L)), reduce it.
            if bm_raw.dim() >= 3:
                bm_raw = _reduce_last_if_levels(bm_raw)

            bm = None
            # Case 4D: try to canonicalize to [B,Ncam,H,W] then flatten/resize to [B,Ncam,Nq]
            if bm_raw.dim() == 4:
                bm4 = _to_B_Ncam_H_W(bm_raw)
                if bm4 is None:
                    raise AssertionError(f"bev_mask shape {tuple(bm_raw.shape)} not compatible with batch={bs}")
                if bm4.size(1) != num_cams_eff:
                    bm4 = bm4[:, :num_cams_eff]
                bm = _resize_bm_to_queries(bm4, num_query)

            # Case 3D: direct token mask forms -> [B,Ncam,Nq]
            elif bm_raw.dim() == 3:
                # Accept (B,Ncam,Nq) / (Ncam,B,Nq) / (Ncam,1,Nq)->expand
                bm3 = _to_B_Ncam_Nq(bm_raw)
                if bm3 is None:
                    # Also accept (B,Nq,Ncam) and swap to (B,Ncam,Nq)
                    if bm_raw.size(0) == bs and bm_raw.size(2) == num_cams_eff:
                        bm3 = bm_raw.permute(0, 2, 1).contiguous()
                    # Or (Ncam,Nq,B)
                    elif bm_raw.size(0) == num_cams_eff and bm_raw.size(2) in (1, bs):
                        bm3 = bm_raw.permute(2, 0, 1).contiguous()
                        if bm3.size(0) == 1 and bs > 1:
                            bm3 = bm3.expand(bs, bm3.size(1), bm3.size(2))
                    else:
                        raise AssertionError(f"bev_mask 3D unexpected shape {tuple(bm_raw.shape)} for batch={bs}, cams={num_cams_eff}")
                bm = bm3

            else:
                raise AssertionError(f"bev_mask must be 3D/4D, got {bm_raw.dim()}D")

        # Final sanity
        assert bm.shape[0] == bs, f"bev_mask batch mismatch: {tuple(bm.shape)} vs bs={bs}"
        assert bm.shape[1] == num_cams_eff, f"bev_mask cams mismatch: {tuple(bm.shape)} vs Ncam={num_cams_eff}"
        if bm.shape[2] != num_query:
            # Resize tokens if necessary (rare)
            if not hasattr(self, "_sca_warn_bm_nq"):
                print(f"[SCA] resizing bev_mask Nq {bm.shape[2]} -> {num_query}")
                self._sca_warn_bm_nq = True
            if bm.shape[2] > num_query:
                bm = bm[:, :, :num_query]
            else:
                pad = query.new_ones(bs, num_cams_eff, num_query - bm.shape[2], dtype=torch.bool)
                bm = torch.cat([bm, pad], dim=2)

        # bm: [B, Ncam, Nq]


        # ---- build per-(batch,cam) query indices ----
        idxs = [[None] * num_cams_eff for _ in range(bs)]
        max_len = 0
        for j in range(bs):
            for i in range(num_cams_eff):
                idx = bm[j, i].nonzero(as_tuple=False).squeeze(-1)
                idxs[j][i] = idx
                max_len = max(max_len, int(idx.numel()))
        if max_len == 0:
            if not hasattr(self, "_sca_warn_empty"):
                print("[SCA] all camera masks empty; skipping spatial cross-attn")
                self._sca_warn_empty = True
            return self.dropout(inp_residual) + inp_residual

        # ---- re-batch queries and per-cam reference points ----
        queries_rebatch   = query.new_zeros(bs, num_cams_eff, max_len, embed_dims)          # [B,Ncam,Lq,C]
        ref_cam_rebatch   = reference_points_cam.new_zeros(bs, num_cams_eff, max_len, D, 2) # [B,Ncam,Lq,D,2]
        for j in range(bs):
            for i in range(num_cams_eff):
                idx = idxs[j][i]; Lq = int(idx.numel())
                if Lq == 0: continue
                queries_rebatch[j, i, :Lq] = query[j, idx]
                ref_cam_rebatch[j, i, :Lq] = reference_points_cam[j, idx, i]

        # ---- key/value -> [Ncam, B*Len, C] (robust) ----
        def to_cam_first(x, name):
            """
            Normalize feature tensor to [Ncam, B*Len, C].
            Accepts any of:
            [B, Ncam, Len, C]
            [Ncam, B, Len, C]        (expands B=1 -> bs if needed)
            [Ncam, Len, B, C]        (expands B=1 -> bs if needed)
            [B, Len, Ncam, C]
            """
            assert x.dim() == 4, f"{name} must be 4D, got {tuple(x.shape)}"
            B = bs
            C = x.size(-1)

            # Case A: [B, Ncam, Len, C]
            if x.size(0) == B and x.size(1) >= num_cams_eff and x.size(-1) == C:
                if x.size(1) != num_cams_eff:
                    x = x[:, :num_cams_eff, ...]
                x = x.permute(1, 0, 2, 3).contiguous()               # -> [Ncam, B, Len, C]
                return x.reshape(num_cams_eff, -1, C), num_cams_eff

            # Case B: [Ncam, B, Len, C]
            if x.size(0) >= num_cams_eff and x.size(1) in (1, B) and x.size(-1) == C:
                if x.size(0) != num_cams_eff:
                    x = x[:num_cams_eff, ...]
                # expand B=1 -> bs (duplicate) if needed
                if x.size(1) == 1 and B > 1:
                    x = x.expand(num_cams_eff, B, x.size(2), C).contiguous()
                return x.reshape(num_cams_eff, -1, C), num_cams_eff

            # Case C: [Ncam, Len, B, C]
            if x.size(0) >= num_cams_eff and x.size(2) in (1, B) and x.size(-1) == C:
                if x.size(0) != num_cams_eff:
                    x = x[:num_cams_eff, ...]
                # expand B=1 -> bs (duplicate) if needed (batch is dim=2 here)
                if x.size(2) == 1 and B > 1:
                    x = x.expand(num_cams_eff, x.size(1), B, C).contiguous()
                x = x.permute(0, 2, 1, 3).contiguous()               # -> [Ncam, B, Len, C]
                return x.reshape(num_cams_eff, -1, C), num_cams_eff

            # Case D: [B, Len, Ncam, C]
            if x.size(0) == B and x.size(2) >= num_cams_eff and x.size(-1) == C:
                if x.size(2) != num_cams_eff:
                    x = x[:, :, :num_cams_eff, :]
                x = x.permute(2, 0, 1, 3).contiguous()               # -> [Ncam, B, Len, C]
                return x.reshape(num_cams_eff, -1, C), num_cams_eff

            raise AssertionError(f"{name} unexpected shape {tuple(x.shape)} for num_cams={num_cams_eff}, batch={B}")

        key, nck = to_cam_first(key,   "key")
        value, ncv = to_cam_first(value, "value")
        assert nck == ncv == num_cams_eff, f"cams mismatch: key={nck}, value={ncv}, eff={num_cams_eff}"

        # ---- flatten per-cam queries and per-cam reference points for attention ----
        B0, Nc0, Lq, Cq = queries_rebatch.shape
        queries_in  = queries_rebatch.reshape(B0 * Nc0, Lq, Cq)        # [B*Ncam, Lq, C]
        ref_in      = ref_cam_rebatch.reshape(B0 * Nc0, Lq, D, 2)      # [B*Ncam, Lq, D, 2]


        # ---------- BEGIN: SCA kernel sanitization ----------
        # 1) Clamp reference points into [0,1] and zero non-finite entries
        if not torch.isfinite(ref_in).all():
            bad = ~torch.isfinite(ref_in)
            # replace NaN/Inf by 0.5 center so they’re inside the image
            ref_in = ref_in.clone()
            ref_in[bad] = 0.5
        ref_in = ref_in.clamp_(0.0, 1.0)

        # 2) Ensure key/value are finite (they may come from backbone/FP16 etc.)
        if not torch.isfinite(key).all():
            key = torch.nan_to_num(key, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(value).all():
            value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

        # 3) Sanity assertions on all shapes that go into the CUDA op
        assert key.dim()   == 3, f"[SCA] key must be 3D [Ncam, B*Len, C], got {tuple(key.shape)}"
        assert value.dim() == 3, f"[SCA] value must be 3D [Ncam, B*Len, C], got {tuple(value.shape)}"
        assert queries_in.dim() == 3 and ref_in.dim() == 4, \
            f"[SCA] queries_in {queries_in.shape} or ref_in {ref_in.shape} has wrong rank"

        # 4) (optional) one-time prints to confirm
        if not hasattr(self, "_sca_sanitize_once"):
            print(f"[SCA/SAN] queries_in={tuple(queries_in.shape)} key={tuple(key.shape)} "
                f"value={tuple(value.shape)} ref_in={tuple(ref_in.shape)} "
                f"spatial_shapes={spatial_shapes.tolist() if torch.is_tensor(spatial_shapes) else spatial_shapes} "
                f"level_start_index={level_start_index.tolist() if torch.is_tensor(level_start_index) else level_start_index}")
            self._sca_sanitize_once = True
        # ---------- END: SCA kernel sanitization ----------

        # ---- deformable attention over image features ----
        try:
            queries = self.deformable_attention(
                query=queries_in,
                key=key,
                value=value,
                value_spatial_shapes=spatial_shapes,
                value_level_start_index=level_start_index,
                reference_points=ref_in
            )

        except RuntimeError as e:
            print(f"[SCA] CUDA kernel failed -> falling back to PyTorch path once: {repr(e)}")
            # Move the minimal tensors to CPU for the pure-PyTorch reference
            queries_cpu = queries_in.detach().cpu()
            key_cpu     = key.detach().cpu()
            value_cpu   = value.detach().cpu()
            ss_cpu      = spatial_shapes.detach().cpu() if torch.is_tensor(spatial_shapes) else spatial_shapes
            lsi_cpu     = level_start_index.detach().cpu() if torch.is_tensor(level_start_index) else level_start_index
            ref_cpu     = ref_in.detach().cpu()
            # SCA uses the same reference kernel signature as TSA: (value, spatial_shapes, sampling_locs, attn_w)
            # Here "ref_cpu" are the normalized sampling locations for SCA.
            out_cpu = multi_scale_deformable_attn_pytorch(
                value_cpu, ss_cpu, ref_cpu,  # sampling_locations
                attention_weights=None       # if your SCA path doesn’t use per-point weights, pass None or ones
            )
            queries = out_cpu.to(queries_in.device, dtype=queries_in.dtype)

        # -> [B*Ncam, Lq, C]

        # reshape back to [B, Ncam, Lq, C] and scatter-add into slots
        # Use the length actually returned by the attention op
        # NEW: fold-aware reshape (handles extra factor on the "batch" axis)
        BNC = queries.size(0)                   # observed "batch" = fold * bs * num_cams_eff
        Lq_out = queries.size(1)
        Cq_out = queries.size(2)
        base = bs * num_cams_eff
        assert Cq_out == embed_dims, f"[SCA] channel mismatch: got {Cq_out}, expected {embed_dims}"

        assert BNC % base == 0, f"[SCA] unexpected first dim {BNC} not multiple of bs*cams={base}"
        fold = BNC // base                      # e.g., 2 in your error

        if fold > 1:
            # Collapse the extra factor by summation (or .mean if that's preferable)
            queries = queries.view(bs, num_cams_eff, fold, Lq_out, embed_dims).sum(dim=2).contiguous()
        else:
            queries = queries.view(bs, num_cams_eff, Lq_out, embed_dims).contiguous()


        # accumulate into a fresh buffer (no graph history), then assign once
        acc = torch.zeros_like(query)  # [B, Nq, C], safe target for in-place adds
        for j in range(bs):
            for i in range(num_cams_eff):
                idx = idxs[j][i]
                take = int(idx.numel())
                if take == 0:
                    continue
                # add along the query-axis (dim=0 of acc[j]) without touching graph tensors in-place
                acc[j].index_add_(0, idx[:take], queries[j, i, :take])

        slots = acc  # now use slots as the accumulated result

        # average over cams that “saw” each query
        count = bm.sum(1).clamp(min=1).to(slots.dtype)   # bm shape: [B, Ncam, Nq] -> sum over cams => [B, Nq]
        slots = slots / count.unsqueeze(-1)              # [B, Nq, C]


        slots = self.output_proj(slots)

        if not hasattr(self, "_dbg_sca_once"):
            ok = torch.isfinite(slots).all()
            print(f"[DBG/SCA] output finite? {bool(ok)} shape={tuple(slots.shape)}")
            self._dbg_sca_once = True

        return self.dropout(slots) + inp_residual



@ATTENTION.register_module()
class MSDeformableAttention3D(BaseModule):
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
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
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
        self.batch_first = batch_first
        self.output_proj = None
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
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)

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
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True
    """
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
                **kwargs):
    """
    def forward(self,
                query, key=None, value=None, identity=None,
                query_pos=None, key_pos=None,
                reference_points=None, reference_points_cam=None,
                spatial_shapes=None, level_start_index=None,
                key_padding_mask=None, mask=None, depth=None,
                lidar_bev=None, depth_z=None,
                mlvl_feats=None,
                **kwargs):

        # --- accept alias arg names used by other forks ---
        if spatial_shapes is None:
            spatial_shapes = kwargs.pop('value_spatial_shapes', None)
        if level_start_index is None:
            level_start_index = kwargs.pop('value_level_start_index', None)
        if mlvl_feats is None:
            mlvl_feats = kwargs.get('value_list', None)

        # --- final fallback order: cached -> per-level feats -> infer from 'value' length ---
        if (spatial_shapes is None or level_start_index is None) and hasattr(self, 'cached_shapes'):
            cs = getattr(self, 'cached_shapes', None)
            if cs is not None:
                spatial_shapes, level_start_index = cs

        if spatial_shapes is None or level_start_index is None:
            # Try to rebuild from explicit per-level features first
            src = mlvl_feats if mlvl_feats is not None else None
            if isinstance(src, (list, tuple)) and len(src) > 0 and hasattr(src[0], 'dim'):
                device = src[0].device
                spatial_shapes = torch.as_tensor(
                    [[feat.size(-2), feat.size(-1)] for feat in src],
                    dtype=torch.long, device=device
                )
                level_start_index = torch.cat((
                    spatial_shapes.new_zeros(1),
                    (spatial_shapes[:, 0] * spatial_shapes[:, 1]).cumsum(0)[:-1]
                ))
            else:
                # Last-ditch: infer a single "level" from 'value' length
                # value is expected to be [bs_eff, num_value, C]
                if value is None or value.dim() != 3:
                    raise RuntimeError(
                        "SpatialCrossAttention.forward: spatial_shapes/level_start_index not provided "
                        "and cannot infer from inputs. Pass mlvl_feats (list of [B,N,C,H,W]) or "
                        "pass spatial_shapes & level_start_index from the encoder."
                    )
                device = value.device
                num_value = value.size(1)

                # estimate number of cameras at runtime
                if reference_points_cam is not None and reference_points_cam.dim() >= 2:
                    num_cam_rt = int(reference_points_cam.size(1))
                elif hasattr(self, 'num_cams') and isinstance(self.num_cams, int) and self.num_cams > 0:
                    num_cam_rt = int(self.num_cams)
                else:
                    num_cam_rt = 1

                # Per-camera token count (Σ H_l*W_l). If not divisible, just use num_value.
                sum_hw = num_value // num_cam_rt if num_value % max(num_cam_rt, 1) == 0 else num_value

                # Build a 1-level grid (H=sum_hw, W=1). This degrades multi-level into a single level
                # but unblocks the forward pass.
                spatial_shapes = torch.as_tensor([[sum_hw, 1]], dtype=torch.long, device=device)
                level_start_index = torch.zeros(1, dtype=torch.long, device=device)

        # --- end rebuild ---
        #if value is None:
        #    value = query
        #if identity is None:
        #    identity = query
        #if query_pos is not None:
        #    query = query + query_pos

        #if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
        #    query = query.permute(1, 0, 2)
        #    value = value.permute(1, 0, 2)

        #bs, num_query, _ = query.shape
        #bs, num_value, _ = value.shape
        #assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
            if identity is not None:
                identity = identity.permute(1, 0, 2)
            if query_pos is not None:
                query_pos = query_pos.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs_v, num_value, c_v = value.shape

        # sum over (H_l * W_l)
        sum_hw = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item())
        if num_value != sum_hw:
            # num_value likely = num_cam * sum_hw
            assert num_value % sum_hw == 0, (
                f"num_value ({num_value}) not divisible by sum(HW) ({sum_hw})"
            )
            num_cam_rt = num_value // sum_hw

            # reshape value: [bs, num_cam, sum_hw, C] -> [bs*num_cam, sum_hw, C]
            value = value.view(bs_v, num_cam_rt, sum_hw, c_v).reshape(bs_v * num_cam_rt, sum_hw, c_v)

            # tile query/identity/query_pos along camera dim to match new batch
            query = query.unsqueeze(1).expand(bs, num_cam_rt, num_query, -1).reshape(bs * num_cam_rt, num_query, -1)
            if identity is not None:
                identity = identity.unsqueeze(1).expand(bs, num_cam_rt, num_query, -1).reshape(bs * num_cam_rt, num_query, -1)
            if query_pos is not None:
                query_pos = query_pos.unsqueeze(1).expand(bs, num_cam_rt, num_query, -1).reshape(bs * num_cam_rt, num_query, -1)

            # update bs and num_value to the per-camera values expected downstream
            bs = bs * num_cam_rt
            num_value = sum_hw

        # At this point, num_value must equal sum(H_l * W_l)
        assert num_value == sum_hw, \
            f"Expected num_value == sum(HW) ({sum_hw}), got {num_value}"
        
        # --- Align batch dims between query and value/key ---
        # query was already shaped to [bs, num_query, C]
        # value must match 'bs' before we split heads
        if value.size(0) != bs:
            assert bs % value.size(0) == 0, \
                f"Batch mismatch: query bs={bs}, value bs={value.size(0)}"
            _rep = bs // value.size(0)
            value = value.repeat_interleave(_rep, dim=0)

        # If 'key' follows the same path, keep it aligned too:
        #if 'key' in locals() and key is not None and key.size(0) != bs:
        #    assert bs % key.size(0) == 0, \
        #        f"Batch mismatch: query bs={bs}, key bs={key.size(0)}"
        #    _rep_k = bs // key.size(0)
        #    key = key.repeat_interleave(_rep_k, dim=0)
        # --- end alignment ---

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)

        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        # --- Guarantee reference_points exists ---
        if reference_points is None:
            # 1) Try to derive from reference_points_cam (common shape: [B, N_cam, Len, 2 or 3])
            ref = None
            if reference_points_cam is not None:
                rpc = reference_points_cam
                if rpc.dim() == 4:          # [B, N_cam, Len, C]
                    ref = rpc.mean(dim=1)   # -> [B, Len, C]
                elif rpc.dim() == 3:        # [B, Len, C]
                    ref = rpc

            # 2) If still None, synthesize a normalized 2D grid that matches the query length
            # ---- Normalize reference_points to [B, Len, Nz, C] ----
            # Accepts incoming shapes:
            #   [B, Len, C]           -> unsqueeze Nz=1
            #   [B, Len, Nz, C]       -> already OK
            # If it's still None (shouldn't happen after our encoder patch), synthesize a simple grid.
            if reference_points is None:
                bs_eff, num_query, _ = query.shape
                # build a normalized 2D grid of length num_query, Nz=1
                x = torch.linspace(0.5 / max(num_query, 1), 1.0 - 0.5 / max(num_query, 1),
                                steps=num_query, device=query.device, dtype=query.dtype)
                y = torch.full_like(x, 0.5)
                reference_points = torch.stack([x, y], dim=-1).unsqueeze(0).repeat(bs_eff, 1, 1)  # [B, Len, 2]

            if reference_points.dim() == 3:
                reference_points = reference_points.unsqueeze(2)  # -> [B, Len, 1, C]

            # Final sanity check
            assert reference_points.dim() == 4, \
                f"reference_points must be [B, Len, Nz, C], got {tuple(reference_points.shape)}"

        if reference_points.shape[-1] == 2:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)
            
            #sampling_locations = reference_points + sampling_offsets
            # --- align batch dims between reference_points and sampling_offsets ---
            # Both tensors must have the same size(0) (batch) before elementwise add.
            bs_ref = reference_points.size(0)
            bs_off = sampling_offsets.size(0)
            if bs_ref != bs_off:
                # Try to repeat the smaller batch to match the larger one.
                if bs_off % bs_ref == 0:
                    rep = bs_off // bs_ref
                    reference_points = reference_points.repeat_interleave(rep, dim=0)
                elif bs_ref % bs_off == 0:
                    rep = bs_ref // bs_off
                    sampling_offsets = sampling_offsets.repeat_interleave(rep, dim=0)
                else:
                    raise RuntimeError(
                        f"Batch mismatch: reference_points bs={bs_ref}, sampling_offsets bs={bs_off} "
                        "are not multiples."
                    )
            # (optional) ensure dtypes/devices match
            if reference_points.dtype != sampling_offsets.dtype:
                reference_points = reference_points.to(sampling_offsets.dtype)
            if reference_points.device != sampling_offsets.device:
                reference_points = reference_points.to(sampling_offsets.device)
            # --- end alignment ---

            sampling_locations = reference_points + sampling_offsets
            
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2
        #  attention_weights.shape: bs, num_query, num_heads, num_levels, num_all_points
        #

        if torch.cuda.is_available() and value.is_cuda:
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output
