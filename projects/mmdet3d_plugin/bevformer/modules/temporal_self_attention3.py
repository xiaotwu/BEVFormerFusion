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
        self.batch_first = True
        print("[TSA:init] batch_first set to", self.batch_first)

        self.fp16_enabled = False

        self.output_proj_in = None  # will lazily match C_in at runtime


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

        self.batch_first = True
        print("[TSA:init] batch_first set to", self.batch_first)

        self.output_proj_in = None  # lazy adapter: maps kernel C_in -> embed_dims when needed

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
        
        if not hasattr(self, "_calls"):
            self._calls = 0
        self._calls += 1
        if self._calls <= 4:  # don’t spam, print a few times
            bf = getattr(self, "batch_first", True)
            bs_q  = query.size(0) if bf else query.size(1)
            Len_q = query.size(1) if bf else query.size(0)
            print(f"[TSA/ENTER] call#{self._calls} bf={bf} bs={bs_q} Len_q={Len_q}")

        # --------------------- PREP & SHAPES ---------------------
        # Identity & layout
        if identity is None:
            identity = query
        bf = getattr(self, "batch_first", True)
        assert bf, "[TSA] This implementation assumes batch_first=True"

        # Query shape
        bs, S_q, C = query.shape  # (B, H*W, C)

        # BEV grid (prefer explicit kwargs; else infer square from S_q)
        H = kwargs.get('bev_h', None)
        W = kwargs.get('bev_w', None)
        if (H is None) or (W is None):
            root = int(S_q ** 0.5)
            if root * root == S_q:
                H = W = root
        assert (H is not None) and (W is not None), "[TSA] bev_h/bev_w not provided and S_q not square"
        H, W = int(H), int(W)
        tokens_per_level = H * W
        assert S_q == tokens_per_level, f"[TSA] query length {S_q} != H*W {tokens_per_level}"

        # Positional enc
        if query_pos is not None:
            query = query + query_pos

        # ---- value normalization: ensure it's 4-D (Bv, Sv, n_heads, head_dim) ----
        # If value is None, fall back to trivial 2-frame stack from query
        if value is None:
            value = torch.cat([query, query], dim=1).contiguous()  # (B, 2*H*W, C)

        # ---- VALUE: project on 3D, then split heads (fix matmul mismatch) ----
        # Accept either 3D (B, Sv, C) or 4D (B, Sv, n_heads, head_dim)
        if value.dim() == 4:
            Bv, Sv, nh, hd = value.shape
            val3 = value.reshape(Bv, Sv, nh * hd).contiguous()  # merge heads -> (Bv, Sv, C_in)
            C_in = nh * hd
        elif value.dim() == 3:
            Bv, Sv, C_in = value.shape
            val3 = value
        else:
            raise AssertionError(f"[TSA] Unsupported value rank {value.dim()}")

        # If needed, adapt to the Linear's expected input channels
        C_proj_in = self.value_proj.in_features  # typically embed_dims (e.g., 256)
        if C_in != C_proj_in:
            # small adapter so we can always feed value_proj with its expected in_features
            if not hasattr(self, "_value_in_adapter") or \
            self._value_in_adapter.in_features != C_in or \
            self._value_in_adapter.out_features != C_proj_in:
                self._value_in_adapter = nn.Linear(C_in, C_proj_in).to(val3.device, dtype=val3.dtype)
            val3 = self._value_in_adapter(val3)  # (Bv, Sv, C_proj_in)

        # Main projection (now last dim matches value_proj.in_features)
        val3 = self.value_proj(val3)  # (Bv, Sv, C_proj_out) usually C_proj_out == embed_dims

        # Split into heads AFTER projection
        assert val3.size(-1) % self.num_heads == 0, \
            f"[TSA] projected C={val3.size(-1)} not divisible by num_heads={self.num_heads}"
        head_dim = val3.size(-1) // self.num_heads
        value = val3.view(Bv, Sv, self.num_heads, head_dim).contiguous()


        # Temporal length present in value
        assert Sv % tokens_per_level == 0, f"[TSA] S_v {Sv} not divisible by H*W {tokens_per_level}"
        L = Sv // tokens_per_level
        num_bev_queue = L  # local, do NOT rely on a class attr here

        import os
        # --- EVIDENCE: tie the pair IDs to TSA’s temporal length L ---
        if os.environ.get("TSA_EVIDENCE", "1") == "1" and not hasattr(self, "_tsa_pair_once"):
            metas = kwargs.get('img_metas', None)
            # metas is a list of dicts per batch in most mmdet codepaths
            if isinstance(metas, (list, tuple)) and len(metas) > 0:
                m0 = metas[0]
                cur_id  = m0.get('timestamp', m0.get('curr_sample_token', m0.get('token', '?')))
                prev_id = m0.get('prev_timestamp', m0.get('prev_sample_token', m0.get('prev_token', '?')))
                print(f"[TSA/PAIR] prev={prev_id} -> curr={cur_id}  L={L} (expect 2)")
            else:
                print(f"[TSA/PAIR] (no metas)  L={L}")
            self._tsa_pair_once = True

        # ---- >>> PLACE THE PROBE HERE <<< ----
        import os, torch
        def _is_main_proc():
            return (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)

        if os.environ.get("TSA_PROBE", "1") == "1" and _is_main_proc() and not hasattr(self, "_dbg_queue_once"):
            bs_q  = query.size(0) if getattr(self, "batch_first", True) else query.size(1)
            Len_q = query.size(1) if getattr(self, "batch_first", True) else query.size(0)
            print(f"[TSA/CHECK] value entering TSA: shape={tuple(value.shape)} "
                f"S_v={Sv} H={H} W={W} -> L={L} | bs={bs_q} Len_q={Len_q} heads={self.num_heads} head_dim={head_dim}",
                flush=True)
            self._dbg_queue_once = True
        # ---- <<< END PROBE >>> ----

        # --------------------- BUILD/VALIDATE SHAPES ---------------------
        # spatial_shapes / level_start_index (single BEV level repeated L times)
        if spatial_shapes is None:
            spatial_shapes = torch.as_tensor([[H, W]], device=value.device, dtype=torch.long)
        if level_start_index is None:
            level_start_index = torch.as_tensor([0], device=value.device, dtype=torch.long)

        # Sanity: S_q vs spatial_shapes if multiple levels were passed in
        sum_hw = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item())
        if sum_hw != S_q:
            # Rebuild as a single BEV level of HxW
            spatial_shapes = torch.as_tensor([[H, W]], device=value.device, dtype=torch.long)
            level_start_index = torch.as_tensor([0], device=value.device, dtype=torch.long)

        # --------------------- PROJECTIONS & MASKING ---------------------
        # Concatenate current query with a summary of history along channel dim (matches original code intent)
        # Using the first temporal slice of value as a simple "history" token source for conditioning
        # value[:, :H*W, :, :].reshape(Bv, H*W, -1) => (B, H*W, C_hist)
        hist_embed = value[:, :tokens_per_level, :, :].reshape(Bv, tokens_per_level, -1)
        # If heads*head_dim != C, we can project hist to C via a small linear; here simply pad/trim via proj layers
        # Safer: project query concatenated with hist using existing learned layers:
        query = torch.cat([hist_embed, query], dim=-1)  # (B, H*W, C_hist + C)

        # Project value to per-head dims for deformable op
        # ---- project on 3D, then split back into heads ----
        if value.dim() == 4:
            Bv, Sv, nh, hd = value.shape
            C_in = nh * hd
            val3 = value.reshape(Bv, Sv, C_in).contiguous()   # (Bv, Sv, C_in)
        elif value.dim() == 3:
            Bv, Sv, C_in = value.shape
            val3 = value
        else:
            raise AssertionError(f"[TSA] Unsupported value rank {value.dim()}")

        # Ensure last dim matches value_proj.in_features
        C_proj_in = self.value_proj.in_features  # typically embed_dims (e.g., 256)
        if C_in != C_proj_in:
            if not hasattr(self, "_value_in_adapter") or \
            self._value_in_adapter.in_features != C_in or \
            self._value_in_adapter.out_features != C_proj_in:
                self._value_in_adapter = nn.Linear(C_in, C_proj_in).to(val3.device, dtype=val3.dtype)
            val3 = self._value_in_adapter(val3)  # (Bv, Sv, C_proj_in)

        # Main projection
        val3 = self.value_proj(val3)  # (Bv, Sv, C_proj_out)

        # Split into heads AFTER projection
        assert val3.size(-1) % self.num_heads == 0, \
            f"[TSA] projected C={val3.size(-1)} not divisible by num_heads={self.num_heads}"
        head_dim = val3.size(-1) // self.num_heads
        value = val3.view(Bv, Sv, self.num_heads, head_dim).contiguous()  # (Bv, Sv, n_heads, head_dim)

        # ==== True batch/length from query (batch_first-aware) ====
        bf    = getattr(self, "batch_first", True)
        bs_q  = query.size(0) if bf else query.size(1)
        Len_q = query.size(1) if bf else query.size(0)

        # value is already (Bv, Sv, n_heads, head_dim) where typically Bv = bs_q * L
        from torch.nn import functional as F

        with torch.no_grad():
            if value.dim() == 4 and value.size(0) % max(bs_q, 1) == 0:
                L_here = value.size(0) // max(bs_q, 1)
                if L_here >= 2:  # only meaningful if we actually have a prev
                    v_prev = value[:bs_q].reshape(bs_q, -1)
                    v_cur  = value[bs_q:2*bs_q].reshape(bs_q, -1)
                    cos = F.cosine_similarity(v_prev, v_cur, dim=1)  # [bs]
                    print(f"[TSA/CHK] cos(prev,cur) per batch = {cos.tolist()}")


        # Optional key padding mask over value sequence tokens (broadcastable)
        if key_padding_mask is not None:
            # key_padding_mask: (B, Sv) -> (B, Sv, 1, 1)
            mask = key_padding_mask[..., None, None]
            value = value.masked_fill(mask, 0.0)

        # Reshape value batch to fold temporal levels into a "levels batch"
        # (Bv*L, H*W, n_heads, head_dim)
        value = value.view(Bv * num_bev_queue, tokens_per_level, self.num_heads, -1)

        # --------------------- OFFSETS & WEIGHTS ---------------------
        sampling_offsets = self.sampling_offsets(query)  # (B, Nq, n_heads * (L * num_levels * num_points * 2))
        sampling_offsets = sampling_offsets.view(
            bs, S_q, self.num_heads, num_bev_queue, self.num_levels, self.num_points, 2
        )

        # weights (before softmax)
        attention_weights = self.attention_weights(query).view(
            bs, S_q, self.num_heads, num_bev_queue, self.num_levels * self.num_points
        )

        # softmax over sampling points per (head, level)
        attention_weights = attention_weights.softmax(-1)

        # keep weights within a sane closed interval
        attention_weights = attention_weights.clamp(1e-6, 1.0)
        attention_weights = attention_weights / attention_weights.sum(-1, keepdim=True).clamp(min=1e-6)

        sampling_offsets = sampling_offsets.clamp(-3.0, 3.0)
        # ---- END INSERT ----

        # reshape for kernel (unchanged)
        attention_weights = attention_weights.view(
            bs, S_q, self.num_heads, num_bev_queue, self.num_levels, self.num_points
        )
        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5) \
            .reshape(bs * num_bev_queue, S_q, self.num_heads, self.num_levels, self.num_points) \
            .contiguous()

        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6) \
            .reshape(bs * num_bev_queue, S_q, self.num_heads, self.num_levels, self.num_points, 2).contiguous()

        # --------------------- REFERENCE POINTS ---------------------
        if reference_points is None:
            # Build normalized [0,1] grid centers for BEV (B, Nq, 1, 2), then tile across L later if needed
            device, dtype = query.device, query.dtype
            gy = torch.linspace(0.5 / H, 1 - 0.5 / H, H, device=device, dtype=dtype)
            gx = torch.linspace(0.5 / W, 1 - 0.5 / W, W, device=device, dtype=dtype)
            grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
            ref_2d = torch.stack((grid_x, grid_y), dim=-1).view(1, tokens_per_level, 1, 2).repeat(bs, 1, 1, 1)
            reference_points = ref_2d  # (B, Nq, 1, 2)

        # Align reference_points with (B*L, ...) if needed
        if reference_points.dim() == 3:
            reference_points = reference_points.unsqueeze(2)  # (B, Nq, 1, 2)
        B_ref, Len_q_rp, L_rp, _ = reference_points.shape
        assert Len_q_rp == S_q, f"[TSA] reference_points Len_q {Len_q_rp} != query {S_q}"

        # Match batch folding
        B_off = sampling_offsets.size(0)  # B*L
        assert B_off % B_ref == 0, f"[TSA] offsets batch {B_off} not multiple of refs batch {B_ref}"
        fold_b = B_off // B_ref
        if fold_b > 1:
            reference_points = reference_points.repeat(fold_b, 1, 1, 1).contiguous()

        # Match level count
        if L_rp != num_bev_queue:
            if L_rp == 1:
                reference_points = reference_points.expand(reference_points.size(0), S_q, num_bev_queue, 2).contiguous()
            else:
                raise AssertionError(f"[TSA] reference_points levels {L_rp} != expected {num_bev_queue}")

        # sampling locations
        if reference_points.shape[-1] == 2:
            # Normalize by (W, H)
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)  # (n_levels, 2)
            sampling_locations = reference_points[:, :, None, :, None, :] + \
                sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + \
                sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(f"[TSA] reference_points last dim must be 2 or 4, got {reference_points.shape[-1]}")

        # ---- sanitize weights and offsets; clamp sampling locations into [0, 1] ----
        # attention_weights is already softmaxed earlier; re-sum after any sanitize
        #if not torch.isfinite(attention_weights).all():
        #    attention_weights = torch.nan_to_num(attention_weights, nan=0.0, posinf=0.0, neginf=0.0)
        #aw_sum = attention_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        #attention_weights = attention_weights / aw_sum

        #if not torch.isfinite(sampling_offsets).all():
        #    sampling_offsets = torch.nan_to_num(sampling_offsets, nan=0.0, posinf=0.0, neginf=0.0)

        # Keep the normalized sampling grid stable
        sampling_locations = sampling_locations.clamp(0.0, 1.0)


        # Contiguity (CUDA op requires .is_contiguous())
        value = value.contiguous()
        sampling_locations = sampling_locations.contiguous()
        attention_weights = attention_weights.contiguous()
        spatial_shapes = spatial_shapes.contiguous()
        level_start_index = level_start_index.contiguous()

        # --------------------- KERNEL CALL (BEV-only shapes) ---------------------
        bev_spatial_shapes = torch.as_tensor([[H, W]] * num_bev_queue, device=value.device, dtype=torch.long)
        bev_level_start_index = torch.as_tensor(
            [i * tokens_per_level for i in range(num_bev_queue)], device=value.device, dtype=torch.long
        )

        if not hasattr(self, "_tsa_dbg_once"):
            print("[TSA] USING BEV shapes",
                "H,W=", (H, W), "L=", num_bev_queue,
                "| bev_spatial_shapes=", bev_spatial_shapes.tolist(),
                "| bev_level_start_index=", bev_level_start_index.tolist())
            self._tsa_dbg_once = True

        if torch.cuda.is_available() and value.is_cuda:
            out = MultiScaleDeformableAttnFunction_fp32.apply(
                value, bev_spatial_shapes, bev_level_start_index, sampling_locations, attention_weights, self.im2col_step
            )
        else:
            out = multi_scale_deformable_attn_pytorch(
                value, bev_spatial_shapes, sampling_locations, attention_weights
            )

        # sanitize once more
        if not torch.isfinite(out).all():
            print("[TSA] non-finite out -> sanitizing")
            out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


        # --------------------- POST / PROJECTION ---------------------
        # Kernel may return (B*L, Nq, n_heads, head_dim) or (B*L, Nq, C)

        # ---- unify kernel output to (bs, Len_q, C) and MEAN over temporal queue ----
        # assumes batch_first=True for query; adjust if you use seq-first
        bs_q  = query.size(0) if self.batch_first else query.size(1)
        Len_q = query.size(1) if self.batch_first else query.size(0)

        if out.dim() == 4:
            # CUDA path often returns (B_fold, Len_q, n_heads, head_dim)
            B_out, Nq_out, n_heads, head_dim = out.shape
            assert Nq_out == Len_q, f"[TSA] kernel Nq {Nq_out} != Len_q {Len_q}"
            out = out.reshape(B_out, Len_q, n_heads * head_dim).contiguous()
        elif out.dim() == 3:
            # PyTorch fallback may already be fused: (B_fold, Len_q, C_fused)
            B_out, Nq_out, _ = out.shape
            assert Nq_out == Len_q, f"[TSA] kernel Nq {Nq_out} != Len_q {Len_q}"
        else:
            raise AssertionError(f"[TSA] Unexpected kernel out rank {out.dim()}")

        # If temporal levels were folded into batch (B_out = bs * L), collapse with MEAN
        if B_out != bs_q:
            assert B_out % bs_q == 0, f"[TSA] batch fold mismatch: B_out={B_out} not multiple of bs={bs_q}"
            L_fold = B_out // bs_q
            # (bs, L, Len_q, C) -> mean over L
            out = out.view(bs_q, L_fold, Len_q, out.size(-1)).mean(dim=1).contiguous()

        # ---- safe projection (adapter if channel count changed) ----
        C_in  = out.size(-1)
        C_out = self.output_proj.out_features
        if self.output_proj.in_features != C_in:
            adapter = getattr(self, "output_proj_in", None)
            if (adapter is None) or (adapter.in_features != C_in) or (adapter.out_features != C_out):
                import torch.nn as nn
                self.output_proj_in = nn.Linear(C_in, C_out).to(out.device, dtype=out.dtype)
                adapter = self.output_proj_in
            out = adapter(out)

        output = self.output_proj(out)

        # --- BEGIN: firewall on both branches of the residual ---
        if not torch.isfinite(output).all():
            print("[TSA] output non-finite -> sanitizing before residual")
            output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)

        if identity is None:
            identity = output.new_zeros(output.shape)
        else:
            if not torch.isfinite(identity).all():
                print("[TSA] identity non-finite -> sanitizing before residual")
                identity = torch.nan_to_num(identity, nan=0.0, posinf=0.0, neginf=0.0)
        # --- END ---

        if not self.batch_first:
            output = output.permute(1, 0, 2).contiguous()

        return self.dropout(output) + identity








