
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import numpy as np
import torch
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, pc_range=None, num_points_in_pillar=4, return_intermediate=False, dataset_type='nuscenes',
                 **kwargs):

        super(BEVFormerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False

        # temporal queue controls (use your config value if present)
        self.queue_length = getattr(self, "queue_length", 2)  # set to 2 for your 6GB GPU, or leave if already defined
        self._bev_queue = []  # stores past BEV tokens; each entry: (bs, H*W, C), batch-first
        print("[ENC] USING encoder.py at runtime")

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    # This function must use fp32!!!
    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, pc_range,  img_metas):
        # NOTE: close tf32 here.
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(
                np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

        return reference_points_cam, bev_mask

    @auto_fp16()
    def forward(self,
                bev_query,
                key,
                value,
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.,
                img_metas=None,
                **kwargs):

        if not hasattr(self, "_dbg_enc_prev_once"):
            print("[ENC/ARGS] prev_bev arg is",
                  "None" if prev_bev is None else f"Tensor {tuple(prev_bev.shape)}")
            self._dbg_enc_prev_once = True

        output = bev_query
        intermediate = []

        # --- reference points (same as your original code) ---
        ref_3d = self.get_reference_points(
            bev_h, bev_w, self.pc_range[5] - self.pc_range[2],
            self.num_points_in_pillar,
            dim='3d',
            bs=bev_query.size(1),
            device=bev_query.device,
            dtype=bev_query.dtype)

        ref_2d = self.get_reference_points(
            bev_h, bev_w,
            dim='2d',
            bs=bev_query.size(1),
            device=bev_query.device,
            dtype=bev_query.dtype)

        metas = img_metas if img_metas is not None else kwargs.get('img_metas', None)
        if metas is None:
            raise KeyError("[Encoder] img_metas not provided to encoder.forward(...)")

        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, metas)

        # bug kept for reproducibility (as in original)
        shift_ref_2d = ref_2d.clone()
        shift_ref_2d += shift[:, None, None, :]

        # (num_query, bs, C) -> (bs, num_query, C)
        bev_query = bev_query.permute(1, 0, 2).contiguous()
        bev_pos   = bev_pos.permute(1, 0, 2).contiguous()

        bs, len_bev, num_bev_level, _ = ref_2d.shape
        bs_q, n_q, c_q = bev_query.shape
        assert n_q == bev_h * bev_w, f"[Encoder] BEV tokens mismatch: expected {bev_h*bev_w}, got {n_q}"

        # simple debug: check prev_bev vs current bev
        if prev_bev is not None and not hasattr(self, "_enc_prev_diff_once"):
            import torch.nn.functional as F
            if prev_bev.dim() == 3 and prev_bev.shape[0] == bev_query.shape[1]:
                prev_bev_bf = prev_bev.permute(1, 0, 2).contiguous()
            else:
                prev_bev_bf = prev_bev.contiguous()
            mean_abs_diff = (prev_bev_bf - bev_query).abs().mean().item()
            cos_sim = F.cosine_similarity(
                prev_bev_bf.reshape(prev_bev_bf.size(0), -1),
                bev_query.reshape(bev_query.size(0), -1),
                dim=1
            ).mean().item()
            print(f"[ENC/CHK] prev_bev present | mean|Δ|={mean_abs_diff:.4f} | cos_sim={cos_sim:.4f}")
            self._enc_prev_diff_once = True

        # --- run the encoder layers, passing prev_bev straight through ---
        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,   # <--- key line: give it to the layer
                img_metas=metas,
                **kwargs
            )

            if not hasattr(self, "_dbg_enc_once"):
                ok = torch.isfinite(output).all()
                print(f"[DBG/ENC] layer {lid} output finite? {bool(ok)} shape={tuple(output.shape)}")
                self._dbg_enc_once = True

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output




@TRANSFORMER_LAYER.register_module()
class BEVFormerLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])
        # temporal queue control (use your config’s queue_length if available)
        self.queue_length = getattr(self, "queue_length", 2)  # recommend 2 for your 6GB GPU
        self._bev_queue = []          # list of BEV tensors, each (bs, H*W, C), detached
        self._step_tag = None         # cache key to avoid appending multiple times per frame
        self._cached_value_bev = None # reuse within the same frame across sublayers
        import inspect
        print("[ENC/LAYER FILE]", inspect.getfile(self.__class__), flush=True)



    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                mlvl_feats=None,
                **kwargs):

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query


        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        # ----- ensure we have per-level feats + shapes BEFORE the loop -----
        # Try to get multi-level features from kwargs or from `value` if it's a list/tuple.
        mlvl_feats_list = kwargs.get('mlvl_feats', None)
        if mlvl_feats_list is None and isinstance(value, (list, tuple)) and len(value) > 0:
            mlvl_feats_list = value

        # If shapes are missing, derive them from per-level features.
        if (spatial_shapes is None or level_start_index is None):
            assert mlvl_feats_list is not None and len(mlvl_feats_list) > 0, \
                "BEVFormerEncoder: need mlvl_feats (list of [B,N_cam,C,H,W]) or precomputed spatial_shapes/level_start_index"
            device = mlvl_feats_list[0].device
            spatial_shapes = torch.as_tensor(
                [[feat.size(-2), feat.size(-1)] for feat in mlvl_feats_list],
                dtype=torch.long, device=device
            )
            level_start_index = torch.cat((
                spatial_shapes.new_zeros(1),
                (spatial_shapes[:, 0] * spatial_shapes[:, 1]).cumsum(0)[:-1]
            ))

        # If `value` is still a list/tuple, flatten it to a tensor [B, N_cam*S, C]
        if isinstance(value, (list, tuple)):
            # Each feat: [B, N_cam, C, H, W]
            per_level = [x.flatten(3).transpose(3, 2).contiguous() for x in value]  # [B, N_cam, HW, C]
            B, N_cam, HW0, C = per_level[0].shape
            S = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item())
            value = torch.cat(per_level, dim=2)         # [B, N_cam, S, C]
            value = value.reshape(B, N_cam * S, C)      # [B, N_cam*S, C]

        # Infer runtime camera count for slicing reference_points_cam later
        num_cam_rt = None
        if 'reference_points_cam' in locals() and reference_points_cam is not None:
            num_cam_rt = reference_points_cam.size(1)
        elif mlvl_feats_list is not None and len(mlvl_feats_list) > 0 and mlvl_feats_list[0].dim() >= 5:
            num_cam_rt = mlvl_feats_list[0].size(1)

        depth = kwargs.get('depth', None)
        depth_z = kwargs.get('depth_z', None)
        lidar_bev = kwargs.get('lidar_bev', None)

        # -------- ensure per-level feats + shapes BEFORE the loop --------
        # Prefer kwargs['mlvl_feats'] if provided; otherwise use `value` when it is a list/tuple
        mlvl_feats_list = kwargs.get('mlvl_feats', None)
        if mlvl_feats_list is None and isinstance(value, (list, tuple)) and len(value) > 0:
            mlvl_feats_list = value

        assert mlvl_feats_list is not None and len(mlvl_feats_list) > 0, \
            "Encoder needs per-level image feats (list of [B,N_cam,C,H,W]). "\
            "Pass them via kwargs['mlvl_feats'] or keep `value` as a list before flattening."

        # Build spatial shapes and level starts from per-level H,W
        spatial_shapes = torch.as_tensor(
            [[feat.size(-2), feat.size(-1)] for feat in mlvl_feats_list],
            dtype=torch.long, device=mlvl_feats_list[0].device
        )
        level_start_index = torch.cat((
            spatial_shapes.new_zeros(1),
            (spatial_shapes[:, 0] * spatial_shapes[:, 1]).cumsum(0)[:-1]
        ))

        # Flatten per-level feats to a single tensor for attention value if your code expects a tensor
        # Each feat is [B, N_cam, C, H, W]
        per_level = [x.flatten(3).transpose(3, 2).contiguous() for x in mlvl_feats_list]  # -> [B, N_cam, HW, C]
        B, N_cam, _, C = per_level[0].shape
        S = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item())
        value = torch.cat(per_level, dim=2).reshape(B, N_cam * S, C)  # [B, N_cam*S, C]
        # ----------------------------------------------------------------

        # -------- ensure per-level feats + shapes BEFORE the loop --------
        mlvl_feats_list = kwargs.get('mlvl_feats', None)
        if mlvl_feats_list is None and isinstance(value, (list, tuple)) and len(value) > 0:
            mlvl_feats_list = value  # each feat: [B, N_cam, C, H, W]

        assert mlvl_feats_list is not None and len(mlvl_feats_list) > 0, \
            "Encoder needs per-level image feats (list of [B,N_cam,C,H,W]) before cross-attn."

        spatial_shapes = torch.as_tensor(
            [[f.size(-2), f.size(-1)] for f in mlvl_feats_list],
            dtype=torch.long, device=mlvl_feats_list[0].device
        )
        level_start_index = torch.cat((
            spatial_shapes.new_zeros(1),
            (spatial_shapes[:, 0] * spatial_shapes[:, 1]).cumsum(0)[:-1]
        ))

        # flatten per-level feats into value tensor if value is still a list/tuple
        if isinstance(value, (list, tuple)):
            per_level = [f.flatten(3).transpose(3, 2).contiguous() for f in mlvl_feats_list]  # [B, N_cam, HW, C]
            B, N_cam, _, C = per_level[0].shape
            S = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item())
            value = torch.cat(per_level, dim=2).reshape(B, N_cam * S, C)  # [B, N_cam*S, C]
        # ----------------------------------------------------------------

        # safe defaults for optional extras
        depth   = kwargs.get('depth', None)
        depth_z = kwargs.get('depth_z', None)
        lidar_bev = kwargs.get('lidar_bev', None)
        
        # ---- ensure BEV reference points exist (2D + 3D) ----
        # ref_2d: [B, bev_h*bev_w, 2] in normalized [0,1] coords
        # ref_3d: [B, bev_h*bev_w, 3] (z is 0), used by spatial cross-attn

        B = query.size(0)
        device = query.device

        need_ref2d = ('ref_2d' not in locals()) or (ref_2d is None)
        need_ref3d = ('ref_3d' not in locals()) or (ref_3d is None)

        if need_ref2d or need_ref3d:
            xs = torch.linspace(0.5 / bev_w, 1.0 - 0.5 / bev_w, bev_w, device=device)
            ys = torch.linspace(0.5 / bev_h, 1.0 - 0.5 / bev_h, bev_h, device=device)
            gy, gx = torch.meshgrid(ys, xs)  # gy: H, gx: W
            ref2d_one = torch.stack([gx, gy], dim=-1).reshape(-1, 2)                # [H*W, 2]
            ref2d_all = ref2d_one.unsqueeze(0).repeat(B, 1, 1).contiguous()         # [B, H*W, 2]

            if need_ref2d:
                ref_2d = ref2d_all
            if need_ref3d:
                zeros_z = torch.zeros(B, ref2d_all.size(1), 1, device=device)
                ref_3d = torch.cat([ref2d_all, zeros_z], dim=-1).contiguous()       # [B, H*W, 3]
        # -----------------------------------------------------


        for layer in self.operation_order:
            # temporal self attention

            # --- sanity checks for TSA inputs ---
            print("[CALLSITE] query", tuple(query.shape),
                "prev_bev", None if prev_bev is None else tuple(prev_bev.shape),
                "spatial_shapes", None if spatial_shapes is None else spatial_shapes.tolist())


            if layer == 'self_attn':

                if prev_bev is None:
                    raise AssertionError("[TSA callsite] prev_bev is None; temporal attention disabled for this step.")
                
                    # Build a CLEAN kwargs just for TSA (do NOT reuse kwargs from SCA)
                tsa_kwargs = {
                    "reference_points": ref_2d,  # should correspond to BEV tokens
                    "spatial_shapes": torch.tensor([[bev_h, bev_w]], device=query.device),
                    "level_start_index": torch.tensor([0], device=query.device),
                    "bev_h": bev_h,
                    "bev_w": bev_w,
                }

                print("[CALLSITE/TSA]",
                    "query", tuple(query.shape),
                    "prev_bev", None if prev_bev is None else tuple(prev_bev.shape),
                    "spatial_shapes", tsa_kwargs["spatial_shapes"].tolist())
    
                # ------------------ build temporal stack for TSA (inside EncoderLayer) ------------------
                bs, S_now, C = query.shape  # query is (bs, H*W, C) here (batch-first)
                # Get BEV size; if you already have bev_h/bev_w in scope, use them; otherwise derive:
                H = int(locals().get('bev_h', None) or kwargs.get('bev_h', None) or (S_now ** 0.5))
                W = int(locals().get('bev_w', None) or kwargs.get('bev_w', None) or (S_now ** 0.5))
                assert H * W == S_now, f"[ENC/TSA] expected H*W tokens; got {S_now} (H={H}, W={W})"

                L = int(getattr(self, "queue_length", 2))
                L = max(L, 1)

                # --- build a per-frame step tag to avoid multiple appends in the same forward ---
                img_metas = kwargs.get("img_metas", None)
                if img_metas and isinstance(img_metas, (list, tuple)) and len(img_metas) > 0 and isinstance(img_metas[0], dict):
                    # prefer stable identifiers if present
                    step_tag = img_metas[0].get("token", None) or img_metas[0].get("timestamp", None) \
                            or img_metas[0].get("frame_id", None)
                else:
                    # fallback: use id(query) so each outer call produces a unique tag
                    step_tag = id(query)

                # If this is a new frame, update the queue and invalidate cache
                if step_tag != self._step_tag:
                    self._step_tag = step_tag
                    # Optional: detect scene reset if your metas expose it
                    is_first = False
                    if img_metas and isinstance(img_metas[0], dict):
                        is_first = (img_metas[0].get("prev_bev", None) is None) or (img_metas[0].get("frame_id", 0) in (0, "0"))
                    if is_first:
                        self._bev_queue = []

                    # push current (detach history to save memory)
                    self._bev_queue.append(query.detach())
                    # keep last L-1 past frames
                    if len(self._bev_queue) > (L - 1):
                        self._bev_queue = self._bev_queue[-(L - 1):]

                    # assemble frames: past (<=L-1) + current, pad with current until warm
                    frames = list(self._bev_queue) + [query]
                    while len(frames) < L:
                        frames.insert(0, query)

                    value_bev = torch.cat(frames, dim=1).contiguous()  # (bs, L*H*W, C)
                    self._cached_value_bev = value_bev
                else:
                    # same frame within this forward; reuse cached stack
                    assert self._cached_value_bev is not None, "[ENC/TSA] cache missing for same-step reuse"
                    value_bev = self._cached_value_bev

                S_v = value_bev.size(1)
                expected = L * (H * W)
                assert S_v == expected, f"[ENC/TSA] S_v {S_v} != L*H*W {expected} (L={L}, H={H}, W={W})"

                # one-time confirmation (will print once per process)
                if not hasattr(self, "_enc_tsa_dbg"):
                    print(f"[ENC/TSA] bs={bs}, H*W={H*W}, L={L}, S_v={S_v}, "
                        f"q={tuple(query.shape)}, v_bev={tuple(value_bev.shape)}")
                    self._enc_tsa_dbg = True
                # ------------------ end temporal stack builder ------------------

                # ------------------ call TSA with guaranteed multi-frame 'value' ------------------
                query = self.attentions[attn_index](
                    query,                # (bs, H*W, C) - current frame as Q
                    value_bev,            # K (TSA will project internally)
                    value_bev,            # V (ditto)
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,                              # BEV positional enc
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    # DO NOT pass external spatial_shapes/level_start_index for TSA
                    bev_h=H, bev_w=W,                             # let TSA build its BEV shapes
                    **kwargs
                )
                attn_index += 1
                identity = query
                # -----------------------------------------------------------------


            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                # cache shapes on the attention module as a last-resort fallback
                attn_mod = self.attentions[attn_index]
                setattr(attn_mod, 'cached_shapes', (spatial_shapes, level_start_index))

                new_query1 = attn_mod(
                    query, key, value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos, key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index] if attn_masks is not None else None,
                    key_padding_mask=key_padding_mask,

                    # pass shapes under BOTH naming styles
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    value_spatial_shapes=spatial_shapes,
                    value_level_start_index=level_start_index,

                    # also pass the per-level list under both common names
                    mlvl_feats=mlvl_feats_list,
                    value_list=mlvl_feats_list,

                    depth=depth, lidar_bev=lidar_bev, depth_z=depth_z,
                    **kwargs
                )
                # ... (your lidar_cross_attn_layer blend stays the same)


                if self.lidar_cross_attn_layer:
                    bs = query.size(0)
                    ref2d_slice = ref_2d[bs:] if (ref_2d is not None and ref_2d.size(0) >= bs * 2) else ref_2d
                    new_query2 = self.lidar_cross_attn_layer(
                        query,
                        lidar_bev,
                        lidar_bev,
                        reference_points=ref2d_slice,
                        spatial_shapes=torch.as_tensor([[bev_h, bev_w]], dtype=torch.long, device=query.device),
                        level_start_index=torch.zeros(1, dtype=torch.long, device=query.device),
                    )
                else:
                    new_query2 = None

                query = new_query1 if new_query2 is None else (
                    new_query1 * self.cross_model_weights + (1 - self.cross_model_weights) * new_query2
                )
                attn_index += 1
                identity = query


            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query




from mmcv.cnn.bricks.transformer import build_feedforward_network, build_attention


@TRANSFORMER_LAYER.register_module()
class MM_BEVFormerLayer(MyCustomBaseTransformerLayer):
    """multi-modality fusion layer.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 lidar_cross_attn_layer=None,
                 **kwargs):
        super(MM_BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])
        self.cross_model_weights = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True) 
        if lidar_cross_attn_layer:
            self.lidar_cross_attn_layer = build_attention(lidar_cross_attn_layer)
            # self.cross_model_weights+=1
        else:
            self.lidar_cross_attn_layer = None


    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                debug=False,
                depth=None,
                depth_z=None,
                lidar_bev=None,
                radar_bev=None,
                **kwargs):
        """Forward function for MM_BEVFormerLayer.

        Args:
            query (Tensor): (bs, num_queries, embed_dims) if batch_first.
            key (Tensor): Camera features.
            value (Tensor): Camera features.
            lidar_bev (Tensor): Raw lidar BEV (unused when lidar_bev_tokens provided).
            **kwargs may contain:
                lidar_bev_tokens (Tensor): Pre-projected lidar tokens
                    (B, HW, C) from PerceptionTransformer.
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        # ----- helpers / locals -----
        mlvl_feats_local = kwargs.get('mlvl_feats', None)
        if mlvl_feats_local is None and isinstance(value, (list, tuple)) and len(value) > 0:
            mlvl_feats_local = value

        # Pre-projected lidar tokens from transformer (encoder-side fusion)
        lidar_bev_tokens = kwargs.get('lidar_bev_tokens', None)

        def _safe_attn_mask(i):
            return attn_masks[i] if (attn_masks is not None and i < len(attn_masks)) else None

        # single-level BEV grid (for temporal SA and LiDAR SCA)
        bev_spatial_shapes = torch.as_tensor([[bev_h, bev_w]],
                                             dtype=torch.long, device=query.device)
        bev_level_start_index = torch.zeros(1, dtype=torch.long, device=query.device)

        # multi-level image feature shapes for cross-attn
        if spatial_shapes is None or level_start_index is None:
            assert mlvl_feats_local is not None and len(mlvl_feats_local) > 0, \
                "Need mlvl_feats (or pass spatial_shapes/level_start_index)"
            spatial_shapes = torch.as_tensor(
                [[x.size(-2), x.size(-1)] for x in mlvl_feats_local],
                dtype=torch.long, device=query.device
            )
            level_start_index = torch.cat((
                spatial_shapes.new_zeros(1),
                (spatial_shapes[:, 0] * spatial_shapes[:, 1]).cumsum(0)[:-1]
            ))

        # infer runtime camera count if available
        num_cam_rt = None
        if reference_points_cam is not None:
            num_cam_rt = reference_points_cam.size(1)
        elif mlvl_feats_local is not None and len(mlvl_feats_local) > 0 and mlvl_feats_local[0].dim() >= 5:
            num_cam_rt = mlvl_feats_local[0].size(1)

        for layer in self.operation_order:
            # -------- temporal self attention --------
            if layer == 'self_attn':
                if prev_bev is None:
                    attn_index += 1
                    continue
                query = self.attentions[attn_index](
                    query,
                    prev_bev, prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos, key_pos=bev_pos,
                    attn_mask=_safe_attn_mask(attn_index),
                    key_padding_mask=query_key_padding_mask,
                    lidar_bev=lidar_bev,
                    reference_points=ref_2d,
                    spatial_shapes=bev_spatial_shapes,
                    level_start_index=bev_level_start_index,
                    mlvl_feats=None,
                    bev_h=bev_h,
                    bev_w=bev_w,
                    **kwargs
                )
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # -------- spatial cross attention --------
            elif layer == 'cross_attn':
                new_query1 = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=(
                        reference_points_cam[:, :num_cam_rt]
                        if (reference_points_cam is not None and num_cam_rt is not None)
                        else reference_points_cam
                    ),
                    mask=mask,
                    attn_mask=_safe_attn_mask(attn_index),
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    depth=depth,
                    lidar_bev=lidar_bev,
                    depth_z=depth_z,
                    mlvl_feats=mlvl_feats_local,
                    num_cam_rt=num_cam_rt,
                    **kwargs
                )

                # ---- LiDAR SCA (encoder-side fusion) ----
                new_query2 = None
                if getattr(self, 'lidar_cross_attn_layer', None) is not None and lidar_bev_tokens is not None:
                    # Use pre-projected lidar_bev_tokens (B, HW, C) as key/value
                    # ref_2d is (B, HW, 1, 2) — BEV grid coords, perfect for deformable attn
                    new_query2 = self.lidar_cross_attn_layer(
                        query,
                        lidar_bev_tokens,
                        lidar_bev_tokens,
                        reference_points=ref_2d,
                        spatial_shapes=bev_spatial_shapes,
                        level_start_index=bev_level_start_index,
                    )

                if new_query2 is not None:
                    query = new_query1 * self.cross_model_weights + (1 - self.cross_model_weights) * new_query2
                else:
                    query = new_query1

                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


