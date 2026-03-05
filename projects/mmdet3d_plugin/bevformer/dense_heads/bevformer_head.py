import copy
import torch
import math

import torch.nn as nn

from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmcv.runner import force_fp32, auto_fp16
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox


def wrap_to_pi(x):
    # x: Tensor
    return torch.remainder(x + math.pi, 2 * math.pi) - math.pi

def yaw_from_sincos(sin_yaw, cos_yaw):
    return torch.atan2(sin_yaw, cos_yaw)  # [-pi, pi]

def encode_yaw_to_bins(yaw, num_bins: int):
    """
    yaw: Tensor in radians (any range). We treat bins over [-pi, pi).
    Returns:
      bin_idx: LongTensor
      res_sin, res_cos: Tensor residual on unit circle relative to bin center
    """
    yaw = wrap_to_pi(yaw)
    bin_size = 2 * math.pi / num_bins

    # map [-pi, pi) -> [0, 2pi)
    yaw_0 = yaw + math.pi

    bin_idx = torch.floor(yaw_0 / bin_size).long()
    bin_idx = torch.clamp(bin_idx, 0, num_bins - 1)

    # bin center in [-pi, pi)
    center = (bin_idx.float() + 0.5) * bin_size - math.pi

    # residual angle in [-pi, pi)
    res = wrap_to_pi(yaw - center)

    res_sin = torch.sin(res)
    res_cos = torch.cos(res)
    return bin_idx, res_sin, res_cos

def encode_yaw_to_bins_from_sincos(sin_yaw, cos_yaw, num_bins):
    yaw = torch.atan2(sin_yaw, cos_yaw)  # [-pi, pi]
    return encode_yaw_to_bins(yaw, num_bins)


def overwrite_sincos_from_bins(bbox_preds, num_bins: int, bin_start: int):
    """
    bbox_preds: [K, code_dim] where:
      - bin logits are bbox_preds[..., bin_start : bin_start+num_bins]
      - residual (sin,cos) are bbox_preds[..., bin_start+num_bins : bin_start+num_bins+2]
    This overwrites bbox_preds[..., 6:8] = final (sin,cos) global.
    """
    logits = bbox_preds[..., bin_start: bin_start + num_bins]
    res_sc = bbox_preds[..., bin_start + num_bins: bin_start + num_bins + 2]

    # argmax bin
    bin_idx = torch.argmax(logits, dim=-1)  # [K]
    two_pi = bbox_preds.new_tensor(2.0 * math.pi)
    bin_size = two_pi / float(num_bins)
    bin_center = (bin_idx.to(bbox_preds.dtype) + 0.5) * bin_size - math.pi

    # residual angle predicted
    res_yaw = torch.atan2(res_sc[..., 0], res_sc[..., 1])

    yaw = bin_center + res_yaw
    yaw = torch.remainder(yaw + math.pi, two_pi) - math.pi

    bbox_preds = bbox_preds.clone()
    bbox_preds[..., 6] = torch.sin(yaw)
    bbox_preds[..., 7] = torch.cos(yaw)
    return bbox_preds


@HEADS.register_module()
class BEVFormerHead(DETRHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=[1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 2.0, 2.0, 0.5, 0.5],
                 bev_h=100,
                 bev_w=100,
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        self.num_yaw_bins = 24  # must be set before super().__init__ which calls _init_layers()
        super(BEVFormerHead, self).__init__(
            *args, transformer=transformer, **kwargs)

        # ---- code_weights (must be after Module.__init__ via super()) ----
        if code_weights is None:
            cw = torch.tensor([1., 1., 1., 1., 1., 1., 2., 2., 0.5, 0.5], dtype=torch.float32)
        else:
            cw = torch.tensor(code_weights, dtype=torch.float32)
        self.register_buffer('code_weights', cw, persistent=False)

        # yaw bin/res losses
        self.loss_yaw_bin = torch.nn.CrossEntropyLoss(reduction='none')
        self.loss_yaw_res = torch.nn.SmoothL1Loss(reduction='none')

        # velocity head loss (camera-only BEV -> vx, vy)
        self.loss_vel = nn.SmoothL1Loss(reduction='none')
        self.loss_vel_weight = 0.25

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        # ---------------- yaw discretization heads (bin + residual) ----------------

        yaw_bin_branch = nn.Sequential(
            Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            Linear(self.embed_dims, self.num_yaw_bins)
        )

        yaw_res_branch = nn.Sequential(
            Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            Linear(self.embed_dims, 2)  # (res_sin, res_cos)
        )
        # --------------------------------------------------------------------------

        # velocity MLP head (camera-only BEV -> vx, vy)
        vel_branch = nn.Sequential(
            Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            Linear(self.embed_dims, 2)  # vx, vy
        )

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
            self.yaw_bin_branches = _get_clones(yaw_bin_branch, num_pred)
            self.yaw_res_branches = _get_clones(yaw_res_branch, num_pred)
            self.vel_branches = _get_clones(vel_branch, num_pred)
        else:
            # IMPORTANT: must deep-copy so each layer has its own head weights
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
            self.yaw_bin_branches = _get_clones(yaw_bin_branch, num_pred)
            self.yaw_res_branches = _get_clones(yaw_res_branch, num_pred)
            self.vel_branches = _get_clones(vel_branch, num_pred)

        # Velocity cross-attention: decoder queries attend to camera-only BEV
        self.vel_cross_attn = nn.ModuleList([
            nn.MultiheadAttention(self.embed_dims, num_heads=8, batch_first=True)
            for _ in range(num_pred)
        ])

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None,  only_bev=False, bev_lidar=None):
        """Forward function."""
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                            device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        if only_bev:
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                            self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                            self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev,
                bev_lidar=bev_lidar,
            )

        bev_embed, hs, init_reference, inter_references, bev_embed_cam = outputs
        hs = hs.permute(0, 2, 1, 3)

        outputs_classes = []
        outputs_coords = []

        outputs_yaw_bin_logits = []  # [nb_dec] each is [bs, num_query, num_bins]
        outputs_yaw_res_preds  = []  # [nb_dec] each is [bs, num_query, 2]
        outputs_vel_preds = []       # [nb_dec] each is [bs, num_query, 2]

        eps = 1e-6  # for sincos normalization

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]

            reference = inverse_sigmoid(reference)

            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            # ===== yaw bin + residual heads =====
            yaw_bin_logits = self.yaw_bin_branches[lvl](hs[lvl])   # [bs, num_query, num_bins]
            yaw_res_pred   = self.yaw_res_branches[lvl](hs[lvl])   # [bs, num_query, 2]
            # ===================================


            # ------------------------------------------------------------
            # ✅ FIX: enforce yaw channels to represent (sin, cos) stably
            # Your bbox format (used in loss/normalize_bbox):
            # [x, y, log(w), log(l), z, log(h), sin(yaw), cos(yaw), vx, vy]
            #
            # Without this, sin/cos stay tiny -> yaw becomes random -> mAOE ~ pi/2
            if tmp.size(-1) >= 8:
                sc = tmp[..., 6:8].tanh()                         # allow [-1, 1]
                sc = sc / (sc.norm(dim=-1, keepdim=True) + eps)   # normalize to unit circle
                tmp[..., 6:8] = sc
            # ------------------------------------------------------------

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3

            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()

            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                            self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                            self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                            self.pc_range[2]) + self.pc_range[2])

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

            outputs_yaw_bin_logits.append(yaw_bin_logits)
            outputs_yaw_res_preds.append(yaw_res_pred)

            # Velocity head: cross-attend decoder queries to camera-only BEV
            vel_context, _ = self.vel_cross_attn[lvl](hs[lvl], bev_embed_cam, bev_embed_cam)
            vel_pred = self.vel_branches[lvl](vel_context)  # (bs, num_query, 2)
            outputs_vel_preds.append(vel_pred)


        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outputs_yaw_bin_logits = torch.stack(outputs_yaw_bin_logits)  # [nb_dec, bs, num_query, num_bins]
        outputs_yaw_res_preds  = torch.stack(outputs_yaw_res_preds)   # [nb_dec, bs, num_query, 2]
        outputs_vel_preds = torch.stack(outputs_vel_preds)            # [nb_dec, bs, num_query, 2]

        outs = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'all_yaw_bin_logits': outputs_yaw_bin_logits,
            'all_yaw_res_preds': outputs_yaw_res_preds,
            'all_vel_preds': outputs_vel_preds,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }
        return outs


    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):


        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        assert gt_c >= 8, "GT bbox must have at least 8 dims (cx,cy,cz,w,l,h,yaw,vx)"
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        # -----------------------
        # yaw discretization GT
        # -----------------------
        NUM_BINS = self.num_yaw_bins
    

        # Allocate per-query targets (same num_query length)
        yaw_bin_targets = gt_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)  # -1 = ignore
        yaw_bin_weights = gt_bboxes.new_zeros((num_bboxes,), dtype=gt_bboxes.dtype)

        yaw_res_targets = gt_bboxes.new_zeros((num_bboxes, 2), dtype=gt_bboxes.dtype)  # (sin,cos)
        yaw_res_weights = gt_bboxes.new_zeros((num_bboxes, 2), dtype=gt_bboxes.dtype)

        if pos_inds.numel() > 0:
            # bbox_targets here is raw GT (pre-normalization): index 6 = yaw in radians
            gt_yaw = bbox_targets[pos_inds, 6]
            bin_idx, res_sin, res_cos = encode_yaw_to_bins(gt_yaw, num_bins=NUM_BINS)

            yaw_bin_targets[pos_inds] = bin_idx
            yaw_bin_weights[pos_inds] = 1.0

            yaw_res_targets[pos_inds, 0] = res_sin
            yaw_res_targets[pos_inds, 1] = res_cos
            yaw_res_weights[pos_inds, :] = 1.0

        return (labels, label_weights, bbox_targets, bbox_weights,
                yaw_bin_targets, yaw_bin_weights, yaw_res_targets, yaw_res_weights,
                pos_inds, neg_inds)


    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
 
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         yaw_bin_targets_list, yaw_bin_weights_list, yaw_res_targets_list, yaw_res_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))

        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
                yaw_bin_targets_list, yaw_bin_weights_list, yaw_res_targets_list, yaw_res_weights_list,
                num_total_pos, num_total_neg)


    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    yaw_bin_logits,
                    yaw_res_preds,
                    vel_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):

        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        assert yaw_bin_logits is not None
        assert yaw_res_preds is not None

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                        gt_bboxes_list, gt_labels_list,
                                        gt_bboxes_ignore_list)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
        yaw_bin_targets_list, yaw_bin_weights_list, yaw_res_targets_list, yaw_res_weights_list,
        num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        yaw_bin_targets = torch.cat(yaw_bin_targets_list, 0)      # [N]
        yaw_bin_weights = torch.cat(yaw_bin_weights_list, 0)      # [N] or [N,1]
        yaw_res_targets = torch.cat(yaw_res_targets_list, 0)      # [N,2] (sin,cos residual)
        yaw_res_weights = torch.cat(yaw_res_weights_list, 0)      # [N,2] or [N,1] broadcastable


        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        # ---- flatten yaw heads to match flattened labels/targets ----
        if yaw_bin_logits is not None:
            # yaw_bin_logits can be [bs, num_query, num_bins]
            if yaw_bin_logits.dim() == 3:
                yaw_bin_logits = yaw_bin_logits.reshape(-1, yaw_bin_logits.size(-1))
            elif yaw_bin_logits.dim() == 2:
                pass  # already [N, num_bins]
            else:
                raise RuntimeError(f"Unexpected yaw_bin_logits shape: {tuple(yaw_bin_logits.shape)}")

        if yaw_res_preds is not None:
            # yaw_res_preds can be [bs, num_query, 2]
            if yaw_res_preds.dim() == 3:
                yaw_res_preds = yaw_res_preds.reshape(-1, 2)
            elif yaw_res_preds.dim() == 2:
                pass  # already [N,2]
            else:
                raise RuntimeError(f"Unexpected yaw_res_preds shape: {tuple(yaw_res_preds.shape)}")
        # ------------------------------------------------------------

        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)

        bbox_weights = bbox_weights * self.code_weights
        # Disable yaw sin/cos in bbox regression — handled by yaw bin/res heads
        bbox_weights[:, 6] = 0.0
        bbox_weights[:, 7] = 0.0
        # Disable velocity in bbox regression — handled by dedicated velocity head
        bbox_weights[:, 8] = 0.0
        bbox_weights[:, 9] = 0.0

        # ---- prepare yaw predictions/targets for loss ----
        # yaw_bin_logits: shape [bs, num_query, num_bins] (for this decoder layer)
        # yaw_res_preds : shape [bs, num_query, 2] (for this decoder layer)
        # We expect loss_single receives yaw_bin_logits/yaw_res_preds arguments (per-layer).

        # flatten per-query dims to match how cls_scores/bbox_preds are flattened
        if yaw_bin_logits is not None and yaw_res_preds is not None:
            # reshape -> [bs * num_query, ...]
            BQ = yaw_bin_logits.shape[0] * yaw_bin_logits.shape[1]
            yaw_bin_logits_flat = yaw_bin_logits.reshape(-1, yaw_bin_logits.size(-1))  # [BQ, num_bins]
            yaw_res_preds_flat  = yaw_res_preds.reshape(-1, 2)                         # [BQ, 2]
        else:
            # safe defaults if caller passed None
            yaw_bin_logits_flat = None
            yaw_res_preds_flat = None

        # yaw targets were provided by get_targets and concatenated above:
        # yaw_bin_targets: [N] long (N = bs * num_query), yaw_bin_weights: [N]
        # yaw_res_targets: [N,2], yaw_res_weights: [N,2]
        # (make sure they exist in this scope — get_targets should return them)
        # If they are not yet available, you must modify get_targets to return them.

        # Mask of valid positions (same mask used for bbox regression)
        valid_mask = isnotnan  # boolean [N] produced earlier (finite targets)
        pos_mask = (labels != self.num_classes)  # positive queries (labels != bg)
        use_mask = valid_mask & pos_mask  # only supervise positives with finite bbox targets

        loss_yaw_bin = bbox_preds.new_tensor(0.0)
        loss_yaw_res = bbox_preds.new_tensor(0.0)

        if yaw_bin_logits_flat is not None and yaw_res_preds_flat is not None:
            # select only the entries we're supervising
            if use_mask.any():
                sel_idx = use_mask.nonzero(as_tuple=False).squeeze(-1)  # indices into flattened N
                # CE: targets need to be long and in [0, num_bins-1]
                bin_targets_sel = yaw_bin_targets[sel_idx].long().to(yaw_bin_logits_flat.device)
                # per-sample CE loss (reduction='none' at construction)
                loss_bin_all = self.loss_yaw_bin(yaw_bin_logits_flat[sel_idx], bin_targets_sel)  # [num_pos]
                # apply per-sample weights if provided
                if yaw_bin_weights is not None:
                    weights_sel = yaw_bin_weights[sel_idx].to(loss_bin_all.device)
                    loss_yaw_bin = (loss_bin_all * weights_sel).sum() / max(1.0, float(use_mask.sum().item()))
                else:
                    loss_yaw_bin = loss_bin_all.mean()

                # residual loss (SmoothL1, reduction='none') -> per-element [num_pos,2]
                res_pred_sel = yaw_res_preds_flat[sel_idx]
                res_tgt_sel  = yaw_res_targets[sel_idx].to(res_pred_sel.device)
                res_loss_all = self.loss_yaw_res(res_pred_sel, res_tgt_sel)  # [num_pos,2]
                # optional per-element weights
                if yaw_res_weights is not None:
                    res_w_sel = yaw_res_weights[sel_idx].to(res_loss_all.device)
                    res_loss_all = res_loss_all * res_w_sel
                # sum over elements then average by number positives
                loss_yaw_res = res_loss_all.sum() / max(1.0, float(use_mask.sum().item()))
            else:
                loss_yaw_bin = bbox_preds.new_tensor(0.0)
                loss_yaw_res = bbox_preds.new_tensor(0.0)

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan,
                                                               :10], bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)

        # --- velocity loss (camera-only BEV head, positives only) ---
        vel_preds = vel_preds.reshape(-1, 2)
        pos_isnotnan = isnotnan & pos_mask  # only positive queries with finite targets
        vel_targets = normalized_bbox_targets[pos_isnotnan, 8:10]  # vx, vy
        if vel_targets.numel() > 0:
            loss_vel_raw = self.loss_vel(vel_preds[pos_isnotnan], vel_targets)
            loss_vel = loss_vel_raw.mean() * self.loss_vel_weight
        else:
            loss_vel = vel_preds.sum() * 0.0

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_vel = torch.nan_to_num(loss_vel)
        return loss_cls, loss_bbox, loss_yaw_bin, loss_yaw_res, loss_vel


    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):

        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        all_yaw_bin_logits = preds_dicts['all_yaw_bin_logits']  # [nb_dec, bs, num_query, num_bins]
        all_yaw_res_preds  = preds_dicts['all_yaw_res_preds']   # [nb_dec, bs, num_query, 2]
        all_vel_preds      = preds_dicts['all_vel_preds']        # [nb_dec, bs, num_query, 2]

        YAW_BIN_W = 0.2
        YAW_RES_W = 0.2
        VEL_W = 1.0

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]
        
        
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox, losses_yaw_bin, losses_yaw_res, losses_vel = multi_apply(
            self.loss_single,
            all_cls_scores,         # [nb_dec, bs, nq, C]
            all_bbox_preds,         # [nb_dec, bs, nq, code]
            all_yaw_bin_logits,     # [nb_dec, bs, nq, bins]
            all_yaw_res_preds,      # [nb_dec, bs, nq, 2]
            all_vel_preds,          # [nb_dec, bs, nq, 2]
            all_gt_bboxes_list,
            all_gt_labels_list,
            all_gt_bboxes_ignore_list
        )

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        loss_dict['loss_yaw_bin'] = losses_yaw_bin[-1] * YAW_BIN_W
        loss_dict['loss_yaw_res'] = losses_yaw_res[-1] * YAW_RES_W
        loss_dict['loss_vel'] = losses_vel[-1] * VEL_W

        # loss from other decoder layers (exclude last)
        num_dec_layer = 0
        for (loss_cls_i, loss_bbox_i, loss_yaw_bin_i, loss_yaw_res_i, loss_vel_i) in zip(
                losses_cls[:-1], losses_bbox[:-1], losses_yaw_bin[:-1], losses_yaw_res[:-1], losses_vel[:-1]):

            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_yaw_bin'] = loss_yaw_bin_i * YAW_BIN_W
            loss_dict[f'd{num_dec_layer}.loss_yaw_res'] = loss_yaw_res_i * YAW_RES_W
            loss_dict[f'd{num_dec_layer}.loss_vel'] = loss_vel_i * VEL_W
            num_dec_layer += 1

        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']

            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']

            ret_list.append([bboxes, scores, labels])

        return ret_list


@HEADS.register_module()
class BEVFormerHead_GroupDETR(BEVFormerHead):
    """GroupDETR variant. NOTE: Does not support LiDAR fusion, yaw bin/res
    heads, or the velocity head. Use BEVFormerHead for fusion training."""

    def __init__(self,
                 *args,
                 group_detr=1,
                 **kwargs):
        self.group_detr = group_detr
        assert 'num_query' in kwargs
        kwargs['num_query'] = group_detr * kwargs['num_query']
        super().__init__(*args, **kwargs)

    def forward(self, mlvl_feats, img_metas, prev_bev=None,  only_bev=False):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        if not self.training:  # NOTE: Only difference to bevformer head
            object_query_embeds = object_query_embeds[:self.num_query // self.group_detr]
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        if only_bev:
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev
        )

        bev_embed, hs, init_reference, inter_references, _bev_embed_cam = outputs
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                             self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                             self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                             self.pc_range[2]) + self.pc_range[2])
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }

        return outs

    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        assert enc_cls_scores is None and enc_bbox_preds is None 

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        loss_dict = dict()
        loss_dict['loss_cls'] = 0
        loss_dict['loss_bbox'] = 0
        for num_dec_layer in range(all_cls_scores.shape[0] - 1):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = 0
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = 0
        num_query_per_group = self.num_query // self.group_detr
        for group_index in range(self.group_detr):
            group_query_start = group_index * num_query_per_group
            group_query_end = (group_index+1) * num_query_per_group
            group_cls_scores =  all_cls_scores[:, :,group_query_start:group_query_end, :]
            group_bbox_preds = all_bbox_preds[:, :,group_query_start:group_query_end, :]
            losses_cls, losses_bbox = multi_apply(
                self.loss_single, group_cls_scores, group_bbox_preds,
                all_gt_bboxes_list, all_gt_labels_list,
                all_gt_bboxes_ignore_list)
            loss_dict['loss_cls'] += losses_cls[-1] / self.group_detr
            loss_dict['loss_bbox'] += losses_bbox[-1] / self.group_detr
            # loss from other decoder layers
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.loss_cls'] += loss_cls_i / self.group_detr
                loss_dict[f'd{num_dec_layer}.loss_bbox'] += loss_bbox_i / self.group_detr
                num_dec_layer += 1
        return loss_dict