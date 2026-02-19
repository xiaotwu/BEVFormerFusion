import copy
import torch
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
import math
import os as _os
import torch.distributed as dist

def is_main():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


@HEADS.register_module()
class BEVFormerHead(DETRHead):

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=None,
                 bev_w=None,
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
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1


        # --- sync code_size with bbox_coder ---
        if hasattr(self.bbox_coder, "code_size"):
            if getattr(self, "code_size", None) is None:
                self.code_size = self.bbox_coder.code_size
            elif self.code_size != self.bbox_coder.code_size:
                print(f"[HEAD] overriding code_size from {self.code_size} "
                    f"to bbox_coder.code_size={self.bbox_coder.code_size}")
                self.code_size = self.bbox_coder.code_size
        # --------------------------------------

        super(BEVFormerHead, self).__init__(
            *args, transformer=transformer, **kwargs)
        
        # Adjust code_weights length to match code_size
        if len(self.code_weights) != self.code_size:
            print(f"[HEAD] adjusting code_weights from {len(self.code_weights)} to {self.code_size}")
            self.code_weights = self.code_weights[:self.code_size]

        self.code_weights = nn.Parameter(
            torch.tensor(self.code_weights, requires_grad=False),
            requires_grad=False
        )


        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)

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

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

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
    def forward(self, mlvl_feats, img_metas, prev_bev=None,  only_bev=False):

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
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

        bev_embed, hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)

        # --- BEGIN: decoder state firewall ---
        # hs: [num_dec_layers, B, Q, C] after your permute
        hs = torch.nan_to_num(hs, nan=0.0, posinf=0.0, neginf=0.0)
        hs = hs.clamp(-20.0, 20.0)
        # --- END ---

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]

            # --- BEGIN: clamp BEFORE inverse_sigmoid ---
            # Protect against exactly 0/1 or NaNs coming from the transformer.
            reference = torch.nan_to_num(reference, nan=0.5, posinf=0.5, neginf=0.5)
            reference = reference.clamp(1e-4, 1 - 1e-4)
            reference = inverse_sigmoid(reference)
            # --- END ---

            # IMPORTANT: sanitize inputs to the linears
            x = hs[lvl]  # shape should match your head expectations
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)

            outputs_class = self.cls_branches[lvl](x)
            tmp = self.reg_branches[lvl](x)

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

            # TODO: check if using sigmoid
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
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights,
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

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))


        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):

        num_imgs = cls_scores.size(0)

        # ----- per-image lists for target assignment -----
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]

        (labels_list, label_weights_list,
         bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = self.get_targets(
            cls_scores_list,
            bbox_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            gt_bboxes_ignore_list
        )

        # ----- concat over batch -----
        labels = torch.cat(labels_list, 0)             # [B*Q]
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0) # [B*Q, T]
        bbox_weights = torch.cat(bbox_weights_list, 0) # [B*Q, T]

        # ================== CLASSIFICATION LOSS ==================
        cls_scores_flat = cls_scores.reshape(-1, self.cls_out_channels)

        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores_flat.new_tensor([cls_avg_factor])
            )
        cls_avg_factor = max(cls_avg_factor, 1.0)

        if not hasattr(self, "_dbg_loss_once"):
            print("[HEAD] cls_scores_flat:", cls_scores_flat.shape)
            print("[HEAD] labels unique:", labels.unique())
            print("[HEAD] label_weights min/max:",
                  label_weights.min().item(), label_weights.max().item())
            print("[HEAD] num_total_pos:", num_total_pos,
                  "num_total_neg:", num_total_neg)
            self._dbg_loss_once = True

        if not torch.isfinite(cls_scores_flat).all():
            bad = ~torch.isfinite(cls_scores_flat)
            #print("[WARN/HEAD] non-finite cls_scores_flat, nan_to_num on",
            #      bad.nonzero().size(0), "entries")
            cls_scores_flat = torch.nan_to_num(
                cls_scores_flat, nan=0.0, posinf=0.0, neginf=0.0
            )

        loss_cls = self.loss_cls(
            cls_scores_flat,
            labels,
            label_weights,
            avg_factor=cls_avg_factor
        )

        # average number of gt boxes across GPUs for regression normalization
        num_total_pos_tensor = loss_cls.new_tensor([num_total_pos])
        num_total_pos_tensor = torch.clamp(
            reduce_mean(num_total_pos_tensor), min=1
        )
        num_total_pos_avg = num_total_pos_tensor.item()

        # ---------------- regression loss (L1 on encoded box codes) ----------------
        # Flatten predictions/targets/weights
        bbox_preds_flat = bbox_preds.reshape(-1, bbox_preds.size(-1))   # [B*Q, P]
        bbox_targets_flat = bbox_targets.reshape(-1, bbox_targets.size(-1))  # [B*Q, T]
        bbox_weights_flat = bbox_weights.reshape(-1, bbox_weights.size(-1))  # [B*Q, T]

        # Use TARGET code dim as the truth
        code_dim_tgt = bbox_targets_flat.size(-1)

        # One-time debug
        if not hasattr(self, "_dbg_bbox_once"):
            print("[HEAD] bbox_preds_flat shape:", bbox_preds_flat.shape)
            print("[HEAD] bbox_targets_flat shape:", bbox_targets_flat.shape)
            print("[HEAD] bbox_weights_flat shape:", bbox_weights_flat.shape)
            print("[HEAD] using code_dim_tgt =", code_dim_tgt)
            self._dbg_bbox_once = True

        # Slice preds and weights to match target dim
        if bbox_preds_flat.size(-1) != code_dim_tgt and not hasattr(self, "_warn_trim_pred_once"):
            #print(f"[WARN/HEAD] trimming bbox_preds_flat last dim "
            #      f"{bbox_preds_flat.size(-1)} -> {code_dim_tgt}")
            bbox_preds_flat = bbox_preds_flat[:, :code_dim_tgt]

        if bbox_weights_flat.size(-1) != code_dim_tgt and not hasattr(self, "_warn_trim_pred_once"):
            #print(f"[WARN/HEAD] trimming bbox_weights_flat last dim "
            #      f"{bbox_weights_flat.size(-1)} -> {code_dim_tgt}")
            bbox_weights_flat = bbox_weights_flat[:, :code_dim_tgt]

        # validity mask from targets
        isnotnan = torch.isfinite(bbox_targets_flat).all(dim=-1)

        bbox_preds_valid   = bbox_preds_flat[isnotnan]
        bbox_targets_valid = bbox_targets_flat[isnotnan]
        bbox_weights_valid = bbox_weights_flat[isnotnan]

        # apply code-wise weights (also slice to code_dim_tgt)
        code_weights = self.code_weights[:code_dim_tgt]
        bbox_weights_valid = bbox_weights_valid * code_weights

        # final safety: if no positives, bbox loss = 0
        if bbox_targets_valid.numel() == 0:
            loss_bbox = bbox_preds_valid.new_tensor(0.0)
        else:
            # now pred and target must match in shape
            # (same N, same code_dim_tgt)
            loss_bbox = self.loss_bbox(
                bbox_preds_valid,
                bbox_targets_valid,
                bbox_weights_valid,
                avg_factor=num_total_pos_avg
            )


        # final NaN safety
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)

        return loss_cls, loss_bbox



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

        # DEBUG: once per run
        if not hasattr(self, "_dbg_cls_once"):
            self._dbg_cls_once = True
            s = all_cls_scores[-1].detach()  # [B, num_query, num_classes]
            print("[DBG][Head] cls logits mean/min/max:",
                float(s.mean()), float(s.min()), float(s.max()))
            sp = s.sigmoid()
            print("[DBG][Head] cls probs max:", float(sp.max()),
                "  frac>1e-2:", float((sp>1e-2).float().mean()))

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

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list)

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

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1

        # ---- logging: GT class -> predicted class (last decoder only) ----
        if not hasattr(self, "_dbg_step"):
            self._dbg_step = 0
        self._dbg_step += 1

        import os
        every = int(os.getenv("BEV_LOG_EVERY", "50"))  # align with your log interval
        self._log_iter = getattr(self, "_log_iter", 0) + 1
        # ---- logging: GT class -> predicted class (last decoder only) ----
        if not hasattr(self, "_dbg_step"):
            self._dbg_step = 0
        self._dbg_step += 1

        # pass DETACHED copies to the logger; never pass graph tensors
        all_cls_scores_det = [t.detach() for t in all_cls_scores]
        all_bbox_preds_det = [t.detach() for t in all_bbox_preds]
        gt_bboxes_list_det = [g.clone().detach() for g in gt_bboxes_list]
        gt_labels_list_det = [g.clone().detach() for g in gt_labels_list]

        self._log_gt_to_pred_last_layer(
            all_cls_scores_det, all_bbox_preds_det,
            gt_bboxes_list_det, gt_labels_list_det, img_metas,
            step_idx=self._dbg_step
        )


        # ------------------------------------------------------------------

        return loss_dict

    # put these near the top of the file (with other imports)
    

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, outs, img_metas, rescale=False):
        """Decode head outputs to 3D boxes for each image in the batch.

        Args:
            outs (dict): Output dict from forward():
                - 'all_cls_scores': [num_layers, B, Q, num_classes]
                - 'all_bbox_preds': [num_layers, B, Q, code_dim_pred]
            img_metas (list[dict]): Meta info for each image in batch.
            rescale (bool): Unused here (kept for API compatibility).

        Returns:
            list[list]: For each batch element:
                [bboxes_3d (LiDARInstance3DBoxes), scores (Tensor), labels (Tensor)]
        """
        import torch
        from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes

        all_cls_scores = outs['all_cls_scores']  # [L, B, Q, C]
        all_bbox_preds = outs['all_bbox_preds']  # [L, B, Q, D]

        num_dec_layers, batch_size, num_query, num_classes = all_cls_scores.shape
        code_dim_pred = all_bbox_preds.size(-1)

        # --- use only the LAST decoder layer for inference ---
        cls_scores = all_cls_scores[-1]   # [B, Q, C]
        bbox_preds = all_bbox_preds[-1]   # [B, Q, D]

        bbox_list = []

        # score threshold from test_cfg (default 0)
        score_thr = 0.0
        if hasattr(self, 'test_cfg') and self.test_cfg is not None:
            score_thr = self.test_cfg.get('score_thr', 0.0)

        for img_id in range(batch_size):
            cls_score = cls_scores[img_id]   # [Q, C]
            bbox_pred = bbox_preds[img_id]   # [Q, D]

            # sigmoid classification scores
            scores_all = cls_score.sigmoid()  # [Q, C]
            # take max over classes to get per-query score + label
            max_scores, labels = scores_all.max(dim=-1)  # [Q], [Q]

            # apply score threshold
            keep = max_scores > score_thr
            scores = max_scores[keep]
            labels = labels[keep]
            bbox_pred = bbox_pred[keep]

            # no detections case
            if bbox_pred.numel() == 0:
                empty_boxes = LiDARInstance3DBoxes(
                    bbox_pred.new_zeros((0, 9)), box_dim=9, origin=(0.5, 0.5, 0.5)
                )
                bbox_list.append([empty_boxes, scores, labels])
                continue

            # --- ensure box code dim == 9: [x, y, z, dx, dy, dz, yaw, vx, vy] ---
            if bbox_pred.size(-1) > 9:
                # trim any extra dims (this matches what we did in the loss)
                bbox_pred = bbox_pred[:, :9]
            elif bbox_pred.size(-1) < 9:
                # if somehow smaller, pad zeros for remaining components
                pad = bbox_pred.new_zeros((bbox_pred.size(0), 9 - bbox_pred.size(-1)))
                bbox_pred = torch.cat([bbox_pred, pad], dim=-1)

            # ---- BEGIN: enforce positive, finite sizes for NuScenes eval ----
            # bbox_pred shape: [N, 9] -> [x, y, z, dx, dy, dz, yaw, vx, vy]
            sizes = bbox_pred[..., 3:6]  # dx, dy, dz

            # remove NaN / inf
            sizes = torch.nan_to_num(sizes, nan=0.0, posinf=0.0, neginf=0.0)

            # make strictly positive
            sizes = sizes.abs()
            sizes = torch.clamp(sizes, min=1e-2)  # small positive lower bound

            bbox_pred[..., 3:6] = sizes

            # also sanitize the whole code vector to avoid weird NaNs in other fields
            bbox_pred = torch.nan_to_num(
                bbox_pred, nan=0.0, posinf=0.0, neginf=0.0
            )
            # ---- END: size / numeric hygiene ----

            # bbox_pred is now [N, 9] in LiDAR/ego frame, meters
            # origin=(0.5, 0.5, 0.5) matches LiDARInstance3DBoxes default usage in NuScenes configs
            bboxes_3d = LiDARInstance3DBoxes(
                bbox_pred, box_dim=9, origin=(0.5, 0.5, 0.5)
            )

            # NOTE: we are intentionally NOT doing NMS here, to keep things simple.
            # If your test_cfg expects NMS, you can either:
            #  - set nms=None in the config, or
            #  - add a 3D NMS call here using mmdet3d.core.post_processing.box3d_multiclass_nms

            bbox_list.append([bboxes_3d, scores, labels])

        return bbox_list


    
    def _log_gt_to_pred_last_layer(
        self,
        all_cls_scores,
        all_bbox_preds,
        gt_bboxes_list,
        gt_labels_list,
        img_metas,
        *,
        step_idx=None,
        max_batches_to_print=1,   # kept for API compat; we’ll force 1 below
        max_gts_per_img=40,
        dist_thresh=6.0,
        print_fn=None,
    ):
        import os
        import torch

        # ---------- ALWAYS have a local printer; ignore external print_fn ----------
        pf = print if (print_fn is None) else print_fn

        # ---------- small helpers ----------
        def is_main() -> bool:
            return (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)

        def _pick_meta(meta_container, b_idx: int):
            """Return a dict-like meta for batch index b_idx.
            Works with: [dict,...], [[dict,...], ...] (queue), or a single dict.
            """
            if isinstance(meta_container, (list, tuple)):
                if not meta_container:
                    return {}
                md = meta_container[b_idx] if b_idx < len(meta_container) else meta_container[0]
                # If metas are a queue per item, take the last frame (current t)
                if isinstance(md, (list, tuple)) and len(md) > 0:
                    md = md[-1]
                return md if isinstance(md, dict) else {}
            return meta_container if isinstance(meta_container, dict) else {}

        def _sample_id_from_meta(md: dict):
            for k in ("curr_sample_token", "sample_id", "sample_idx", "token", "timestamp", "frame_id"):
                v = md.get(k, None)
                if v is not None:
                    return v
            return "?"

        # ---------- print-at-most once per two calls (handles grad_accum=2/double forward) ----------
        self._gt2pred_call = getattr(self, "_gt2pred_call", 0) + 1
        acc_skip = int(os.environ.get("BEV_LOG_ACC_SKIP", "2"))  # default: skip every other call
        if acc_skip > 1 and (self._gt2pred_call % acc_skip) != 0:
            return

        try:
            # ---------- gate by env + frequency ----------
            if os.environ.get("BEV_LOG_MATCH", "1") != "1":
                return
            every = int(os.environ.get("BEV_LOG_EVERY", "50"))
            if step_idx is not None and every > 0 and (step_idx % every) != 0:
                return

            # ---------- reentrancy guard: print once per train step ----------
            if step_idx is not None:
                last = getattr(self, "_gt2pred_last_step", None)
                if last == step_idx:
                    return
                self._gt2pred_last_step = step_idx

            # ---------- pick last decoder layer (detach for safety) ----------
            cls_last = all_cls_scores[-1]
            box_last = all_bbox_preds[-1]
            B, Q, C = cls_last.shape

            # ---------- pick exactly ONE sample to print ----------
            gt_counts = []
            for b in range(B):
                try:
                    gt_counts.append(int(gt_labels_list[b].numel()))
                except Exception:
                    gt_counts.append(0)
            b_pick = int(torch.as_tensor(gt_counts).argmax().item()) if len(gt_counts) else 0
            b_pick = max(0, min(b_pick, B - 1))  # clamp

            # predictions (DETACHED; logging only)
            logits  = cls_last[b_pick].detach().float()         # [Q, C]
            scores  = logits.sigmoid()
            centers = box_last[b_pick, :, :3].detach().float()  # [Q, 3]

            # assume BEVFormer box format: [x, y, z, w, l, h, sin(yaw), cos(yaw), vx, vy]
            pred_sin = box_last[b_pick, :, 6].detach().float()  # [Q]
            pred_cos = box_last[b_pick, :, 7].detach().float()  # [Q]
            pred_yaw = torch.atan2(pred_sin, pred_cos)          # [Q]

            # ---- GT boxes as a tensor ----
            gtb = gt_bboxes_list[b_pick]

            # Most nuScenes boxes in mmdet3d are Instance3DBoxes with a `.tensor` of shape [N, 9]
            if hasattr(gtb, 'tensor'):
                gt_tensor = gtb.tensor.detach().float()         # [Ng, D]
            else:
                gt_tensor = torch.as_tensor(
                    gtb, dtype=centers.dtype, device=centers.device
                ).detach().float()

            gt_centers = gt_tensor[..., :3]                     # [Ng, 3]
            Ng_total = gt_centers.size(0)
            Ng = Ng_total  # or Ng = min(Ng_total, 40) if you want to truncate prints
            gt_yaw = gt_tensor[..., 6] if gt_tensor.size(-1) >= 7 else None  # [Ng] or None

            labels = gt_labels_list[b_pick].detach().long()
            Ng = gt_centers.size(0)

            # sample identifier from metas
            md = _pick_meta(img_metas, b_pick)
            sample_id = _sample_id_from_meta(md)

            if is_main():
                pf(f"[GT2Pred][b{b_pick}] step={step_idx} sample_id={sample_id} "
                f"Q={Q} C={C} GTs={Ng_total} (showing {Ng})")

            if centers.numel() == 0 or scores.numel() == 0 or Ng == 0:
                if is_main():
                    pf("  (no predictions or no GTs)")
                return

            # pairwise XY distances: [Q, Ng]
            dists = torch.cdist(centers[:, :2], gt_centers[:Ng, :2], p=2)
            dmin, q_idx = dists.min(dim=0)  # per GT

            if is_main():
                for g in range(Ng):
                    di = float(dmin[g])
                    qi = int(q_idx[g])
                    gt_cls = int(labels[g].item())
                    pred_cls = int(scores[qi].argmax().item())
                    pred_score = float(scores[qi, pred_cls].item())

                    # centers and z
                    gt_center = gt_centers[g]
                    pred_center = centers[qi]
                    dz = float(pred_center[2] - gt_center[2])

                    base = (
                        f"  gt#{g:02d}  gt_cls={gt_cls}  ->  pred_cls={pred_cls}  "
                        f"score={pred_score:.3f}  d_xy={di:.2f}  "
                        f"gt_xyz=({gt_center[0]:.2f},{gt_center[1]:.2f},{gt_center[2]:.2f})  "
                        f"pred_xyz=({pred_center[0]:.2f},{pred_center[1]:.2f},{pred_center[2]:.2f})  "
                        f"dz={dz:.2f}"
                    )

                    # yaw terms if we have them
                    if gt_yaw is not None:
                        y_gt = float(gt_yaw[g])
                        y_pred = float(pred_yaw[qi])
                        d_yaw = ((y_pred - y_gt + math.pi) % (2 * math.pi)) - math.pi
                        base += f"  yaw_gt={y_gt:.2f}  yaw_pred={y_pred:.2f}  d_yaw={d_yaw:.2f}"

                    pf(base)


        except Exception as e:
            # Never break training if logging fails.
            print(f"[GT2Pred] logging failed: {repr(e)}")

@HEADS.register_module()
class BEVFormerHead_GroupDETR(BEVFormerHead):
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

        bev_embed, hs, init_reference, inter_references = outputs
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
    

