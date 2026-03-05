import torch
import math

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox


def wrap_to_pi(x):
    return torch.remainder(x + math.pi, 2 * math.pi) - math.pi


def encode_yaw_to_bins(yaw, num_bins: int):
    """
    yaw: Tensor in radians (any range). Bins span [-pi, pi).
    Returns:
      bin_idx: LongTensor
      res_sin, res_cos: Tensor residual on unit circle relative to bin center
    """
    yaw = wrap_to_pi(yaw)
    bin_size = 2 * math.pi / num_bins

    yaw_0 = yaw + math.pi  # map [-pi, pi) -> [0, 2pi)

    bin_idx = torch.floor(yaw_0 / bin_size).long()
    bin_idx = torch.clamp(bin_idx, 0, num_bins - 1)

    center = (bin_idx.float() + 0.5) * bin_size - math.pi
    res = wrap_to_pi(yaw - center)

    return bin_idx, torch.sin(res), torch.cos(res)


def decode_yaw_from_bins(bin_logits, res_sin, res_cos, num_bins: int):
    two_pi = 2.0 * math.pi
    bin_size = two_pi / float(num_bins)

    bin_idx = torch.argmax(bin_logits, dim=-1)
    bin_center = (bin_idx.to(res_sin.dtype) + 0.5) * bin_size

    residual = torch.atan2(res_sin, res_cos)
    yaw_0_2pi = bin_center + residual
    yaw = torch.remainder(yaw_0_2pi + math.pi, two_pi) - math.pi
    return yaw


@BBOX_CODERS.register_module()
class NMSFreeCoder(BaseBBoxCoder):

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10,
                 dims_order='wlh',
                 z_origin='center',
                 clamp_sizes=False):
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes
        self.dims_order = dims_order
        self.z_origin = z_origin
        self.clamp_sizes = clamp_sizes

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds, yaw_bin_logits=None, yaw_res_preds=None, vel_preds=None):

        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]

        # ---------- yaw from bins+res (separate heads) ----------
        if yaw_bin_logits is not None and yaw_res_preds is not None:
            yb = yaw_bin_logits[bbox_index]   # [K, num_bins]
            yr = yaw_res_preds[bbox_index]    # [K, 2]

            bin_idx = yb.argmax(dim=-1)       # [K]
            num_bins = yb.size(-1)
            bin_width = (2.0 * math.pi) / float(num_bins)
            bin_center = -math.pi + (bin_idx.float() + 0.5) * bin_width  # [K]

            yr = yr / (yr.norm(dim=-1, keepdim=True) + 1e-6)
            res_ang = torch.atan2(yr[..., 0], yr[..., 1])  # [-pi, pi]

            yaw = bin_center + res_ang
            yaw = torch.remainder(yaw + math.pi, 2 * math.pi) - math.pi

            bbox_preds = bbox_preds.clone()
            bbox_preds[..., 6] = torch.sin(yaw)
            bbox_preds[..., 7] = torch.cos(yaw)

        # Override velocity with dedicated velocity head predictions
        if vel_preds is not None:
            vp = vel_preds[bbox_index]  # [K, 2]
            bbox_preds = bbox_preds.clone()
            bbox_preds[..., 8:10] = vp

        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)

        # --- normalize / fix conventions ---
        pi = final_box_preds.new_tensor(math.pi)

        # 1) dims: if the network outputs (l, w, h) convert to (w, l, h)
        if final_box_preds.size(-1) >= 6:
            if getattr(self, 'dims_order', 'wlh').lower() == 'lwh':
                final_box_preds[..., [3, 4]] = final_box_preds[..., [4, 3]]

        # 2) z origin: convert from bottom -> gravity-center if required
        if final_box_preds.size(-1) >= 6 and getattr(self, 'z_origin', 'center').lower() == 'bottom':
            final_box_preds[..., 2] = final_box_preds[..., 2] + 0.5 * final_box_preds[..., 5]

        # 3) clamp sizes to a sane range (safety)
        if getattr(self, 'clamp_sizes', True) and final_box_preds.size(-1) >= 6:
            final_box_preds[..., 3:6] = final_box_preds[..., 3:6].clamp(min=0.2, max=8.0)

        final_scores = scores
        final_preds = labels

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score

        if self.post_center_range is not None:
            post_center_range = torch.tensor(
                self.post_center_range,
                device=final_box_preds.device,
                dtype=final_box_preds.dtype
            )

            mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(dim=1)
            mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(dim=1)

            if self.score_threshold is not None:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]

            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict

    def decode(self, preds_dicts):
        # last decoder layer
        all_cls_scores = preds_dicts['all_cls_scores'][-1]   # [bs, num_query, C]
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1]   # [bs, num_query, code]

        all_yaw_bin_logits = preds_dicts.get('all_yaw_bin_logits', None)
        all_yaw_res_preds  = preds_dicts.get('all_yaw_res_preds', None)
        all_vel_preds      = preds_dicts.get('all_vel_preds', None)

        # slice last layer if yaw tensors are 4D (nb_dec, bs, num_query, ...)
        if all_yaw_bin_logits is not None:
            if isinstance(all_yaw_bin_logits, (list, tuple)):
                all_yaw_bin_logits = all_yaw_bin_logits[-1]
            elif all_yaw_bin_logits.dim() == 4:
                all_yaw_bin_logits = all_yaw_bin_logits[-1]

        if all_yaw_res_preds is not None:
            if isinstance(all_yaw_res_preds, (list, tuple)):
                all_yaw_res_preds = all_yaw_res_preds[-1]
            elif all_yaw_res_preds.dim() == 4:
                all_yaw_res_preds = all_yaw_res_preds[-1]

        if all_vel_preds is not None:
            if isinstance(all_vel_preds, (list, tuple)):
                all_vel_preds = all_vel_preds[-1]
            elif all_vel_preds.dim() == 4:
                all_vel_preds = all_vel_preds[-1]

        if all_yaw_bin_logits is not None:
            assert all_yaw_bin_logits.shape[:2] == all_cls_scores.shape[:2], \
                f"{all_yaw_bin_logits.shape}, {all_cls_scores.shape}"

        batch_size = all_cls_scores.size(0)
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(
                self.decode_single(
                    all_cls_scores[i],
                    all_bbox_preds[i],
                    yaw_bin_logits=(all_yaw_bin_logits[i] if all_yaw_bin_logits is not None else None),
                    yaw_res_preds=(all_yaw_res_preds[i] if all_yaw_res_preds is not None else None),
                    vel_preds=(all_vel_preds[i] if all_vel_preds is not None else None),
                )
            )
        return predictions_list
