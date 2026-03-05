import torch
import math

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
import numpy as np

import torch.nn.functional as F


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

def encode_yaw_to_bins_from_sincos(gt_sin, gt_cos, num_bins: int):
    yaw = yaw_from_sincos(gt_sin, gt_cos)
    return encode_yaw_to_bins(yaw, num_bins)

def encode_yaw_to_bins(gt_yaw, num_bins: int):
    two_pi = 2.0 * math.pi
    bin_size = two_pi / float(num_bins)

    yaw_0_2pi = torch.remainder(gt_yaw + two_pi, two_pi)
    bin_idx = torch.floor(yaw_0_2pi / bin_size).long().clamp_(0, num_bins - 1)

    bin_center = (bin_idx.to(gt_yaw.dtype) + 0.5) * bin_size
    residual = yaw_0_2pi - bin_center
    residual = torch.remainder(residual + math.pi, two_pi) - math.pi

    res_sin = torch.sin(residual)
    res_cos = torch.cos(residual)
    return bin_idx, res_sin, res_cos


def decode_yaw_from_bins(bin_logits, res_sin, res_cos, num_bins: int):
    two_pi = 2.0 * math.pi
    bin_size = two_pi / float(num_bins)

    bin_idx = torch.argmax(bin_logits, dim=-1)
    bin_center = (bin_idx.to(res_sin.dtype) + 0.5) * bin_size

    residual = torch.atan2(res_sin, res_cos)
    yaw_0_2pi = bin_center + residual
    yaw = torch.remainder(yaw_0_2pi + math.pi, two_pi) - math.pi
    return yaw


def overwrite_sincos_from_bins(bbox_preds, num_bins: int, bin_start: int):
    out = bbox_preds.clone()

    bin_logits = out[..., bin_start:bin_start + num_bins]
    res = out[..., bin_start + num_bins: bin_start + num_bins + 2]

    res_sin = res[..., 0]
    res_cos = res[..., 1]

    yaw = decode_yaw_from_bins(bin_logits, res_sin, res_cos, num_bins=num_bins)

    out[..., 6] = torch.sin(yaw)
    out[..., 7] = torch.cos(yaw)
    return out


@BBOX_CODERS.register_module()
class NMSFreeCoder(BaseBBoxCoder):

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10,
                 # NEW:
                 dims_order='wlh',          # your net’s dims order: 'lwh' or 'wlh'
                 z_origin='center',         # your z meaning: 'bottom' or 'center'
                 clamp_sizes=False):         # clamp dims to sane range during debug
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes
        # NEW:
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
            # select same queries
            yb = yaw_bin_logits[bbox_index]   # [K, num_bins]
            yr = yaw_res_preds[bbox_index]    # [K, 2]

            # predicted bin
            bin_idx = yb.argmax(dim=-1)       # [K]
            num_bins = yb.size(-1)
            bin_width = (2.0 * math.pi) / float(num_bins)
            bin_center = -math.pi + (bin_idx.float() + 0.5) * bin_width  # [K]

            # residual -> angle (normalize to unit)
            yr = yr / (yr.norm(dim=-1, keepdim=True) + 1e-6)
            res_ang = torch.atan2(yr[..., 0], yr[..., 1])  # [-pi, pi]

            # final yaw
            yaw = bin_center + res_ang
            yaw = torch.remainder(yaw + math.pi, 2 * math.pi) - math.pi

            # write sin/cos into slots expected by denormalize_bbox()
            bbox_preds = bbox_preds.clone()
            bbox_preds[..., 6] = torch.sin(yaw)
            bbox_preds[..., 7] = torch.cos(yaw)

        # -------------------------------------------------------

        # Override velocity with dedicated velocity head predictions
        if vel_preds is not None:
            vp = vel_preds[bbox_index]  # [K, 2]
            bbox_preds = bbox_preds.clone()
            bbox_preds[..., 8:10] = vp

        # --- QUICK YAW SANITY CHECK (paste right after `bbox_preds = bbox_preds[bbox_index]`) ---
        if not hasattr(self, "_dbg_once_yaw"):
            self._dbg_once_yaw = True

            # handle empty selection safely
            if bbox_preds.numel() == 0 or bbox_preds.size(-1) < 8:
                print("[DBG pre-denorm] yaw: no boxes or missing sin/cos")
            else:
                sc = bbox_preds[..., 6:8]           # (sin, cos) as currently encoded
                # atan2 expects (sin, cos) -> angle in radians
                yaw_rad = torch.atan2(sc[..., 0], sc[..., 1])
                yaw_deg = yaw_rad * (180.0 / math.pi)
                y_min = float(yaw_deg.min().detach().cpu().item())
                y_max = float(yaw_deg.max().detach().cpu().item())
                y_mean = float(yaw_deg.mean().detach().cpu().item())
                print(f"[DBG pre-denorm] yaw deg min={y_min:.3f} max={y_max:.3f} mean={y_mean:.3f}")
        # --- end quick yaw sanity check ---

        # >>> ADD THIS BLOCK HERE (sincos norm debug)
        # >>> ADD THIS BLOCK HERE (sincos norm debug + summary log)
        if getattr(self, "debug_decode", False) and not hasattr(self, "_dbg_once_sc"):
            self._dbg_once_sc = True
            with torch.no_grad():
                # bbox_preds is normalized-code (sin,cos at 6:8) before denormalize
                if bbox_preds.numel() == 0 or bbox_preds.size(-1) < 8:
                    sc = bbox_preds.new_zeros((0, 2))
                else:
                    sc = bbox_preds[..., 6:8]  # (sin, cos)

                # norm per-box: sqrt(sin^2 + cos^2)
                # avoid in-place ops; keep on device until moving to cpu for printing
                sc_norm = (sc[..., 0].pow(2) + sc[..., 1].pow(2)).sqrt()

                # prepare summary only if any boxes present
                if sc_norm.numel() > 0:
                    sc_norm_cpu = sc_norm.detach().to('cpu')
                    mean = float(sc_norm_cpu.mean().item())
                    std = float(sc_norm_cpu.std().item())
                    mn = float(sc_norm_cpu.min().item())
                    mx = float(sc_norm_cpu.max().item())

                    # also show a few raw pred sin/cos pairs for quick inspection
                    sample = sc.detach().to('cpu')[:5].numpy()  # up to first 5 pairs

                    # print/log — replace print(...) with logger.info(...) if you prefer
                    print(f"[DBG s/c norm] n={sc_norm_cpu.numel()} mean={mean:.4f} std={std:.4f} "
                        f"min={mn:.4f} max={mx:.4f}")
                    print(f"[DBG s/c sample (first up to 5)] {sample.tolist()}")

        # put this near the top of decode_single(), after bbox_preds = bbox_preds[bbox_index]
        if not hasattr(self, "_dbg_once_denorm"):
            self._dbg_once_denorm = True
            if bbox_preds.numel() > 0:
                x = bbox_preds.detach()
                print("[DBG pre-denorm] cx min/max:", float(x[...,0].min()), float(x[...,0].max()))
                print("[DBG pre-denorm] cy min/max:", float(x[...,1].min()), float(x[...,1].max()))
                print("[DBG pre-denorm] cz min/max:", float(x[...,4].min()), float(x[...,4].max()))
                print("[DBG pre-denorm] w  min/max:", float(x[...,2].min()), float(x[...,2].max()))
                print("[DBG pre-denorm] l  min/max:", float(x[...,3].min()), float(x[...,3].max()))
                print("[DBG pre-denorm] h  min/max:", float(x[...,5].min()), float(x[...,5].max()))


        # then continue with denormalize
        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)
 
        # --- normalize / fix conventions ---

        # make tensor constants on same device/dtype
        pi = final_box_preds.new_tensor(math.pi)
        half_pi = pi / 2
        two_pi = pi * 2

        # 1) dims: if the network outputs (l, w, h) convert to (w, l, h)
        if final_box_preds.size(-1) >= 6:
            if getattr(self, 'dims_order', 'wlh').lower() == 'lwh':
                final_box_preds[..., [3, 4]] = final_box_preds[..., [4, 3]]

        # 2) z origin: convert from bottom -> gravity-center if required
        if final_box_preds.size(-1) >= 6 and getattr(self, 'z_origin', 'center').lower() == 'bottom':
            final_box_preds[..., 2] = final_box_preds[..., 2] + 0.5 * final_box_preds[..., 5]


        # 4) clamp sizes to a sane range (debugging / safety)
        if getattr(self, 'clamp_sizes', True) and final_box_preds.size(-1) >= 6:
            final_box_preds[..., 3:6] = final_box_preds[..., 3:6].clamp(min=0.2, max=8.0)
        # --- end normalization ---

        
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

        # >>> slice last layer if yaw tensors are 4D (nb_dec, bs, num_query, ...)
        if all_yaw_bin_logits is not None:
            if isinstance(all_yaw_bin_logits, (list, tuple)):
                all_yaw_bin_logits = all_yaw_bin_logits[-1]
            elif all_yaw_bin_logits.dim() == 4:
                all_yaw_bin_logits = all_yaw_bin_logits[-1]   # [bs, num_query, num_bins]

        if all_yaw_res_preds is not None:
            if isinstance(all_yaw_res_preds, (list, tuple)):
                all_yaw_res_preds = all_yaw_res_preds[-1]
            elif all_yaw_res_preds.dim() == 4:
                all_yaw_res_preds = all_yaw_res_preds[-1]     # [bs, num_query, 2]

        if all_vel_preds is not None:
            if isinstance(all_vel_preds, (list, tuple)):
                all_vel_preds = all_vel_preds[-1]
            elif all_vel_preds.dim() == 4:
                all_vel_preds = all_vel_preds[-1]             # [bs, num_query, 2]

        # now this assert will pass
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


