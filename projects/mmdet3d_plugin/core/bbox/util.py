import torch 
import math
import torch.nn.functional as F


def normalize_bbox(bboxes, pc_range):

    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8] 
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes

def encode_yaw_to_bins(gt_yaw, num_bins: int):
    """
    Encode GT yaw (radians, ideally in [-pi, pi]) into:
      - bin index in [0, num_bins-1]
      - residual sin/cos relative to bin center

    Args:
        gt_yaw: Tensor [...], radians
        num_bins: int

    Returns:
        bin_idx: LongTensor [...]
        res_sin: Tensor [...]
        res_cos: Tensor [...]
    """
    two_pi = 2.0 * math.pi
    bin_size = two_pi / float(num_bins)

    # map yaw to [0, 2pi)
    yaw_0_2pi = torch.remainder(gt_yaw + two_pi, two_pi)

    # bin index
    bin_idx = torch.floor(yaw_0_2pi / bin_size).long().clamp_(0, num_bins - 1)

    # bin center in [0, 2pi)
    bin_center = (bin_idx.to(gt_yaw.dtype) + 0.5) * bin_size

    # residual in (-bin_size/2, +bin_size/2) (still in [0,2pi) domain)
    residual = yaw_0_2pi - bin_center
    # wrap residual to [-pi, pi] so sin/cos is stable
    residual = torch.remainder(residual + math.pi, two_pi) - math.pi

    res_sin = torch.sin(residual)
    res_cos = torch.cos(residual)
    return bin_idx, res_sin, res_cos


def decode_yaw_from_bins(bin_logits, res_sin, res_cos, num_bins: int):
    """
    Decode predicted yaw from:
      - bin_logits: Tensor [..., num_bins]
      - res_sin/res_cos: Tensor [...] (unit-ish)

    Returns:
        yaw: Tensor [...], radians in [-pi, pi]
    """
    two_pi = 2.0 * math.pi
    bin_size = two_pi / float(num_bins)

    bin_idx = torch.argmax(bin_logits, dim=-1)  # [...]

    bin_center = (bin_idx.to(res_sin.dtype) + 0.5) * bin_size  # [...]

    # residual angle in [-pi, pi]
    residual = torch.atan2(res_sin, res_cos)

    yaw_0_2pi = bin_center + residual
    yaw = torch.remainder(yaw_0_2pi + math.pi, two_pi) - math.pi
    return yaw


def overwrite_sincos_from_bins(bbox_preds, num_bins: int, bin_start: int):
    """
    Convert (bin logits + residual sin/cos) -> yaw sin/cos and write into bbox_preds[..., 6:8]
    so that existing denormalize_bbox() can be used unchanged.

    Layout assumed:
      bbox_preds[..., bin_start : bin_start+num_bins]        = bin logits
      bbox_preds[..., bin_start+num_bins : bin_start+num_bins+2] = (res_sin, res_cos)

    This function returns a modified copy (no in-place on input).
    """
    out = bbox_preds.clone()

    bin_logits = out[..., bin_start:bin_start + num_bins]
    res = out[..., bin_start + num_bins: bin_start + num_bins + 2]
    res_sin = res[..., 0]
    res_cos = res[..., 1]

    yaw = decode_yaw_from_bins(bin_logits, res_sin, res_cos, num_bins=num_bins)

    # write back to expected sin/cos slots for denormalize_bbox()
    out[..., 6] = torch.sin(yaw)
    out[..., 7] = torch.cos(yaw)
    return out


def denormalize_bbox(normalized_bboxes, pc_range):
    # rotation 
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]
   
    # size
    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = w.exp() 
    l = l.exp() 
    h = h.exp() 
    if normalized_bboxes.size(-1) > 8:
         # velocity 
        vx = normalized_bboxes[..., 8:9]
        vy = normalized_bboxes[..., 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes