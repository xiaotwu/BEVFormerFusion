import torch

from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import AssignResult
from mmdet.core.bbox.assigners import BaseAssigner
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.models.utils.transformer import inverse_sigmoid
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@BBOX_ASSIGNERS.register_module()
class HungarianAssigner3D(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.
    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', weight=0.0),
                 pc_range=None):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)
        self.pc_range = pc_range

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes, 
               gt_labels,
               gt_bboxes_ignore=None,
               eps=1e-7):

        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # regression L1 cost
       
        normalized_gt_bboxes = normalize_bbox(gt_bboxes, self.pc_range)
    
        reg_cost = self.reg_cost(bbox_pred[:, :8], normalized_gt_bboxes[:, :8])
      
        # weighted sum of above two costs
        cost = cls_cost + reg_cost
        
        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')        
        
        # ----- BEGIN: sanitize cost BEFORE linear_sum_assignment -----
        def _isfinite(x: torch.Tensor) -> torch.Tensor:
            return torch.isfinite(x) if hasattr(torch, "isfinite") else (~torch.isnan(x)) & (~torch.isinf(x))

        # ensure float tensor
        cost = cost.to(dtype=torch.float32)

        if cost.numel() == 0:
            # handle degenerate case early (no queries or no gt)
            cost_np = cost.detach().cpu().numpy()
        else:
            if not _isfinite(cost).all():
                nan_cnt = torch.isnan(cost).sum().item()
                inf_cnt = torch.isinf(cost).sum().item()
                print(f"[ASSIGN] non-finite cost -> nan:{nan_cnt} inf:{inf_cnt}")
                if hasattr(torch, "nan_to_num"):
                    cost = torch.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=1e6)
                else:
                    cost = torch.where(torch.isnan(cost), torch.full_like(cost, 1e6), cost)
                    cost = torch.where(torch.isinf(cost), torch.full_like(cost, 1e6), cost)

            # clamp to reasonable range
            cost = cost.clamp(min=0.0, max=1e6)
            cost_np = cost.detach().cpu().numpy()
        # ----- END: sanitize cost -----

        # SciPy returns NumPy arrays on CPU
        matched_row_inds_np, matched_col_inds_np = linear_sum_assignment(cost_np)

        # Convert BOTH to torch.long on the same device as bbox_pred
        device = bbox_pred.device
        matched_row_inds = torch.from_numpy(matched_row_inds_np).to(device=device, dtype=torch.long)
        matched_col_inds = torch.from_numpy(matched_col_inds_np).to(device=device, dtype=torch.long)

        # 4) assign backgrounds and foregrounds
        assigned_gt_inds[:] = 0  # background
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)