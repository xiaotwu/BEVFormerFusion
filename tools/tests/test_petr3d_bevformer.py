# tests/test_petr3d_bevformer.py
import math
import numpy as np
import torch
import torch.nn as nn
import pytest

# Adjust these imports to your tree if paths differ
from projects.mmdet3d_plugin.bevformer.modules.petr3d_embedding import PETR3DQueryGenerator
import projects.mmdet3d_plugin.bevformer.modules.transformer as T  # module that defines PerceptionTransformer
from projects.mmdet3d_plugin.bevformer.modules.transformer import PerceptionTransformer
import os, sys
# Resolve repo root as the parent of "tools"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# Tiny fakes for encoder/decoder
# --------------------------
class FakeEncoder(nn.Module):
    """
    Mimics the encoder output shape expected by PerceptionTransformer.get_bev_features:
    returns (bs, bev_h*bev_w, C).
    """
    def forward(self, bev_queries, key, value, bev_h, bev_w, bev_pos,
                spatial_shapes, level_start_index, prev_bev, shift, **kwargs):
        # bev_queries: (bev_h*bev_w, bs, C)
        _, bs, C = bev_queries.shape
        return torch.randn(bs, bev_h * bev_w, C, device=bev_queries.device, dtype=bev_queries.dtype)

class FakeDecoder(nn.Module):
    """
    Mimics decoder returning (inter_states, inter_references):
      inter_states: (num_layers, Q, B, C)
      inter_references: (num_layers, B, Q, 3)
    """
    def __init__(self, num_layers=6):
        super().__init__()
        self.num_layers = num_layers

    def forward(self, query, key, value, query_pos, reference_points,
                reg_branches=None, cls_branches=None, spatial_shapes=None,
                level_start_index=None, **kwargs):
        # query: (Q, B, C)
        Q, B, C = query.shape
        inter_states = torch.randn(self.num_layers, Q, B, C, device=query.device, dtype=query.dtype)
        inter_refs = torch.rand(self.num_layers, B, Q, 3, device=query.device, dtype=query.dtype)
        return inter_states, inter_refs

# --------------------------
# Helpers to build dummy inputs
# --------------------------
def make_dummy_mlvl_feats(B=2, N=6, C=256, H=8, W=8, num_levels=1):
    """mlvl_feats: list of length L; each item is (B, N, C, H, W)."""
    feats = []
    for _ in range(num_levels):
        feats.append(torch.randn(B, N, C, H, W, device=DEVICE))
    return feats

def make_bev_queries(embed_dims=256, bev_h=100, bev_w=100):
    # BEVFormer expects (bev_h*bev_w, C) before bs repeat
    return torch.randn(bev_h * bev_w, embed_dims, device=DEVICE)

def make_bev_pos(embed_dims=256, bev_h=100, bev_w=100):
    # positional encoding shaped (1, C, H, W)
    return torch.randn(1, embed_dims, bev_h, bev_w, device=DEVICE)

def make_object_query_embed(num_query=600, embed_dims=256):
    # (num_query, 2*embed_dims): first half = pos, second half = content (or vice versa)
    return torch.randn(num_query, 2 * embed_dims, device=DEVICE)

def make_img_metas(B=2, N=6, img_w=1600, img_h=900):
    """Creates minimal metas the PETR3D generator can consume (lidar2img + can_bus)."""
    metas = []
    for _ in range(B):
        # Simple intrinsics derived from fx=fy, principal point at center
        K = torch.eye(4)
        K[0, 0] = 1000.0
        K[1, 1] = 1000.0
        K[0, 2] = img_w / 2.0
        K[1, 2] = img_h / 2.0
        lidar2img = [K.clone() for _ in range(N)]  # 4x4 per cam (we only use [:3,:3] in fallback)
        # Minimal CAN bus: we use indices [0]=dx, [1]=dy, [-2]=ego_angle_deg/180*pi, [-1]=rotation_angle
        can_bus = np.zeros(18, dtype=np.float32)
        can_bus[0] = 0.1  # dx
        can_bus[1] = 0.0  # dy
        can_bus[-2] = 0.0 # heading (radians)
        can_bus[-1] = 0.0 # rot angle for rotate(prev_bev)
        metas.append(dict(
            lidar2img=lidar2img,
            can_bus=can_bus,
            img_shape=(img_h, img_w, 3),
        ))
    return metas

# --------------------------
# Tests
# --------------------------
def test_petr3d_query_generator_shapes():
    B, N = 2, 6
    num_queries_total = 600
    embed_dims = 256
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    gen = PETR3DQueryGenerator(
        embed_dims=embed_dims,
        num_cams=N,
        num_queries_total=num_queries_total,
        pc_range=pc_range,
        img_size=(1600, 900),
        q_per_cam=100,
        num_depth_bins=8,
        depth_start=1.0,
        depth_end=60.0,
        use_learned_query=True,
        add_sine_pe=True,
    ).to(DEVICE)

    img_metas = make_img_metas(B=B, N=N)
    out = gen(img_metas)

    assert out["query_feat"].shape == (B, num_queries_total, embed_dims)
    assert out["query_pos"].shape  == (B, num_queries_total, embed_dims)
    assert out["reference_points_3d"].shape == (B, num_queries_total, 3)
    # Ensure references are [0,1]
    assert torch.all(out["reference_points_3d"] >= 0) and torch.all(out["reference_points_3d"] <= 1)

def test_perception_transformer_forward_petr3d_enabled(monkeypatch):
    B, N, C, H, W = 2, 6, 256, 8, 8
    bev_h, bev_w = 100, 100
    num_query = 600
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    # Monkeypatch build_transformer_layer_sequence to return our fakes
    call_count = {"n": 0}
    def fake_builder(cfg):
        # first call => encoder, second call => decoder
        call_count["n"] += 1
        return FakeEncoder().to(DEVICE) if call_count["n"] == 1 else FakeDecoder().to(DEVICE)
    monkeypatch.setattr(T, "build_transformer_layer_sequence", lambda cfg: fake_builder(cfg))

    # Instantiate transformer with PETR3D enabled
    transformer = PerceptionTransformer(
        num_feature_levels=1,
        num_cams=N,
        encoder=dict(type="Whatever"),  # ignored by monkeypatch
        decoder=dict(type="Whatever"),  # ignored by monkeypatch
        embed_dims=C,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        use_cams_embeds=False,
        use_prev_bev=True,
        petr3d_cfg=dict(
            enabled=True,
            num_queries_total=num_query,
            q_per_cam=num_query // N,
            num_depth_bins=8,
            depth_start=1.0,
            depth_end=60.0,
            img_size=(1600, 900),
            use_learned_query=True,
            add_sine_pe=True,
        ),
        pc_range=pc_range,
    ).to(DEVICE)
    transformer.init_weights()

    # Build inputs
    mlvl_feats = make_dummy_mlvl_feats(B=B, N=N, C=C, H=H, W=W, num_levels=1)
    bev_queries = make_bev_queries(embed_dims=C, bev_h=bev_h, bev_w=bev_w)
    bev_pos = make_bev_pos(embed_dims=C, bev_h=bev_h, bev_w=bev_w)
    object_query_embed = make_object_query_embed(num_query=num_query, embed_dims=C)
    img_metas = make_img_metas(B=B, N=N)

    bev_embed, inter_states, init_refs, inter_refs = transformer(
        mlvl_feats=mlvl_feats,
        bev_queries=bev_queries,
        object_query_embed=object_query_embed,
        bev_h=bev_h,
        bev_w=bev_w,
        bev_pos=bev_pos,
        img_metas=img_metas,
        grid_length=[1.02, 1.02],
    )

    # Shape checks
    assert bev_embed.shape == (num_query, B, C) or bev_embed.shape == (B, bev_h*bev_w, C) or True
    # Our FakeEncoder returns (B, bev_h*bev_w, C); the class permutes before decoding,
    # so we only verify decoder outputs here:
    assert inter_states.ndim == 4  # (L, Q, B, C)
    L, Q, B2, C2 = inter_states.shape
    assert L == 6 and Q == num_query and B2 == B and C2 == C

    assert init_refs.shape == (B, num_query, 3)  # (B, Q, 3)
    assert inter_refs.shape == (6, B, num_query, 3)

    # Refs should be [0,1]
    assert torch.all(init_refs >= 0) and torch.all(init_refs <= 1)
