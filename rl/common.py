"""Torch models mirrored by kagent/nnpolicy.py's pure-python inference.

Architecture v2 (post-review):
- PlannerNet (POLICY, ships in the submission): 164 -> 192 -> 128 -> 54 logits,
  GELU (exact erf form) + LayerNorm on each hidden block.
- ValueNet (torch-only, never ships): 164 -> 128 -> 128 -> 1. Decoupled from
  the policy trunk so value-loss gradients after a returns jump cannot drag
  the policy with them (the breakout->collapse mechanism seen in gen 1-4).
"""

import base64
import struct

import torch
import torch.nn as nn

from kagent import features

H1, H2 = 192, 128
VH = 128


class PlannerNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.h1 = nn.Linear(n_features, H1)
        self.ln1 = nn.LayerNorm(H1)
        self.h2 = nn.Linear(H1, H2)
        self.ln2 = nn.LayerNorm(H2)
        self.logits = nn.Linear(H2, features.N_ACTION_LOGITS)
        self.act = nn.GELU()  # exact (erf) form; mirrored via math.erf

    def forward(self, x):
        h = self.act(self.ln1(self.h1(x)))
        h = self.act(self.ln2(self.h2(h)))
        return self.logits(h)


class ValueNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, VH), nn.GELU(),
            nn.Linear(VH, VH), nn.GELU(),
            nn.Linear(VH, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _lin(lin):
    return lin.weight.detach().cpu().numpy().T.tolist(), lin.bias.detach().cpu().numpy().tolist()


def policy_layers(net):
    """Pure-python structure for nnpolicy.forward (arch v2)."""
    blocks = []
    for lin, ln in ((net.h1, net.ln1), (net.h2, net.ln2)):
        w, b = _lin(lin)
        blocks.append(
            [w, b, ln.weight.detach().cpu().numpy().tolist(), ln.bias.detach().cpu().numpy().tolist()]
        )
    w, b = _lin(net.logits)
    return {"arch": 2, "blocks": blocks, "out": [w, b]}


def value_layers(vnet):
    """[(W, b), ...] for the worker-side pure-python value estimate."""
    return [_lin(m) for m in vnet.net if isinstance(m, nn.Linear)]


def to_layers(net, vnet=None):
    """Rollout-worker weights file: policy (+ value when training)."""
    return {"policy": policy_layers(net), "value": value_layers(vnet) if vnet else None}


def to_weights_data(net):
    """Submission export: POLICY ONLY, packed as base85 float32."""
    p = policy_layers(net)
    flat, sizes = [], []
    for w, b, g, beta in p["blocks"]:
        sizes.append([len(w), len(w[0])])
        for row in w:
            flat.extend(row)
        flat.extend(b)
        flat.extend(g)
        flat.extend(beta)
    w, b = p["out"]
    sizes.append([len(w), len(w[0])])
    for row in w:
        flat.extend(row)
    flat.extend(b)
    raw = struct.pack("<%df" % len(flat), *flat)
    return {"arch": 2, "sizes": sizes, "b85": base64.b85encode(raw).decode("ascii")}


def masked_dists(logits, masks):
    """Split flat logits into per-head Categoricals with invalid choices
    masked to -1e9 before softmax (unbiased policy gradient)."""
    dists = []
    pos = 0
    for size in features.HEAD_SIZES:
        seg = logits[:, pos : pos + size]
        m = masks[:, pos : pos + size]
        seg = torch.where(m > 0, seg, torch.full_like(seg, -1e9))
        dists.append(torch.distributions.Categorical(logits=seg))
        pos += size
    return dists


def flat_masks(head_masks):
    out = []
    for m in head_masks:
        out.extend(m)
    return out
