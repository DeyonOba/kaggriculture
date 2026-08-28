"""Torch model mirroring kagent/nnpolicy.py's pure-python MLP, plus
weight conversion and heuristic-plan -> head-choice quantization for BC."""

import base64
import struct

import torch
import torch.nn as nn

from kagent import features

HIDDEN = 96


class PlannerNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.h1 = nn.Linear(n_features, HIDDEN)
        self.h2 = nn.Linear(HIDDEN, HIDDEN)
        self.logits = nn.Linear(HIDDEN, features.N_ACTION_LOGITS)
        self.value = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        h = torch.tanh(self.h1(x))
        h = torch.tanh(self.h2(h))
        return self.logits(h), self.value(h).squeeze(-1)


def to_layers(net):
    """PlannerNet -> pure-python layers [(W rows, b), ...] as in nnpolicy."""
    layers = []
    for lin in (net.h1, net.h2, net.logits, net.value):
        w = lin.weight.detach().cpu().numpy()  # (out, in)
        b = lin.bias.detach().cpu().numpy()
        layers.append((w.T.tolist(), b.tolist()))
    return layers


def to_weights_data(net):
    """PlannerNet -> {"sizes": [...], "b85": ...} for kagent/weights.py."""
    flat = []
    sizes = []
    for lin in (net.h1, net.h2, net.logits, net.value):
        w = lin.weight.detach().cpu().numpy().T  # (in, out)
        b = lin.bias.detach().cpu().numpy()
        sizes.append([w.shape[0], w.shape[1]])
        flat.extend(w.reshape(-1).tolist())
        flat.extend(b.tolist())
    raw = struct.pack("<%df" % len(flat), *flat)
    return {"sizes": sizes, "b85": base64.b85encode(raw).decode("ascii")}


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
    """List-of-lists head masks -> flat list matching N_ACTION_LOGITS."""
    out = []
    for m in head_masks:
        out.extend(m)
    return out


# (BC target quantization lives in rl/rollout.py, which must stay torch-free.)
