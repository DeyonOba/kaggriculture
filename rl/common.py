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

# Board trunk. The policy has only ever seen `kagent.features` day-level
# aggregates -- 168 numbers in which the board is a set of counts. The LP's
# provably optimal ALLOCATION executed by our own scheduler tied champ-h8
# (+$828 +/- 2,317), so allocation is not where the money is; what a count
# cannot express is WHERE things are and how the two farms sit against each
# other, and `tools/board_tensor.py` has encoded exactly that all along
# without ever entering the training loop.
#
# It is trained END TO END with the policy rather than pretrained and frozen:
# a frozen trunk fitted on a corpus is a self-play potential by another name,
# and that scored -0.398 R^2 out of distribution
# (docs/phi_ood_and_phase_timing.md). The representation has to move with the
# states the policy actually visits.
BOARD_CH = 91          # tools.board_tensor.N_FULL
BOARD_W = 48           # trunk output width


class BoardTrunk(nn.Module):
    """(B, 91, 10, 10) -> (B, BOARD_W). Small on purpose.

    Simulation/fitting is ~3000:1 here (docs/training_algorithms.md): games
    cost everything and parameters cost nothing, so capacity is not the
    constraint and a big trunk only slows the rollout that feeds it.
    """

    def __init__(self, in_ch=BOARD_CH, width=BOARD_W):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 32, 3, padding=1)
        self.ln = nn.LayerNorm(64)
        self.fc = nn.Linear(64, width)
        self.act = nn.GELU()

    def forward(self, b):
        h = self.act(self.c1(b))
        h = self.act(self.c2(h))
        # mean AND max pooling: mean carries "how much", max carries "is there
        # one anywhere", and several channels (a ripe melon, a starving cow)
        # matter as existence rather than as a count.
        h = torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)
        return self.act(self.fc(self.ln(h)))


class PlannerNet(nn.Module):
    def __init__(self, n_features, board=False):
        super().__init__()
        self.board = BoardTrunk() if board else None
        n_in = n_features + (BOARD_W if board else 0)
        self.h1 = nn.Linear(n_in, H1)
        self.ln1 = nn.LayerNorm(H1)
        self.h2 = nn.Linear(H1, H2)
        self.ln2 = nn.LayerNorm(H2)
        self.logits = nn.Linear(H2, features.N_ACTION_LOGITS)
        self.act = nn.GELU()  # exact (erf) form; mirrored via math.erf

    def forward(self, x, b=None):
        if self.board is not None:
            if b is None:
                raise ValueError(
                    "PlannerNet was built with board=True but no board tensor "
                    "was passed -- set KAGG_BOARD=1 so rl/rollout.py records it")
            x = torch.cat([x, self.board(b)], dim=1)
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
