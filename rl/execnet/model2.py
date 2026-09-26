"""execnet v2: a small CNN over our own board that scores every DESTINATION.

Input: N_PLANE x 10 x 10 (24 tile channels + the unit's position + the
scheduler's 8 request planes) and the v1 scalar vector, which is projected
and added to every cell as a global conditioning bias. Two heads:
  dest  a 1x1 conv gives one logit per tile (100), and the pooled trunk gives
        the SHED and STAY logits -> 102, masked to the legal destinations
  op    a 1x1 conv gives 8 request-class logits per tile; the loss reads them
        at the label tile, so the model also says WHAT it would do there
Saved as {"state", "cfg", "extra"} like v1, with cfg["v"] = 2 so the policy
can tell the two model families apart.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from rl.execnet import features2 as F2


class ExecNet2(nn.Module):
    def __init__(self, ch=64, blocks=3, scal_hidden=64, n_scalar=31):
        super().__init__()
        self.ch = ch
        layers = [nn.Conv2d(F2.N_PLANE, ch, 3, padding=1), nn.ReLU()]
        for _ in range(blocks - 1):
            layers += [nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)
        self.scal = nn.Sequential(nn.Linear(n_scalar, scal_hidden), nn.ReLU(),
                                  nn.Linear(scal_hidden, ch))
        self.dest_tile = nn.Conv2d(ch, 1, 1)
        self.op_tile = nn.Conv2d(ch, F2.N_REQ, 1)
        self.global_head = nn.Sequential(nn.Linear(ch, ch), nn.ReLU(), nn.Linear(ch, 2))

    def forward(self, planes, scal, mask=None):
        """planes [B, N_PLANE, 10, 10], scal [B, n_scalar], mask [B, 102] bool."""
        h = self.trunk(planes)
        h = h + self.scal(scal).unsqueeze(-1).unsqueeze(-1)
        b = h.shape[0]
        tile_logits = self.dest_tile(h).reshape(b, F2.N_TILES)
        glob = self.global_head(h.mean(dim=(2, 3)))
        logits = torch.cat([tile_logits, glob], dim=-1)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1e9)
        op_logits = self.op_tile(h).reshape(b, F2.N_REQ, F2.N_TILES)
        return logits, op_logits


def save(model, path, extra=None):
    torch.save({"state": model.state_dict(),
                "cfg": {"v": 2, "ch": model.ch,
                        "blocks": sum(1 for m in model.trunk if isinstance(m, torch.nn.Conv2d)),
                        "scal_hidden": model.scal[0].out_features,
                        "n_scalar": model.scal[0].in_features},
                "extra": extra or {}}, path)


def load(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(blob["cfg"])
    cfg.pop("v", None)
    model = ExecNet2(**cfg)
    model.load_state_dict(blob["state"])
    model.eval()
    return model, blob.get("extra", {})


def is_v2(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return int(blob.get("cfg", {}).get("v", 1)) == 2
