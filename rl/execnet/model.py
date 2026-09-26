"""The execnet model: a small MLP over the 5x5 tile window plus the scalar
vector, with a legality mask applied to the logits. Saved as a dict
{"state": state_dict, "cfg": {...}} so the policy can rebuild it."""
from __future__ import annotations

import torch
import torch.nn as nn

from rl.execnet import features as F


class ExecNet(nn.Module):
    def __init__(self, hidden=256, tile_hidden=256, scal_hidden=64):
        super().__init__()
        self.tile = nn.Sequential(nn.Linear(F.N_TILE, tile_hidden), nn.ReLU(),
                                  nn.Linear(tile_hidden, tile_hidden), nn.ReLU())
        self.scal = nn.Sequential(nn.Linear(F.N_SCALAR, scal_hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(tile_hidden + scal_hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, len(F.OPS)))

    def forward(self, tiles, scal, mask=None):
        h = torch.cat([self.tile(tiles), self.scal(scal)], dim=-1)
        logits = self.head(h)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1e9)
        return logits


def save(model, path, extra=None):
    torch.save({"state": model.state_dict(), "cfg": {"hidden": model.head[0].out_features,
                                                     "tile_hidden": model.tile[0].out_features,
                                                     "scal_hidden": model.scal[0].out_features},
                "extra": extra or {}}, path)


def load(path):
    blob = torch.load(path, map_location="cpu")
    model = ExecNet(**blob["cfg"])
    model.load_state_dict(blob["state"])
    model.eval()
    return model, blob.get("extra", {})
