"""The learned readout as a plan source: predicted stock 3 dawns on, committed.

Division of labour, measured (tools/bc_tile_error.py, 3,540 leader dawns):

    what the leader holds on a given day    readout 0.72 tiles   priced 7.13
    what the leader's CAPITAL RATIO is      priced within 0.03-0.10

The readout knows the TRAJECTORY -- how many of each asset a top player has
standing on a given day, learned from 1,188 top-band sessions on labels that
finally describe them (v4, 98.7% expressible). `rlv3.portfolio.target_stock`
knows the DESTINATION and the risk tilt but not the pace. So the readout sets
the pace and the pricing sets the order; neither is asked for the thing it is
bad at.

THE COMMITMENT. The readout is trained at a 3-day horizon: dawn d is labelled
with the stock at the end of day d+2. So its prediction IS the 3-day target,
and the commitment is the delta from what stands now:

    plant_quota[c]  = max(0, predicted_crops[c] - standing_crops[c])
    animal_buys[k]  = max(0, predicted_herd[k]  - (herd + shed + pocket)[k])
    structures      = derived from animal_buys, the planner's own pipeline rule
    hires_target    = predicted hires, under the planner's own budget clamp

Asked for in FULL today; tomorrow's dawn re-predicts from the new board and the
delta shrinks by whatever was delivered. That re-derivation is the
self-correction -- it needs no separate machinery.

Land is NOT taken from the readout. It is the one decision with a hard gate on
record (labour gates land, the fourth quadrant -$6,266 replicated), and the
leaders themselves stop at two quadrants in 37 of 40 games. The readout's land
signal is logged for the S0 read and left to the planner's own rule.
"""
from __future__ import annotations

import collections
import os

from kagent import econ, features
from kagent.constants import ANIMALS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS = os.path.join(REPO, "rl", "runs", "headfix", "readout_v5_h3", "bc.pt")
VERSION = 5
HORIZON = 3                     # days the prediction is FOR; masks() bounds capacity at it
DAILY_ANIMAL_CAP = 4            # the champion plan's own per-day cap on a kind
# The planner's own last purchase day per kind (planner._champ_herd_targets:
# COW and GOOSE targets go to 0 after day 18, SHEEP after 20) -- an animal
# bought later cannot pay back its cost and feed before the season ends. The
# readout override bypassed this and the S0 read showed it: livestock share
# 0.35-0.41 on days 18-25 and 0.57 on day 29 against the leaders' 0.00.
LAST_BUY_DAY = {"GOOSE": 18, "COW": 18, "SHEEP": 20}


def last_plant_day(crop):
    """The planner's own last useful planting day for `crop`, or the engine
    bound where the planner has none: a crop planted later cannot yield
    before the season ends. The horizon-3 label clamps to day 29, so on the
    last dawns the readout predicts today's stock and the persistence ask
    says "replant everything" -- S0 read 0.68 seed share on day 29 against
    the leaders' 0.00 before this existed."""
    from kagent import planner as P
    from kagent.constants import CROPS as _C
    named = {"WHEAT": getattr(P, "WHEAT_LAST_PLANT_DAY", None),
             "CARROT": getattr(P, "CARROT_LAST_PLANT_DAY", None),
             "TOMATO": getattr(P, "TOMATO_LAST_PLANT_DAY", None),
             "MELON": getattr(P, "MELON_LAST_PLANT_DAY", None)}
    if named.get(crop) is not None:
        return int(named[crop])
    c = _C.get(crop, {})
    lead = c.get("first_yield_day") if c.get("ongoing") else c.get("max_yield_day")
    return 29 - int(lead or 0)


class Readout:
    def __init__(self, weights=DEFAULT_WEIGHTS, version=VERSION, horizon=HORIZON):
        import torch
        from rl.common import PlannerNet

        if features.HEADS_VERSION != version:
            features.set_heads_version(version)
        sd = torch.load(weights, map_location="cpu")["policy"]
        n_in = sd["h1.weight"].shape[1]
        self.net = PlannerNet(n_in, board=False)
        self.net.load_state_dict(sd)
        self.net.eval()
        self.n_in = n_in
        self.version = version
        self.horizon = horizon
        self.heads = list(features.HEADS)
        self.path = weights

    def predict(self, view, memory):
        """-> {"crops": {crop: n}, "herd": {kind: n}, "hires": n, "land": 0|1}
        -- the standing stock this readout expects three dawns on."""
        import torch
        from rl.common import masked_dists

        x = features.encode(view, memory)
        if len(x) != self.n_in:
            raise RuntimeError(f"readout wants {self.n_in} features, encode gave "
                               f"{len(x)} -- heads version mismatch "
                               f"(features.HEADS_VERSION={features.HEADS_VERSION})")
        m = features.masks(view, horizon=self.horizon)
        flat = sum(m, []) if m and isinstance(m[0], list) else list(m)
        with torch.no_grad():
            logits = self.net.forward_z(torch.tensor([x], dtype=torch.float32), None)
            dists = masked_dists(logits, torch.tensor([flat], dtype=torch.float32))
            idx = [int(d.probs.argmax(-1)[0]) for d in dists]
        out = {"crops": {}, "herd": {}, "hires": None, "land": 0, "fert": None}
        for (name, vals), i in zip(self.heads, idx):
            v = vals[i]
            if name in features.QUOTA_CROP:
                out["crops"][features.QUOTA_CROP[name]] = int(v)
            elif name in features.BUY_KIND:
                out["herd"][features.BUY_KIND[name]] = int(v)
            elif name == "hires":
                out["hires"] = int(v)
            elif name == "land":
                out["land"] = int(v)
            elif name == "fert_budget":
                out["fert"] = int(v)
        return out


def standing(view):
    crops, herd = collections.Counter(), collections.Counter()
    for _x, _y, t in view.me.plants():
        if t.get("crop"):
            crops[t["crop"]] += 1
    for _x, _y, t in view.me.animals():
        if t.get("animal"):
            herd[t["animal"]] += 1
    pocket = collections.Counter()
    for i, _px, _py in view.me.units:
        for it, n in view.unit_inv(i).items():
            pocket[it] += n
    empty = collections.Counter(t["kind"] for _x, _y, t in view.me.empty_structures())
    return crops, herd, pocket, empty


def commit(plan, view, pred, order=None, horizon=HORIZON):
    """Overwrite the allocation fields with the 3-day commitment. Returns the
    plan, and the delta it asked for, so the S0 read can score it.

    THE COMMITMENT SUBTRACTS ONLY WHAT PERSISTS. The target is a standing
    stock, and a stock delta -- target minus standing -- cannot see a
    replant: a wheat tile that harvests inside the window and is planted
    again ends the window standing, so its delta is zero while the leader
    planted it. Measured on a leader's board (2026-09-29, one game): wheat
    `freeing` equalled `standing` on EVERY dawn, because the crop's cycle is
    shorter than the window, so the leader's 3-day flow (7-17 plants) was
    mostly replanting and the stock delta read 0-6. So:

        ask[c] = target[c] - (standing[c] - freeing[c])

    i.e. the tiles of c that will still be standing at the horizon are the
    only ones already counted toward the target; the rest must be planted,
    replants included. For an ongoing crop nothing frees and this IS the
    stock delta. On that same board the persistence-adjusted ask read 17 vs
    the leader's 16 on day 10 and 17 vs 15 on day 18.

    It can over-ask on a dawn the readout overshoots (day 8: 22 asked, 7
    planted). That is bounded by the executor, which plants only what is
    bare, and corrected at the next dawn's re-derivation; the plan is
    re-made every day.
    """
    crops, herd, pocket, empty = standing(view)
    shed = view.shed or {}
    freeing = features.freeing_by_crop(view, horizon)

    quota = {c: max(0, int(n) - (crops.get(c, 0) - freeing.get(c, 0)))
             for c, n in pred["crops"].items()
             if view.day <= last_plant_day(c)}          # past payback: no ask
    # ORDER MATTERS: the scheduler consumes plant_quota in dict order and a
    # queue dealt in fixed order starved the last crop (melon realised 0.07
    # against a 0.15 share). The pricing's rank is the order.
    if order:
        ranked = [c for c in order if quota.get(c, 0) > 0]
        rest = [c for c in quota if c not in ranked and quota[c] > 0]
        quota = {c: quota[c] for c in ranked + rest}
    else:
        quota = {c: n for c, n in quota.items() if n > 0}

    buys = {}
    for k, n in pred["herd"].items():
        if view.day > LAST_BUY_DAY.get(k, 29):
            continue                          # past payback; the planner's own rule
        want = int(n) - herd.get(k, 0) - shed.get(k, 0) - pocket.get(k, 0)
        if want > 0:
            buys[k] = min(want, DAILY_ANIMAL_CAP)

    # The planner's own structure pipeline, verbatim in shape.
    builds = {"COOP": 0, "PASTURE": 0}
    empty = dict(empty)
    for kind, structure in (("GOOSE", "COOP"), ("COW", "PASTURE"), ("SHEEP", "PASTURE")):
        pipeline = buys.get(kind, 0) + shed.get(kind, 0) + pocket.get(kind, 0)
        need = pipeline - empty.get(structure, 0)
        if need > 0:
            builds[structure] += need
            empty[structure] = 0
        else:
            empty[structure] = -need

    hires = plan.hires_target if pred["hires"] is None else int(pred["hires"])
    budget = float(view.me.money)
    while hires > 2 and econ.hire_total_cost(hires) > budget:
        hires -= 1

    plan.plant_quota = quota
    plan.animal_buys = buys
    plan.structure_builds = builds
    plan.hires_target = hires
    if pred.get("fert") is not None:
        plan.fert_budget = int(pred["fert"])      # v5: FERTILIZE actions today
    delta = {"crops": dict(quota), "herd": dict(buys), "hires": hires,
             "fert": pred.get("fert"),
             "land_signal": pred["land"], "pred": pred,
             "freeing": dict(freeing), "standing": dict(crops)}
    return plan, delta
