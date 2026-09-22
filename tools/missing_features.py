"""The observation fields `tools/board_tensor.py` and the 48-metric panel drop.

Audited in `docs/state_capture_spec.md` §3. Every quantity here is present in
the observation on every turn, is free to read, and is encoded nowhere. None
of it is history -- these are all properties of the CURRENT frame, which is
the point: the three-frame temporal stack was refuted per band
(`tools/temporal_bands.py`), so if the state is missing something it is
missing it *now*, not yesterday.

Grouped by why each one should matter.

DEATH CLOCKS. The encoder carries `consecutive_unwatered` for plants and
nothing equivalent for animals, in a game where an unfed animal dies and CARE
doubles yield. `max_lifespan_step` is a hard per-tile deadline the model has
never seen.

BANKED VALUE. `pending_care_bonus` is yield already earned and not yet
realised; `fertilized_until_day` is a paid-for multiplier with an expiry.
Both are invisible, and both are exactly the kind of "value on the board that
is not in the bank" that `docs/game_phases.md` says decides the mid-game.

MARKET SIGN. `scalars()` encodes `min(abs(inv - 10000), 600) / 600`. The
absolute value makes a glut and a shortage identical on the one shared
variable that costs $52,652 a game (z 17.52), and the 600 clip saturates long
before a real glut does. Here the deviation keeps its sign and its range.

SHOP IDENTITY. `len(shops) / 8` cannot express that WHEAT is demanded by five
of the eight shop types and MELON by none -- which is the exact structure the
+$676 future-demand planner term priced. Eight bits instead of one count.

    from tools.missing_features import features, NAMES
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

TPD = 24
CROPS_L = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
PRODUCTS_L = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
              "EGG", "MILK", "WOOL", "FERTILIZER"]
# Taken from kagent.constants.SHOPS, not from memory -- three of the eight
# names I first wrote (PIZZERIA / YARN_SHOP / SMOOTHIE_BAR) do not exist in
# the engine, and those bits would have read zero all season and quietly
# voided the whole shop-identity arm.
def _shop_names():
    from kagent.constants import SHOPS
    return list(SHOPS)


SHOPS_L = _shop_names()

NAMES: List[str] = (
    # death clocks
    ["lifespan_left", "dying_3d", "unfed_streak", "starving", "animal_age"]
    # banked value
    + ["pending_care", "care_cover", "fert_active", "fert_days_left"]
    # labour and cash, both farms, both visible
    + ["opp_money", "hires_today", "opp_hires_today", "opp_hands"]
    # seeds we hold, per crop
    + [f"seed_{c.lower()}" for c in CROPS_L]
    # shop identity, one bit each
    + [f"shop_{s.lower()}" for s in SHOPS_L]
    # SIGNED market inventory deviation, per product
    + [f"minv_{p.lower()}" for p in PRODUCTS_L]
)


def _sat(x: float, scale: float) -> float:
    """tanh squash, the same convention tools/phase_metrics.py uses.

    `min(x, C) / C` discards every distinction above C; tanh keeps the
    ordering everywhere and still bounds the range.
    """
    return float(np.tanh(max(0.0, float(x)) / float(scale)))


def _signed(x: float, scale: float) -> float:
    return float(np.tanh(float(x) / float(scale)))


def features(obs, seat: int = 0, private=None) -> np.ndarray:
    """-> float32 (len(NAMES),) for one seat at one moment."""
    farm = obs["farms"][seat]
    opp = obs["farms"][1 - seat]
    day = int(obs["day"])
    step = int(obs.get("step", day * TPD + int(obs.get("hour", 0))))
    v: Dict[str, float] = {k: 0.0 for k in NAMES}

    life, n_plant, dying = [], 0, 0
    fert_on, fert_left = 0, []
    unfed, ages, care, cared, n_anim = [], [], [], 0, 0
    for row in farm["tiles"]:
        for c in row:
            if not isinstance(c, dict):
                continue
            if c.get("kind") == "PLANT" and c.get("crop"):
                n_plant += 1
                left = (c.get("max_lifespan_step", step) - step) / float(TPD)
                life.append(max(0.0, left))
                if left <= 3.0:
                    dying += 1
                fu = c.get("fertilized_until_day", -1)
                if fu is not None and fu >= day:
                    fert_on += 1
                    fert_left.append(float(fu - day))
            elif "animal" in c and c.get("animal"):
                n_anim += 1
                unfed.append(float(c.get("consecutive_unfed", 0)))
                ages.append(float(day - c.get("placed_day", day)))
                care.append(float(c.get("pending_care_bonus", 0)))
                cared += 1 if c.get("cared_today") else 0

    v["lifespan_left"] = _sat(np.mean(life) if life else 0.0, 12.0)
    v["dying_3d"] = _sat(dying, 8.0)
    v["unfed_streak"] = _sat(np.mean(unfed) if unfed else 0.0, 1.5)
    v["starving"] = _sat(sum(1 for u in unfed if u >= 1.0), 6.0)
    v["animal_age"] = _sat(np.mean(ages) if ages else 0.0, 12.0)
    v["pending_care"] = _sat(sum(care), 20.0)
    v["care_cover"] = (cared / n_anim) if n_anim else 0.0
    v["fert_active"] = _sat(fert_on, 10.0)
    v["fert_days_left"] = _sat(np.mean(fert_left) if fert_left else 0.0, 4.0)

    v["opp_money"] = _sat(float(opp.get("money", 0.0)), 30000.0)
    v["hires_today"] = _sat(float(farm.get("hires_today", 0)), 6.0)
    v["opp_hires_today"] = _sat(float(opp.get("hires_today", 0)), 6.0)
    v["opp_hands"] = _sat(len(opp.get("hands") or []), 12.0)

    seeds = ((private if private is not None else obs.get("private")) or {})
    seeds = (seeds.get("seeds") or {}) if hasattr(seeds, "get") else {}
    for c in CROPS_L:
        v[f"seed_{c.lower()}"] = _sat(float(seeds.get(c, 0)), 10.0)

    shops = set((obs.get("town") or {}).get("unlocked_shops") or [])
    for s in SHOPS_L:
        v[f"shop_{s.lower()}"] = 1.0 if s in shops else 0.0

    inv = (obs.get("market") or {}).get("inventory") or {}
    for p in PRODUCTS_L:
        # SIGNED, and on a scale that does not saturate in a real glut.
        v[f"minv_{p.lower()}"] = _signed(float(inv.get(p, 10000)) - 10000.0,
                                         1500.0)

    return np.asarray([v[k] for k in NAMES], dtype="float32")
