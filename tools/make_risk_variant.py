"""Three risk-adjusted rent rules, sharing one consumer so they are comparable.

`_rent_pick` maximises EXPECTED revenue per tile-day and has no notion of
variance, so it cannot express "bet on stable items" or "they are on stable,
so the risky ones are uncontested". Three ways to add that, tested against
each other rather than chosen a priori:

  --mode computed    rent - lam * sd, sd COMPUTED from the engine. The shop
                     draw is eight uniform draws and price is closed form on
                     inventory, so the distribution is exact
                     (tools/volatility.py). Scaled by how many draws are still
                     to come -- risk falls as the season resolves.

  --mode learnt      rent - lam * sd, sd FITTED from 51,122 corpus samples
                     (tools/fit_risk.py). Note the variance model is weak
                     (best R2 0.135, negative for TOMATO and WOOL), so this is
                     close to a per-product constant. That is itself the
                     finding: risk LEVEL is knowable, risky STATES are not.

  --mode diff        rent - lam * (their visible holding of that product).
                     The game-theoretic one: both farms sell into ONE
                     market["inventory"], so a product their board is loaded
                     with is contested and one they ignore is free. This is
                     the only arm that targets MARGIN rather than money.

alpha=0 / lam=0 is bit-identical to the base by construction.

    PYTHONPATH=. uv run python tools/make_risk_variant.py --name k-comp --mode computed --lam 0.5
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tools.rollout_plan import _split_bundle   # noqa: E402

# sd of realised $/unit as a FRACTION of base.
# computed: tools/volatility.py, sd/E over the shop-draw distribution
# learnt:   tools/fit_risk.py, sd[ratio] over 51,122 corpus samples
SD_COMPUTED = {"WHEAT": 0.09, "CARROT": 1.07, "TOMATO": 1.60, "STRAWBERRY": 0.16,
               "MELON": 0.00, "EGG": 0.28, "MILK": 0.21, "WOOL": 0.73,
               "FERTILIZER": 0.00}
SD_LEARNT = {"WHEAT": 0.20, "CARROT": 0.74, "TOMATO": 0.93, "STRAWBERRY": 0.52,
             "MELON": 0.25, "EGG": 0.16, "MILK": 0.43, "WOOL": 0.37,
             "FERTILIZER": 0.15}

ANCHOR = '''        rev = econ.sell_revenue(crop, int(round(inv[crop][k])), total)
        rent = rev / float(free * occupies)
        if best is None or rent > best[1]:
            best = (crop, rent)'''

REPL = '''        rev = econ.sell_revenue(crop, int(round(inv[crop][k])), total)
        rent = rev / float(free * occupies)
        rent = _risk_adjust(rent, crop, view, day)
        if best is None or rent > best[1]:
            best = (crop, rent)'''

GLUE = '''

# --- risk-adjusted rent -----------------------------------------------------
_RISK_MODE = "__MODE__"
_RISK_LAM = __LAM__
_RISK_SD = __SD__


def _risk_adjust(rent, crop, view, day):
    if _RISK_LAM <= 0:
        return rent
    try:
        if _RISK_MODE == "diff":
            # Their visible holding of this product: both farms sell into one
            # market["inventory"], so what they are loaded with is contested.
            n = 0
            for _x, _y, _t in view.opp.plants():
                if _t.get("crop") == crop:
                    n += 1
            pen = min(n / 30.0, 1.0)
        else:
            sd = _RISK_SD.get(crop, 0.3)
            if _RISK_MODE == "computed":
                # risk resolves as the shop draws land
                left = max(0, 8 - len(view.unlocked_shops)) / 8.0
                sd = sd * (0.35 + 0.65 * left)
            pen = sd
        return rent * (1.0 - _RISK_LAM * pen)
    except Exception:
        return rent

'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="r-mrg100")
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=["computed", "learnt", "diff"], required=True)
    ap.add_argument("--lam", type=float, required=True)
    args = ap.parse_args()

    sd = SD_COMPUTED if args.mode == "computed" else SD_LEARNT
    src = open(os.path.join(REPO, "arena", "agents", args.base, "main.py")).read()
    head, mods, tail = _split_bundle(src)
    out, done = [], False
    for mod, text in mods:
        if mod == "planner":
            if ANCHOR not in text:
                raise SystemExit("rent anchor not found; is the base r-mrg100?")
            text = (GLUE.replace("__MODE__", args.mode)
                        .replace("__LAM__", repr(args.lam))
                        .replace("__SD__", repr(sd))
                    + text.replace(ANCHOR, REPL, 1))
            done = True
        out.append((mod, text))
    if not done:
        raise SystemExit("no planner module in bundle")

    p = os.path.join(REPO, "arena", "agents", args.name)
    os.makedirs(p, exist_ok=True)
    f = os.path.join(p, "main.py")
    code = head + "_SOURCES = " + repr(out) + tail
    open(f, "w").write(code)
    compile(code, f, "exec")
    print(f"{args.name}: {args.base} + {args.mode} risk, lam={args.lam}")


if __name__ == "__main__":
    main()
