"""Risk = how hard the price falls when you SELL, not how the shop draw lands.

`k-comp5`'s `_RISK_SD` is computed by `tools/volatility.py` from the shop
draw -- eight uniform draws every three days -- so it measures **demand
uncertainty**. Ranked against the actual glut curve it is inverted:

    crop         shops   base   P(I0+T)   glut drop   k-comp5 sd
    MELON            0    250        $1        100%         0.00   <- least risky
    STRAWBERRY       4    120        $1         99%         0.16
    CARROT           3     35       $10         71%         1.07
    TOMATO           2     60       $24         60%         1.60   <- most risky
    WHEAT            5     25       $20         20%         0.09

Melon is the one plantable crop in **no shop at all** (README Town Buildings:
bakery/pizza/brunch/yarn/ice-cream/pet-cafe/smoothie/farmers-market -- melon
is in none of them), so its only demand is the town centre's flat 1 unit a
day. Zero shop demand means zero shop-draw variance, which is why the fitted
model calls it risk-free -- and zero shop demand is exactly what makes it
dangerous. It carries the steepest glut curve in the game (`sq` at
`above_target` 3.60) and hits the $1 floor inside one field's production.

So this replaces the sd with the **structural glut coefficient**: the
fractional price drop after selling one 5x5 field's worth (T units),
computed from MARKET_PARAMS. No fitting -- it is arithmetic on the price
function, which is the only thing that has ever paid here.

Mode `glut` also drops the `0.35 + 0.65 * shops_left` decay that `computed`
applies. That decay says risk resolves as the season's draws land; glut risk
does not resolve, and for melon it cannot -- no draw can ever add a melon
shop.

    PYTHONPATH=. uv run python tools/make_glut_variant.py --name x-glut50 --lam 0.5
"""
import argparse
import ast
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def glut_table():
    """Fractional price drop after T units of supply, per product."""
    from kagent.constants import MARKET_PARAMS
    from kagent import econ
    out = {}
    for p, mp in MARKET_PARAMS.items():
        base, at_T = mp["base"], econ.price(p, mp["I0"] + mp["T"])
        out[p] = round((base - at_T) / float(base), 3)
    return out


def build(base, name, lam):
    table = glut_table()
    src_path = os.path.join(REPO, "arena", "agents", base, "main.py")
    src = open(src_path).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, mods, tail = src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]

    out, hits = [], 0
    for mod, text in mods:
        if mod == "planner" and "_RISK_SD" in text:
            text, n1 = re.subn(r'_RISK_MODE = "[a-z]+"', '_RISK_MODE = "glut"', text, count=1)
            text, n2 = re.subn(r"_RISK_LAM = [0-9.]+", f"_RISK_LAM = {lam}", text, count=1)
            text, n3 = re.subn(r"_RISK_SD = \{[^}]*\}", "_RISK_SD = " + repr(table), text, count=1)
            hits = n1 + n2 + n3
            if hits != 3:
                raise SystemExit(f"expected 3 substitutions, made {hits}")
        out.append((mod, text))
    if not hits:
        raise SystemExit(f"{base} carries no _RISK_SD -- build it from a risk bundle")

    dst_dir = os.path.join(REPO, "arena", "agents", name)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "main.py")
    with open(dst, "w") as f:
        f.write(head + "_SOURCES = " + repr(out) + tail)
    rules = os.path.join(REPO, "arena", "agents", base, "rules.json")
    if os.path.isfile(rules):
        import shutil
        shutil.copyfile(rules, os.path.join(dst_dir, "rules.json"))

    text = open(dst).read()
    compile(text, dst, "exec")
    g = {"__name__": "probe"}
    exec(compile(text, dst, "exec"), g)
    if not callable(g.get("agent")):
        raise SystemExit("bundle exec'd but defines no callable `agent`")
    print(f"wrote {dst}  ({os.path.getsize(dst)/1024:.0f} KiB)  lam={lam}")
    print(f"  glut sd: " + "  ".join(f"{k} {v}" for k, v in sorted(table.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="x-riskfx")
    ap.add_argument("--name", required=True)
    ap.add_argument("--lam", type=float, default=0.5)
    args = ap.parse_args()
    build(args.base, args.name, args.lam)


if __name__ == "__main__":
    main()
