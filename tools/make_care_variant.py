"""Variants for the two rule-compliance misses that survived diagnosis.

tools/executor_audit.py counted 97.9 skipped CAREs and 1.0 escaped animal
per game. tools/care_diagnose.py then classified them, and most of the CARE
number is not a defect:

    54.7%  product below 35% of base   deliberate crash-pivot
    22.0%  animal was not fed          a FEED problem, not a CARE problem
    12.6%  day >= 28                   deliberate, cannot pay before the bell
    10.7%  emitted, nobody got to it   <-- the real slice, 13.3/game

Escapes are the opposite: **100% gateable**. Wheat was on the farm in every
single case, so a P_SURVIVAL FEED task existed, wheat existed, and the
animal starved anyway. That is routing, not supply.

Three knobs, each defaulting to the SHIPPED value so the variant is
bit-identical to the base at defaults:

    CARE_PRI         1.5   -> lower number = scheduled earlier
    FEED_SURV_PRI    0.0   -> priority of FEED on an animal about to escape
    CARE_PRICE_GATE  0.35  -> skip CARE below this fraction of base price

    PYTHONPATH=. uv run python tools/make_care_variant.py
"""
import ast
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(REPO, "arena/agents")
sys.path.insert(0, REPO)

FIND = '''            pri = P_SURVIVAL if t.get("consecutive_unfed", 0) >= 1 else P_ANIMAL
            tasks.append(Task(pri, x, y, ["FEED"], need="WHEAT"))'''
REPL = '''            pri = FEED_SURV_PRI if t.get("consecutive_unfed", 0) >= 1 else P_ANIMAL
            tasks.append(Task(pri, x, y, ["FEED"], need="WHEAT"))'''

FIND2 = '''                or view.market_prices.get(product, 999) >= 0.35 * MARKET_PARAMS[product]["base"]'''
REPL2 = '''                or view.market_prices.get(product, 999) >= CARE_PRICE_GATE * MARKET_PARAMS[product]["base"]'''

FIND3 = '''                tasks.append(Task(P_ANIMAL, x, y, ["CARE"]))'''
REPL3 = '''                tasks.append(Task(CARE_PRI, x, y, ["CARE"]))'''

CONSTS = '''CARE_PRI = __CARE_PRI__
FEED_SURV_PRI = __FEED_SURV__
CARE_PRICE_GATE = __GATE__

'''


def build(base, name, care_pri=1.5, feed_surv=0.0, gate=0.35):
    src = open(os.path.join(AGENTS, base, "main.py")).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head = src[:i]
    mods = ast.literal_eval(src[i + len("_SOURCES = "):j])
    tail = src[j:]

    out, hits = [], 0
    for mod, text in mods:
        if FIND in text:
            text = text.replace(FIND, REPL, 1)
            text = text.replace(FIND2, REPL2, 1)
            text = text.replace(FIND3, REPL3, 1)
            text = (CONSTS.replace("__CARE_PRI__", repr(float(care_pri)))
                          .replace("__FEED_SURV__", repr(float(feed_surv)))
                          .replace("__GATE__", repr(float(gate)))) + text
            hits += 1
        out.append((mod, text))
    if hits != 1:
        raise RuntimeError(f"animal FEED/CARE block matched {hits} modules")

    d = os.path.join(AGENTS, name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "main.py")
    open(p, "w").write(head + "_SOURCES = " + repr(out) + tail)
    compile(open(p).read(), p, "exec")
    return p


ARMS = {
    # identity: every knob at its shipped value. MUST be bit-identical.
    "care-id":    dict(care_pri=1.5, feed_surv=0.0, gate=0.35),
    # A: the 10.7% labour-bound CARE slice -- schedule CARE ahead of HARVEST
    "care-pri":   dict(care_pri=0.9, feed_surv=0.0, gate=0.35),
    # B: escapes. P_SURVIVAL is already 0, the minimum, so a strictly
    #    NEGATIVE priority is the only way to put it ahead of everything.
    "care-surv":  dict(care_pri=1.5, feed_surv=-1.0, gate=0.35),
    # C: both
    "care-both":  dict(care_pri=0.9, feed_surv=-1.0, gate=0.35),
}


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "champ-h8"
    for name, kw in ARMS.items():
        p = build(base, name, **kw)
        print(f"  {name:<12} {kw}  -> {p}")


if __name__ == "__main__":
    main()
