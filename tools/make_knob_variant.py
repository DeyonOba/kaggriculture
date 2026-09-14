"""Clone a frozen agent bundle with extra constant knob rules patched in.

Why not just edit rules.json: `rl/export.py --freeze` bakes the rules into
`BANDIT_RULES` inside the bundle's embedded weights module, and the shipped
`main.py` never reads rules.json at run time. Rebuilding from `kagent/` instead
risks the head-value-semantics trap (a dev rebuild of rl-v3 once lost 0-10 to
its own frozen bundle), so a knob gate must start from the frozen champion and
change exactly one literal.

    PYTHONPATH=. uv run python tools/make_knob_variant.py \
        --base champ-h8 --name wheat48 --set wheat_engine=48

The variant differs from its base by those knob rules and nothing else; verify
with `diff <(...) <(...)` or the byte count this prints.
"""

import argparse
import json
import os
import re
import shutil

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(REPO, "arena", "agents")


def patch_rules(src, extra):
    """Insert `extra` condition dicts at the front of BANDIT_RULES.conditions.

    The rules live inside a repr'd module source, so the literal in the file is
    escaped (\\'knob\\'). Rules are evaluated in order and a later rule
    overwrites an earlier one for the same knob, so prepending means an
    explicit --set always loses to a base rule for the same knob; we therefore
    APPEND, and assert the knob is not already present.
    """
    marker = "BANDIT_RULES = {"
    start = src.index(marker)
    end = src.index("}\\n')", start)          # end of the weights module source
    block = src[start:end]

    rules_start = block.index("\\'conditions\\': [")
    insert_at = start + block.rindex("}]") + 1   # just before the closing ]

    parts = []
    for knob, value in extra:
        lit = json.dumps(value)
        parts.append("{\\'knob\\': \\'%s\\', \\'value\\': %s}" % (knob, lit))
    payload = ", " + ", ".join(parts)
    return src[:insert_at] + payload + src[insert_at:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", dest="sets", action="append", required=True,
                    metavar="KNOB=VALUE", help="repeatable, e.g. --set wheat_engine=48")
    args = ap.parse_args()

    base_dir = os.path.join(AGENTS, args.base)
    out_dir = os.path.join(AGENTS, args.name)
    src_path = os.path.join(base_dir, "main.py")
    if not os.path.isfile(src_path):
        raise SystemExit(f"no frozen bundle at {src_path}")

    extra = []
    for item in args.sets:
        knob, _, raw = item.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        extra.append((knob.strip(), value))

    src = open(src_path).read()
    for knob, _ in extra:
        if f"\\'{knob}\\'" in src[src.index("BANDIT_RULES = {"):][:2000]:
            raise SystemExit(f"{args.base} already carries a rule for {knob}; "
                             f"patching would fight it")
    out = patch_rules(src, extra)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "main.py"), "w") as f:
        f.write(out)
    rules = os.path.join(base_dir, "rules.json")
    if os.path.isfile(rules):
        data = json.load(open(rules))
        data.setdefault("conditions", []).extend(
            {"knob": k, "value": v} for k, v in extra)
        json.dump(data, open(os.path.join(out_dir, "rules.json"), "w"))

    sets = " ".join(f"{k}={v}" for k, v in extra)
    print(f"{args.name}: {args.base} + [{sets}]  "
          f"({len(src)} -> {len(out)} bytes, +{len(out)-len(src)})")


if __name__ == "__main__":
    main()
