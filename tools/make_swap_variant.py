"""Swap ONE dev-tree module into a frozen bundle, to price the drift per file.

`dev-exec` (everything in `kagent/`) gated −$777 against champ-h8 while the
same executor fix on the frozen champion gated +$454. A single pooled number
cannot say which file costs the money, and `tools/submit.sh` rebuilds from
`kagent/`, so the drift has to be priced before anything ships from there.

Each arm is `--base` with exactly one module's source replaced by the current
`kagent/<module>.py`, imports stripped the same way `build_submission.py`
strips them. Gate every arm against the same base and the effects decompose.

    PYTHONPATH=. uv run python tools/make_swap_variant.py --name h13-planner \\
        --base h12-ufh --module planner
"""
import argparse
import ast
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

INTRA_IMPORT = re.compile(r"^\s*(from kagent[.\w]* import|import kagent)")


def strip_imports(text):
    """Same rule the bundler uses: intra-package imports go, the rest stays.

    A multi-line `from kagent.x import (a, b, c)` has to be consumed to its
    closing paren or the leftover names become a syntax error in the bundle.
    """
    out, skip = [], False
    for line in text.splitlines():
        if skip:
            if ")" in line:
                skip = False
            continue
        if INTRA_IMPORT.match(line):
            if "(" in line and ")" not in line:
                skip = True
            continue
        out.append(line)
    return "\n".join(out)


def _module_source(mod, donor):
    """Dev tree by default; another frozen bundle with --from."""
    if donor is None:
        p = os.path.join(REPO, "kagent", f"{mod}.py")
        if not os.path.isfile(p):
            raise SystemExit(f"no dev module at {p}")
        return strip_imports(open(p).read())
    d = os.path.join(REPO, "arena", "agents", donor, "main.py")
    if not os.path.isfile(d):
        raise SystemExit(f"no donor bundle at {d}")
    src = open(d).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    got = dict(ast.literal_eval(src[i + len("_SOURCES = "):j]))
    if mod not in got:
        raise SystemExit(f"donor {donor} has no module {mod}")
    return got[mod]


def build(base, name, modules, donor=None):
    src_path = os.path.join(REPO, "arena", "agents", base, "main.py")
    src = open(src_path).read()
    i = src.index("_SOURCES = ")
    j = src.index("\n_G = {", i)
    head, mods, tail = src[:i], ast.literal_eval(src[i + len("_SOURCES = "):j]), src[j:]
    have = {m for m, _ in mods}
    missing = [m for m in modules if m not in have]
    if missing:
        raise SystemExit(f"{base} has no module(s) {missing}; it carries {sorted(have)}")

    out, swapped = [], []
    for mod, text in mods:
        if mod in modules:
            new = _module_source(mod, donor)
            if new == text:
                raise SystemExit(f"{mod} is already identical in {base} -- "
                                 f"the arm would be a copy of its base")
            text = new
            swapped.append(mod)
        out.append((mod, text))

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
    print(f"wrote {dst}  ({os.path.getsize(dst)/1024:.0f} KiB)  "
          f"swapped {','.join(swapped)} from {donor or 'kagent/'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="h12-ufh")
    ap.add_argument("--name", required=True)
    ap.add_argument("--module", nargs="+", required=True)
    ap.add_argument("--from", dest="donor", default=None,
                    help="take the module(s) from this frozen bundle instead "
                         "of from kagent/")
    args = ap.parse_args()
    build(args.base, args.name, set(args.module), args.donor)


if __name__ == "__main__":
    main()
