---
name: build-variant
description: Build a Kaggriculture agent variant by patching a frozen bundle, and pre-flight it so it is not silently dead. Use whenever creating a new candidate agent or exporting a tape.
---

# Building a variant

Candidates are made by **textually patching a frozen bundle**, never by editing
`kagent/` and rebuilding — the dev tree and the frozen champion have diverged
before, and `tools/submit.sh` rebuilds from `kagent/`, so it would ship
different code than you tested.

## The bundle shape

`arena/agents/<name>/main.py` is:

```python
_SOURCES = [("constants", "..."), ("econ", "..."), ("planner", "..."), ...]
_G = {"__name__": "kagent_bundle"}
for _name, _src in _SOURCES: exec(compile(_src, ...), _G)
```

`_SOURCES` is `ast.literal_eval`-able. **All modules share ONE globals dict**, so
anything defined in any module is visible from any other (`econ.forecast_drain`,
`supply_projection` etc. resolve at call time).

Pattern — copy an existing `tools/make_*_variant.py`:

```python
src = open(f"arena/agents/{base}/main.py").read()
i = src.index("_SOURCES = "); j = src.index("\n_G = {", i)
head, mods, tail = src[:i], ast.literal_eval(src[i+len("_SOURCES = "):j]), src[j:]
for mod, text in mods:
    if FIND in text: text = text.replace(FIND, REPL, 1)
open(path, "w").write(head + "_SOURCES = " + repr(out) + tail)
compile(open(path).read(), path, "exec")
```

**Always `raise` if an anchor is not found.** A silent no-match produces a
variant identical to its base, and you will gate a copy of the baseline against
itself. (This happened: three `c-opp*` thresholds came back bit-identical.)

## Pre-flight — non-negotiable

```python
src = open(p).read()
compile(src, p, "exec")                       # compiles
g = {"__name__": "probe"}; exec(compile(src, p, "exec"), g)   # IMPORTS
assert callable(g.get("agent"))
# then play 2-3 games and check the score is NOT exactly $3,000
```

**`$3,000` exactly is the silent-death signature.** The bundle raised on import
and kaggle-environments no-opped every turn. It compiles, it looks valid, it does
nothing. This shipped undetected once: `EXPORT_TMPL` grew placeholders
(`__SE_BLOCK__`, `__FREE_WORK_BLOCK__`, …) that only `export_selector`
substituted, so **every single-tape export raised NameError** until fixed.
`compile()` does NOT catch it — only `exec` plus a smoke game does.

Also check: size ≤ 100 MiB (we ship 0.01–0.11), and **no file I/O** — the
submission lives at `/kaggle_simulations/agent/`, so a bundle must be
self-contained.

## Verify the artifact, not the prototype

If a candidate was measured in-process (e.g. `make_tape_agent(actions)`), the
exported bundle carries its **own copy** of the logic and can diverge. Re-run the
exported bundle on the same seed block and compare before trusting the number.

## Gitignore

Throwaway variants are ignored: `arena/agents/{f,match,lay,sell,c,x,v,b,r,w,o,onm,bc,tape-112,tape-277}-*`.
Add new prefixes rather than committing agent dirs.
