"""Materialise SEVERAL per-day schedules into one frozen bundle.

`tools/rollout_plan.py` can rewrite one schedule at a time, which is all a
greedy per-day rollout needs. Searching the plan space jointly needs all of
them applied to the same planner source in one pass, so the edits have to
compose: each `find` anchor must still be present after the previous edits ran.
They are disjoint today, and this asserts it rather than trusting it.

Also provides the curve parameterisation the search uses. A 30-day schedule is
30 numbers, which is too many to search directly at ~4 s/game; but the useful
schedules found so far (the hiring rollout) are smooth ramps, so each schedule
is represented by `k` control points interpolated over its day range. That
turns ~120 dimensions into ~20.

    from tools.sched_bundle import bundle_for_multi, curve
    path = bundle_for_multi({"hires": curve([5, 9, 11, 9], 30, (0, 29))})
"""

import ast
import hashlib
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from tools.rollout_plan import SCHEDULES, BASE_AGENT, AGENT_DIR, _split_bundle  # noqa: E402


def curve(points, n_days=30, days=None, lo=0, hi=None, integer=True):
    """Linearly interpolate `points` across `days`, holding the base outside it.

    Outside the schedule's active window the value is clamped to the first/last
    control point, which is what every schedule in SCHEDULES already does with
    its zero padding.
    """
    d0, d1 = days if days else (0, n_days - 1)
    out = []
    span = max(1, d1 - d0)
    k = len(points) - 1
    for d in range(n_days):
        if d < d0:
            v = points[0]
        elif d > d1:
            v = points[-1]
        else:
            t = (d - d0) / span * k
            i = min(int(t), k - 1) if k else 0
            v = points[i] + (t - i) * (points[i + 1] - points[i]) if k else points[0]
        v = max(lo, v if hi is None else min(hi, v))
        out.append(int(round(v)) if integer else float(v))
    return out


def bundle_for_multi(schedules, base_agent=None, tag="multi"):
    """schedules: {name: [30 ints]} -> path to a frozen bundle carrying them all."""
    if not schedules:
        raise ValueError("no schedules given")
    for name in schedules:
        if name not in SCHEDULES:
            raise KeyError(f"unknown schedule {name!r}; have {sorted(SCHEDULES)}")

    sig = ";".join(f"{k}:{list(schedules[k])}" for k in sorted(schedules))
    key = hashlib.sha1(f"{tag}:{base_agent}:{sig}".encode()).hexdigest()[:12]
    d = os.path.join(AGENT_DIR, f"s-{key}")
    p = os.path.join(d, "main.py")
    if os.path.exists(p):
        return p

    src = open(os.path.join(REPO, base_agent or BASE_AGENT)).read()
    head, mods, tail = _split_bundle(src)

    out, applied = [], []
    for mod, text in mods:
        if mod == "planner":
            inject = []
            for name in sorted(schedules):
                spec, sched = SCHEDULES[name], list(schedules[name])
                if spec["array"] is None:                 # hires: swap the literal
                    text, n = re.subn(r"CHAMP_HIRES = \[[0-9,\s\n]*?\]",
                                      "CHAMP_HIRES = " + repr(sched), text, count=1)
                    if n != 1:
                        raise RuntimeError("CHAMP_HIRES literal not found")
                else:
                    if spec["find"] not in text:
                        raise RuntimeError(
                            f"anchor for {name!r} not found -- an earlier edit "
                            f"in this pass consumed it; the edits are not disjoint")
                    text = text.replace(spec["find"], spec["repl"], 1)
                    inject.append(f"{spec['array']} = {sched!r}")
                applied.append(name)
            if inject:
                k = text.index("CHAMP_HIRES = ")
                text = text[:k] + "\n".join(inject) + "\n" + text[k:]
        out.append((mod, text))

    if sorted(applied) != sorted(schedules):
        raise RuntimeError(f"applied {applied} but asked for {sorted(schedules)}")

    os.makedirs(d, exist_ok=True)
    open(p, "w").write(head + "_SOURCES = " + repr(out) + tail)
    compile(open(p).read(), p, "exec")     # a bundle that cannot import scores $3,000
    return p


def base_schedules(names):
    return {n: list(SCHEDULES[n]["base"]) for n in names}


if __name__ == "__main__":
    import sys
    names = sys.argv[1:] or ["hires", "straw", "wheat", "land"]
    p = bundle_for_multi(base_schedules(names))
    print(f"{len(names)} schedules applied together -> {p}")
    print("this bundle should be behaviourally identical to the base:")
    print("  " + " ".join(names))
