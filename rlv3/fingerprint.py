"""What an artifact was made of -- the anti-stale layer under every skip.

WHY THIS EXISTS. `kaggriculture-gate-cache-stale`: ladder_gate reused a cached
base without checking it covered the CURRENT worlds. The corpus had grown, 61
of 64 arms covered only 119/270 games, and the two win columns of one printed
table came from different world sets. Nobody noticed, because "the output file
is there" is indistinguishable from "the output file is right".

`tools/kaggle_job/job.py` resolves a step by exactly that test -- `skip_if`
hits when the path exists -- and this driver adopts it deliberately, for every
stage, across process and machine boundaries. So the test has to be made safe
rather than merely convenient. Every artifact a stage produces is written with
a sidecar `<artifact>.stamp.json` recording the fingerprint it was made under.
A skip is honoured only when that stamp still matches today's fingerprint on
the keys that artifact actually depends on. Anything else is quarantined into
`<run>/stale/<ts>/` and the stage re-runs. Nothing is ever silently reused.

THE FAILURE THIS IS AIMED AT, concretely. `job_topboard.json`'s run-3 note:
`goose_target` went from [0, 8, 16, 24, 32] to [0, 4, 6, 10, 16], every stored
`choices` index in the BC corpus silently came to mean a different number of
geese, and the operator had to hand-curate the checkpoint dataset so the steps
would not skip themselves back into the old action space. `head_sizes_sha`
catches that in milliseconds, with no games played.

The 995-game draw itself changed once already (785 -> 995 on 26 September),
which would silently make a resumed gate incomparable to the run it continues.
`corpus_draw()` therefore hashes the DRAW -- the actual (path, seat) list that
`rl.session_eval.held_out_entries` returns at the gate's own seed and per-bin
-- not just the index file.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCHEMA = 1

# The keys that describe the ACTION SPACE. A learned artifact -- the BC corpus,
# bc.pt, a PPO run, an eval.json -- is meaningless under a different one.
ACTION_KEYS = (
    "heads_version",
    "heads_mode",
    "head_sizes_sha",
    "n_action_logits",
    "n_features",
    "board_arm",
    "board_ch",
)

# `board` (KAGG_BOARD on/off) is not in ACTION_KEYS because the spec pins that
# tuple, but flipping it changes the net's input width just as surely. Stages
# declare the group "action", which expands to ACTION_KEYS + BOARD_KEYS.
BOARD_KEYS = ("board",)

DEFAULT_INDEX = os.path.join(REPO, "rl", "datasets", "session_index.json")
DEFAULT_SEED = 0
DEFAULT_PER_BIN = 175            # rl/session_eval.py's 995-game draw

# Roots whose *.py content defines "the code that made this".
CODE_ROOTS = ("rlv3", "kagent", "rl")
_CODE_SKIP = ("__pycache__", "runs", "datasets", "stale", ".git")


def _sha1(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def _sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------- action space
def probe_n_features(cache_path=None, play=False):
    """The observation width, WITHOUT playing a game unless asked.

    `rl/train.py:1136` takes it from one BC episode (`len(probe["records"][0]
    ["x"])`). That costs a game, and this runs at the top of every slice on a
    machine that may already be saturated, so: use the value the encoder has
    already resolved (`features.N_FEATURES`, set on first encode), else a
    cached number from a previous slice, else None -- and only play when the
    caller explicitly passes play=True.
    """
    from kagent import features

    if features.N_FEATURES:
        return int(features.N_FEATURES)
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path) as fh:
                v = json.load(fh).get("n_features")
            if v:
                return int(v)
        except (ValueError, OSError):
            pass
    if not play:
        return None
    from rl import rollout

    probe = rollout.run_episode({"weights_path": None, "mode": "bc", "seed": 0,
                                 "side": 0, "opponent": "heuristic",
                                 "temperature": 0.0})
    n = len(probe["records"][0]["x"])
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "w") as fh:
            json.dump({"n_features": n, "written": _now()}, fh)
    return n


def action_space(n_features=None):
    """The live action space, read out of the modules that define it.

    `rl.common.policy_layers` writes heads_version/heads_mode INTO the weights
    and `kagent.nnpolicy.NeuralPolicy.make_plan` calls
    `features.set_heads_version(want)` before encoding, so the head version is
    already the contract between a checkpoint and the agent that runs it. This
    records it so a resumed slice cannot cross that contract.
    """
    from kagent import features
    from rl import common

    board_on = os.environ.get("KAGG_BOARD", "") not in ("", "0")
    return {
        "heads_version": int(features.HEADS_VERSION),
        "heads_mode": str(features.MODE),
        "head_sizes_sha": _sha1(json.dumps(list(features.HEAD_SIZES),
                                           sort_keys=True).encode()),
        "n_action_logits": int(features.N_ACTION_LOGITS),
        "n_features": None if n_features is None else int(n_features),
        "board_arm": os.environ.get("KAGG_BOARD_ARM", "T_crop_103"),
        "board_ch": int(common.BOARD_CH),
        "board": bool(board_on),
        "head_names": [h[0] for h in features.HEADS],   # provenance, not compared
    }


# -------------------------------------------------------------------- corpus
def corpus_draw(index_path=None, seed=DEFAULT_SEED, per_bin=DEFAULT_PER_BIN):
    """The gate's own draw, hashed.

    Calls `rl.session_eval.held_out_entries` -- the SAME function the 995-game
    instrument calls -- so this cannot drift from what the gate will actually
    play. `draw_sha` covers the ordered (path, seat) list: the corpus growing
    by one session reshuffles the draw and changes this hash even when the
    counts per bin are unchanged.
    """
    from rl import session_eval

    index_path = index_path or DEFAULT_INDEX
    if not os.path.exists(index_path):
        return {"index": None, "index_sha": None, "seed": seed,
                "per_bin": per_bin, "counts": [], "n": 0, "draw_sha": None,
                "missing": True}
    entries = session_eval.held_out_entries(index_path, seed=seed, per_bin=per_bin)
    counts = [sum(1 for e in entries if e[2] == b)
              for b in range(len(session_eval.EDGES))]
    draw = json.dumps([[e[0], e[1]] for e in entries]).encode()
    return {
        "index": os.path.relpath(os.path.abspath(index_path), REPO),
        "index_sha": _sha1_file(index_path),
        "seed": int(seed), "per_bin": int(per_bin),
        "counts": counts, "n": len(entries),
        "draw_sha": _sha1(draw),
    }


# ---------------------------------------------------------------------- code
def code_digest(roots=CODE_ROOTS):
    """sha1 over sorted (relpath, sha1(bytes)) of every *.py under `roots`.

    Deliberately coarse: any edit anywhere in rlv3/kagent/rl invalidates the
    artifacts that declare `code`. A stage that only depends on the action
    space (the gate) does not declare it, so an unrelated edit does not throw
    away an hour of eval.
    """
    rows = []
    for root in roots:
        base = root if os.path.isabs(root) else os.path.join(REPO, root)
        if not os.path.isdir(base):
            continue
        for d, dirs, files in os.walk(base):
            dirs[:] = [x for x in dirs if x not in _CODE_SKIP]
            for name in sorted(files):
                if not name.endswith(".py"):
                    continue
                p = os.path.join(d, name)
                rows.append((os.path.relpath(p, REPO), _sha1_file(p)))
    rows.sort()
    return _sha1(json.dumps(rows).encode())


# --------------------------------------------------------------- fingerprint
def fingerprint(cfg=None):
    """Today's fingerprint: action space, corpus draw, code, engine pin.

    Recomputed at the top of EVERY slice -- this is the component's "dawn".
    The plan then self-corrects to what it finds rather than to a number
    written in a config yesterday.
    """
    cfg = cfg or {}
    fp = {"schema": SCHEMA}
    fp.update(action_space(cfg.get("n_features")))
    fp["corpus"] = corpus_draw(cfg.get("index_path"),
                               cfg.get("seed", DEFAULT_SEED),
                               cfg.get("per_bin", DEFAULT_PER_BIN))
    fp["code"] = code_digest(cfg.get("code_roots", CODE_ROOTS))
    fp["engine"] = cfg.get("engine")
    fp["computed"] = _now()
    return fp


# -------------------------------------------------------------------- stamps
def stamp_path(artifact: str) -> str:
    return artifact.rstrip("/") + ".stamp.json"


def write_stamp(artifact, fp, keys, extra=None):
    """Write `<artifact>.stamp.json`. Returns the stamp.

    A directory artifact (a PPO run dir) gets sha256=None: its contents change
    every iteration and the stamp exists to pin the FINGERPRINT the run began
    under, not its bytes. The driver writes a run-dir stamp BEFORE launching
    the trainer, so a slice killed by the wall still leaves a valid stamp and
    the next slice resumes instead of quarantining an hour of work.
    """
    keys = tuple(keys)
    is_file = os.path.isfile(artifact)
    stamp = {
        "schema": SCHEMA,
        "keys": list(keys),
        "fp": {k: fp.get(k) for k in keys},
        "sha256": _sha256_file(artifact) if is_file else None,
        "bytes": os.path.getsize(artifact) if is_file else None,
        "written": _now(),
    }
    if extra:
        stamp["extra"] = dict(extra)
    p = stamp_path(artifact)
    os.makedirs(os.path.dirname(os.path.abspath(p)) or ".", exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(stamp, fh, indent=1, sort_keys=True)
    os.replace(tmp, p)
    return stamp


def read_stamp(artifact):
    p = stamp_path(artifact)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as fh:
            return json.load(fh)
    except (ValueError, OSError):
        return None


def check(artifact, fp, keys, content=True):
    """-> (fresh, reason). A missing stamp is NOT fresh.

    `content=True` also compares the recorded sha256, which catches an
    artifact rewritten out of band between slices.
    """
    if not os.path.exists(artifact):
        return False, "missing"
    stamp = read_stamp(artifact)
    if stamp is None:
        return False, "unstamped"
    want = dict(stamp.get("fp") or {})
    for k in keys:
        if k not in want:
            return False, f"stamp does not cover {k!r}"
        if want[k] != fp.get(k):
            return False, (f"{k} changed: stamped {_short(want[k])} "
                           f"!= now {_short(fp.get(k))}")
    if content and stamp.get("sha256") and os.path.isfile(artifact):
        if _sha256_file(artifact) != stamp["sha256"]:
            return False, "content changed since stamp"
    return True, "fresh"


def _short(v):
    s = json.dumps(v, sort_keys=True) if not isinstance(v, str) else v
    return s if len(s) <= 44 else s[:20] + ".." + s[-16:]


def quarantine(artifact, run_dir, reason):
    """Move the artifact and its stamp into `<run>/stale/<ts>/`. -> new path.

    Never deleted. Two arms that disagree because one of them ran under an old
    head version is the sort of thing you want to be able to prove afterwards.
    """
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    dest_dir = os.path.join(run_dir, "stale", ts)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(artifact.rstrip("/")))
    if os.path.exists(dest):
        dest = dest + "." + _sha1(artifact.encode())[:6]
    shutil.move(artifact, dest)
    sp = stamp_path(artifact)
    if os.path.exists(sp):
        shutil.move(sp, stamp_path(dest))
    with open(os.path.join(dest_dir, "reason.txt"), "a") as fh:
        fh.write(f"{_now()}  {artifact}  {reason}\n")
    return dest


# ------------------------------------------------------- head-version guard
class HeadMismatch(RuntimeError):
    """A checkpoint's action space is not the one running today."""


def verify_weights_head(weights_json_path, fp):
    """Open a `rl.common.to_layers` JSON and refuse a crossed action space.

    This is the invariant `kagent.nnpolicy` relies on: `choose` walks a cursor
    through the flat logits using `features.HEADS`, so a net trained on a
    different head set is split at the wrong offsets and SafeAgent swallows
    the error into a silent heuristic fallback -- a checkpoint that looks like
    it is playing and is not.
    """
    with open(weights_json_path) as fh:
        data = json.load(fh)
    pol = data.get("policy")
    if not isinstance(pol, dict):
        raise HeadMismatch(f"{weights_json_path}: no 'policy' block")
    hv, hm = pol.get("heads_version"), pol.get("heads_mode")
    if hv != fp.get("heads_version"):
        raise HeadMismatch(
            f"{weights_json_path}: policy heads_version={hv!r} but the live "
            f"action space is {fp.get('heads_version')!r} -- every stored "
            f"choices index means a different target. Refusing.")
    if hm != fp.get("heads_mode"):
        raise HeadMismatch(
            f"{weights_json_path}: policy heads_mode={hm!r} != "
            f"{fp.get('heads_mode')!r}")
    has_board = pol.get("board") is not None
    if has_board != bool(fp.get("board")):
        raise HeadMismatch(
            f"{weights_json_path}: policy {'carries' if has_board else 'has no'} "
            f"board trunk but KAGG_BOARD is "
            f"{'not set' if has_board else 'set'} -- rl/rollout.py would feed "
            f"the wrong input width (Stage B's silent truncation). Refusing.")
    return True
