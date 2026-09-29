"""The ONLY learned part of the rlv3 executor. Optional, gated, OFF.

TWO USES, both switchable and both expected by the record to fail:

`pl_net` reads ExecNet2's OP head across the candidate tiles and returns a
placement PRIOR -- a ranking over tiles, never a destination and never a
move. rl/execnet/train2.py trains the op head as

    op_logits.gather(2, tgt.view(-1, 1, 1).expand(-1, N_REQ, 1))

i.e. AT THE LABEL TILE ONLY, so every off-label cell is UNSUPERVISED.

MEASURED ON THIS REPO'S CHECKPOINT, BEFORE ANY ARM WAS RUN -- and the
answer is not the one the unsupervised-head argument predicts:

  off-label agreement with the request class   0.71   (uniform 0.125)
  agreement at the trained label tile          0.69
  prior dispersion across candidate tiles      cv 0.018-0.019, 25 tiles

The first number looks like skill and is not. Its target is the request
class, and the request map is an INPUT channel (features2 planes 24..31),
so the trunk can copy it straight through; 0.71 says the head echoes its
input. The third number is the one that decides, and it says the prior is
FLAT: every candidate carries a PLANT request at the same priority, the
echo gives them all the same value, and max/min comes to about 1.06 against
`pl_wheat_adj`'s weight of 3.0. There is no ranking information in it.

VERDICT: pl_net is INERT, not wrong. It ships off, its arm exists only so
the measurement is reproducible, and nobody should spend 995 games on it.
Run `prior_dispersion()` again if the checkpoint is ever retrained with the
op head supervised off-label; until then this is settled.

`mv_dest` is the v2 per-destination take-over, unchanged, as a CONTROL arm.
It is expected to lose: v1 per-op scored −10.3 / −8.0 / −21.5 points at
conf 0 / 0.5 / 0.7 and v2 per-destination −19.6 / −13.7 / −8.0 at conf
0.4 / 0.4 / 0.6. The series is monotone in "override less" while the model
is a GOOD fit (0.776 top-player agreement against our 0.355) that
reproduces the leaders' headline executor statistics exactly. That is the
observational/interventional law in its sharpest form, and the reason this
module ships defaulted off and exists mainly so the seam has a measured
negative control beside its positive claims.
"""

from __future__ import annotations

import os

DEFAULT_PT = "rl/datasets/execnet2_cond/execnet2_cond.pt"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ExecNet2 request classes read as a placement prior: 4 = PLANT, 7 = BUILD /
# PLACE (rl.execnet.features2.REQ_OF_OP).
PLANT_CLASS = 4
BUILD_CLASS = 7

_CACHE = {}


def _abs(path):
    return path if os.path.isabs(path) else os.path.join(REPO, path)


def load(path=DEFAULT_PT):
    """-> (model, cond). Wraps rl.execnet.policy.load_model, cached here too
    so a missing checkpoint is a clean error rather than an import failure
    inside a scheduler call."""
    p = _abs(path)
    if p not in _CACHE:
        if not os.path.exists(p):
            raise FileNotFoundError(f"rlv3.placenet: no checkpoint at {p}")
        from rl.execnet.policy import load_model
        _CACHE[p] = load_model(p)
    return _CACHE[p]


# ------------------------------------------------------------ pl_net prior
def placement_prior(view, plan, tasks, model, cond, conf):
    """-> {(x, y): weight} over the tiles with a pending growth request.

    The op head's PLANT and BUILD classes are softmaxed over the request
    classes at each candidate tile and renormalised over the candidates.
    Returns {} -- the scheduler's own order stands -- whenever the top
    probability is below `conf`.
    """
    import numpy as np
    import torch
    from rl.execnet import features as F
    from rl.execnet import features2 as F2

    cand = []
    for t in tasks:
        if t.op[0] in ("PLANT", "BUILD_COOP", "BUILD_PASTURE") and (t.x, t.y) not in cand:
            cand.append((t.x, t.y))
    if not cand or not view.me.units:
        return {}

    board = F2.board_planes(view)
    req, _ = F2.request_planes(tasks)
    base = np.asarray(board + req, dtype=np.float32)
    _, ux, uy = view.me.units[0]
    up = np.zeros((1, F2.N_TILES), dtype=np.float32)
    up[0, F2.tile_index(ux, uy)] = 1.0
    planes = np.concatenate([base, up], axis=0).reshape(1, F2.N_PLANE, F2.BOARD, F2.BOARD)
    scal = list(F.unit_features(view, 0, F.farm_counts(view))[1])
    if cond is not None:
        scal = scal + list(cond)
    with torch.no_grad():
        _, op_logits, _ = model(torch.tensor(planes),
                             torch.tensor(np.asarray([scal], dtype=np.float32)))
        probs = torch.softmax(op_logits[0], dim=0).numpy()   # [N_REQ, N_TILES]

    raw = {}
    for (x, y) in cand:
        i = F2.tile_index(x, y)
        raw[(x, y)] = float(probs[PLANT_CLASS][i] + probs[BUILD_CLASS][i])
    top = max(raw.values()) if raw else 0.0
    if top < conf:
        return {}
    total = sum(raw.values()) or 1.0
    return {k: v / total for k, v in raw.items()}


# ----------------------------------------------------- mv_dest control arm
def override_destinations(view, plan, tasks, actions, labels, model, cond, conf):
    """The v2 per-destination take-over verbatim, in place on `actions`.

    `ExecNetPolicy._override_dest` already skips any unit carrying an animal
    and any unit whose assigning pass is in `KEEP_PASSES`, which is the
    contract this seam needs.
    """
    pol = _policy_for(model, cond, conf)
    pol._override_dest(view, plan, tasks, actions, labels)
    return pol.stats


_BY_MODEL = {}


def _policy_for(model, cond, conf):
    key = (id(model), round(float(conf), 6))
    if key not in _BY_MODEL:
        from rl.execnet.policy import ExecNetPolicy
        _BY_MODEL[key] = ExecNetPolicy(model, None, conf=conf, mode="dest", cond=cond)
    return _BY_MODEL[key]


# ------------------------------------------------- the pre-arm diagnostic
def op_head_offlabel_check(npz_paths, path=DEFAULT_PT, n=200, seed=0):
    """Does the op head say anything AT NON-LABEL TILES?

    For `n` held-out rows: take the model's op-class distribution at a tile
    that is NOT the label tile but DOES carry a pending request of a known
    class, and ask whether argmax matches that request's class more often
    than a uniform prior over the 8 classes (1/8 = 0.125). At the LABEL tile
    the same number is reported as the trained-cell control.

    -> {"n", "offlabel_acc", "label_acc", "uniform", "verdict"}
    """
    import random

    import numpy as np
    import torch
    from rl.execnet import features2 as F2

    model, cond = load(path)
    rng = random.Random(seed)
    off_hit = off_tot = lab_hit = lab_tot = 0
    for p in npz_paths:
        if off_tot >= n:
            break
        d = np.load(_abs(p))
        planes, scal = d["planes"], d["scal"]
        sidx, dest, opc, ours = d["sidx"], d["dest"], d["opc"], d["ours"]
        rows = [i for i in range(len(dest))
                if ours[i] and 0 <= dest[i] < F2.N_TILES and opc[i] >= 0]
        rng.shuffle(rows)
        for i in rows:
            if off_tot >= n:
                break
            s = int(sidx[i])
            pl = planes[s].astype(np.float32) / 255.0
            up = np.zeros((1, F2.N_TILES), dtype=np.float32)
            up[0, int(dest[i])] = 0.0            # unit plane rebuilt below
            ux, uy = int(d["ux"][i]), int(d["uy"][i])
            up[0, F2.tile_index(ux, uy)] = 1.0
            x = np.concatenate([pl, up], axis=0).reshape(1, F2.N_PLANE, F2.BOARD, F2.BOARD)
            sv = list(scal[i].astype(np.float32))
            if cond is not None:
                sv = sv + list(cond)
            with torch.no_grad():
                _, op_logits, _ = model(torch.tensor(x),
                                     torch.tensor(np.asarray([sv], dtype=np.float32)))
            pr = op_logits[0].numpy()            # [N_REQ, N_TILES]
            # request planes are channels F.TILE_C .. F.TILE_C + N_REQ
            from rl.execnet import features as F
            req = pl[F.TILE_C:F.TILE_C + F2.N_REQ]
            label = int(dest[i])
            lab_tot += 1
            if int(pr[:, label].argmax()) == int(opc[i]):
                lab_hit += 1
            others = [t for t in range(F2.N_TILES) if t != label and req[:, t].max() > 0]
            if not others:
                continue
            t = rng.choice(others)
            want = int(req[:, t].argmax())
            off_tot += 1
            if int(pr[:, t].argmax()) == want:
                off_hit += 1
    uniform = 1.0 / F2.N_REQ
    acc = (off_hit / off_tot) if off_tot else None
    return {
        "n": off_tot,
        "offlabel_echo_acc": acc,
        "label_acc": (lab_hit / lab_tot) if lab_tot else None,
        "uniform": uniform,
        "note": "THIS IS AN ECHO MEASURE, NOT A SKILL MEASURE. The target is "
                "the request-plane class, and the request planes are INPUT "
                "channels (features2 planes 24..31), so a conv trunk can copy "
                "them straight through. A high number here says the head "
                "echoes its input. The decisive test is prior_dispersion().",
    }


def prior_dispersion(views_plans_tasks, path=DEFAULT_PT, conf=0.0):
    """THE decisive check for pl_net: does the prior VARY across candidates?

    Every candidate tile carries a PLANT request at the same priority, so if
    the op head echoes the request planes the prior is identical on all of
    them -- flat, and worth nothing as a ranking. Measured on this repo's
    checkpoint the coefficient of variation is 0.018-0.019 across 25
    candidate tiles (max/min about 1.06), against `pl_wheat_adj`'s weight of
    3.0 on a 0..1 scale. That is not a tie-break, it is noise.

    -> {"n_cases", "mean_cv", "max_cv", "verdict"}
    """
    import statistics as st

    model, cond = load(path)
    cvs = []
    for view, plan, tasks in views_plans_tasks:
        pr = placement_prior(view, plan, tasks, model, cond, conf)
        if len(pr) < 2:
            continue
        vals = list(pr.values())
        flat = 1.0 / len(vals)
        cvs.append(st.pstdev(vals) / flat)
    if not cvs:
        return {"n_cases": 0, "mean_cv": None, "max_cv": None,
                "verdict": "no candidate sets with >= 2 tiles"}
    mean_cv, max_cv = st.mean(cvs), max(cvs)
    return {
        "n_cases": len(cvs), "mean_cv": mean_cv, "max_cv": max_cv,
        "verdict": ("USABLE" if mean_cv > 0.25 else
                    "INERT -- the prior is flat across candidates; the op head "
                    "is echoing the request planes, not ranking tiles. Do not "
                    "spend an evaluation on pl_net."),
    }
