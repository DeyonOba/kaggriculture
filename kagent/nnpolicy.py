"""Neural day-planner: masked multi-discrete MLP inference in pure stdlib.

The network only replaces the STRATEGIC fields of DayPlan (targets, hires,
land, sell floors) once per day; the scripted scheduler and market controller
execute the micro. Inference is ~30 forward passes per episode, so cost is
irrelevant next to the 1s/step budget.

Arch v2 (current): 2 blocks of Linear -> LayerNorm -> GELU(exact), then a
logits layer. Arch v1 (legacy bundles): 2 tanh layers + logits + value.
Weights ship as base85 float32 in kagent/weights.py (policy only); training
workers additionally receive a small value net for GAE.
"""

import base64
import math
import struct

from kagent import features, planner

_LN_EPS = 1e-5


def decode_weights(data):
    """kagent/weights.py payload -> forward-ready structure (policy only)."""
    if not data:
        return None
    raw = base64.b85decode(data["b85"].encode("ascii"))
    flat = list(struct.unpack("<%df" % (len(raw) // 4), raw))
    sizes = data["sizes"]
    pos = 0

    def take(n):
        nonlocal pos
        out = flat[pos : pos + n]
        pos += n
        return out

    def matrix(n_in, n_out):
        return [take(n_out) for _ in range(n_in)]

    if data.get("arch") == 2:
        blocks = []
        for n_in, n_out in sizes[:-1]:
            blocks.append([matrix(n_in, n_out), take(n_out), take(n_out), take(n_out)])
        n_in, n_out = sizes[-1]
        out = [matrix(n_in, n_out), take(n_out)]
        return {"policy": {"arch": 2, "blocks": blocks, "out": out}, "value": None}

    # Legacy v1: [(W, b)] x 4 (h1, h2, logits, value), tanh activations.
    layers = []
    for n_in, n_out in sizes:
        layers.append((matrix(n_in, n_out), take(n_out)))
    return {"policy": {"arch": 1, "layers": layers}, "value": None}


def _matvec(w, b, x):
    out = list(b)
    for i, xi in enumerate(x):
        if xi == 0.0:
            continue
        row = w[i]
        for j in range(len(out)):
            out[j] += xi * row[j]
    return out


def _layernorm(x, gamma, beta):
    n = len(x)
    mean = sum(x) / n
    var = sum((v - mean) ** 2 for v in x) / n
    inv = 1.0 / math.sqrt(var + _LN_EPS)
    return [(x[i] - mean) * inv * gamma[i] + beta[i] for i in range(n)]


def _gelu(x):
    return [0.5 * v * (1.0 + math.erf(v / 1.4142135623730951)) for v in x]


def forward_policy(policy, x):
    """Policy structure -> flat action logits."""
    if policy.get("arch") == 2:
        h = x
        for w, b, g, beta in policy["blocks"]:
            h = _gelu(_layernorm(_matvec(w, b, h), g, beta))
        w, b = policy["out"]
        return _matvec(w, b, h)
    layers = policy["layers"]
    h = [math.tanh(v) for v in _matvec(layers[0][0], layers[0][1], x)]
    h = [math.tanh(v) for v in _matvec(layers[1][0], layers[1][1], h)]
    return _matvec(layers[2][0], layers[2][1], h)


def forward_board(bl, t):
    """(C,10,10) -> (48,) matching rl.common.BoardTrunk exactly.

    numpy, not pure python: the bundle already ships numpy (the S4 net
    variant inlines a numpy forward at 0.3 ms) and a 103x32 3x3 convolution
    in interpreted loops would not fit the 60 s season budget.
    """
    import numpy as np

    def conv3(x, w, b):
        # x (Ci,H,W), w (Co,Ci,3,3) -> (Co,H,W), zero padding 1
        ci, h, wd = x.shape
        xp = np.zeros((ci, h + 2, wd + 2), dtype=np.float32)
        xp[:, 1:h + 1, 1:wd + 1] = x
        co = w.shape[0]
        out = np.empty((co, h, wd), dtype=np.float32)
        cols = np.stack([xp[:, i:i + h, j:j + wd].reshape(ci, -1)
                         for i in range(3) for j in range(3)])      # (9,Ci,HW)
        cols = cols.transpose(1, 0, 2).reshape(ci * 9, -1)          # (Ci*9,HW)
        wm = w.transpose(1, 2, 3, 0).reshape(ci * 9, co)            # (Ci*9,Co)
        out = (cols.T @ wm).T.reshape(co, h, wd) + b[:, None, None]
        return out

    def gelu(x):
        # exact erf form, as nn.GELU() defaults to
        from math import sqrt
        return 0.5 * x * (1.0 + np.vectorize(_erf)(x / sqrt(2.0)))

    x = np.asarray(t, dtype=np.float32)
    h = gelu(conv3(x, np.asarray(bl["c1w"], np.float32),
                   np.asarray(bl["c1b"], np.float32)))
    h = gelu(conv3(h, np.asarray(bl["c2w"], np.float32),
                   np.asarray(bl["c2b"], np.float32)))
    g = np.concatenate([h.mean((1, 2)), h.max((1, 2))])
    mu, var = g.mean(), g.var()
    g = (g - mu) / np.sqrt(var + 1e-5)
    g = g * np.asarray(bl["lnw"], np.float32) + np.asarray(bl["lnb"], np.float32)
    return gelu(np.asarray(bl["fcw"], np.float32) @ g
                + np.asarray(bl["fcb"], np.float32)).tolist()


def _erf(v):
    return math.erf(v)


def forward_value(value_layers, x):
    """Worker-side value estimate ([(W,b),...] GELU MLP); 0.0 if absent."""
    if not value_layers:
        return 0.0
    h = x
    for w, b in value_layers[:-1]:
        h = _gelu(_matvec(w, b, h))
    w, b = value_layers[-1]
    return _matvec(w, b, h)[0]


def choose(logits, head_masks, rng=None, temperature=1.0):
    """Split flat logits by head, mask, and pick (argmax, or sample if rng).

    Masked-before-softmax keeps the policy gradient unbiased
    (Huang & Ontanon 2020).
    """
    choices = []
    logprob = 0.0
    pos = 0
    t = temperature if (rng is not None and temperature > 0) else 1.0
    for (name, vals), mask in zip(features.HEADS, head_masks):
        n = len(vals)
        seg = logits[pos : pos + n]
        pos += n
        masked = [(seg[i] / t if mask[i] else -1e9) for i in range(n)]
        mx = max(masked)
        exps = [math.exp(v - mx) for v in masked]
        total = sum(exps)
        probs = [e / total for e in exps]
        if rng is None:
            idx = max(range(n), key=lambda i: probs[i])
        else:
            r = rng.random()
            acc = 0.0
            idx = n - 1
            for i, p in enumerate(probs):
                acc += p
                if r <= acc:
                    idx = i
                    break
        choices.append(idx)
        logprob += math.log(max(probs[idx], 1e-12))
    return choices, logprob


class NeuralPolicy:
    """Day-plan provider: heuristic base plan, strategic fields overridden by
    the network. Plugs into HeuristicPolicy via its plan_provider seam.
    `layers` is {"policy": ..., "value": ... or None} (decode_weights or
    rl/common.to_layers output). Falls back to the pure heuristic plan when
    no weights are loaded."""

    def __init__(self, layers, temperature=1.0, rng=None, residual=False):
        self.layers = layers
        self.temperature = temperature
        self.rng = rng
        # residual: champion base plan + offset heads (features.RESIDUAL_HEADS)
        self.residual = residual
        self.last_decision = None  # (features, masks, choices, logprob, value)

    def make_plan(self, view, memory):
        plan = planner.make_day_plan(view, memory, speculative=self.residual)
        if self.layers is None:
            return plan
        x = features.encode(view, memory)
        head_masks = features.masks(view)
        bl = self.layers["policy"].get("board")
        board = getattr(self, "board", None)
        xp = x
        if bl is not None:
            if board is None:
                raise ValueError(
                    "policy weights carry a board trunk but no board tensor "
                    "was supplied -- set provider.board before act()")
            # ONLY the policy sees the trunk output. ValueNet has no board and
            # takes the bare 168; feeding it the concatenated 216 walked off
            # the end of its first weight matrix, and act_safe swallowed the
            # IndexError into a silent heuristic fallback -- zero records,
            # status DONE.
            xp = list(x) + forward_board(bl, board)
        logits = forward_policy(self.layers["policy"], xp)
        value = forward_value(self.layers.get("value"), x)
        choices, logprob = choose(
            logits, head_masks, rng=self.rng, temperature=self.temperature
        )
        self.last_decision = (x, head_masks, choices, logprob, value)
        if self.residual:
            return features.apply_residual(plan, choices, view)
        return features.apply_to_plan(plan, choices, view)
