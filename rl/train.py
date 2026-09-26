"""Train the neural day-planner: behavior cloning from the heuristic (v3),
then PPO self-play against an opponent pool {heuristic, starter, snapshots}.

Design follows the research notes:
- BC first, then conservative on-policy improvement (AlphaStar Unplugged).
- Masked multi-discrete heads, mask-before-softmax (Huang & Ontanon).
- Shaped (portfolio-delta) -> sparse (final money + win) reward curriculum,
  opponent-pool evaluation (RAISocketAI / microRTS).

Usage:
    PYTHONPATH=. uv run python rl/train.py --bc-games 90 --iters 60 \
        --episodes 96 --procs 10 --run rl/runs/r1
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from multiprocessing import Pool  # noqa: E402

from kagent import features  # noqa: E402
from rl import rollout  # noqa: E402
from tools import shaping as _shaping  # noqa: E402
from rl.common import (  # noqa: E402
    PlannerNet,
    ValueNet,
    masked_dists,
    to_layers,
    to_weights_data,
)

GAMMA = 0.997
LAM = 0.95
CLIP = 0.2
LR = 2.5e-4          # hot early (breakout-capable), scheduled down to consolidate
LR_FLOOR = 0.15      # linear decay floor as a fraction of LR
PLATEAU_COOL = 0.6   # extra LR multiplier per eval without improvement
ENT_COEF = 0.02      # initial; decays with the LR schedule (field study: the
                     # policy under-explored portfolio switches per shop draw)
VF_COEF = 0.5
# Profit-first objective: the game scores final bank (only SOLD product
# counts), so money dominates the terminal reward; the win term is a small
# tiebreaker, not the target.
MONEY_COEF = 1.0 / 1000.0
RESIDUAL = False
PAIRED = False
MARGIN_COEF = 3.0 / 1000.0  # within-episode margin: the two banks of one episode
                            # correlate at +0.73, so margin is the cleaner signal
WIN_COEF = 6.0            # ordinary (league/heuristic) games
SESSION_WIN_COEF = 15.0   # session-replay games: beating a real recorded
                          # opponent on its true world IS the objective now
# Both of these were per-step / terminal costs on STATE VARIABLES, which is
# not a potential-based form, so they moved the optimum rather than the
# convergence rate -- and both moved it onto behaviour the ladder has already
# refuted (see tests/test_shaping.py H4b/H4c and docs/shaping_spec.md S3).
# They are kept only to reproduce the old curriculum under --legacy-shaping.
CARRY_COEF = 0.005   # per-decision cost of holding unsold stock (softened: gen-4
UNSOLD_COEF = 0.2    # coefficients handicapped raw profit vs a penalty-free champion)
# `unsold` needs no penalty of its own under PBRS: Phi(terminal) = 0 means the
# final shaping term is exactly -Phi(s_last), so ending the season holding a
# large non-cash portfolio is already charged, at the potential's own scale
# and with no free coefficient to tune.
TRUNK_LR_SCALE = 1.0  # --trunk-lr-scale; 0 is equivalent to --freeze-trunk
KL_STOP = 0.03       # per-iteration early stop on policy drift
WARMUP_ITERS = 3     # value-head-only updates while advantages are garbage

# --reward. "legacy" is the mix above. "win" makes the win THE terminal
# reward and margin a bounded tiebreaker: under the legacy coefficients the
# win term was a few percent of terminal-reward variance (money at 1e-3 x
# $40-100k, margin at 3e-3 x a ~$25k spread, against a +/-6..15 win), so the
# trainer was optimising bank size while the ladder scores wins.
REWARD = "legacy"
MARGIN_TIE = 2.0            # max reward a margin can add, via tanh
MARGIN_SCALE = 20_000.0     # dollars at which the tiebreaker is 76% saturated
# Phi = PHI_SCALE * portfolio_value. Under PBRS the scale cannot move the
# optimum, but Phi should sit near the terminal reward's scale so the critic
# is not learning a potential ten times the size of the signal. 1e-3 puts
# Phi at 30..150 against a +/-15 win; --reward win defaults to 1e-4.
PHI_SCALE = 1e-3
# The worker's logged log pi_old(a|s) and the trainer's recomputation must
# agree before any update -- same weights, same states, same actions. This
# is the tolerance for float32 rounding; anything larger means the rollouts
# were played by a different policy than the one being trained.
ONPOLICY_KL_TOL = 1e-3
# --potential. "pv": Phi = PHI_SCALE x own portfolio value (stationary, a
# LEVEL). "lead": rl/phase_potential.py, the phase reward -- asset lead over
# the opponent early, money lead late, tanh-bounded at the session win
# coefficient. Both are potentials, so neither can move the optimum
# (tests/test_shaping.py H3); "lead" is the dense, opponent-relative signal
# for the regime Stage B3 measured, where the terminal win arrives 1-5 times
# in 64 episodes and a win-only reward carries almost no gradient.
POTENTIAL = "pv"


def _boards(records):
    """(B, C, 10, 10) float32, or None when the board hook is off.

    Recorded as float16 to survive the Pool; cast once here rather than per
    forward pass.
    """
    if not records or records[0].get("board") is None:
        return None
    import numpy as np
    return torch.from_numpy(
        np.stack([r["board"] for r in records]).astype("float32"))


def _tensor_batch(records):
    x = torch.tensor([r["x"] for r in records], dtype=torch.float32)
    masks = torch.tensor(
        [sum(r["masks"], []) for r in records], dtype=torch.float32
    )
    choices = torch.tensor([r["choices"] for r in records], dtype=torch.long)
    return x, masks, choices


def _sample_weights(records):
    """Outcome-weighted imitation: winners count 1-2x scaled by profit
    margin; losers stay as low-weight contrast (state coverage without
    imitating losing plans). Unlabeled samples (teacher games) weigh 1."""
    w = []
    for r in records:
        if "win" not in r:
            w.append(1.0)
        elif r["win"]:
            w.append(1.0 + min(max(r.get("margin", 0), 0), 60000) / 60000.0)
        else:
            w.append(0.25)
    return torch.tensor(w, dtype=torch.float32)


def _bc_params(net):
    """Every policy parameter EXCEPT the board trunk's."""
    trunk_ids = ({id(q) for q in net.board.parameters()}
                 if getattr(net, "board", None) is not None else set())
    return [q for q in net.parameters() if id(q) not in trunk_ids]


def _onpolicy_check(old_lp, lp_now, it, tol=None):
    """mean(log pi_old - log pi) over the batch, which must be ~0 before any
    update. Raises when it is not.

    The worker logs log pi_old(a|s) from the serialised weights; the trainer
    recomputes it from the torch net on the same (s, a). Same weights, same
    states, same actions: the two agree up to float32 rounding, or the
    rollouts were played by a DIFFERENT policy than the one being trained.
    In Stage B this was +3.7 to +4.0 nats on every board arm -- the KL from
    the truncated policy to the full one -- and because KL_STOP is 0.03 the
    early stop fired on the first minibatch of every iteration. No policy
    update ran in any board arm, and nothing said so.
    """
    tol = ONPOLICY_KL_TOL if tol is None else tol
    kl0 = float((old_lp - lp_now).mean())
    if not math.isfinite(kl0) or abs(kl0) > tol:
        raise RuntimeError(
            f"iter {it}: worker and trainer disagree on the policy: mean "
            f"log pi_old - log pi = {kl0:+.4f} (tolerance {tol}). The weights "
            f"the rollouts played are not the net being trained -- a "
            f"serialisation or feature mismatch. Refusing to update.")
    return kl0


def _joint_logprob_entropy(net, x, masks, choices, b=None):
    logits = net(x, b) if b is not None else net(x)
    dists = masked_dists(logits, masks)
    lp = torch.zeros(x.shape[0])
    ent = torch.zeros(x.shape[0])
    for h, d in enumerate(dists):
        lp = lp + d.log_prob(choices[:, h])
        ent = ent + d.entropy()
    return lp, ent


def bc_phase(net, pool, n_games, epochs=40, extra=None, ft_epochs=15):
    print(f"== BC: collecting {n_games} heuristic games (50% speculative teacher) ==")
    jobs = []
    for i in range(n_games):
        opponent = ["heuristic", "starter", "heuristic"][i % 3]
        jobs.append(
            {"weights_path": None, "seed": 5000 + i, "side": i % 2,
             "opponent": opponent, "mode": "bc",
             "style": "spec" if (i % 2 == 0 or RESIDUAL) else "react"}
        )
    records = []
    for ep in pool.imap_unordered(rollout.run_episode, jobs):
        records.extend(ep["records"])
    print(f"   {len(records)} teacher day-decisions")
    if extra:
        records = records + list(extra)
        print(f"   + {len(extra)} champion day-decisions (harvested replays)")

    x, masks, choices = _tensor_batch(records)
    boards = _boards(records)
    weights = _sample_weights(records)
    # The trunk is EXCLUDED from BC. The teacher is the heuristic planner,
    # which never reads the board, so its choices carry nothing the trunk
    # could learn from -- BC on it only rewrites what the trunk already
    # knows. Measured: 40 BC epochs moved the margin-pretrained trunk 0.23
    # relative-L2 from its file in every trainable Stage B arm, and PPO then
    # resumed from THAT. h1's board columns still train here, so the policy
    # learns to read the trunk's output; the trunk itself waits for PPO.
    bc_params = _bc_params(net)
    opt = torch.optim.Adam(bc_params, lr=1e-3)
    n = x.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total, correct, seen = 0.0, 0, 0
        for s in range(0, n, 512):
            idx = perm[s : s + 512]
            lp, _ = _joint_logprob_entropy(net, x[idx], masks[idx], choices[idx],
                                           None if boards is None else boards[idx])
            loss = -(lp * weights[idx]).sum() / weights[idx].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            with torch.no_grad():
                logits = (net(x[idx]) if boards is None
                          else net(x[idx], boards[idx]))
                dists = masked_dists(logits, masks[idx])
                hit = torch.ones(len(idx), dtype=torch.bool)
                for h, d in enumerate(dists):
                    hit &= d.probs.argmax(-1) == choices[idx][:, h]
                correct += hit.sum().item()
                seen += len(idx)
        if epoch % 10 == 9 or epoch == 0:
            print(f"   bc epoch {epoch+1}: nll={total/n:.3f} exact-plan acc={correct/seen:.2%}")

    if extra and ft_epochs > 0:
        # Fine-tune on champion data only (AlphaStar Unplugged: train on the
        # full dataset, then FT on the high-quality subset).
        fx, fm, fc = _tensor_batch(list(extra))
        fb = _boards(list(extra))
        fw = _sample_weights(list(extra))
        fopt = torch.optim.Adam(bc_params, lr=3e-4)
        fn = fx.shape[0]
        for epoch in range(ft_epochs):
            perm = torch.randperm(fn)
            total = 0.0
            for s in range(0, fn, 256):
                idx = perm[s : s + 256]
                lp, _ = _joint_logprob_entropy(net, fx[idx], fm[idx], fc[idx],
                                               None if fb is None else fb[idx])
                loss = -(lp * fw[idx]).sum() / fw[idx].sum()
                fopt.zero_grad()
                loss.backward()
                fopt.step()
                total += loss.item() * len(idx)
            if epoch % 5 == 4 or epoch == 0:
                print(f"   champion-FT epoch {epoch+1}: nll={total/fn:.3f}")


def _potentials(recs):
    """Phi(s_t) per decision, in reward units, under the selected POTENTIAL."""
    if POTENTIAL == "lead":
        from rl.phase_potential import phase_potential
        return [phase_potential(r["lead"]["asset_lead"], r["lead"]["money_lead"],
                                r["lead"]["day"]) for r in recs]
    return [r["pv"] * PHI_SCALE for r in recs]


def bc_session_phase(net, path, epochs=40, batch=1024, val_frac=0.05, patience=5):
    """Behaviour cloning from `rl/session_bc.py` samples: the planner readout
    on TOP PLAYERS' implied standing targets, with the board.

    Every BC run so far cloned our own heuristic, which never reads the
    board, so the board columns of the readout learned nothing worth
    keeping. Here the teacher is the recorded top players, whose choices do
    depend on their boards. The trunk stays frozen in this phase and its
    output is computed ONCE for the whole corpus, so a 700k-sample epoch is
    a readout pass, not a conv pass; PPO trains the trunk afterwards.

    Outcome-weighted imitation as in bc_phase: winners 1-2x by margin,
    losers 0.25. Split by SESSION for validation, via the sidecar
    `<path>_sessions.json` the extractor writes.
    """
    import numpy as np

    boards_path = path.replace(".jsonl", "_boards.npy")
    side_path = path.replace(".jsonl", "_sessions.json")
    n = sum(1 for _ in open(path))
    print(f"== BC from sessions: {n:,} samples in {path} ==")
    x = np.zeros((n, 0), dtype="float32")
    masks = choices = None
    w = np.ones(n, dtype="float32")
    bi = np.zeros(n, dtype="int64")
    with open(path) as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            if i == 0:
                x = np.zeros((n, len(r["x"])), dtype="float32")
                masks = np.zeros((n, sum(len(m) for m in r["masks"])), dtype="float32")
                choices = np.zeros((n, len(r["choices"])), dtype="int64")
            x[i] = r["x"]
            masks[i] = sum(r["masks"], [])
            choices[i] = r["choices"]
            bi[i] = r.get("bi", i)
            win, margin = r.get("w"), r.get("m")
            if win is not None and margin is not None and margin == margin:
                w[i] = (1.0 + min(max(margin, 0.0), 60000.0) / 60000.0) if win >= 0.5 else 0.25
    if x.shape[1] != features.N_FEATURES and features.N_FEATURES is not None:
        raise SystemExit(f"session samples carry {x.shape[1]} features, the net "
                         f"expects {features.N_FEATURES}: different encoder")

    # session id per row, from the sidecar; val split by session
    with open(side_path) as fh:
        used = json.load(fh)["sessions"]
    counts = np.asarray([u["n"] for u in used], int)
    if counts.sum() != n:
        raise SystemExit(f"sidecar lists {counts.sum():,} samples, file has {n:,}")
    sid = np.repeat(np.arange(len(used)), counts)
    rng = np.random.default_rng(0)
    gids = np.arange(len(used)); rng.shuffle(gids)
    nv = max(1, int(len(gids) * val_frac))
    is_val = np.isin(sid, gids[:nv])
    tr_i, va_i = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    print(f"   {len(used):,} sessions; split by session: {len(tr_i):,} train / {len(va_i):,} val")

    # precompute the frozen trunk's output once
    z = None
    if getattr(net, "board", None) is not None:
        B = np.load(boards_path, mmap_mode="r")
        if B.shape[0] < n:
            raise SystemExit(f"{boards_path} has {B.shape[0]:,} boards for {n:,} samples")
        z = np.zeros((n, net.h1.in_features - x.shape[1]), dtype="float32")
        net.board.eval()
        t0 = time.time()
        with torch.no_grad():
            for s in range(0, n, 2048):
                idx = bi[s:s + 2048]
                zb = net.board(torch.from_numpy(np.asarray(B[idx], dtype="float32")))
                z[s:s + 2048] = zb.numpy()
        print(f"   trunk features precomputed for {n:,} boards in {time.time()-t0:.0f}s")

    xt, mt, ct, wt = (torch.from_numpy(a) for a in (x, masks, choices, w))
    zt = torch.from_numpy(z) if z is not None else None
    opt = torch.optim.Adam(_bc_params(net), lr=1e-3)

    def nll_acc(rows):
        tot, hit = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(rows), 4096):
                idx = rows[s:s + 4096]
                logits = net.forward_z(xt[idx], None if zt is None else zt[idx])
                dists = masked_dists(logits, mt[idx])
                lp = sum(d.log_prob(ct[idx][:, h]) for h, d in enumerate(dists))
                tot += float(-lp.sum())
                ok = torch.ones(len(idx), dtype=torch.bool)
                for h, d in enumerate(dists):
                    ok &= d.probs.argmax(-1) == ct[idx][:, h]
                hit += int(ok.sum())
        return tot / max(1, len(rows)), hit / max(1, len(rows))

    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        perm = tr_i[np.random.default_rng(ep).permutation(len(tr_i))]
        tot = 0.0
        for s in range(0, len(perm), batch):
            idx = perm[s:s + batch]
            logits = net.forward_z(xt[idx], None if zt is None else zt[idx])
            dists = masked_dists(logits, mt[idx])
            lp = sum(d.log_prob(ct[idx][:, h]) for h, d in enumerate(dists))
            loss = -(lp * wt[idx]).sum() / wt[idx].sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        v_nll, v_acc = nll_acc(va_i)
        print(f"   epoch {ep+1:>3}: train nll {tot/len(perm):.3f}  val nll {v_nll:.3f}  "
              f"val exact-plan acc {v_acc:.2%}", flush=True)
        if v_nll < best - 1e-3:
            best, bad = v_nll, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print("   early stop"); break
    if best_state is not None:
        net.load_state_dict(best_state)
    return best


def _episode_advantages(ep, shaping_w, legacy=False):
    """Per-decision rewards, then GAE.

    The intermediate term is potential-based (Ng, Harada & Russell 1999):

        F_t = GAMMA * Phi(s_{t+1}) - Phi(s_t),   Phi(terminal) = 0

    which telescopes to -Phi(s_0) over the episode -- a constant per world,
    so it cannot change which policy is optimal, only how fast the critic
    finds it. `tools/shaping.py` carries the derivation and the three ways
    this was previously got wrong; `tests/test_shaping.py` holds them shut.

    `legacy=True` restores the pre-2026-09-21 curriculum verbatim, for
    reproducing old runs. It is not policy-invariant and should not be used
    to train anything new.
    """
    recs = ep["records"]
    n = len(recs)
    if n == 0:
        return [], []
    phis = _potentials(recs)
    shape = (_shaping.legacy_shaping_terms(
                 phis, terminal_potential=ep["final_money"] * PHI_SCALE)
             if legacy else
             _shaping.shaping_terms(phis, GAMMA,
                                    terminal_potential=_shaping.TERMINAL_POTENTIAL))
    rewards = []
    for i in range(n):
        r = shaping_w * shape[i]
        if legacy:
            r -= CARRY_COEF * recs[i].get("stock", 0.0) / 1000.0
        if i == n - 1:
            wc = SESSION_WIN_COEF if ep.get("is_session") else WIN_COEF
            margin = ep["final_money"] - ep.get("opp_money", ep["final_money"])
            if REWARD == "win":
                # WIN-FIRST. The ladder is win-based, so the win is the
                # reward. Margin enters through tanh so it stays inside
                # +/-MARGIN_TIE whatever the world does -- a blowout cannot
                # outvote a win, and a narrow win outranks any loss. Money
                # enters nowhere: an absolute level measures the world, not
                # the player (docs/reward_design.md Rule 1).
                tie = MARGIN_TIE * math.tanh(margin / MARGIN_SCALE)
                if "base_final" in ep:
                    base_margin = ep["base_final"] - ep.get("base_opp", ep["base_final"])
                    r += wc * (ep["win"] - ep["base_win"])
                    r += tie - MARGIN_TIE * math.tanh(base_margin / MARGIN_SCALE)
                else:
                    r += wc * (2 * ep["win"] - 1) + tie
            elif "base_final" in ep:
                # Paired baseline: the zero-offset champion plan played the
                # SAME world (seed, seat, opponent). Rewarding the difference
                # removes the +/-$30k shared-world variance that drowned the
                # per-decision advantage (the two banks of one episode
                # correlate at +0.73; a policy and its prior on one world far
                # more).
                base_margin = ep["base_final"] - ep.get("base_opp", ep["base_final"])
                r += MONEY_COEF * (ep["final_money"] - ep["base_final"])
                r += wc * (ep["win"] - ep["base_win"])
                r += MARGIN_COEF * (margin - base_margin)
            else:
                r += MONEY_COEF * ep["final_money"] + wc * (2 * ep["win"] - 1)
                r += MARGIN_COEF * margin
            if legacy:
                r -= UNSOLD_COEF * ep.get("unsold_value", 0.0) / 1000.0
        rewards.append(r)
    values = [r["value"] for r in recs]
    adv = [0.0] * n
    ret = [0.0] * n
    gae = 0.0
    next_v = 0.0
    for i in reversed(range(n)):
        delta = rewards[i] + GAMMA * next_v - values[i]
        gae = delta + GAMMA * LAM * gae
        adv[i] = gae
        ret[i] = gae + values[i]
        next_v = values[i]
    return adv, ret


# Rating bins shared by the curriculum, the in-loop eval and rl/session_eval.py.
EDGES = [(0, 600), (600, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 99999)]
EVAL_PER_BIN = 175   # rl/session_eval.py's draw; --session-mix eval matches its composition


def _eval_band_weights(idx_path, edges=None, per_bin=EVAL_PER_BIN):
    """Per-bin weights equal to the paired eval's composition.

    The eval draws `per_bin` rated held-out seats a bin, or every seat when a
    bin has fewer (0-600 holds 64, 600-1000 holds 21), so its composition is
    min(count, per_bin) normalised. Training opponents are then drawn from the
    TRAIN sessions with these weights: the held-out set fixes the weights and
    never supplies an opponent.
    """
    from rl import session_eval
    edges = EDGES if edges is None else edges
    if list(session_eval.EDGES) != list(edges):
        raise RuntimeError("rl/session_eval.EDGES and rl/train.EDGES disagree")
    ent = session_eval.held_out_entries(idx_path, seed=0, per_bin=per_bin)
    counts = [sum(1 for x in ent if x[2] == b) for b in range(len(edges))]
    tot = float(sum(counts)) or 1.0
    return [c / tot for c in counts]


def _draw_opponent(rng, roll, entries, bins, unrated, bin_wins, band_w,
                   session_mix, session_frac, opp_bundles, league, snapshots):
    """One episode's opponent, and the learner's seat when it is a session.

    Returns (opp_spec, side_or_None).

    session_mix
      frontier  the legacy curriculum: 70% of session draws from the first
                bin whose rolling win rate is below 0.55, 30% from any rated
                entry. In Stage B2 the frontier never left the 0-600 bin (its
                win rate peaked at 27%), so ~35% of ALL episodes were against
                0-600 tapes and ~35-42% against the heuristic, while the eval
                is 89% opponents rated 1000+. Board arms beat that training
                mix by $13-19k and lost or tied held out.
      eval      draw the BIN with the paired eval's band weights, then an
                entry from it: train on the deployment distribution.
      uniform   equal weight per non-empty bin.

    The non-session remainder keeps the legacy split among bundles / league /
    heuristic / snapshots (40 / 16 / 28 / 16 with fall-throughs), rescaled to
    1 - session_frac, so session_frac=0.5 with 'frontier' IS the old code.
    """
    nb = len(bins)
    if (entries or unrated) and roll < session_frac:
        if not entries:
            path, script_side = rng.choice(unrated)
        elif session_mix in ("eval", "uniform"):
            w = ([band_w[b] if bins[b] else 0.0 for b in range(nb)]
                 if session_mix == "eval" else
                 [1.0 if bins[b] else 0.0 for b in range(nb)])
            if sum(w) <= 0:
                w = [1.0 if bins[b] else 0.0 for b in range(nb)]
            b = rng.choices(range(nb), weights=w)[0]
            path, script_side, _ = rng.choice(bins[b])
        else:
            rates = [w_ / g for w_, g in bin_wins]
            frontier = next((b for b in range(nb) if bins[b] and rates[b] < 0.55), None)
            if frontier is not None and rng.random() < 0.7:
                path, script_side, _ = rng.choice(bins[frontier])
            elif not unrated or rng.random() < 0.8:
                path, script_side, _ = rng.choice(entries)
            else:
                path, script_side = rng.choice(unrated)
        return ("session", path, script_side), 1 - script_side
    rest = max(1e-9, 1.0 - session_frac)
    u = (roll - session_frac) / rest if roll >= session_frac else rng.random()
    if opp_bundles and u < 0.40:
        return ("bundle", rng.choice(opp_bundles)), None
    if league and u < 0.56:
        return ("league", rng.choice(league[-4:])), None
    if u < 0.84 or not snapshots:
        return "heuristic", None
    return ("snapshot", rng.choice(snapshots[-5:])), None


def ppo_phase(net, vnet, pool, run_dir, iters, episodes, procs, eval_every=5, opp_bundles=(),
              prior_net=None, prior_kl=0.0,
              patience=4, legacy_shaping=False, eval_per_bin=3,
              session_mix="frontier", session_frac=0.5, world_seed=None, start_iter=0):
    """`start_iter`: resume a run that already completed that many iterations
    (--resume its last.pt, --start-iter k): the iteration counter, hence the
    worlds, the opponent draw, the eval cadence and the value warmup, carry on
    from k instead of replaying the first worlds and warming up again."""
    snapshots = []
    league = []          # weight paths of every new-best: future opponents
    best_score = -1.0
    evals_since_best = 0
    os.makedirs(run_dir, exist_ok=True)

    # Session-replay pool (primary training signal): real recorded opponents
    # on their true worlds. Held-out fifth is never trained on — it is the
    # honest field-difficulty eval.
    import glob as _glob
    sessions = sorted(_glob.glob("rl/datasets/sessions/session-*.json"))
    train_sessions = [p for p in sessions if int(p.split("-")[-1][:-5]) % 5 != 0]
    held_out = [p for p in sessions if int(p.split("-")[-1][:-5]) % 5 == 0]
    # RATING-BIN curriculum (user design): each rated recorded side is a
    # (session, script_side) entry binned by that opponent's rating at game
    # time. Frontier = first bin with rolling win rate < 0.55; rl-v9 ladder:
    # 0.85 / 0.10 / 0 / 0 / 0 / 0 -> frontier is the 600-1000 bin.
    NB = len(EDGES)
    entries = []          # (path, script_side, bin)
    unrated = []          # (path, winner script_side)
    for p in train_sessions:
        s = json.load(open(p))
        sc = s.get("scores") or [None, None]
        any_rated = False
        for ss in (0, 1):
            r = sc[ss]
            if r is None:
                continue
            any_rated = True
            b = next(i for i, (lo, hi) in enumerate(EDGES) if lo <= r < hi)
            entries.append((p, ss, b))
        if not any_rated:
            unrated.append((p, 0 if s["finals"][0] >= s["finals"][1] else 1))
    bins = [[e for e in entries if e[2] == b] for b in range(NB)]
    bin_wins = [[1, 2] for _ in range(NB)]
    entry_bin = {(p, ss): b for p, ss, b in entries}
    print(f"session pool: {len(train_sessions)} train / {len(held_out)} held-out; "
          f"rating-bins={[len(b) for b in bins]} unrated={len(unrated)}")
    band_w = [1.0 / NB] * NB
    if session_mix == "eval":
        band_w = _eval_band_weights("rl/datasets/session_index.json")
    print(f"opponent mix: sessions {session_frac:.0%}, drawn by '{session_mix}'"
          + (f" with band weights {[round(w, 3) for w in band_w]}"
             if session_mix != "frontier" else "")
          + "; remainder bundles/league/heuristic/snapshots 40/16/28/16")
    # Gates: the heuristic plus every incumbent bundle. Training against ONE
    # static incumbent breeds a brittle exploiter (measured: a v3 < rl-v1 <
    # exploiter < v3 cycle). Selection is PROFIT-first: mean final money
    # across all gates (diverse contested play), win rates reported for info.
    gates = ["heuristic"] + [("bundle", b) for b in opp_bundles]

    # Replication. --seed pins torch and python, which is the minibatch
    # shuffle; the worlds (10_000 + it*episodes + e) and the opponent draw
    # (Random(it)) are fixed per iteration by construction, so two "seeds"
    # share every world and every opponent. With --vary-worlds both are
    # re-seeded from the training seed, so replicates are independent runs.
    world_base = (10_000 if world_seed is None
                  else 10_000 + (world_seed % 997) * 100_000)
    if world_seed is not None:
        print(f"replicate: worlds from {world_base:,}, opponent draw seeded by {world_seed}")

    total_iters = start_iter + iters
    for it in range(start_iter, total_iters):
        t0 = time.time()
        shaping_w = max(0.0, 1.0 - it / (total_iters * 0.5))
        wpath = os.path.join(run_dir, f"w{it}.json")
        with open(wpath, "w") as f:
            json.dump(to_layers(net, vnet), f)

        jobs = []
        rng = random.Random(it if world_seed is None else it * 1000 + world_seed)
        for e in range(episodes):
            roll = rng.random()
            opp, sess_side = _draw_opponent(
                rng, roll, entries, bins, unrated, bin_wins, band_w,
                session_mix, session_frac, opp_bundles, league, snapshots)
            side = sess_side if sess_side is not None else e % 2
            jobs.append(
                {"weights_path": wpath, "temperature": 1.0,
                 "seed": world_base + it * episodes + e, "side": side,
                 "opponent": opp, "mode": "rl"}
            )
        if PAIRED:
            # Twin of every episode: the champion plan (zero offsets) on the
            # identical world; its outcome is the reward baseline.
            twins = [dict(j, weights_path=None, mode="bc", style="spec") for j in jobs]
            outs = pool.map(rollout.run_episode, jobs + twins)  # ordered
            episodes_out = outs[: len(jobs)]
            for ep, tw in zip(episodes_out, outs[len(jobs):]):
                ep["base_final"] = tw["final_money"]
                ep["base_opp"] = tw["opp_money"]
                ep["base_win"] = tw["win"]
            gain = sum(ep["final_money"] - ep["base_final"] for ep in episodes_out) / len(episodes_out)
            print(f"   paired: mean gain vs champion plan on same worlds = {gain:+,.0f}")
        else:
            episodes_out = pool.map(rollout.run_episode, jobs)  # ordered: bin stats zip below
        snapshots.append(wpath)
        for ep, j in zip(episodes_out, jobs):
            o = j["opponent"]
            if isinstance(o, tuple) and o[0] == "session":
                b = entry_bin.get((o[1], o[2]))
                if b is not None:
                    bin_wins[b][0] += ep["win"]
                    bin_wins[b][1] += 1
                    if bin_wins[b][1] > 60:  # rolling window
                        bin_wins[b][0] *= 0.5
                        bin_wins[b][1] *= 0.5

        all_recs, all_adv, all_ret = [], [], []
        for ep in episodes_out:
            adv, ret = _episode_advantages(ep, shaping_w, legacy=legacy_shaping)
            all_recs.extend(ep["records"])
            all_adv.extend(adv)
            all_ret.extend(ret)

        x, masks, choices = _tensor_batch(all_recs)
        boards = _boards(all_recs)
        old_lp = torch.tensor([r["logprob"] for r in all_recs], dtype=torch.float32)
        adv_t = torch.tensor(all_adv, dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-6)
        ret_t = torch.tensor(all_ret, dtype=torch.float32)

        # ON-POLICY IDENTITY. Cheap (one forward over the batch) and the one
        # check that would have caught Stage B from inside the trainer: see
        # _onpolicy_check. Runs during warmup too, so a mismatch is refused
        # at iteration 0 rather than after three iterations of value fitting.
        with torch.no_grad():
            lp_now, _ = _joint_logprob_entropy(net, x, masks, choices, boards)
        onpolicy_kl = _onpolicy_check(old_lp, lp_now, it)

        opt = getattr(ppo_phase, "_opt", None)
        if opt is None or getattr(ppo_phase, "_opt_net", None) is not net:
            # The trunk gets its OWN learning rate. Frozen recovered 45% of
            # what the trainable board arm lost (-10,746 -> -5,922), so PPO
            # was damaging the encoder -- but frozen also gave up the arm's
            # win-vs-gate edge (+0.052 -> +0.004). A small non-zero rate is
            # the middle: the representation keeps adapting to the states the
            # policy visits, slowly enough not to be overwritten by a gradient
            # far noisier than the 132k supervised samples that built it.
            trunk_ps = (list(net.board.parameters())
                        if getattr(net, "board", None) is not None else [])
            trunk_ids = {id(q) for q in trunk_ps}
            rest = [q for q in list(net.parameters()) + list(vnet.parameters())
                    if id(q) not in trunk_ids]
            groups = [{"params": rest, "scale": 1.0}]
            if trunk_ps and TRUNK_LR_SCALE > 0:
                groups.append({"params": trunk_ps, "scale": TRUNK_LR_SCALE})
            opt = torch.optim.Adam(groups, lr=LR)
            ppo_phase._opt = opt
            ppo_phase._opt_net = net

        # LR schedule: linear decay across the run, cooled further on each
        # eval without improvement. Constant LR kept overshooting the profit
        # peak (breakout -> collapse the very next eval, every generation).
        decay = max(LR_FLOOR, 1.0 - it / max(1, iters))
        cool = PLATEAU_COOL ** min(evals_since_best, 4)
        lr_now = LR * decay * cool
        ent_now = ENT_COEF * decay
        for gparam in opt.param_groups:
            gparam["lr"] = lr_now * gparam.get("scale", 1.0)

        n = x.shape[0]
        warmup = it < WARMUP_ITERS
        stop = False
        # Counted and logged. A trainer that can perform zero updates and
        # still print twenty iterations of money means is how Stage B went
        # unnoticed; `policy_updates` and `kl_stop_at` land in telemetry.
        n_updates, kl_stop_at = 0, None
        for ep_i in range(4):
            if stop:
                break
            perm = torch.randperm(n)
            for s in range(0, n, 512):
                idx = perm[s : s + 512]
                values = vnet(x[idx])
                vloss = F.mse_loss(values, ret_t[idx])
                if warmup:
                    # Value warmup: advantages from an untrained critic would
                    # wreck the BC policy (AlphaStar Unplugged: anchor
                    # improvement to a settled value estimate). The value net
                    # is decoupled, so this cannot touch the policy at all.
                    loss = VF_COEF * vloss
                else:
                    lp, ent = _joint_logprob_entropy(
                        net, x[idx], masks[idx], choices[idx],
                        None if boards is None else boards[idx])
                    approx_kl = (old_lp[idx] - lp).mean()
                    if approx_kl > KL_STOP:
                        stop = True
                        kl_stop_at = [ep_i, s // 512]
                        break
                    ratio = torch.exp(lp - old_lp[idx])
                    a = adv_t[idx]
                    pg = -torch.min(
                        ratio * a, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * a
                    ).mean()
                    loss = pg + VF_COEF * vloss - ent_now * ent.mean()
                    if prior_net is not None and prior_kl > 0:
                        # Trust region around the base controller (residual
                        # RL): keep the policy close to the BC prior so
                        # exploration is a few offsets per game, not twenty.
                        with torch.no_grad():
                            p_dists = masked_dists(
                                prior_net(x[idx]) if boards is None
                                else prior_net(x[idx], boards[idx]), masks[idx])
                        c_dists = masked_dists(
                            net(x[idx]) if boards is None
                            else net(x[idx], boards[idx]), masks[idx])
                        kl_prior = sum(
                            torch.distributions.kl_divergence(pd, cd)
                            for pd, cd in zip(p_dists, c_dists)
                        ).mean()
                        loss = loss + prior_kl * kl_prior
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(net.parameters()) + list(vnet.parameters()), 0.5
                )
                opt.step()
                if not warmup:
                    n_updates += 1
        if not warmup and n_updates == 0:
            print(f"   WARNING iter {it}: ZERO policy updates (kl stop at {kl_stop_at})")

        money = [ep["final_money"] for ep in episodes_out]
        gate_wins = [
            ep["win"]
            for ep, j in zip(episodes_out, jobs)
            if j["opponent"] != "starter" and not (
                isinstance(j["opponent"], tuple) and j["opponent"][0] == "snapshot"
            )
        ]
        durs = sorted(ep["duration"] for ep in episodes_out)
        bad = sum(1 for ep in episodes_out if any(s != "DONE" for s in ep["statuses"]))
        # Behavioural telemetry. The shaping defects are predicted to move
        # LIQUIDATION, not money: the missing gamma pays to hold, the carry
        # cost pays to dump. `unsold` is what the engine scores at zero, so
        # it is the direct read-out. Logged per iteration so the two arms can
        # be compared on mechanism rather than on a noisy final rating.
        unsold = [ep.get("unsold_value", 0.0) for ep in episodes_out]
        held = [sum(r.get("stock", 0.0) for r in ep["records"])
                / max(1, len(ep["records"])) for ep in episodes_out]
        telemetry = {
            "iter": it, "shaping_w": shaping_w, "legacy": bool(legacy_shaping),
            "money": sum(money) / len(money),
            "win_vs_gate": sum(gate_wins) / max(1, len(gate_wins)),
            "unsold": sum(unsold) / len(unsold),
            "held_stock": sum(held) / len(held),
            "episodes": len(episodes_out), "steps": n, "bad": bad,
            "onpolicy_kl": onpolicy_kl, "policy_updates": n_updates,
            "kl_stop_at": kl_stop_at, "warmup": warmup, "reward": REWARD,
        }
        with open(os.path.join(run_dir, "telemetry.jsonl"), "a") as _tf:
            _tf.write(json.dumps(telemetry) + "\n")
        # Every iteration, not only at eval: a run killed between evals, or
        # one with evals disabled, otherwise leaves no final weights at all.
        torch.save({"policy": net.state_dict(), "value": vnet.state_dict()},
                   os.path.join(run_dir, "last.pt"))
        print(
            f"iter {it:3}: money mean={sum(money)/len(money):8.0f} "
            f"win-vs-gate={sum(gate_wins)/max(1,len(gate_wins)):.2f} "
            f"unsold={telemetry['unsold']:7.0f} held={telemetry['held_stock']:7.0f} "
            f"shaping={shaping_w:.2f} lr={lr_now:.1e} steps={n} bad={bad} "
            f"upd={n_updates} kl0={onpolicy_kl:+.1e} "
            f"ep_dur p50={durs[len(durs)//2]:.1f}s max={durs[-1]:.1f}s ({time.time()-t0:.0f}s)"
        )

        if (it + 1) % eval_every == 0:
            results = []
            for gate in gates:
                win, mean_money = evaluate(
                    net, pool, run_dir, n_seeds=10, opponent=gate, seed_base=90_000 + it * 100
                )
                results.append((gate, win, mean_money))
            profit = sum(m for _, _, m in results) / len(results)
            mean_win = sum(w for _, w, _ in results) / len(results)
            # Held-out sessions: the honest field-difficulty gate (rl-v9
            # baseline: 8%). Weighted heaviest in selection — this is the
            # primary track.
            # Bin-stratified held-out eval: up to 3 rated opponents per rating
            # bin so low-bin mastery is visible, not drowned by champion games.
            strat, per_bin_n = [], [0] * NB
            for sp in held_out:
                s_ = json.load(open(sp))
                for ss, r in enumerate(s_.get("scores") or [None, None]):
                    if r is None:
                        continue
                    b = next(i for i, (lo, hi) in enumerate(EDGES) if lo <= r < hi)
                    if per_bin_n[b] < eval_per_bin:
                        per_bin_n[b] += 1
                        strat.append((sp, ss))
            sess_jobs = []
            for k, (sp, script_side) in enumerate(strat):
                sess_jobs.append({
                    "weights_path": os.path.join(run_dir, "eval.json"),
                    "temperature": 0.0, "seed": 0, "side": 1 - script_side,
                    "opponent": ("session", sp, script_side), "mode": "rl"})
            sess_out = list(pool.imap_unordered(rollout.run_episode, sess_jobs))
            session_wr = sum(ep["win"] for ep in sess_out) / max(1, len(sess_out))
            session_money = sum(ep["final_money"] for ep in sess_out) / max(1, len(sess_out))
            # The standard error goes next to the number. At the old default
            # of 3 a bin this was 18 games: 0.06 was one win, 0.11 two, and
            # six arms were ranked on that. Selection here is still noisy at
            # any affordable size; the decisive comparison is
            # rl/session_eval.py on the final nets, paired, ~750 games.
            n_sess = max(1, len(sess_out))
            session_se = math.sqrt(session_wr * (1 - session_wr) / n_sess)
            bin_report = "/".join(f"{w/g:.0%}" for w, g in bin_wins)
            profit = 100_000 * session_wr + (profit + 15_000 * mean_win) / 10.0
            min_win = min(w for _, w, _ in results)
            detail = "  ".join(
                f"{g if isinstance(g, str) else os.path.basename(os.path.dirname(g[1]))}"
                f":{w:.2f}/${m/1000:.1f}k"
                for g, w, m in results
            )
            print(f"   EVAL argmax score={profit:8.0f} mean-win={mean_win:.2f} min-win={min_win:.2f} "
                  f"SESSION-WR={session_wr:.2f}+/-{session_se:.2f} (n={n_sess}, ${session_money/1000:.1f}k) "
                  f"train-bins={bin_report}  [{detail}]")
            torch.save({"policy": net.state_dict(), "value": vnet.state_dict()},
                       os.path.join(run_dir, "last.pt"))
            if profit > best_score:
                best_score = profit
                evals_since_best = 0
                league.append(wpath)
                torch.save({"policy": net.state_dict(), "value": vnet.state_dict()},
                           os.path.join(run_dir, "best.pt"))
                if getattr(net, "board", None) is None:
                    with open(os.path.join(run_dir, "best_weights.json"), "w") as f:
                        json.dump(to_weights_data(net), f)
                else:
                    # No b85 export for a board net: the bundle cannot build
                    # a board tensor yet, so the export would be the silent
                    # truncation again. best.pt and eval.json carry it.
                    print("   (board net: b85 export skipped; best.pt + eval.json hold the weights)")
                print(f"   new best (score={profit:.0f}) saved; league size {len(league)}")
            else:
                evals_since_best += 1
                if evals_since_best >= patience:
                    print(
                        f"   converged: no profit improvement in {patience} evals "
                        f"({patience * eval_every} iters); stopping at iter {it}"
                    )
                    return


def evaluate(net, pool, run_dir, n_seeds=10, opponent="heuristic", seed_base=90_000):
    wpath = os.path.join(run_dir, "eval.json")
    with open(wpath, "w") as f:
        json.dump(to_layers(net), f)
    jobs = []
    for i in range(n_seeds):
        for side in (0, 1):
            jobs.append(
                {"weights_path": wpath, "temperature": 0.0, "seed": seed_base + i,
                 "side": side, "opponent": opponent, "mode": "rl"}
            )
    out = list(pool.imap_unordered(rollout.run_episode, jobs))
    wins = sum(ep["win"] for ep in out) / len(out)
    money = sum(ep["final_money"] for ep in out) / len(out)
    return wins, money


def apply_checkpoints(net, vnet, resume=None, trunk_init=None, freeze_trunk=False):
    """Load in the ONE order that cannot lose the trunk: resume first, then
    the trunk file OVER it, then freeze.

    The old order was trunk-init, then resume. `load_state_dict` is strict
    and a BC checkpoint carries `board.*` keys, so the resume silently put
    the BC-trained trunk back over the pretrained one. Measured on Stage B:
    four of five board arms ran PPO on a trunk 0.23 relative-L2 from the
    file the log said they had loaded.

    Returns {"resume_trunk_dist": how far the resume's trunk sat from the
    file (None when either is absent), "trunk_dist": distance after loading,
    0.0 by construction}. Both are printed so the log is the evidence.
    """
    if resume:
        ckpt = torch.load(resume)
        if isinstance(ckpt, dict) and "policy" in ckpt:
            net.load_state_dict(ckpt["policy"])
            vnet.load_state_dict(ckpt["value"])
        else:
            net.load_state_dict(ckpt)
    info = {"resume_trunk_dist": None, "trunk_dist": None}
    if trunk_init:
        if getattr(net, "board", None) is None:
            raise SystemExit("--trunk-init needs --board")
        ck = torch.load(trunk_init)
        arm = ck.get("arm", "?")
        want = os.environ.get("KAGG_BOARD_ARM", "T_crop_103")
        if arm != want:
            raise SystemExit(f"trunk was pretrained on arm {arm} but "
                             f"KAGG_BOARD_ARM is {want} -- the channel sets "
                             f"differ, refusing to load")
        with torch.no_grad():
            cur = net.board.state_dict()
            if resume:
                info["resume_trunk_dist"] = max(
                    float((cur[k] - v).abs().max()) for k, v in ck["trunk"].items())
            net.board.load_state_dict(ck["trunk"])
            cur = net.board.state_dict()
            info["trunk_dist"] = max(
                float((cur[k] - v).abs().max()) for k, v in ck["trunk"].items())
        info["arm"] = arm
        info["val_r2"] = ck.get("val_r2", float("nan"))
    if freeze_trunk:
        if getattr(net, "board", None) is None:
            raise SystemExit("--freeze-trunk needs --board")
        for prm in net.board.parameters():
            prm.requires_grad_(False)
        net.board.eval()
    return info


def main():
    global RESIDUAL, ENT_COEF, LR, PLATEAU_COOL, PAIRED, TRUNK_LR_SCALE
    global REWARD, MARGIN_TIE, MARGIN_SCALE, PHI_SCALE, POTENTIAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-games", type=int, default=90)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--episodes", type=int, default=96)
    ap.add_argument("--procs", type=int, default=max(2, (os.cpu_count() or 8) - 2))
    ap.add_argument("--run", default=None)
    ap.add_argument("--resume", default=None, help="path to a .pt to resume from")
    ap.add_argument("--heads-version", type=int, default=1,
                    help="planner action space: 1 = shipped (13 heads, 57 logits); "
                         "2 adds sell timing; 3 makes every quantity head a fraction of "
                         "a demand anchor and adds the flow observation. Selected BEFORE "
                         "the net is built and before any encode(), and recorded in the "
                         "exported weights so inference restores it.")
    ap.add_argument("--start-iter", type=int, default=0,
                    help="continue an interrupted PPO run: iterations already done (with --resume its last.pt)")
    ap.add_argument(
        "--opp-bundles",
        default=None,
        help="comma-separated frozen main.py bundles used as incumbent opponents + eval gates",
    )
    ap.add_argument(
        "--bc-extra",
        default=None,
        help="JSONL of champion day-decisions from rl/replay_bc.py, mixed into BC + used for FT",
    )
    ap.add_argument("--ent-coef", type=float, default=None, help="entropy bonus (default ENT_COEF)")
    ap.add_argument("--prior-kl", type=float, default=0.0,
                    help="KL penalty toward the frozen BC prior (residual trust region)")
    ap.add_argument("--lr", type=float, default=None, help="policy/value learning rate (default LR)")
    ap.add_argument("--board", action="store_true",
                    help="give PlannerNet a conv trunk over the board tensor "
                         "(needs KAGG_BOARD=1 so rollout records it)")
    ap.add_argument("--trunk-lr-scale", type=float, default=1.0,
                    help="trunk learning rate as a fraction of the policy's. "
                         "1.0 is the Stage B board arm, 0.0 is --freeze-trunk; "
                         "a small value lets the representation keep adapting "
                         "without being overwritten")
    ap.add_argument("--freeze-trunk", action="store_true",
                    help="load the pretrained trunk and do NOT train it. The "
                         "board arm's money gap WIDENED over iterations "
                         "(-5,336 at 3-9 to -14,533 at 10-19), which is what "
                         "a well-fitted encoder being overwritten by a noisy "
                         "policy gradient looks like: the trunk saw 132k "
                         "supervised samples, PPO gives it 38k frames and a "
                         "far noisier signal")
    ap.add_argument("--trunk-init", default=None,
                    help="rl/pretrain_trunk.py checkpoint. An INITIALISATION, "
                         "not a freeze -- PPO keeps training these weights, "
                         "because a trunk frozen on an offline distribution is "
                         "the self-play-potential failure again")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed torch + python so two runs differ ONLY in what "
                         "you changed. Rollouts are already deterministic in "
                         "(weights, world seed, side); this pins net init and "
                         "minibatch shuffling, which are the rest of it")
    ap.add_argument("--legacy-shaping", action="store_true",
                    help="restore the pre-2026-09-21 curriculum: undiscounted "
                         "portfolio delta + per-step carry cost + terminal "
                         "unsold penalty. NOT policy-invariant (see "
                         "docs/shaping_spec.md); for reproducing old runs only")
    ap.add_argument("--paired", action="store_true",
                    help="reward = outcome minus the champion plan's outcome on the identical world (twin episodes)")
    ap.add_argument("--reward", choices=("legacy", "win"), default="legacy",
                    help="terminal reward. legacy: money + margin + win mix "
                         "(the win is a few percent of its variance). win: "
                         "the win IS the reward, margin a tanh-bounded "
                         "tiebreaker of at most --margin-tie, money absent")
    ap.add_argument("--margin-tie", type=float, default=None,
                    help=f"--reward win: max reward a margin can add (default {MARGIN_TIE})")
    ap.add_argument("--margin-scale", type=float, default=None,
                    help=f"--reward win: dollars at 76%% tanh saturation (default {MARGIN_SCALE:.0f})")
    ap.add_argument("--phi-scale", type=float, default=None,
                    help="Phi = phi-scale * portfolio value. Default 1e-3 "
                         "(legacy) or 1e-4 under --reward win, which puts "
                         "Phi on the terminal reward's scale")
    ap.add_argument("--potential", choices=("pv", "lead"), default="pv",
                    help="the PBRS potential. pv: own portfolio value, a "
                         "stationary level. lead: rl/phase_potential.py, the "
                         "phase reward -- asset lead over the opponent early, "
                         "money lead late, bounded at the win coefficient. "
                         "Neither can move the optimum; lead is the dense "
                         "signal for opponents the policy rarely beats")
    ap.add_argument("--eval-every", type=int, default=5, help="iterations between evals")
    ap.add_argument("--eval-per-bin", type=int, default=3,
                    help="held-out session games per rating bin at each eval. "
                         "3 -> 18 games, se ~0.1: telemetry, not a verdict. "
                         "The verdict is rl/session_eval.py on the final nets")
    ap.add_argument("--session-mix", choices=("frontier", "eval", "uniform"),
                    default="frontier",
                    help="how session opponents are drawn across rating bins. "
                         "frontier: legacy, 70%% on the first bin below a 0.55 "
                         "win rate (stuck at 0-600 all of Stage B2). eval: the "
                         "paired eval's band weights -- the deployment "
                         "distribution. uniform: equal per bin")
    ap.add_argument("--session-frac", type=float, default=0.5,
                    help="share of episodes against recorded sessions; the rest "
                         "is bundles/league/heuristic/snapshots in the legacy "
                         "40/16/28/16 split. The eval has no heuristic games")
    ap.add_argument("--bc-session", default=None,
                    help="rl/session_bc.py JSONL: clone the planner readout from "
                         "RECORDED (top-player) choices instead of our heuristic; "
                         "boards from the companion memmap, trunk frozen and "
                         "precomputed in this phase. Replaces --bc-games")
    ap.add_argument("--bc-session-epochs", type=int, default=40)
    ap.add_argument("--vary-worlds", action="store_true",
                    help="also re-seed the world seeds and the opponent draw from "
                         "--seed, so a replicate is an independent training run "
                         "rather than the same worlds with a different minibatch "
                         "order. Off by default: every prior run stays reproducible")
    ap.add_argument("--plateau-cool", type=float, default=None,
                    help="LR multiplier per eval without improvement (1.0 = off; residual argmax evals stay flat by design)")
    ap.add_argument("--prior-pt", default=None, help="checkpoint to use as the frozen prior for --prior-kl (default: the resumed/BC net)")
    ap.add_argument(
        "--residual", action="store_true",
        help="offset heads over the champion plan (champ-h3 lineage); zero = the shipped heuristic",
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=4,
        help="stop after this many evals without a profit improvement",
    )
    args = ap.parse_args()
    if args.heads_version != 1:
        # MUST happen before PlannerNet is sized (rl/common sizes the logit layer
        # from features.N_ACTION_LOGITS) and before any encode(), or the net and the
        # observation disagree and nothing reports it.
        features.set_heads_version(args.heads_version)
        print(f"action space: heads v{args.heads_version} -- "
              f"{len(features.HEADS)} heads, {features.N_ACTION_LOGITS} logits")
    # Pinned threads. torch.manual_seed pins a fit WITHIN a process; thread
    # order broke it ACROSS processes and silently un-paired every ablation
    # (memory: fit determinism). The PPO update is ~2k rows, so one thread
    # costs nothing; the rollouts run in the Pool without torch.
    torch.set_num_threads(int(os.environ.get("KAGG_THREADS", "1")))
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"seeded torch+python with {args.seed}; torch threads "
              f"{torch.get_num_threads()}")
    REWARD = args.reward
    if args.margin_tie is not None:
        MARGIN_TIE = args.margin_tie
    if args.margin_scale is not None:
        MARGIN_SCALE = args.margin_scale
    PHI_SCALE = (args.phi_scale if args.phi_scale is not None
                 else (1e-4 if REWARD == "win" else 1e-3))
    POTENTIAL = args.potential
    print(f"reward={REWARD} win_coef={WIN_COEF}/{SESSION_WIN_COEF}(session) "
          + (f"margin_tie=+/-{MARGIN_TIE} @${MARGIN_SCALE:.0f} money=0 "
             if REWARD == "win" else
             f"money_coef={MONEY_COEF} margin_coef={MARGIN_COEF} ")
          + (f"potential=lead (phase: asset lead -> money lead, |Phi|<=15)"
             if POTENTIAL == "lead" else f"potential=pv phi_scale={PHI_SCALE:g}"))
    TRUNK_LR_SCALE = 0.0 if args.freeze_trunk else args.trunk_lr_scale
    RESIDUAL = bool(args.residual)
    PAIRED = bool(args.paired)
    if args.plateau_cool is not None:
        PLATEAU_COOL = args.plateau_cool
    if args.ent_coef is not None:
        ENT_COEF = args.ent_coef
    if args.lr is not None:
        LR = args.lr
    print(f"ENT_COEF={ENT_COEF} LR={LR} prior_kl={args.prior_kl}")
    if RESIDUAL:
        os.environ["KAGG_RESIDUAL"] = "1"  # inherited by Pool workers
        features.set_mode("residual")
        print("mode: RESIDUAL (offset heads over the champion plan)")

    run_dir = args.run or f"rl/runs/{time.strftime('%m%d_%H%M')}"
    os.makedirs(run_dir, exist_ok=True)

    # Feature size probe: one tiny BC episode.
    probe = rollout.run_episode(
        {"weights_path": None, "seed": 1, "side": 0, "opponent": "starter", "mode": "bc"}
    )
    n_features = len(probe["records"][0]["x"])
    print(f"n_features={n_features} n_action_logits={features.N_ACTION_LOGITS}")

    net = PlannerNet(n_features, board=args.board)
    vnet = ValueNet(n_features)
    if args.freeze_trunk and not args.board:
        raise SystemExit("--freeze-trunk needs --board")
    info = apply_checkpoints(net, vnet, resume=args.resume,
                             trunk_init=args.trunk_init,
                             freeze_trunk=args.freeze_trunk)
    if args.resume:
        print(f"resumed from {args.resume}")
    if args.trunk_init:
        state = "FROZEN" if args.freeze_trunk else "trainable by PPO"
        print(f"trunk loaded from {args.trunk_init} (arm {info['arm']}, "
              f"held-out R2 {info['val_r2']:+.4f}) -> {state}; "
              f"distance from file after loading {info['trunk_dist']:.2e}"
              + (f"; the resume's trunk sat {info['resume_trunk_dist']:.4f} "
                 f"from the file and was OVERRIDDEN" if args.resume else ""))
        if info["trunk_dist"] > 0:
            raise SystemExit("trunk does not match its file after loading")
    if args.board:
        n_tr = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"  trainable policy params: {n_tr:,}")

    bundles = tuple(b for b in (args.opp_bundles or "").split(",") if b)
    extra = None
    if args.bc_extra:
        with open(args.bc_extra) as f:
            extra = [json.loads(line) for line in f if line.strip()]
        print(f"loaded {len(extra)} champion samples from {args.bc_extra}")

    with Pool(args.procs) as pool:
        if args.bc_session and not args.resume:
            bc_session_phase(net, args.bc_session, epochs=args.bc_session_epochs)
            torch.save({"policy": net.state_dict(), "value": vnet.state_dict()},
                       os.path.join(run_dir, "bc.pt"))
        elif args.bc_games > 0 and not args.resume:
            bc_phase(net, pool, args.bc_games, extra=extra)
            torch.save({"policy": net.state_dict(), "value": vnet.state_dict()},
                       os.path.join(run_dir, "bc.pt"))
        prior_net = None
        if args.prior_kl > 0:
            prior_net = copy.deepcopy(net).eval()
            if args.prior_pt:
                pck = torch.load(args.prior_pt)
                prior_net.load_state_dict(pck["policy"] if isinstance(pck, dict) and "policy" in pck else pck)
                print(f"prior for KL: {args.prior_pt}")
            for prm in prior_net.parameters():
                prm.requires_grad_(False)
        ppo_phase(
            net, vnet, pool, run_dir, args.iters, args.episodes, args.procs,
            eval_every=args.eval_every, start_iter=args.start_iter,
            opp_bundles=bundles, patience=args.patience,
            prior_net=prior_net, prior_kl=args.prior_kl,
            legacy_shaping=args.legacy_shaping, eval_per_bin=args.eval_per_bin,
            session_mix=args.session_mix, session_frac=args.session_frac,
            world_seed=(args.seed if args.vary_worlds else None),
        )


if __name__ == "__main__":
    main()
