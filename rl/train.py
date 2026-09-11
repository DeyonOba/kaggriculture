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
MARGIN_COEF = 3.0 / 1000.0  # within-episode margin: the two banks of one episode
                            # correlate at +0.73, so margin is the cleaner signal
WIN_COEF = 6.0            # ordinary (league/heuristic) games
SESSION_WIN_COEF = 15.0   # session-replay games: beating a real recorded
                          # opponent on its true world IS the objective now
CARRY_COEF = 0.005   # per-decision cost of holding unsold stock (softened: gen-4
UNSOLD_COEF = 0.2    # coefficients handicapped raw profit vs a penalty-free champion)
KL_STOP = 0.03       # per-iteration early stop on policy drift
WARMUP_ITERS = 3     # value-head-only updates while advantages are garbage


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


def _joint_logprob_entropy(net, x, masks, choices):
    logits = net(x)
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
    weights = _sample_weights(records)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = x.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total, correct, seen = 0.0, 0, 0
        for s in range(0, n, 512):
            idx = perm[s : s + 512]
            lp, _ = _joint_logprob_entropy(net, x[idx], masks[idx], choices[idx])
            loss = -(lp * weights[idx]).sum() / weights[idx].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            with torch.no_grad():
                logits = net(x[idx])
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
        fw = _sample_weights(list(extra))
        fopt = torch.optim.Adam(net.parameters(), lr=3e-4)
        fn = fx.shape[0]
        for epoch in range(ft_epochs):
            perm = torch.randperm(fn)
            total = 0.0
            for s in range(0, fn, 256):
                idx = perm[s : s + 256]
                lp, _ = _joint_logprob_entropy(net, fx[idx], fm[idx], fc[idx])
                loss = -(lp * fw[idx]).sum() / fw[idx].sum()
                fopt.zero_grad()
                loss.backward()
                fopt.step()
                total += loss.item() * len(idx)
            if epoch % 5 == 4 or epoch == 0:
                print(f"   champion-FT epoch {epoch+1}: nll={total/fn:.3f}")


def _episode_advantages(ep, shaping_w):
    recs = ep["records"]
    n = len(recs)
    if n == 0:
        return [], []
    pvs = [r["pv"] for r in recs] + [ep["final_money"]]
    rewards = []
    for i in range(n):
        r = shaping_w * (pvs[i + 1] - pvs[i]) / 1000.0
        r -= CARRY_COEF * recs[i].get("stock", 0.0) / 1000.0
        if i == n - 1:
            wc = SESSION_WIN_COEF if ep.get("is_session") else WIN_COEF
            r += MONEY_COEF * ep["final_money"] + wc * (2 * ep["win"] - 1)
            r += MARGIN_COEF * (ep["final_money"] - ep.get("opp_money", ep["final_money"]))
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


def ppo_phase(net, vnet, pool, run_dir, iters, episodes, procs, eval_every=5, opp_bundles=(),
              prior_net=None, prior_kl=0.0,
              patience=4):
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
    EDGES = [(0, 600), (600, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 99999)]
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
    # Gates: the heuristic plus every incumbent bundle. Training against ONE
    # static incumbent breeds a brittle exploiter (measured: a v3 < rl-v1 <
    # exploiter < v3 cycle). Selection is PROFIT-first: mean final money
    # across all gates (diverse contested play), win rates reported for info.
    gates = ["heuristic"] + [("bundle", b) for b in opp_bundles]

    for it in range(iters):
        t0 = time.time()
        shaping_w = max(0.0, 1.0 - it / (iters * 0.5))
        wpath = os.path.join(run_dir, f"w{it}.json")
        with open(wpath, "w") as f:
            json.dump(to_layers(net, vnet), f)

        jobs = []
        rng = random.Random(it)
        for e in range(episodes):
            roll = rng.random()
            side = e % 2
            if train_sessions and roll < 0.50:
                # Curriculum: 70% mass on the frontier rating bin, 30% anywhere.
                rates = [w / g for w, g in bin_wins]
                frontier = next((b for b in range(NB) if bins[b] and rates[b] < 0.55), None)
                if frontier is not None and rng.random() < 0.7:
                    path, script_side, _ = rng.choice(bins[frontier])
                elif entries and (not unrated or rng.random() < 0.8):
                    path, script_side, _ = rng.choice(entries)
                else:
                    path, script_side = rng.choice(unrated)
                opp = ("session", path, script_side)
                side = 1 - script_side
            elif opp_bundles and roll < 0.70:
                opp = ("bundle", rng.choice(opp_bundles))
            elif league and roll < 0.78:
                opp = ("league", rng.choice(league[-4:]))
            elif roll < 0.92 or not snapshots:
                opp = "heuristic"
            else:
                opp = ("snapshot", rng.choice(snapshots[-5:]))
            jobs.append(
                {"weights_path": wpath, "temperature": 1.0,
                 "seed": 10_000 + it * episodes + e, "side": side,
                 "opponent": opp, "mode": "rl"}
            )
        episodes_out = list(pool.imap_unordered(rollout.run_episode, jobs))
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
            adv, ret = _episode_advantages(ep, shaping_w)
            all_recs.extend(ep["records"])
            all_adv.extend(adv)
            all_ret.extend(ret)

        x, masks, choices = _tensor_batch(all_recs)
        old_lp = torch.tensor([r["logprob"] for r in all_recs], dtype=torch.float32)
        adv_t = torch.tensor(all_adv, dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-6)
        ret_t = torch.tensor(all_ret, dtype=torch.float32)

        opt = getattr(ppo_phase, "_opt", None)
        if opt is None or getattr(ppo_phase, "_opt_net", None) is not net:
            opt = torch.optim.Adam(
                list(net.parameters()) + list(vnet.parameters()), lr=LR
            )
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
            gparam["lr"] = lr_now

        n = x.shape[0]
        warmup = it < WARMUP_ITERS
        stop = False
        for _ in range(4):
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
                    lp, ent = _joint_logprob_entropy(net, x[idx], masks[idx], choices[idx])
                    approx_kl = (old_lp[idx] - lp).mean()
                    if approx_kl > KL_STOP:
                        stop = True
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
                            p_dists = masked_dists(prior_net(x[idx]), masks[idx])
                        c_dists = masked_dists(net(x[idx]), masks[idx])
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
        print(
            f"iter {it:3}: money mean={sum(money)/len(money):8.0f} "
            f"win-vs-gate={sum(gate_wins)/max(1,len(gate_wins)):.2f} "
            f"shaping={shaping_w:.2f} lr={lr_now:.1e} steps={n} bad={bad} "
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
                    if per_bin_n[b] < 3:
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
            bin_report = "/".join(f"{w/g:.0%}" for w, g in bin_wins)
            profit = 100_000 * session_wr + (profit + 15_000 * mean_win) / 10.0
            min_win = min(w for _, w, _ in results)
            detail = "  ".join(
                f"{g if isinstance(g, str) else os.path.basename(os.path.dirname(g[1]))}"
                f":{w:.2f}/${m/1000:.1f}k"
                for g, w, m in results
            )
            print(f"   EVAL argmax score={profit:8.0f} mean-win={mean_win:.2f} min-win={min_win:.2f} "
                  f"SESSION-WR={session_wr:.2f} (${session_money/1000:.1f}k) train-bins={bin_report}  [{detail}]")
            torch.save({"policy": net.state_dict(), "value": vnet.state_dict()},
                       os.path.join(run_dir, "last.pt"))
            if profit > best_score:
                best_score = profit
                evals_since_best = 0
                league.append(wpath)
                torch.save({"policy": net.state_dict(), "value": vnet.state_dict()},
                           os.path.join(run_dir, "best.pt"))
                with open(os.path.join(run_dir, "best_weights.json"), "w") as f:
                    json.dump(to_weights_data(net), f)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-games", type=int, default=90)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--episodes", type=int, default=96)
    ap.add_argument("--procs", type=int, default=max(2, (os.cpu_count() or 8) - 2))
    ap.add_argument("--run", default=None)
    ap.add_argument("--resume", default=None, help="path to a .pt to resume from")
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
    global RESIDUAL, ENT_COEF, LR
    RESIDUAL = bool(args.residual)
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

    net = PlannerNet(n_features)
    vnet = ValueNet(n_features)
    if args.resume:
        ckpt = torch.load(args.resume)
        if isinstance(ckpt, dict) and "policy" in ckpt:
            net.load_state_dict(ckpt["policy"])
            vnet.load_state_dict(ckpt["value"])
        else:
            net.load_state_dict(ckpt)
        print(f"resumed from {args.resume}")

    bundles = tuple(b for b in (args.opp_bundles or "").split(",") if b)
    extra = None
    if args.bc_extra:
        with open(args.bc_extra) as f:
            extra = [json.loads(line) for line in f if line.strip()]
        print(f"loaded {len(extra)} champion samples from {args.bc_extra}")

    with Pool(args.procs) as pool:
        if args.bc_games > 0 and not args.resume:
            bc_phase(net, pool, args.bc_games, extra=extra)
            torch.save({"policy": net.state_dict(), "value": vnet.state_dict()},
                       os.path.join(run_dir, "bc.pt"))
        prior_net = None
        if args.prior_kl > 0:
            prior_net = copy.deepcopy(net).eval()
            for prm in prior_net.parameters():
                prm.requires_grad_(False)
        ppo_phase(
            net, vnet, pool, run_dir, args.iters, args.episodes, args.procs,
            opp_bundles=bundles, patience=args.patience,
            prior_net=prior_net, prior_kl=args.prior_kl,
        )


if __name__ == "__main__":
    main()
