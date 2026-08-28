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
from rl.common import PlannerNet, masked_dists, to_layers, to_weights_data  # noqa: E402

GAMMA = 0.997
LAM = 0.95
CLIP = 0.2
LR = 1.5e-4          # conservative: protect the BC initialization
ENT_COEF = 0.01
VF_COEF = 0.5
# Profit-first objective: the game scores final bank (only SOLD product
# counts), so money dominates the terminal reward; the win term is a small
# tiebreaker, not the target.
MONEY_COEF = 1.0 / 1000.0
WIN_COEF = 2.0
KL_STOP = 0.03       # per-iteration early stop on policy drift
WARMUP_ITERS = 3     # value-head-only updates while advantages are garbage


def _tensor_batch(records):
    x = torch.tensor([r["x"] for r in records], dtype=torch.float32)
    masks = torch.tensor(
        [sum(r["masks"], []) for r in records], dtype=torch.float32
    )
    choices = torch.tensor([r["choices"] for r in records], dtype=torch.long)
    return x, masks, choices


def _joint_logprob_entropy(net, x, masks, choices):
    logits, values = net(x)
    dists = masked_dists(logits, masks)
    lp = torch.zeros(x.shape[0])
    ent = torch.zeros(x.shape[0])
    for h, d in enumerate(dists):
        lp = lp + d.log_prob(choices[:, h])
        ent = ent + d.entropy()
    return lp, ent, values


def bc_phase(net, pool, n_games, epochs=40):
    print(f"== BC: collecting {n_games} heuristic games ==")
    jobs = []
    for i in range(n_games):
        opponent = ["heuristic", "starter", "heuristic"][i % 3]
        jobs.append(
            {"weights_path": None, "seed": 5000 + i, "side": i % 2,
             "opponent": opponent, "mode": "bc"}
        )
    records = []
    for ep in pool.imap_unordered(rollout.run_episode, jobs):
        records.extend(ep["records"])
    print(f"   {len(records)} day-decisions")

    x, masks, choices = _tensor_batch(records)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = x.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total, correct, seen = 0.0, 0, 0
        for s in range(0, n, 512):
            idx = perm[s : s + 512]
            lp, _, _ = _joint_logprob_entropy(net, x[idx], masks[idx], choices[idx])
            loss = -lp.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            with torch.no_grad():
                logits, _ = net(x[idx])
                dists = masked_dists(logits, masks[idx])
                hit = torch.ones(len(idx), dtype=torch.bool)
                for h, d in enumerate(dists):
                    hit &= d.probs.argmax(-1) == choices[idx][:, h]
                correct += hit.sum().item()
                seen += len(idx)
        if epoch % 10 == 9 or epoch == 0:
            print(f"   bc epoch {epoch+1}: nll={total/n:.3f} exact-plan acc={correct/seen:.2%}")


def _episode_advantages(ep, shaping_w):
    recs = ep["records"]
    n = len(recs)
    if n == 0:
        return [], []
    pvs = [r["pv"] for r in recs] + [ep["final_money"]]
    rewards = []
    for i in range(n):
        r = shaping_w * (pvs[i + 1] - pvs[i]) / 1000.0
        if i == n - 1:
            r += MONEY_COEF * ep["final_money"] + WIN_COEF * (2 * ep["win"] - 1)
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


def ppo_phase(net, pool, run_dir, iters, episodes, procs, eval_every=5, opp_bundles=(),
              patience=4):
    snapshots = []
    league = []          # weight paths of every new-best: future opponents
    best_score = -1.0
    evals_since_best = 0
    os.makedirs(run_dir, exist_ok=True)
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
            json.dump(to_layers(net), f)

        jobs = []
        rng = random.Random(it)
        for e in range(episodes):
            roll = rng.random()
            if opp_bundles and roll < 0.30:
                opp = ("bundle", rng.choice(opp_bundles))
            elif league and roll < 0.42:
                # Past bests join the opposition (league growth).
                opp = ("league", rng.choice(league[-4:]))
            elif roll < 0.62 or not snapshots:
                opp = "heuristic"
            elif roll < 0.68:
                opp = "starter"
            else:
                opp = ("snapshot", rng.choice(snapshots[-5:]))
            jobs.append(
                {"weights_path": wpath, "temperature": 1.0,
                 "seed": 10_000 + it * episodes + e, "side": e % 2,
                 "opponent": opp, "mode": "rl"}
            )
        episodes_out = list(pool.imap_unordered(rollout.run_episode, jobs))
        snapshots.append(wpath)

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
            opt = torch.optim.Adam(net.parameters(), lr=LR)
            ppo_phase._opt = opt
            ppo_phase._opt_net = net

        n = x.shape[0]
        warmup = it < WARMUP_ITERS
        stop = False
        for _ in range(4):
            if stop:
                break
            perm = torch.randperm(n)
            for s in range(0, n, 512):
                idx = perm[s : s + 512]
                lp, ent, values = _joint_logprob_entropy(net, x[idx], masks[idx], choices[idx])
                vloss = F.mse_loss(values, ret_t[idx])
                if warmup:
                    # Value-head warmup: advantages from an untrained critic
                    # would wreck the BC policy (AlphaStar Unplugged: anchor
                    # improvement to a settled value estimate).
                    loss = VF_COEF * vloss
                else:
                    approx_kl = (old_lp[idx] - lp).mean()
                    if approx_kl > KL_STOP:
                        stop = True
                        break
                    ratio = torch.exp(lp - old_lp[idx])
                    a = adv_t[idx]
                    pg = -torch.min(
                        ratio * a, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * a
                    ).mean()
                    loss = pg + VF_COEF * vloss - ENT_COEF * ent.mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
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
            f"shaping={shaping_w:.2f} steps={n} bad={bad} "
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
            min_win = min(w for _, w, _ in results)
            detail = "  ".join(
                f"{g if isinstance(g, str) else os.path.basename(os.path.dirname(g[1]))}"
                f":{w:.2f}/${m/1000:.1f}k"
                for g, w, m in results
            )
            print(f"   EVAL argmax profit={profit:8.0f} min-win={min_win:.2f}  [{detail}]")
            torch.save(net.state_dict(), os.path.join(run_dir, "last.pt"))
            if profit > best_score:
                best_score = profit
                evals_since_best = 0
                league.append(wpath)
                torch.save(net.state_dict(), os.path.join(run_dir, "best.pt"))
                with open(os.path.join(run_dir, "best_weights.json"), "w") as f:
                    json.dump(to_weights_data(net), f)
                print(f"   new best (profit={profit:.0f}) saved; league size {len(league)}")
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
        "--patience",
        type=int,
        default=4,
        help="stop after this many evals without a profit improvement",
    )
    args = ap.parse_args()

    run_dir = args.run or f"rl/runs/{time.strftime('%m%d_%H%M')}"
    os.makedirs(run_dir, exist_ok=True)

    # Feature size probe: one tiny BC episode.
    probe = rollout.run_episode(
        {"weights_path": None, "seed": 1, "side": 0, "opponent": "starter", "mode": "bc"}
    )
    n_features = len(probe["records"][0]["x"])
    print(f"n_features={n_features} n_action_logits={features.N_ACTION_LOGITS}")

    net = PlannerNet(n_features)
    if args.resume:
        net.load_state_dict(torch.load(args.resume))
        print(f"resumed from {args.resume}")

    bundles = tuple(b for b in (args.opp_bundles or "").split(",") if b)
    with Pool(args.procs) as pool:
        if args.bc_games > 0 and not args.resume:
            bc_phase(net, pool, args.bc_games)
            torch.save(net.state_dict(), os.path.join(run_dir, "bc.pt"))
        ppo_phase(
            net, pool, run_dir, args.iters, args.episodes, args.procs,
            opp_bundles=bundles, patience=args.patience,
        )


if __name__ == "__main__":
    main()
