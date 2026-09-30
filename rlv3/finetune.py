"""Fine-tune a trained readout on a fresh, small leader corpus.

    PYTHONPATH=. uv run python rlv3/finetune.py --init rl/runs/headfix/readout_v5_h3/bc.pt \
        --corpus rl/runs/headfix/bc_now.jsonl --out rl/runs/headfix/readout_v5_now/bc.pt

WHY. The leaders' agents are learned and move week to week. A readout trained
on the 2026-09-2x corpus generalised to the 09-29 board in the contest and
consolidation stages (S0 at or under their floor) and NOT in the opening
(0.330 vs 0.188): the new leaders buy a cow and a goose on day 2 off the
first wheat income, run 5x the tomato and restock sheep late, none of which
the old corpus contains. Forty fresh games are too few to train on alone
(1,800 dawns against 71,280), so this continues from the trained weights at a
low rate -- the corpus-wide structure is kept, the fresh behaviour is layered
on -- with the OPENING up-weighted, because that is the stage that failed.

HELD OUT BY GAME. Two games per team are never seen here (replays/now_holdout)
and the S0 check runs on those alone, paired against the un-tuned readout on
the same dawns. Scoring on the games trained on would be the training-set
trap the record warns about everywhere.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--init", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--opening-weight", type=float, default=3.0,
                    help="sample weight for dawns on days 0-8")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--train-trunk", action="store_true",
                    help="board readouts: train the conv trunk end-to-end on the targets "
                         "instead of caching its (frozen) features")
    args = ap.parse_args()

    from kagent import features
    from rl.common import PlannerNet, masked_dists
    features.set_heads_version(args.version)
    torch.manual_seed(args.seed)

    rows = [json.loads(l) for l in open(args.corpus)]
    x = torch.tensor([r["x"] for r in rows], dtype=torch.float32)
    m = torch.tensor([sum(r["masks"], []) if isinstance(r["masks"][0], list) else r["masks"]
                      for r in rows], dtype=torch.float32)
    c = torch.tensor([r["choices"] for r in rows], dtype=torch.long)
    w = torch.tensor([args.opening_weight if r.get("day", 99) <= 8 else 1.0 for r in rows],
                     dtype=torch.float32)
    n_open = int((w > 1).sum())
    print(f"{len(rows):,} dawns, {n_open} in the opening (x{args.opening_weight}); "
          f"x width {x.shape[1]}, heads {len(features.HEADS)}, logits {features.N_ACTION_LOGITS}")

    sd = torch.load(args.init, map_location="cpu")["policy"]
    from rl.common import BOARD_W
    has_board = any(k.startswith("board.") for k in sd)
    net = PlannerNet(sd["h1.weight"].shape[1] - (BOARD_W if has_board else 0), board=has_board)
    net.load_state_dict(sd)
    z = None
    B = None
    if has_board:
        B = np.load(args.corpus.replace(".jsonl", "_boards.npy"), mmap_mode="r")
        assert len(B) == len(rows), f"boards {len(B)} vs samples {len(rows)}"
        if args.train_trunk:
            # END-TO-END: the conv trunk learns the leaders' targets with the
            # heads (the S2-passing setting), a conv pass per minibatch
            print(f"   board trunk: TRAINABLE, conv pass per minibatch over {len(rows):,} dawns")
        else:
            # the trunk stays frozen, as in the BC fit: its features are cached once
            with torch.no_grad():
                z = torch.cat([net.board(torch.tensor(np.asarray(B[s_:s_ + 512], dtype=np.float32)))
                               for s_ in range(0, len(rows), 512)])
            for p_ in net.board.parameters():
                p_.requires_grad_(False)
            print(f"   board trunk: features cached for {len(rows):,} dawns, trunk frozen")
    assert net.logits.out_features == features.N_ACTION_LOGITS, "head set mismatch with the init weights"

    def nll(idx):
        if B is not None and z is None:
            bb = torch.tensor(np.stack([B[i] for i in idx.tolist()]).astype(np.float32))
            logits = net(x[idx], bb)
        else:
            logits = net.forward_z(x[idx], None if z is None else z[idx])
        dists = masked_dists(logits, m[idx])
        lp = sum(d.log_prob(c[idx][:, h]) for h, d in enumerate(dists))
        return -(lp * w[idx]).sum() / w[idx].sum()

    with torch.no_grad():
        print(f"   before: nll {float(nll(torch.arange(len(rows)))):.3f}")
    opt = torch.optim.Adam([p_ for p_ in net.parameters() if p_.requires_grad], lr=args.lr)
    rng = np.random.default_rng(args.seed)
    for ep in range(args.epochs):
        perm = torch.from_numpy(rng.permutation(len(rows)))
        tot = 0.0
        for s in range(0, len(perm), 256):
            idx = perm[s:s + 256]
            loss = nll(idx)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        print(f"   epoch {ep + 1:>2}: nll {tot / len(perm):.3f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"policy": net.state_dict()}, args.out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
