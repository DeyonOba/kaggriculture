"""Summaries for docs/action_modeling_research.md: the dataset by player
(games, samples, rating), the label distribution by player, and the
held-out report of rl/execnet/train.py rendered as tables.

    PYTHONPATH=. uv run python rl/execnet/report.py --data rl/datasets/execnet \\
        --report rl/datasets/execnet/execnet_report.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from rl.execnet import features as F  # noqa: E402


def dataset_tables(data_dir):
    meta = json.load(open(os.path.join(data_dir, "meta.json")))
    by_player = defaultdict(lambda: {"games": 0, "samples": 0, "ratings": [], "labels": Counter(), "ours_eq": 0, "ours_n": 0})
    for f in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        gid = os.path.splitext(os.path.basename(f))[0]
        m = meta.get(gid, {})
        player = m.get("player", "?")
        key = player if m.get("source") != "session" else "sessions >= 3000 (many teams)"
        z = np.load(f)
        y = z["y"].astype(int)
        ours = z["ours"].astype(int)
        d = by_player[key]
        d["games"] += 1
        d["samples"] += len(y)
        if m.get("rating"):
            d["ratings"].append(m["rating"])
        d["labels"].update(F.OPS[i] for i in y)
        v = ours >= 0
        d["ours_eq"] += int((ours[v] == y[v]).sum())
        d["ours_n"] += int(v.sum())
    lines = ["| player | games | samples | rating | moves | PASS | WATER | HARVEST | PLANT | FEED | CARE | FERT | PICKUP | DROP | COLLECT | our scheduler agrees |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, d in sorted(by_player.items(), key=lambda kv: -kv[1]["samples"]):
        n = d["samples"]
        sh = lambda ops: sum(d["labels"][o] for o in ops) / n  # noqa: E731
        rating = f"{np.mean(d['ratings']):.0f}" if d["ratings"] else "replay"
        lines.append(f"| {key} | {d['games']} | {n:,} | {rating} | {sh(['NORTH','SOUTH','EAST','WEST']):.2f} | {sh(['PASS']):.2f} | "
                     f"{sh(['WATER']):.2f} | {sh(['HARVEST']):.2f} | {sh(['PLANT']):.2f} | {sh(['FEED']):.2f} | {sh(['CARE']):.2f} | "
                     f"{sh(['FERTILIZE']):.2f} | {sh(['PICKUP']):.2f} | {sh(['DROP']):.2f} | {sh(['COLLECT_FERTILIZER']):.2f} | "
                     f"{d['ours_eq'] / max(1, d['ours_n']):.2f} |")
    return "\n".join(lines)


def report_tables(report_path):
    r = json.load(open(report_path))
    out = [f"Held-out: {r['games_held']} games, {r['n_held']:,} samples (train {r['games_train']} games, {r['n_train']:,}).",
           f"Accuracy on the top players' ops: model **{r['accuracy_model']:.3f}**, our scheduler {r['accuracy_ours_vs_top']:.3f}; "
           f"model-vs-scheduler agreement {r['agreement_model_vs_ours']:.3f}. Moves only: model {r['accuracy_model_moves_only']:.3f}. "
           f"Chores only (non-move, non-PASS): model {r['accuracy_model_chores_only']:.3f}, ours {r['accuracy_ours_chores_only']:.3f}.",
           "", "| op | support | model precision | model recall | our scheduler precision | our scheduler recall |", "|---|---|---|---|---|---|"]
    pm, po = r["per_op_model"], r["per_op_ours"]
    fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
    for op in F.OPS:
        a, b = pm[op], po.get(op, {})
        if a["support"] == 0:
            continue
        out.append(f"| {op} | {a['support']:,} | {fmt(a['precision'])} | {fmt(a['recall'])} | {fmt(b.get('precision'))} | {fmt(b.get('recall'))} |")
    out += ["", "Where the top players and our scheduler part ways most (held-out rows, top player's op → our op, count):", ""]
    conf = r["confusion_top_vs_ours"]
    pairs = [(a, b, n) for a, row in conf.items() for b, n in row.items() if a != b]
    pairs.sort(key=lambda t: -t[2])
    out += [f"- {a} → we {b}: {n:,}" for a, b, n in pairs[:12]]
    out += ["", "Accuracy by model confidence (threshold: accuracy of the covered rows, share covered): "
            + ", ".join(f"≥{th}: {v[0]:.3f} on {v[1]:.2f}" for th, v in r["accuracy_by_confidence"].items() if v[0] is not None)]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO, "rl/datasets/execnet"))
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    print(dataset_tables(args.data))
    if args.report and os.path.exists(args.report):
        print()
        print(report_tables(args.report))


if __name__ == "__main__":
    main()
