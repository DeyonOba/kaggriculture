"""Assemble the HTML replay viewer from one or more tools/game_trace.py traces.

The page (tools/game_viewer.html) is static and self-contained: the traces
are inlined as JSON, nothing is fetched. Every step of the season is
scrubbable; both farms, the market, the dawn plan, this step's actions and
the trainer's signals (portfolio value, asset lead, money lead, Phi) are shown
for the current step.

    PYTHONPATH=. uv run python tools/game_viewer.py --trace win.json loss.json --out replay.html
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PLACEHOLDER = "/*__GAMES__*/null"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trace", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    games = [json.load(open(p)) for p in args.trace]
    with open(os.path.join(HERE, "game_viewer.html")) as fh:
        html = fh.read()
    if PLACEHOLDER not in html:
        raise SystemExit("template has no data placeholder")
    blob = json.dumps(games, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace(PLACEHOLDER, blob)
    with open(args.out, "w") as fh:
        fh.write(html)
    print(f"-> {args.out}  {os.path.getsize(args.out)/1e6:.1f} MB, {len(games)} game(s)")


if __name__ == "__main__":
    main()
