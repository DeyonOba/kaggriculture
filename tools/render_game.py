"""Render one game to a standalone HTML replay you can open in a browser.

Nothing in this repo has ever produced a watchable game -- every result is a
number in a table. That makes some questions very hard to ask: the executor's
remaining gap is spatial (our consecutive worked tiles are 1.92 Manhattan steps
apart against ahmed's 1.60, and we spend 58.6% of unit-turns walking against
their 42.2%), and a table cannot show you what that looks like.

    PYTHONPATH=. uv run python tools/render_game.py --a j-tile --b pub-ahmed-v38
    PYTHONPATH=. uv run python tools/render_game.py --a tape-sel --seed 71003 --out /tmp/x.html

`--pick` plays several seeds and renders the one where `--a` does best, which
is usually what you want to look at; without it, `--seed` is played directly.
"""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import tools.fast_env as fast_env  # noqa: E402

# The renderer needs the real step records; fast_env only changes how state is
# cloned between steps, and rewards/observations stay bit-identical, so it is
# safe here and roughly halves the wait.
fast_env.install()

from kaggle_environments import make   # noqa: E402


def path_for(name):
    p = os.path.join(REPO, "arena", "agents", name, "main.py")
    if not os.path.isfile(p):
        raise SystemExit(f"no agent bundle at {p}")
    return p


def play(a, b, seed):
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.run([path_for(a), path_for(b)])
    r = [env.steps[-1][i]["reward"] for i in (0, 1)]
    return env, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="j-tile", help="seat 0 (left)")
    ap.add_argument("--b", default="pub-ahmed-v38", help="seat 1 (right)")
    ap.add_argument("--seed", type=int, default=71000)
    ap.add_argument("--pick", type=int, default=0,
                    help="play N seeds from --seed and render A's best one")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.pick:
        best = None
        for k in range(args.pick):
            env, r = play(args.a, args.b, args.seed + k)
            margin = r[0] - r[1]
            print(f"  seed {args.seed+k}: {args.a} ${r[0]:>9,.0f}  "
                  f"{args.b} ${r[1]:>9,.0f}  margin {margin:>+9,.0f}", flush=True)
            if best is None or margin > best[0]:
                best = (margin, env, r, args.seed + k)
        _, env, r, seed = best
        print(f"\n  rendering seed {seed} (best margin for {args.a})")
    else:
        seed = args.seed
        env, r = play(args.a, args.b, seed)
        print(f"  seed {seed}: {args.a} ${r[0]:,.0f}  {args.b} ${r[1]:,.0f}")

    out = args.out or os.path.join(REPO, f"replay-{args.a}-vs-{args.b}-{seed}.html")
    html = env.render(mode="html", width=1000, height=700)
    with open(out, "w") as f:
        f.write(html)
    n = os.path.getsize(out)
    print(f"\n  wrote {out}  ({n/1_000_000:.1f} MB)")
    print(f"  winner: {args.a if r[0] > r[1] else args.b}"
          f"   final ${r[0]:,.0f} vs ${r[1]:,.0f}")
    print(f"\n  open it with:  open {out}")


if __name__ == "__main__":
    main()
