"""Reserve market-order slots for selling.

The engine caps market orders at 10 per turn (maxMarketOrdersPerTurn) and does
q[:max_orders] -- anything past the tenth is silently dropped. Each order
carries its own quantity, so one SELL of 30 units costs ONE slot, but HIRE is
one order PER HAND.

kagent/market.py emits HIRE before feed and sells, up to 8 per turn. Measured:
champ-h8 saturates the cap on 21 turns a game and HIRE takes 165 of those 210
slots -- 79%. Hiring also consumes more slots all game (273) than selling does
(168). It is also why h-front11 could never field 11 hands: 11 hires needs 11
slots against a cap of 10.

Across the recorded corpus, SELL order count is the ONE op-mix metric that
tracks rating -- 631 orders/game at rating <800 against 263 at 3000+ -- so how
market slots are spent is a quality signal, and ours are spent on hiring.

    PYTHONPATH=. uv run python tools/make_slot_variant.py --name m-hire5 --cap 5
"""
import argparse, os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tools.rollout_plan import _split_bundle   # noqa: E402

ANCHOR = "while need > 0 and hired < 8:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="champ-h8")
    ap.add_argument("--name", required=True)
    ap.add_argument("--cap", type=int, required=True,
                    help="max HIRE orders per turn; the rest of the 10 stay "
                         "free for feed and sells")
    args = ap.parse_args()

    src = open(os.path.join(REPO, "arena", "agents", args.base, "main.py")).read()
    head, mods, tail = _split_bundle(src)
    out, done = [], False
    for mod, text in mods:
        if mod == "market":
            if ANCHOR not in text:
                raise SystemExit("hire-cap anchor not found")
            text = text.replace(ANCHOR, f"while need > 0 and hired < {args.cap}:", 1)
            done = True
        out.append((mod, text))
    if not done:
        raise SystemExit("no market module")
    d = os.path.join(REPO, "arena", "agents", args.name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "main.py")
    code = head + "_SOURCES = " + repr(out) + tail
    open(p, "w").write(code)
    compile(code, p, "exec")
    print(f"{args.name}: {args.base}, HIRE capped at {args.cap}/turn "
          f"(was 8) -- {10-args.cap} slots reserved")


if __name__ == "__main__":
    main()
