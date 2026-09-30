"""Pool paired gate blocks: per file and pooled -- win rates, delta, z (McNemar),
discordant counts, and BOTH banks' paired deltas. usage:
  PYTHONPATH=. uv run python tools/gate_pool.py ARM BASELINE file.json [file.json ...]"""
import sys, json, math, os
arm, base_name, files = sys.argv[1], sys.argv[2], sys.argv[3:]
tot = dict(n=0, a=0, b=0, wa=0, wb=0, d_ours=0.0, d_opp=0.0, ours=0.0, opp=0.0)
print(f"{'block':28s} {'arm':>6s} {'base':>6s} {'delta':>7s} {'z':>6s} {'a-only':>6s} {'b-only':>6s} {'d_ours':>8s} {'d_opp':>8s}")
for f in files:
    d = json.load(open(f)); res = d["results"]
    key = lambda r: (r["path"], r["script_side"])
    base = {key(r): r for r in res[base_name]}
    n = a = b = wa = wb = 0; do = dp = 0.0
    for r in res[arm]:
        bb = base[key(r)]; n += 1
        a += int(r["win"] > bb["win"]); b += int(r["win"] < bb["win"]); wa += r["win"]; wb += bb["win"]
        do += r["money"] - bb["money"]; dp += r["opp"] - bb["opp"]
    z = (a - b) / math.sqrt(max(1, a + b))
    print(f"{os.path.basename(f):28s} {wa/n:6.3f} {wb/n:6.3f} {(a-b)/n:+7.3f} {z:+6.2f} {a:6d} {b:6d} {do/n:+8,.0f} {dp/n:+8,.0f}")
    for k, v in (("n", n), ("a", a), ("b", b), ("wa", wa), ("wb", wb), ("d_ours", do), ("d_opp", dp)): tot[k] += v
n, a, b = tot["n"], tot["a"], tot["b"]
z = (a - b) / math.sqrt(max(1, a + b)); se = math.sqrt(a + b) / n
print(f"{'POOLED ' + str(n) + ' games':28s} {tot['wa']/n:6.3f} {tot['wb']/n:6.3f} {(a-b)/n:+7.3f} {z:+6.2f} {a:6d} {b:6d} {tot['d_ours']/n:+8,.0f} {tot['d_opp']/n:+8,.0f}   (delta +/- {se:.3f})")
