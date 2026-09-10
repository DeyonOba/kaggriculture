# rl/datasets — stored outside git

The training/evaluation corpus is ~424 MB raw (one file alone is 208 MB, above GitHub's
100 MB limit), so it is **not tracked here**. It lives in a private Kaggle Dataset as
compressed artifacts (~45 MB total); `MANIFEST.json` in this directory is the committed
reference (artifact sha256, sizes, item counts, dataset id).

```bash
PYTHONPATH=. uv run python tools/data.py pull     # restore everything into rl/datasets/
PYTHONPATH=. uv run python tools/data.py verify   # local files vs MANIFEST counts
PYTHONPATH=. uv run python tools/data.py push -m "what changed"   # after a new harvest
```

`push` rebuilds the artifacts, uploads a new dataset version, and rewrites `MANIFEST.json`
— commit the manifest with the code that depends on the new data.

| Path | What | Produced by |
|---|---|---|
| `sessions/session-<episode>.json` | seed + both sides' full action scripts + finals + per-side ratings; replays exactly on the local engine | `tools/harvest_stream.py`, `tools/build_sessions.py` |
| `field/samples.jsonl` | per-day features + head targets from both sides of harvested games, labelled `{win, margin, money}` | `tools/harvest_stream.py` |
| `field/stats.jsonl`, `field/seen.txt` | per-side outcome rows; harvested episode ids | `tools/harvest_stream.py` |
| `champions.jsonl` | BC samples from top-tier replay sides | `rl/replay_bc.py` |
| `decode/profiles.jsonl` | per-day state + cash-flow profiles of rated sides and of our agent on their worlds | `tools/decode_sessions.py` |

Everything else here (`*.log`, `field/tmp/`) is scratch and ignored.
