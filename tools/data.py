"""Dataset store for rl/datasets (too large for GitHub: 424 MB raw, one 208 MB
file). The corpus lives in a private Kaggle Dataset as compressed artifacts;
git tracks only rl/datasets/MANIFEST.json (sha256 + counts) and README.md.

    PYTHONPATH=. uv run python tools/data.py push  -m "683 sessions, 77k field samples"
    PYTHONPATH=. uv run python tools/data.py pull            # restore rl/datasets in place
    PYTHONPATH=. uv run python tools/data.py verify          # local files vs MANIFEST

Requires the Kaggle CLI (already used for submissions). Artifacts:
  sessions.tar.gz          rl/datasets/sessions/session-*.json (replay scripts + seeds)
  field-samples.jsonl.gz   rl/datasets/field/samples.jsonl (outcome-labelled BC samples)
  field-stats.jsonl.gz     rl/datasets/field/stats.jsonl
  field-seen.txt.gz        rl/datasets/field/seen.txt (harvested episode ids)
  champions.jsonl.gz       rl/datasets/champions.jsonl
  decode-profiles.jsonl.gz rl/datasets/decode/profiles.jsonl
"""

import argparse
import glob
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(REPO, "rl", "datasets")
MANIFEST = os.path.join(ROOT, "MANIFEST.json")
DATASET = "gideonoba/kaggriculture-datasets"
TITLE = "Kaggriculture agent datasets"

# artifact name -> (kind, source path relative to ROOT)
ARTIFACTS = {
    # Sessions ship as one JSONL (one session per line): Kaggle decompresses
    # single .gz files on mount and unpacks tarballs under a folder named
    # after them, so the JSONL is the reliable form; pull and the Kaggle job
    # runner split it back into session-<id>.json files.
    "sessions.jsonl.gz": ("jsonl_dir", "sessions"),
    "field-samples.jsonl.gz": ("gz", "field/samples.jsonl"),
    "field-stats.jsonl.gz": ("gz", "field/stats.jsonl"),
    "field-seen.txt.gz": ("gz", "field/seen.txt"),
    "champions.jsonl.gz": ("gz", "champions.jsonl"),
    "decode-profiles.jsonl.gz": ("gz", "decode/profiles.jsonl"),
    # Per-seat ratings for the official daily episodes (community crawl,
    # georgymamarin/kaggriculture-episodes agents.csv): the ratings join for
    # tools/extract_episodes.py.
    "ratings-agents.csv.gz": ("gz", "ratings/agents.csv"),
    "ratings-leaderboard.csv.gz": ("gz", "ratings/leaderboard.csv"),  # current team ratings (fallback join)
}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _count(src):
    """Files for a directory, lines for a text file."""
    if os.path.isdir(src):
        return len(glob.glob(os.path.join(src, "*")))
    with open(src, "rb") as f:
        return sum(1 for _ in f)


def build(stage):
    manifest = {"dataset": DATASET, "artifacts": {}}
    for name, (kind, rel) in ARTIFACTS.items():
        src = os.path.join(ROOT, rel)
        if not os.path.exists(src):
            print(f"  skip {name}: {rel} missing")
            continue
        out = os.path.join(stage, name)
        if kind == "tar":
            with tarfile.open(out, "w:gz") as tf:
                for p in sorted(glob.glob(os.path.join(src, "*.json"))):
                    tf.add(p, arcname=os.path.join(rel, os.path.basename(p)))
        elif kind == "jsonl_dir":
            with gzip.open(out, "wt", compresslevel=6) as fo:
                for p in sorted(glob.glob(os.path.join(src, "*.json"))):
                    with open(p) as fi:
                        fo.write(fi.read().strip() + "\n")
        else:
            with open(src, "rb") as fi, gzip.open(out, "wb", compresslevel=6) as fo:
                shutil.copyfileobj(fi, fo, 1 << 20)
        manifest["artifacts"][name] = {
            "source": rel,
            "count": _count(src),
            "raw_bytes": sum(
                os.path.getsize(p) for p in (glob.glob(os.path.join(src, "*")) if os.path.isdir(src) else [src])
            ),
            "bytes": os.path.getsize(out),
            "sha256": _sha256(out),
        }
        print(f"  {name}: {manifest['artifacts'][name]['bytes'] / 1e6:.1f} MB "
              f"({manifest['artifacts'][name]['count']} items)")
    return manifest


def push(message):
    stage = tempfile.mkdtemp(prefix="kagg-data-")
    print(f"staging in {stage}")
    manifest = build(stage)
    with open(os.path.join(stage, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(stage, "dataset-metadata.json"), "w") as f:
        json.dump({"title": TITLE, "id": DATASET, "licenses": [{"name": "CC0-1.0"}]}, f)
    # (the status endpoint returns 403 with a legacy API key; use the listing)
    listing = subprocess.run(
        ["uv", "run", "kaggle", "datasets", "list", "--mine"], capture_output=True, text=True
    )
    if DATASET in listing.stdout:
        cmd = ["uv", "run", "kaggle", "datasets", "version", "-p", stage, "-m", message, "--dir-mode", "skip"]
    else:
        cmd = ["uv", "run", "kaggle", "datasets", "create", "-p", stage, "--dir-mode", "skip"]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"MANIFEST written to {MANIFEST}; commit it.")


def pull():
    with open(MANIFEST) as f:
        manifest = json.load(f)
    stage = tempfile.mkdtemp(prefix="kagg-data-")
    subprocess.run(
        ["uv", "run", "kaggle", "datasets", "download", "-d", manifest["dataset"], "-p", stage, "--unzip"],
        check=True,
    )
    ok = True
    for name, meta in manifest["artifacts"].items():
        path = os.path.join(stage, name)
        if not os.path.exists(path):
            print(f"  MISSING {name}")
            ok = False
            continue
        if _sha256(path) != meta["sha256"]:
            print(f"  CHECKSUM MISMATCH {name} (dataset version differs from MANIFEST)")
            ok = False
        dst = os.path.join(ROOT, meta["source"])
        if name.endswith(".tar.gz"):
            with tarfile.open(path, "r:gz") as tf:
                tf.extractall(ROOT)
        elif name == "sessions.jsonl.gz":
            os.makedirs(dst, exist_ok=True)
            with gzip.open(path, "rt") as fi:
                for line in fi:
                    if line.strip():
                        ep = json.loads(line)["episode"]
                        with open(os.path.join(dst, f"session-{ep}.json"), "w") as fo:
                            fo.write(line.strip())
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with gzip.open(path, "rb") as fi, open(dst, "wb") as fo:
                shutil.copyfileobj(fi, fo, 1 << 20)
        print(f"  restored {meta['source']} ({meta['count']} items)")
    shutil.rmtree(stage, ignore_errors=True)
    print("pull OK" if ok else "pull finished with problems")
    return ok


def verify():
    with open(MANIFEST) as f:
        manifest = json.load(f)
    ok = True
    for name, meta in manifest["artifacts"].items():
        src = os.path.join(ROOT, meta["source"])
        if not os.path.exists(src):
            print(f"  missing {meta['source']} (run: tools/data.py pull)")
            ok = False
            continue
        n = _count(src)
        flag = "ok" if n == meta["count"] else f"COUNT {n} != {meta['count']}"
        ok &= n == meta["count"]
        print(f"  {meta['source']}: {flag}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "pull", "verify"])
    ap.add_argument("-m", "--message", default="update")
    args = ap.parse_args()
    if args.cmd == "push":
        push(args.message)
    elif args.cmd == "pull":
        sys.exit(0 if pull() else 1)
    else:
        sys.exit(0 if verify() else 1)


if __name__ == "__main__":
    main()
