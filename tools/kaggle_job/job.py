"""Kaggle-notebook job runner: run one of our compute jobs (bandit collection,
BC/PPO training, paired evaluation) on Kaggle's free CPU/GPU tier.

Pushed with `tools/kaggle_job/push.sh`; the job is selected by
`job_config.json` in this directory (Kaggle scripts take no CLI args).
Everything the job writes into OUT_DIR lands in the kernel's output, which
`kaggle kernels output gideonoba/kaggriculture-job -p <dir>` fetches locally.

Layout on Kaggle:
  /tmp/kaggriculture               the repo, restored from the code Dataset (not published)
  /kaggle/input/kaggriculture-datasets/  our private Kaggle Dataset (sessions etc.)
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
import gzip
import time

WORK = "/kaggle/working"            # = the kernel's OUTPUT; keep it small
REPO = "/tmp/kaggriculture"         # restored repo + corpus live outside the output
INPUT = "/kaggle/input"


def _discover(marker_file=None, marker_dir=None, exclude=()):
    """Find the mounted Dataset containing a marker (mount slugs can differ
    from the dataset id, and Kaggle may auto-extract uploaded archives)."""
    if not os.path.isdir(INPUT):
        return None
    for top in sorted(os.listdir(INPUT)):
        root = os.path.join(INPUT, top)
        if root in exclude:
            continue
        for d, dirs, files in os.walk(root):
            if (marker_file and marker_file in files) or (marker_dir and marker_dir in dirs):
                return root
    return None


CODE_IN = _discover(marker_dir="kagent") or "/kaggle/input/kaggriculture-code"
DATA_IN = (_discover(marker_file="sessions.jsonl", exclude=(CODE_IN,))
           or _discover(marker_file="sessions.jsonl.gz", exclude=(CODE_IN,))
           or _discover(marker_file="sessions.tar.gz", exclude=(CODE_IN,))
           or _discover(marker_dir="sessions", exclude=(CODE_IN,))
           or "/kaggle/input/kaggriculture-datasets")
OUT_DIR = os.path.join(WORK, "out")
CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "job_config.json")
# push.sh replaces this with the contents of job_config.json (Kaggle uploads
# only the single script file of a kernel).
CONFIG_JSON = None


def sh(cmd, **kw):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kw)


def _listing(root, depth=2):
    out = []
    base = root.rstrip("/").count("/")
    for d, dirs, files in os.walk(root):
        if d.count("/") - base >= depth:
            dirs[:] = []
        out.append(f"{d}: {sorted(dirs)[:8]} {sorted(files)[:8]}")
    return "\n".join(out[:30])


def _find_dir(root, name):
    for d, dirs, _ in os.walk(root):
        if name in dirs:
            return os.path.join(d, name)
    return None


def setup(cfg):
    print("inputs:", sorted(os.listdir(INPUT)) if os.path.isdir(INPUT) else "none", flush=True)
    print("code mount:\n" + _listing(CODE_IN), flush=True)
    print("data mount:\n" + _listing(DATA_IN), flush=True)
    if not os.path.exists(os.path.join(REPO, "kagent")):
        # The GitHub repo is private; the code arrives as a Dataset. Kaggle
        # may serve the uploaded tarball as-is or auto-extracted.
        os.makedirs(REPO, exist_ok=True)
        tarball = os.path.join(CODE_IN, "kaggriculture-code.tar.gz")
        if os.path.exists(tarball):
            with tarfile.open(tarball, "r:gz") as tf:
                tf.extractall(REPO)
        else:
            src = os.path.dirname(_find_dir(CODE_IN, "kagent") or CODE_IN)
            shutil.copytree(src, REPO, dirs_exist_ok=True)
        print("code restored:", sorted(os.listdir(REPO))[:12], flush=True)
    # Engine parity: the agent mirrors kaggle_environments 1.32.7 constants.
    sh(f"{sys.executable} -m pip install -q 'kaggle-environments=={cfg.get('engine', '1.32.7')}'")
    os.makedirs(os.path.join(REPO, "rl", "datasets"), exist_ok=True)
    # Restore the corpus from the Kaggle Dataset mount (tools/data.py layout).
    ds_root = os.path.join(REPO, "rl", "datasets")
    if os.path.isdir(DATA_IN):
        plain_map = {"field-samples.jsonl": "field/samples.jsonl", "field-stats.jsonl": "field/stats.jsonl",
                     "field-seen.txt": "field/seen.txt", "champions.jsonl": "champions.jsonl",
                     "decode-profiles.jsonl": "decode/profiles.jsonl"}
        for d, _, files in os.walk(DATA_IN):
            for name in files:
                src = os.path.join(d, name)
                if name == "sessions.tar.gz":
                    with tarfile.open(src, "r:gz") as tf:
                        tf.extractall(ds_root)
                    continue
                base = name[:-3] if name.endswith(".gz") else name
                if base in plain_map:
                    dst = os.path.join(ds_root, plain_map[base])
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    opener = gzip.open if name.endswith(".gz") else open
                    with opener(src, "rb") as fi, open(dst, "wb") as fo:
                        shutil.copyfileobj(fi, fo, 1 << 20)
        # Preferred: the sessions JSONL (Kaggle decompresses .gz on mount).
        sess_dir = os.path.join(ds_root, "sessions")
        for d, _, files in os.walk(DATA_IN):
            for name in files:
                if name in ("sessions.jsonl", "sessions.jsonl.gz"):
                    os.makedirs(sess_dir, exist_ok=True)
                    opener = gzip.open if name.endswith(".gz") else open
                    with opener(os.path.join(d, name), "rt") as fi:
                        for line in fi:
                            if line.strip():
                                ep = json.loads(line)["episode"]
                                with open(os.path.join(sess_dir, f"session-{ep}.json"), "w") as fo:
                                    fo.write(line.strip())
        # Auto-extracted layout: a `sessions` directory somewhere under the mount.
        if not os.path.isdir(os.path.join(ds_root, "sessions")):
            found = _find_dir(DATA_IN, "sessions")
            if found:
                shutil.copytree(found, os.path.join(ds_root, "sessions"), dirs_exist_ok=True)
        # Kaggle unpacks an uploaded tarball under a folder named after it:
        # sessions/sessions/*.json -> flatten.
        nested = os.path.join(ds_root, "sessions", "sessions")
        if os.path.isdir(nested):
            for f in os.listdir(nested):
                shutil.move(os.path.join(nested, f), os.path.join(ds_root, "sessions", f))
            os.rmdir(nested)
        n = len(glob.glob(os.path.join(ds_root, "sessions", "session-*.json")))
        print(f"restored datasets: {n} sessions", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)


def run(cfg):
    env = dict(os.environ, PYTHONPATH=REPO)
    procs = cfg.get("procs", os.cpu_count() or 4)
    budget = cfg.get("budget_s", 11 * 3600)  # Kaggle CPU sessions cap at 12h
    t0 = time.time()
    for step in cfg["steps"]:
        cmd = step.format(procs=procs, out=OUT_DIR, repo=REPO, python=sys.executable)
        remaining = budget - (time.time() - t0)
        if remaining < 300:
            print("budget exhausted; stopping before:", cmd, flush=True)
            break
        print(f"== step ({remaining / 3600:.1f}h left): {cmd}", flush=True)
        try:
            rc = subprocess.run(cmd, shell=True, cwd=REPO, env=env, timeout=remaining).returncode
            if rc != 0:
                print(f"step failed (rc={rc}); continuing", flush=True)
        except subprocess.TimeoutExpired:
            print("step hit the time budget; partial outputs kept", flush=True)
            break
    # Ship anything the job left in rl/datasets/bandit or rl/runs to the output.
    for rel in cfg.get("collect", []):
        src = os.path.join(REPO, rel)
        if os.path.exists(src):
            dst = os.path.join(OUT_DIR, os.path.basename(rel.rstrip("/")))
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
            print("collected", rel, flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    if CONFIG_JSON:
        cfg = json.loads(CONFIG_JSON)
    else:
        with open(CONFIG) as f:
            cfg = json.load(f)
    print("job config:", json.dumps(cfg), flush=True)
    setup(cfg)
    run(cfg)
