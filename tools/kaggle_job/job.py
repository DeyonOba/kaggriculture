"""Kaggle-notebook job runner: run one of our compute jobs (bandit collection,
BC/PPO training, paired evaluation) on Kaggle's free CPU/GPU tier.

Pushed with `tools/kaggle_job/push.sh`; the job is selected by
`job_config.json` in this directory (Kaggle scripts take no CLI args).
Everything the job writes into OUT_DIR lands in the kernel's output, which
`kaggle kernels output gideonoba/kaggriculture-job -p <dir>` fetches locally.

Layout on Kaggle:
  /kaggle/working/kaggriculture   the repo, restored from the code Dataset tarball
  /kaggle/input/kaggriculture-datasets/  our private Kaggle Dataset (sessions etc.)
"""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import gzip
import time

WORK = "/kaggle/working"
REPO = os.path.join(WORK, "kaggriculture")
CODE_IN = "/kaggle/input/kaggriculture-code"      # private Dataset: repo tarball (push.sh)
DATA_IN = "/kaggle/input/kaggriculture-datasets"  # private Dataset: corpus (tools/data.py)
OUT_DIR = os.path.join(WORK, "out")
CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "job_config.json")
# push.sh replaces this with the contents of job_config.json (Kaggle uploads
# only the single script file of a kernel).
CONFIG_JSON = None


def sh(cmd, **kw):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kw)


def setup(cfg):
    if not os.path.exists(REPO):
        # The GitHub repo is private; the code arrives as a Dataset tarball.
        tarball = os.path.join(CODE_IN, "kaggriculture-code.tar.gz")
        os.makedirs(REPO, exist_ok=True)
        with tarfile.open(tarball, "r:gz") as tf:
            tf.extractall(REPO)
        print("code restored:", sorted(os.listdir(REPO))[:12], flush=True)
    # Engine parity: the agent mirrors kaggle_environments 1.32.7 constants.
    sh(f"{sys.executable} -m pip install -q 'kaggle-environments=={cfg.get('engine', '1.32.7')}'")
    os.makedirs(os.path.join(REPO, "rl", "datasets"), exist_ok=True)
    # Restore the corpus from the Kaggle Dataset mount (tools/data.py layout).
    if os.path.isdir(DATA_IN):
        for name in os.listdir(DATA_IN):
            src = os.path.join(DATA_IN, name)
            if name.endswith(".tar.gz"):
                with tarfile.open(src, "r:gz") as tf:
                    tf.extractall(os.path.join(REPO, "rl", "datasets"))
            elif name.endswith(".gz"):
                rel = {"field-samples.jsonl.gz": "field/samples.jsonl", "field-stats.jsonl.gz": "field/stats.jsonl",
                       "field-seen.txt.gz": "field/seen.txt", "champions.jsonl.gz": "champions.jsonl",
                       "decode-profiles.jsonl.gz": "decode/profiles.jsonl"}.get(name)
                if rel:
                    dst = os.path.join(REPO, "rl", "datasets", rel)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    with gzip.open(src, "rb") as fi, open(dst, "wb") as fo:
                        shutil.copyfileobj(fi, fo, 1 << 20)
        n = len(os.listdir(os.path.join(REPO, "rl", "datasets", "sessions"))) if os.path.isdir(os.path.join(REPO, "rl", "datasets", "sessions")) else 0
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
