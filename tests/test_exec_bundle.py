"""A bundle carries its executor rules in EXEC_RULES (written by
rl/export.py --exec) and switches them on at import; dev mode is inert.

    PYTHONPATH=. uv run pytest tests/test_exec_bundle.py -q
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from kagent import scheduler  # noqa: E402


def test_dev_mode_leaves_the_flags_off():
    # importing the package the normal way must not set any flag
    out = subprocess.run(
        [sys.executable, "-c",
         "import kagent.policy as p, kagent.scheduler as s;"
         "print(p.EXEC_RULES, any(s.EXEC.values()))"],
        capture_output=True, text=True, cwd=REPO, env={**os.environ, "PYTHONPATH": REPO},
    )
    assert out.returncode == 0, out.stderr[-400:]
    assert out.stdout.strip() == "None False", out.stdout


def test_a_bundle_namespace_switches_them_on():
    # the flat-bundle shape: every module body shares one namespace, so the
    # weights body's EXEC_RULES is a global by the time policy.py runs
    src = open(os.path.join(REPO, "kagent", "policy.py")).read()
    start = src.index("EXEC_RULES = globals()")
    body = src[start:src.index("\nif BANDIT_RULES:", start)]
    ns = {"EXEC_RULES": ["lastwater", "h23", "harvest_visit"], "scheduler": scheduler}
    scheduler.reset_exec()
    try:
        exec(compile(body, "bundle", "exec"), ns)
        assert scheduler.EXEC["lastwater"] and scheduler.EXEC["h23"] and scheduler.EXEC["harvest_visit"]
        assert not scheduler.EXEC["oneshed"] and not scheduler.EXEC["quadq"]
    finally:
        scheduler.reset_exec()


def test_exporter_rejects_an_unknown_flag():
    out = subprocess.run(
        [sys.executable, "rl/export.py", "--clear", "--champion", "--exec", "lastwater,nosuchflag"],
        capture_output=True, text=True, cwd=REPO, env={**os.environ, "PYTHONPATH": REPO},
    )
    assert out.returncode != 0 and "unknown executor flag" in out.stderr
