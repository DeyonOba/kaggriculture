"""tools/agent_scorecard.py: the spec parser, the band-spread game picker,
the gap arithmetic, and metric assembly on a synthetic trace.

    PYTHONPATH=. uv run pytest tests/test_scorecard.py -q
"""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))

import agent_scorecard as A  # noqa: E402


# ------------------------------------------------------------------ specs
def test_parse_spec_every_form():
    assert A.parse_spec("champion") == {"kind": "champion"}
    assert A.parse_spec(None) == {"kind": "champion"}
    b = A.parse_spec("bundle:arena/agents/champ-h10/main.py")
    assert b["kind"] == "bundle" and b["path"].endswith("champ-h10/main.py")
    assert os.path.isabs(b["path"]), "a trace runs from elsewhere, so the path must be absolute"
    w = A.parse_spec("weights:rl/runs/draw995/baselines.json")
    assert w["kind"] == "weights" and os.path.isabs(w["path"])
    k = A.parse_spec("knobs:tools/rules/rules_h8.json")
    assert k["kind"] == "knobs" and "conditions" in k["rules"]


def test_parse_spec_rejects_nonsense():
    with pytest.raises(ValueError):
        A.parse_spec("nosuchkind:foo")


def test_spec_label_names_the_agent_not_the_file():
    assert A.spec_label(A.parse_spec("bundle:arena/agents/champ-h10/main.py")) == "champ-h10"
    assert A.spec_label(A.parse_spec("champion")) == "champion"


def test_pick_games_spreads_across_bands():
    games = A.pick_games(6)
    assert len(games) == 6
    assert sorted(b for _, _, b in games) == [0, 1, 2, 3, 4, 5]
    for path, side, _ in games:
        assert os.path.exists(path) and side in (0, 1)


# -------------------------------------------------------------------- gaps
def test_gap_closed_is_a_share_of_the_reference_to_leader_distance():
    ref = {"moves_per_productive": 1.4, "chores_per_visit": 1.4}
    leaders = {"a": {"moves_per_productive": 0.7, "chores_per_visit": 2.1}}
    # exactly halfway on both, in the right direction for each
    agent = {"moves_per_productive": 1.05, "chores_per_visit": 1.75}
    frac, detail = A.gap_closed(agent, ref, leaders, ["moves_per_productive", "chores_per_visit"])
    assert frac == pytest.approx(0.5, abs=1e-6)
    assert len(detail) == 2
    # level with the reference is 0.0, level with the leader is 1.0
    assert A.gap_closed(ref, ref, leaders, ["moves_per_productive"])[0] == pytest.approx(0.0)
    assert A.gap_closed(leaders["a"], ref, leaders, ["moves_per_productive"])[0] == pytest.approx(1.0)
    # moving the WRONG way is negative even though the raw number went up
    worse = {"moves_per_productive": 1.75}
    assert A.gap_closed(worse, ref, leaders, ["moves_per_productive"])[0] < 0


def test_gap_picks_the_best_leader_per_metric_and_skips_unknowns():
    ref = {"pickups_per_feed": 1.0}
    leaders = {"a": {"pickups_per_feed": 0.8}, "b": {"pickups_per_feed": 0.5}}
    # lower is better, so the target is 0.5, and 0.75 is half the way there
    assert A.gap_closed({"pickups_per_feed": 0.75}, ref, leaders, ["pickups_per_feed"])[0] \
        == pytest.approx(0.5)
    # a metric with no direction contributes nothing
    assert A.gap_closed({"eval_games": 9}, {"eval_games": 1}, leaders, ["eval_games"])[0] is None


# ------------------------------------------------------- synthetic metrics
def _tile_plant(crop_idx, age, held=0, watered=1):
    return [4, crop_idx, age, held, watered, 0]


def _synthetic_trace(path, sell_price=100.0, sell_units=10, base_wheat_day=6):
    """Two days, one unit, a wheat tile and a cow, one SELL a day."""
    steps = []
    for day in (base_wheat_day, base_wheat_day + 6):
        for hour in range(24):
            tiles = [[1] * 10 for _ in range(10)]
            tiles[0][0] = _tile_plant(0, 3, held=2)            # wheat
            tiles[1][1] = [5, 1, 1, 1, 0, 2]                   # a fed, cared cow
            acts = {"farmer": ["WATER"] if hour else ["HARVEST"], "hands": [],
                    "market": [["SELL", "WHEAT", sell_units]] if hour == 2 else []}
            steps.append({
                "t": len(steps), "day": day, "hour": hour,
                "me": {"tiles": tiles, "farmer": [0, 0], "hands": [], "money": 1000.0,
                       "quadrants": ["NW"] if day == base_wheat_day else ["NW", "NE"],
                       "counts": {}},
                "opp": {"tiles": tiles, "farmer": [9, 9], "hands": [], "money": 900.0,
                        "quadrants": ["NW"], "counts": {}},
                "market": {"prices": [sell_price] * 9, "inv": [10000] * 9,
                           "shops": ["BAKERY"]},
                "shed": {}, "seeds": {},
                "act": {"me": acts, "opp": {"farmer": ["PASS"], "hands": [], "market": []}},
            })
    trace = {"label": "synthetic", "session": "s", "seed": 1, "our_seat": 0, "tape_seat": 1,
             "final": {"me": 5000.0, "opp": 4000.0, "win": True, "margin": 1000.0},
             "products": A.PRODUCTS_L, "crops": A.CROPS_L, "animals": A.ANIMALS_L,
             "base_prices": [1.0] * 9, "plans": {}, "steps": steps}
    with open(path, "w") as fh:
        json.dump(trace, fh)
    return path


def test_metric_groups_assemble_from_a_trace(tmp_path):
    p = _synthetic_trace(str(tmp_path / "t.json"))
    row = A.build_row([p])
    # outcome-ish numbers come straight off the trace
    assert row["final_money"] == 5000.0 and row["margin"] == 1000.0
    assert row["trace_win_rate"] == 1.0
    # executor group
    assert row["harvest_ops"] == 2 and row["moves_per_productive"] == 0.0
    # market group: two SELL orders of 10 units each
    assert row["units_offered"] == 20
    assert 0.0 <= row["demand_satisfied_share"] <= 1.0
    # planner group: the second quadrant is bought between the two days
    assert row["quadrant2_day"] == 12
    assert row["hands_peak"] == 0


def test_crashed_sale_share_counts_selling_into_a_collapse(tmp_path):
    from kagent.constants import MARKET_PARAMS
    base = MARKET_PARAMS["WHEAT"]["base"]
    healthy = A.build_row([_synthetic_trace(str(tmp_path / "ok.json"), sell_price=base)])
    crashed = A.build_row([_synthetic_trace(str(tmp_path / "bad.json"),
                                            sell_price=base * 0.1)])
    assert healthy["crashed_sale_share"] == 0.0
    assert crashed["crashed_sale_share"] == 1.0


def test_outcome_reads_an_existing_paired_eval():
    row = A.outcome_metrics("rl/runs/draw995/batch2.json:ship2_hv")
    assert row["eval_games"] == 995
    assert row["win_rate"] == pytest.approx(0.354, abs=0.002)
    assert all(f"win_band{b}" in row for b in range(6))


def test_toward_target_metrics_close_from_either_side():
    # a crop count has no better/worse direction: the leaders' mean is a
    # target, and overshooting it is just as far away as undershooting
    ref = {"day20_strawberry": 40.0}
    leaders = {"a": {"day20_strawberry": 20.0}, "b": {"day20_strawberry": 24.0}}
    target = 22.0
    assert A._is_toward("day20_strawberry")
    assert A.gap_closed({"day20_strawberry": 31.0}, ref, leaders, ["day20_strawberry"])[0] \
        == pytest.approx(0.5)
    assert A.gap_closed({"day20_strawberry": target}, ref, leaders, ["day20_strawberry"])[0] \
        == pytest.approx(1.0)
    # overshooting past the target loses ground again
    assert A.gap_closed({"day20_strawberry": 4.0}, ref, leaders, ["day20_strawberry"])[0] < 1.0
