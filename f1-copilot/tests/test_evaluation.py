import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.claims import ClaimStatus, InvalidatingEvent
from evaluation.registry import ClaimRegistry
from evaluation.scoring import detect_tire_cliff, score_gap_trend, score_tire_cliff
from evaluation.storage import LocalJSONClaimStorage


@pytest.fixture
def registry(tmp_path):
    storage = LocalJSONClaimStorage(tmp_path / "claims.json")
    return ClaimRegistry(storage)


def test_log_and_retrieve_gap_trend_claim(registry):
    claim = registry.log_gap_trend(
        session_key="2026_monza_race",
        driver="VER",
        rival="NOR",
        predicted_delta_seconds=-0.3,
        horizon_laps=5,
        created_at_lap=10,
        assumptions=["no_pit", "no_sc"],
    )
    assert claim.status == ClaimStatus.PENDING
    fetched = registry.storage.get(claim.id)
    assert fetched.rival == "NOR"
    assert fetched.horizon_laps == 5


def test_void_claim(registry):
    claim = registry.log_gap_trend(
        session_key="2026_monza_race",
        driver="VER",
        rival="NOR",
        predicted_delta_seconds=-0.3,
        horizon_laps=5,
        created_at_lap=10,
    )
    voided = registry.void_claim(claim.id, InvalidatingEvent.SAFETY_CAR)
    assert voided.status == ClaimStatus.VOIDED
    assert voided.voided_reason == InvalidatingEvent.SAFETY_CAR


def test_grounding_check_pass_and_fail(registry):
    ok = registry.log_grounding_check(
        session_key="2026_monza_race",
        driver="VER",
        metric_name="delta_frenata_T4",
        stated_value=0.15,
        computed_value=0.16,
        created_at_lap=12,
    )
    assert ok.score["passed"] is True

    hallucination = registry.log_grounding_check(
        session_key="2026_monza_race",
        driver="VER",
        metric_name="delta_frenata_T4",
        stated_value=0.50,
        computed_value=0.10,
        created_at_lap=12,
    )
    assert hallucination.score["passed"] is False


def test_score_gap_trend_direction_correct():
    claim_data = dict(
        session_key="s",
        driver="VER",
        rival="NOR",
        predicted_delta_seconds=-0.3,
        horizon_laps=5,
        created_at_lap=10,
    )
    from evaluation.claims import GapTrendClaim

    claim = GapTrendClaim(**claim_data)
    gap_series = pd.Series({10: 2.0, 15: 1.6})  # si e' avvicinato di 0.4s, previsti -0.3
    result = score_gap_trend(claim, gap_series)
    assert result["direction_correct"] is True
    assert result["abs_error_seconds"] == pytest.approx(0.1, abs=1e-6)


def test_score_gap_trend_wrong_direction():
    from evaluation.claims import GapTrendClaim

    claim = GapTrendClaim(
        session_key="s", driver="VER", rival="NOR",
        predicted_delta_seconds=-0.3, horizon_laps=5, created_at_lap=10,
    )
    gap_series = pd.Series({10: 2.0, 15: 2.5})  # in realta' si e' allontanato
    result = score_gap_trend(claim, gap_series)
    assert result["direction_correct"] is False


def test_detect_tire_cliff_finds_dropoff():
    # 4 giri normali di degrado lineare, poi crollo prestazionale evidente
    normal = [90.0, 90.3, 90.6, 90.9]
    cliff = [92.5, 93.2, 93.8]
    lap_times = pd.Series(normal + cliff, index=range(1, len(normal) + len(cliff) + 1))
    cliff_lap = detect_tire_cliff(lap_times, min_laps_for_trend=4, consecutive_laps_required=2)
    assert cliff_lap == 5


def test_detect_tire_cliff_no_cliff_in_stable_stint():
    stable = [90.0, 90.2, 90.4, 90.6, 90.8, 91.0]
    lap_times = pd.Series(stable, index=range(1, len(stable) + 1))
    cliff_lap = detect_tire_cliff(lap_times, min_laps_for_trend=4, consecutive_laps_required=2)
    assert cliff_lap is None


def test_score_tire_cliff_voids_on_early_pit():
    from evaluation.claims import TireCliffClaim

    claim = TireCliffClaim(
        session_key="s", driver="VER", compound="MEDIUM", stint_number=1,
        predicted_cliff_lap_min=18, predicted_cliff_lap_max=22, created_at_lap=5,
    )
    lap_times = pd.Series([90.0, 90.2], index=[1, 2])
    result = score_tire_cliff(claim, lap_times, pitted_early_before_lap=15)
    assert result is None
