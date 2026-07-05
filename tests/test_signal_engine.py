import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signal_engine.degradation import current_degradation_rate, estimate_cliff_horizon
from signal_engine.gap_trend import gap_trend_slope, project_gap
from signal_engine.sector_delta import sector_deltas, worst_sector
from signal_engine.snapshot import build_driver_snapshot


def test_gap_trend_closing():
    # gap che si chiude di circa 0.1s a giro
    gap_series = pd.Series({10: 2.0, 11: 1.9, 12: 1.8, 13: 1.7, 14: 1.6})
    result = gap_trend_slope(gap_series, window_laps=5)
    assert result["slope_seconds_per_lap"] == pytest.approx(-0.1, abs=0.01)
    assert result["current_gap_seconds"] == 1.6


def test_gap_trend_insufficient_data():
    gap_series = pd.Series({10: 2.0})
    result = gap_trend_slope(gap_series)
    assert result["slope_seconds_per_lap"] is None


def test_project_gap():
    assert project_gap(current_gap=1.6, slope_per_lap=-0.1, horizon_laps=5) == pytest.approx(1.1)


def test_degradation_rate_with_fuel_correction():
    # tempi che peggiorano di 0.15s/giro "grezzi", ma parte di quello e' carburante
    lap_times = pd.Series({1: 90.0, 2: 90.15, 3: 90.30, 4: 90.45})
    result = current_degradation_rate(lap_times, window_laps=4, fuel_correction_per_lap=0.055)
    # il degrado "vero" dovrebbe essere piu' alto del grezzo, perche' il carburante
    # nasconde parte del peggioramento
    assert result["degradation_seconds_per_lap"] > 0.15


def test_degradation_insufficient_data():
    lap_times = pd.Series({1: 90.0, 2: 90.2})
    result = current_degradation_rate(lap_times)
    assert result["degradation_seconds_per_lap"] is None


def test_estimate_cliff_horizon_within_range():
    horizon = estimate_cliff_horizon(
        current_lap_time=91.0, degradation_seconds_per_lap=0.2, cliff_threshold_seconds=1.0
    )
    assert horizon == 5  # 1.0 / 0.2 = 5 giri


def test_estimate_cliff_horizon_out_of_range():
    horizon = estimate_cliff_horizon(
        current_lap_time=91.0, degradation_seconds_per_lap=0.01, cliff_threshold_seconds=1.0, max_horizon_laps=15
    )
    assert horizon is None  # ci vorrebbero 100 giri, fuori dall'orizzonte utile


def test_sector_deltas_and_worst():
    driver = pd.Series({"S1": 28.1, "S2": 31.5, "S3": 22.0})
    rival = pd.Series({"S1": 28.0, "S2": 31.2, "S3": 22.1})
    deltas = sector_deltas(driver, rival)
    assert deltas["S2"] == pytest.approx(0.3)
    worst = worst_sector(deltas)
    assert worst[0] == "S2"


def test_build_driver_snapshot_end_to_end():
    gap_series = pd.Series({10: 2.0, 11: 1.9, 12: 1.8, 13: 1.7, 14: 1.6})
    stint_lap_times = pd.Series({1: 90.0, 2: 90.3, 3: 90.6, 4: 90.9})
    driver_sectors = pd.Series({"S1": 28.1, "S2": 31.5, "S3": 22.0})
    rival_sectors = pd.Series({"S1": 28.0, "S2": 31.2, "S3": 22.1})

    snapshot = build_driver_snapshot(
        driver="VER", rival="NOR",
        gap_series=gap_series, stint_lap_times=stint_lap_times,
        driver_sectors=driver_sectors, rival_sectors=rival_sectors,
    )

    assert snapshot["driver"] == "VER"
    assert snapshot["gap"]["slope_seconds_per_lap"] < 0
    assert snapshot["worst_sector"]["sector"] == "S2"
    assert "estimated_laps_to_cliff" in snapshot
