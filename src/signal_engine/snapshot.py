"""
Combina i segnali dei singoli moduli in un unico "snapshot" per pilota.
E' questo oggetto (in forma di dict, serializzabile a JSON) che finisce nel
prompt del capo ingegnere: l'AI non vede mai dati grezzi, vede solo numeri
gia' calcolati qui, deterministicamente.
"""
from __future__ import annotations

import pandas as pd

from .degradation import current_degradation_rate, estimate_cliff_horizon
from .gap_trend import gap_trend_slope
from .sector_delta import sector_deltas, worst_sector


def build_driver_snapshot(
    driver: str,
    rival: str,
    gap_series: pd.Series,
    stint_lap_times: pd.Series,
    driver_sectors: pd.Series,
    rival_sectors: pd.Series,
) -> dict:
    gap_info = gap_trend_slope(gap_series)
    degradation_info = current_degradation_rate(stint_lap_times)
    cliff_horizon = estimate_cliff_horizon(
        current_lap_time=float(stint_lap_times.dropna().iloc[-1]) if len(stint_lap_times.dropna()) else None,
        degradation_seconds_per_lap=degradation_info["degradation_seconds_per_lap"],
    )
    deltas = sector_deltas(driver_sectors, rival_sectors)
    worst = worst_sector(deltas)

    return {
        "driver": driver,
        "rival": rival,
        "gap": gap_info,
        "tire_degradation": degradation_info,
        "estimated_laps_to_cliff": cliff_horizon,
        "sector_deltas": deltas,
        "worst_sector": {"sector": worst[0], "delta_seconds": worst[1]} if worst else None,
    }
