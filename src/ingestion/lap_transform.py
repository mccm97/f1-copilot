"""
Trasformazioni condivise da dati OpenF1 grezzi (liste di giri) a pandas.Series
utilizzabili dal signal engine. Usato sia dallo script di backfill manuale
che dal worker automatico, per non duplicare la stessa logica due volte.
"""
from __future__ import annotations

import pandas as pd


def laps_to_series(laps: list[dict]) -> tuple[pd.Series, pd.Series]:
    """Ritorna (lap_times, sector_times_ultimo_giro)."""
    lap_times = pd.Series({lap["lap_number"]: lap["lap_duration"] for lap in laps})
    last_lap = laps[-1]
    sectors = pd.Series({
        "S1": last_lap.get("duration_sector_1"),
        "S2": last_lap.get("duration_sector_2"),
        "S3": last_lap.get("duration_sector_3"),
    }).dropna()
    return lap_times, sectors


def build_gap_series(driver_laps: list[dict], rival_laps: list[dict]) -> pd.Series:
    """Approssimazione: gap = tempo cumulato pilota - tempo cumulato rivale,
    per ogni numero di giro presente in entrambi. Da migliorare in futuro
    con l'endpoint 'intervals' di OpenF1 per maggiore precisione."""
    driver_cum, total = {}, 0.0
    for lap in driver_laps:
        total += lap["lap_duration"]
        driver_cum[lap["lap_number"]] = total

    rival_cum, total = {}, 0.0
    for lap in rival_laps:
        total += lap["lap_duration"]
        rival_cum[lap["lap_number"]] = total

    common_laps = sorted(set(driver_cum) & set(rival_cum))
    return pd.Series({lap: driver_cum[lap] - rival_cum[lap] for lap in common_laps})
