"""
Segnale "gap trend": non solo il gap attuale, ma con che velocita' si sta
aprendo o chiudendo. Questo e' il numero che alimenta sia il prompt del
capo ingegnere (per formulare una previsione) sia lo scoring a posteriori
(evaluation.scoring.score_gap_trend legge lo stesso tipo di serie).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def gap_trend_slope(
    gap_series: pd.Series,  # indice = numero giro, valore = gap in secondi dal rivale
    window_laps: int = 5,
) -> dict:
    """
    Calcola il trend recente del gap usando una regressione lineare sugli
    ultimi `window_laps` giri disponibili. Uno slope negativo significa che
    il gap si sta chiudendo (ci si avvicina), positivo che si allarga.
    """
    recent = gap_series.dropna().iloc[-window_laps:]
    if len(recent) < 2:
        return {
            "current_gap_seconds": float(gap_series.dropna().iloc[-1]) if len(gap_series.dropna()) else None,
            "slope_seconds_per_lap": None,
            "laps_used": len(recent),
            "note": "dati insufficienti per calcolare un trend affidabile",
        }

    laps = recent.index.to_numpy(dtype=float)
    values = recent.to_numpy(dtype=float)
    slope, intercept = np.polyfit(laps, values, 1)

    return {
        "current_gap_seconds": round(float(values[-1]), 3),
        "slope_seconds_per_lap": round(float(slope), 4),
        "laps_used": len(recent),
    }


def project_gap(current_gap: float, slope_per_lap: float, horizon_laps: int) -> float:
    """Proietta il gap futuro assumendo che il trend recente continui invariato.
    Usato solo come base di partenza per una previsione, mai spacciato per
    previsione definitiva: il capo ingegnere deve comunque dichiarare le
    assunzioni (no_pit, no_sc...) nel claim che genera da qui."""
    return round(current_gap + slope_per_lap * horizon_laps, 3)
