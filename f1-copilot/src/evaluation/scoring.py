"""
Funzioni di scoring, deterministiche: prendono un claim gia' registrato piu'
i dati reali della sessione (gia' calcolati dal signal engine, mai dall'AI)
e restituiscono un punteggio o un motivo di annullamento.

Principio guida: l'AI non giudica mai se stessa. Il "cliff" delle gomme,
per esempio, viene rilevato qui con una regola matematica esplicita,
non chiedendo a un modello linguistico se il grafico "sembra" un cliff.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .claims import GapTrendClaim, TireCliffClaim


def score_gap_trend(
    claim: GapTrendClaim,
    gap_series: pd.Series,  # indice = numero giro, valore = gap in secondi dal rivale
) -> dict:
    """
    Confronta il gap previsto con quello osservato all'orizzonte previsto.
    gap_series deve coprire almeno da created_at_lap a created_at_lap + horizon_laps.
    """
    target_lap = claim.created_at_lap + claim.horizon_laps
    if target_lap not in gap_series.index or claim.created_at_lap not in gap_series.index:
        raise ValueError(
            f"gap_series non copre l'orizzonte richiesto (giro {claim.created_at_lap} -> {target_lap})"
        )

    starting_gap = gap_series.loc[claim.created_at_lap]
    actual_gap = gap_series.loc[target_lap]
    actual_delta = actual_gap - starting_gap

    predicted_delta = claim.predicted_delta_seconds
    abs_error = abs(actual_delta - predicted_delta)

    # "direzione giusta": segno del delta previsto coincide con quello osservato
    # (tolleranza: se il delta reale e' quasi zero, non penalizziamo la direzione)
    if abs(actual_delta) < 0.02:
        direction_correct = abs(predicted_delta) < 0.1
    else:
        direction_correct = np.sign(actual_delta) == np.sign(predicted_delta)

    return {
        "predicted_delta_seconds": predicted_delta,
        "actual_delta_seconds": round(actual_delta, 3),
        "abs_error_seconds": round(abs_error, 3),
        "direction_correct": bool(direction_correct),
    }


def detect_tire_cliff(
    lap_times: pd.Series,  # indice = numero giro all'interno dello stint, valore = tempo sul giro (s)
    min_laps_for_trend: int = 3,
    residual_threshold_sigma: float = 2.0,
    consecutive_laps_required: int = 2,
) -> int | None:
    """
    Rilevamento deterministico del "cliff": fitta un trend lineare di degrado
    sui primi giri dello stint, poi cerca il primo punto in cui il residuo
    (tempo reale - tempo atteso dal trend) supera una soglia per N giri di fila.
    Ritorna il numero di giro (relativo allo stint) del cliff, o None se non rilevato.

    Semplice e trasparente di proposito: e' una baseline, va validata su dati
    reali e i parametri vanno tarati, ma la logica resta sempre esplicita e
    ispezionabile, mai delegata al giudizio del modello linguistico.
    """
    laps = lap_times.index.to_numpy()
    times = lap_times.to_numpy()

    if len(laps) < min_laps_for_trend + consecutive_laps_required:
        return None

    # trend lineare sui primi min_laps_for_trend giri (degrado "normale" atteso)
    trend_laps = laps[:min_laps_for_trend]
    trend_times = times[:min_laps_for_trend]
    slope, intercept = np.polyfit(trend_laps, trend_times, 1)

    residual_std = np.std(trend_times - (slope * trend_laps + intercept))
    residual_std = max(residual_std, 0.05)  # evita soglie assurdamente strette

    expected = slope * laps + intercept
    residuals = times - expected

    threshold = residual_threshold_sigma * residual_std
    over_threshold = residuals > threshold

    for i in range(len(over_threshold) - consecutive_laps_required + 1):
        if all(over_threshold[i : i + consecutive_laps_required]):
            return int(laps[i])
    return None


def score_tire_cliff(
    claim: TireCliffClaim,
    lap_times: pd.Series,
    pitted_early_before_lap: int | None = None,
) -> dict | None:
    """
    Ritorna None se il claim va annullato (es. pit prima che l'orizzonte
    previsto potesse verificarsi), altrimenti un dizionario di punteggio.
    """
    if pitted_early_before_lap is not None and pitted_early_before_lap <= claim.predicted_cliff_lap_max:
        return None  # da annullare con InvalidatingEvent.UNPLANNED_PIT

    actual_cliff_lap = detect_tire_cliff(lap_times)

    if actual_cliff_lap is None:
        return {
            "predicted_range": [claim.predicted_cliff_lap_min, claim.predicted_cliff_lap_max],
            "actual_cliff_lap": None,
            "within_predicted_range": False,
            "note": "nessun cliff rilevato nello stint osservato",
        }

    within_range = claim.predicted_cliff_lap_min <= actual_cliff_lap <= claim.predicted_cliff_lap_max
    if actual_cliff_lap < claim.predicted_cliff_lap_min:
        error_laps = claim.predicted_cliff_lap_min - actual_cliff_lap
    elif actual_cliff_lap > claim.predicted_cliff_lap_max:
        error_laps = actual_cliff_lap - claim.predicted_cliff_lap_max
    else:
        error_laps = 0

    return {
        "predicted_range": [claim.predicted_cliff_lap_min, claim.predicted_cliff_lap_max],
        "actual_cliff_lap": actual_cliff_lap,
        "within_predicted_range": within_range,
        "error_laps": error_laps,
    }
