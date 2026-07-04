"""
Segnale "degrado in corso": a differenza di evaluation.scoring.detect_tire_cliff
(che verifica A POSTERIORI se un cliff e' avvenuto, con dati completi dello
stint), questo modulo stima IN CORSA la velocita' di degrado attuale, per
alimentare previsioni sul quando arrivera' il cliff.

Sono due funzioni deliberatamente separate anche se simili: una guarda
indietro per giudicare, una guarda avanti per prevedere. Mescolarle
porterebbe a validare le previsioni con lo stesso codice che le genera.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def current_degradation_rate(
    lap_times: pd.Series,  # indice = giro nello stint, valore = tempo sul giro (s)
    window_laps: int = 4,
    fuel_correction_per_lap: float = 0.055,  # secondi guadagnati a giro per calo carburante, valore tipico
) -> dict:
    """
    Regressione lineare sugli ultimi `window_laps` giri disponibili, DOPO aver
    corretto per l'effetto carburante (altrimenti il calo di peso auto
    maschera parte del degrado reale, o viceversa lo esagera a inizio stint).
    """
    clean = lap_times.dropna()
    recent = clean.iloc[-window_laps:]
    if len(recent) < 3:
        return {
            "degradation_seconds_per_lap": None,
            "laps_used": len(recent),
            "note": "servono almeno 3 giri puliti per una stima affidabile",
        }

    laps = recent.index.to_numpy(dtype=float)
    # correggo il tempo per l'effetto carburante: mano a mano che il giro
    # avanza nello stint l'auto e' piu' leggera, quindi "aggiungo" il tempo
    # che il carburante le ha fatto guadagnare per isolare il degrado puro
    corrected_times = recent.to_numpy(dtype=float) + laps * fuel_correction_per_lap

    slope, _ = np.polyfit(laps, corrected_times, 1)

    return {
        "degradation_seconds_per_lap": round(float(slope), 4),
        "laps_used": len(recent),
        "fuel_correction_applied": fuel_correction_per_lap,
    }


def estimate_cliff_horizon(
    current_lap_time: float,
    degradation_seconds_per_lap: float,
    cliff_threshold_seconds: float = 1.0,
    max_horizon_laps: int = 15,
) -> int | None:
    """
    Stima grezza: quanti giri mancano prima che il degrado cumulato superi
    una soglia critica, assumendo che il tasso di degrado attuale continui
    linearmente (assunzione forte, va dichiarata come tale nel claim: i cliff
    reali sono spesso non lineari, questo e' un punto di partenza, non oro colato).
    """
    if degradation_seconds_per_lap is None or degradation_seconds_per_lap <= 0:
        return None
    laps_to_cliff = cliff_threshold_seconds / degradation_seconds_per_lap
    if laps_to_cliff > max_horizon_laps:
        return None  # fuori dall'orizzonte in cui ha senso fare una previsione
    return int(np.ceil(laps_to_cliff))
