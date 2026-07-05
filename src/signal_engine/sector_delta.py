"""
Segnale "delta per settore": la base numerica per qualsiasi consiglio tipo
"perdi tempo in T4". Questo e' il numero che il grounding check (vedi
evaluation.registry.log_grounding_check) confronta con quello che l'AI scrive
nel commento, per intercettare eventuali allucinazioni.
"""
from __future__ import annotations

import pandas as pd


def sector_deltas(
    driver_sectors: pd.Series,  # indice = nome/numero settore, valore = tempo (s)
    rival_sectors: pd.Series,
) -> dict:
    """
    Ritorna, per ogni settore in comune tra i due piloti, il delta
    (positivo = il pilota e' piu' lento del rivale in quel settore).
    """
    common = driver_sectors.index.intersection(rival_sectors.index)
    deltas = {}
    for sector in common:
        deltas[sector] = round(float(driver_sectors[sector] - rival_sectors[sector]), 3)
    return deltas


def worst_sector(deltas: dict) -> tuple[str, float] | None:
    """Il settore dove si perde di piu', utile come punto di partenza per
    il commento del capo ingegnere invece di elencare tutti i settori."""
    if not deltas:
        return None
    worst_key = max(deltas, key=lambda k: deltas[k])
    return worst_key, deltas[worst_key]
