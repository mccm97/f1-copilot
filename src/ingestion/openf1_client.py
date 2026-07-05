"""
Client minimale per OpenF1 (https://api.openf1.org). Nessuna chiave richiesta
per i dati storici. Ogni funzione ritorna dati "grezzi" cosi' come li da'
l'API: la trasformazione in segnali resta compito del signal engine, non di
questo modulo.
"""
from __future__ import annotations

import requests

BASE_URL = "https://api.openf1.org/v1"


def get_session_info(session_key: int) -> dict:
    resp = requests.get(f"{BASE_URL}/sessions", params={"session_key": session_key}, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise ValueError(f"Nessuna sessione trovata per session_key={session_key}")
    return data[0]


def get_laps(session_key: int, driver_number: int) -> list[dict]:
    resp = requests.get(
        f"{BASE_URL}/laps",
        params={"session_key": session_key, "driver_number": driver_number},
        timeout=20,
    )
    resp.raise_for_status()
    laps = resp.json()
    # scarta giri senza tempo valido (es. giro di rientro ai box, safety car)
    return [lap for lap in laps if lap.get("lap_duration")]
