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


def get_current_or_upcoming_session() -> dict | None:
    """
    Cerca tra le sessioni dell'anno corrente quella in corso ADESSO
    (date_start <= ora <= date_end). Filtriamo lato Python su una lista
    non troppo grande (una stagione F1 ha circa 100-120 sessioni) invece di
    usare i filtri per data dell'API OpenF1, la cui sintassi esatta va
    verificata con attenzione: piu' robusto controllare qui.
    Ritorna None se nessuna sessione e' live in questo momento.
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    resp = requests.get(f"{BASE_URL}/sessions", params={"year": now.year}, timeout=20)
    resp.raise_for_status()

    for session in resp.json():
        try:
            start = datetime.fromisoformat(session["date_start"])
            end = datetime.fromisoformat(session["date_end"])
        except (KeyError, ValueError):
            continue
        if start <= now <= end:
            return session
    return None


def get_drivers(session_key: int) -> list[int]:
    resp = requests.get(f"{BASE_URL}/drivers", params={"session_key": session_key}, timeout=20)
    resp.raise_for_status()
    return [d["driver_number"] for d in resp.json() if d.get("driver_number") is not None]


def get_latest_positions(session_key: int) -> dict[int, int]:
    """Ritorna {driver_number: posizione} usando il campione di posizione
    piu' recente per ogni pilota."""
    resp = requests.get(f"{BASE_URL}/position", params={"session_key": session_key}, timeout=20)
    resp.raise_for_status()

    latest_by_driver: dict[int, dict] = {}
    for row in resp.json():
        driver = row.get("driver_number")
        if driver is None:
            continue
        if driver not in latest_by_driver or row["date"] > latest_by_driver[driver]["date"]:
            latest_by_driver[driver] = row

    return {driver: row["position"] for driver, row in latest_by_driver.items() if row.get("position")}


def get_latest_weather(session_key: int) -> dict | None:
    resp = requests.get(f"{BASE_URL}/weather", params={"session_key": session_key}, timeout=20)
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        return None
    latest = max(rows, key=lambda r: r["date"])
    return {
        "air_temp": latest.get("air_temperature"),
        "track_temp": latest.get("track_temperature"),
        # OpenF1 da' un valore di pioggia rilevata, non una probabilita' vera:
        # lo trattiamo come approssimazione, non come previsione meteo.
        "rain_probability": latest.get("rainfall"),
    }
