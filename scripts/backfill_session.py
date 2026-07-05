"""
Primo test end-to-end vero: prende una sessione F1 REALE gia' conclusa da
OpenF1, calcola i segnali con il signal engine, e scrive tutto in Supabase
cosi' la dashboard mostra dati veri.

Semplificazioni deliberate di questo primo script (da migliorare nei
prossimi passi, non e' ancora il worker live definitivo):
- il gap tra i due piloti e' approssimato come differenza cumulata dei tempi
  sul giro, non usa l'endpoint "intervals" di OpenF1 (piu' preciso ma piu'
  complesso da allineare nel tempo) — va bene per validare la pipeline,
  non e' ancora il numero definitivo da mostrare in produzione
- il degrado gomme viene calcolato sugli ultimi giri disponibili, assumendo
  che siano tutti nello stesso stint (niente rilevamento cambio gomma qui)

Uso (da riga di comando, o dal workflow GitHub Actions):
    python scripts/backfill_session.py --session-key 9161 --driver 1 --rival 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingestion.openf1_client import get_laps, get_session_info
from ingestion.supabase_writer import SupabaseWriter
from signal_engine.snapshot import build_driver_snapshot


def laps_to_series(laps: list[dict]) -> tuple[pd.Series, pd.Series]:
    """Ritorna (lap_times, sector_times_ultimo_giro)."""
    lap_times = pd.Series(
        {lap["lap_number"]: lap["lap_duration"] for lap in laps}
    )
    last_lap = laps[-1]
    sectors = pd.Series(
        {
            "S1": last_lap.get("duration_sector_1"),
            "S2": last_lap.get("duration_sector_2"),
            "S3": last_lap.get("duration_sector_3"),
        }
    ).dropna()
    return lap_times, sectors


def build_gap_series(driver_laps: list[dict], rival_laps: list[dict]) -> pd.Series:
    """Approssimazione: gap = tempo cumulato pilota - tempo cumulato rivale,
    per ogni numero di giro presente in entrambi."""
    driver_cum = {}
    total = 0.0
    for lap in driver_laps:
        total += lap["lap_duration"]
        driver_cum[lap["lap_number"]] = total

    rival_cum = {}
    total = 0.0
    for lap in rival_laps:
        total += lap["lap_duration"]
        rival_cum[lap["lap_number"]] = total

    common_laps = sorted(set(driver_cum) & set(rival_cum))
    return pd.Series({lap: driver_cum[lap] - rival_cum[lap] for lap in common_laps})


def run(session_key: int, driver_number: int, rival_number: int) -> None:
    print(f"Recupero sessione {session_key} da OpenF1...")
    session_info = get_session_info(session_key)
    circuit = session_info.get("circuit_short_name", "sconosciuto")
    session_type = session_info.get("session_type", "sconosciuto")

    print(f"Recupero giri pilota {driver_number} e rivale {rival_number}...")
    driver_laps = get_laps(session_key, driver_number)
    rival_laps = get_laps(session_key, rival_number)

    if not driver_laps or not rival_laps:
        raise RuntimeError("Nessun giro valido trovato per uno dei due piloti, controlla i numeri pilota")

    current_lap = max(lap["lap_number"] for lap in driver_laps)

    gap_series = build_gap_series(driver_laps, rival_laps)
    stint_lap_times, driver_sectors = laps_to_series(driver_laps)
    _, rival_sectors = laps_to_series(rival_laps)

    print("Calcolo lo snapshot dei segnali...")
    snapshot = build_driver_snapshot(
        driver=str(driver_number),
        rival=str(rival_number),
        gap_series=gap_series,
        stint_lap_times=stint_lap_times.tail(6),  # ultimi giri come "stint corrente"
        driver_sectors=driver_sectors,
        rival_sectors=rival_sectors,
    )
    print(snapshot)

    print("Scrivo su Supabase...")
    writer = SupabaseWriter()

    writer.insert("session_context", {
        "circuit": circuit,
        "session_type": session_type,
        "current_lap": int(current_lap),
        "total_laps": int(current_lap),  # OpenF1 non da' il totale gare direttamente qui
        "air_temp": None,
        "track_temp": None,
        "rain_probability": None,
    })

    writer.insert("driver_snapshots", {
        "driver": snapshot["driver"],
        "rival": snapshot["rival"],
        "created_at_lap": int(current_lap),
        "current_gap_seconds": snapshot["gap"]["current_gap_seconds"],
        "gap_slope_seconds_per_lap": snapshot["gap"]["slope_seconds_per_lap"],
        "degradation_seconds_per_lap": snapshot["tire_degradation"]["degradation_seconds_per_lap"],
        "estimated_laps_to_cliff": snapshot["estimated_laps_to_cliff"],
        "sector_deltas": snapshot["sector_deltas"],
        "worst_sector": snapshot["worst_sector"]["sector"] if snapshot["worst_sector"] else None,
        "worst_sector_delta": snapshot["worst_sector"]["delta_seconds"] if snapshot["worst_sector"] else None,
    })

    print("Fatto. Controlla la dashboard e le tabelle Supabase.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-key", type=int, required=True)
    parser.add_argument("--driver", type=int, required=True)
    parser.add_argument("--rival", type=int, required=True)
    args = parser.parse_args()
    run(args.session_key, args.driver, args.rival)
