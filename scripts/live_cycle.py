"""
Ciclo automatico del capo ingegnere. Pensato per essere lanciato a intervalli
regolari (cron GitHub Actions). Se non c'e' nessuna sessione F1 in corso in
questo momento, non fa nulla e termina — questo e' il comportamento normale
per la stragrande maggioranza delle esecuzioni, non un errore.

Semplificazioni deliberate (documentate, da migliorare in futuro):
- il rivale di ogni pilota e' sempre "il pilota davanti in classifica",
  determinato dalla posizione piu' recente disponibile
- il leader di gara non ha un rivale davanti: viene escluso da questo ciclo
- il gap resta l'approssimazione cumulata vista nel backfill, non ancora
  l'endpoint "intervals" di OpenF1 (piu' preciso ma piu' complesso)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_layer.client import HuggingFaceLLMClient
from ai_layer.orchestrator import EngineerCycleError, run_engineer_cycle
from evaluation.registry import ClaimRegistry
from evaluation.storage import SupabaseClaimStorage
from ingestion.lap_transform import build_gap_series, laps_to_series
from ingestion.openf1_client import (
    get_current_or_upcoming_session,
    get_drivers,
    get_laps,
    get_latest_positions,
    get_latest_weather,
)
from ingestion.supabase_writer import SupabaseWriter
from signal_engine.snapshot import build_driver_snapshot


def build_snapshots_for_session(session_key: int) -> tuple[list[dict], int]:
    """Ritorna (lista di snapshot, giro corrente). Salta i piloti senza dati
    sufficienti (es. ritirati, o senza rivale davanti in classifica)."""
    positions = get_latest_positions(session_key)
    if not positions:
        return [], 0

    # ordina i piloti per posizione, cosi' possiamo abbinare ognuno al
    # pilota immediatamente davanti
    ordered = sorted(positions.items(), key=lambda item: item[1])  # (driver_number, position)

    snapshots = []
    current_lap = 0

    for i in range(1, len(ordered)):  # salta il primo (il leader, nessun rivale davanti)
        driver_number, _ = ordered[i]
        rival_number, _ = ordered[i - 1]

        driver_laps = get_laps(session_key, driver_number)
        rival_laps = get_laps(session_key, rival_number)
        if not driver_laps or not rival_laps:
            continue

        current_lap = max(current_lap, max(lap["lap_number"] for lap in driver_laps))

        gap_series = build_gap_series(driver_laps, rival_laps)
        stint_lap_times, driver_sectors = laps_to_series(driver_laps)
        _, rival_sectors = laps_to_series(rival_laps)

        if gap_series.empty or driver_sectors.empty or rival_sectors.empty:
            continue

        snapshot = build_driver_snapshot(
            driver=str(driver_number),
            rival=str(rival_number),
            gap_series=gap_series,
            stint_lap_times=stint_lap_times.tail(6),
            driver_sectors=driver_sectors,
            rival_sectors=rival_sectors,
        )
        snapshots.append(snapshot)

    return snapshots, current_lap


def run() -> None:
    print("Controllo se c'e' una sessione F1 in corso...")
    session = get_current_or_upcoming_session()
    if session is None:
        print("Nessuna sessione live in questo momento. Esco senza fare nulla.")
        return

    session_key = session["session_key"]
    circuit = session.get("circuit_short_name", "sconosciuto")
    session_type = session.get("session_type", "sconosciuto")
    print(f"Sessione live trovata: {circuit} - {session_type} (session_key={session_key})")

    drivers = get_drivers(session_key)
    if len(drivers) < 2:
        print("Meno di due piloti con dati disponibili, esco.")
        return

    print("Calcolo gli snapshot per tutti i piloti...")
    snapshots, current_lap = build_snapshots_for_session(session_key)
    if not snapshots:
        print("Nessuno snapshot calcolabile in questo momento (sessione appena iniziata?), esco.")
        return
    print(f"{len(snapshots)} snapshot calcolati, giro corrente stimato: {current_lap}")

    writer = SupabaseWriter()
    weather = get_latest_weather(session_key) or {}

    writer.insert("session_context", {
        "circuit": circuit,
        "session_type": session_type,
        "current_lap": current_lap,
        "total_laps": current_lap,  # OpenF1 non da' il totale gara direttamente
        "air_temp": weather.get("air_temp"),
        "track_temp": weather.get("track_temp"),
        "rain_probability": weather.get("rain_probability"),
    })

    for snapshot in snapshots:
        writer.insert("driver_snapshots", {
            "driver": snapshot["driver"],
            "rival": snapshot["rival"],
            "created_at_lap": current_lap,
            "current_gap_seconds": snapshot["gap"]["current_gap_seconds"],
            "gap_slope_seconds_per_lap": snapshot["gap"]["slope_seconds_per_lap"],
            "degradation_seconds_per_lap": snapshot["tire_degradation"]["degradation_seconds_per_lap"],
            "estimated_laps_to_cliff": snapshot["estimated_laps_to_cliff"],
            "sector_deltas": snapshot["sector_deltas"],
            "worst_sector": snapshot["worst_sector"]["sector"] if snapshot["worst_sector"] else None,
            "worst_sector_delta": snapshot["worst_sector"]["delta_seconds"] if snapshot["worst_sector"] else None,
        })

    print("Chiamo il capo ingegnere AI (una chiamata per tutti i piloti insieme)...")
    registry = ClaimRegistry(SupabaseClaimStorage())
    client = HuggingFaceLLMClient()

    try:
        result = run_engineer_cycle(
            snapshots=snapshots,
            created_at_lap=current_lap,
            session_key=str(session_key),
            registry=registry,
            llm_client=client,
        )
    except EngineerCycleError as exc:
        print(f"ERRORE nel ciclo AI: {exc}")
        raise

    for driver_result in result["results"]:
        writer.write_engineer_analysis(driver_result, created_at_lap=current_lap)

    print(f"Ciclo completato: {len(result['results'])} analisi scritte.")


if __name__ == "__main__":
    run()
