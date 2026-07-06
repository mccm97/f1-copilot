"""
Test end-to-end del layer AI con dati reali:
1. Legge l'ultimo snapshot scritto su Supabase (dal test di backfill precedente)
2. Lo passa al capo ingegnere vero (HuggingFace), non piu' al MockLLMClient
3. Esegue il grounding check sulla risposta reale del modello
4. Scrive il risultato in engineer_analyses, cosi' appare nel feed muretto
   della dashboard

I claim (gap_trend, tire_cliff, grounding_check) restano per ora in un file
locale nel runner GitHub Actions: e' un test dell'AI layer, non ancora la
persistenza definitiva dei claim (quella verra' migrata su Supabase quando
costruiremo il worker live).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_layer.client import HuggingFaceLLMClient
from ai_layer.orchestrator import EngineerCycleError, run_engineer_cycle
from evaluation.registry import ClaimRegistry
from evaluation.storage import LocalJSONClaimStorage
from ingestion.supabase_writer import SupabaseWriter


def row_to_snapshot(row: dict) -> dict:
    """Ricostruisce la forma di snapshot attesa dall'AI layer a partire
    dalla riga salvata su Supabase."""
    return {
        "driver": row["driver"],
        "rival": row["rival"],
        "gap": {
            "current_gap_seconds": row["current_gap_seconds"],
            "slope_seconds_per_lap": row["gap_slope_seconds_per_lap"],
        },
        "tire_degradation": {
            "degradation_seconds_per_lap": row["degradation_seconds_per_lap"],
        },
        "estimated_laps_to_cliff": row["estimated_laps_to_cliff"],
        "sector_deltas": row["sector_deltas"],
        "worst_sector": (
            {"sector": row["worst_sector"], "delta_seconds": row["worst_sector_delta"]}
            if row.get("worst_sector")
            else None
        ),
    }


def main() -> None:
    writer = SupabaseWriter()

    print("Leggo l'ultimo snapshot da Supabase...")
    rows = writer.select_latest("driver_snapshots", order_by="created_at_lap", limit=1)
    if not rows:
        raise RuntimeError("Nessuno snapshot trovato in driver_snapshots: esegui prima il backfill")

    row = rows[0]
    snapshot = row_to_snapshot(row)
    created_at_lap = row["created_at_lap"]
    print("Snapshot letto:", json.dumps(snapshot, indent=2))

    registry = ClaimRegistry(LocalJSONClaimStorage("ai_test_claims.json"))
    client = HuggingFaceLLMClient()  # usa HF_TOKEN dall'ambiente

    print("Chiamo il capo ingegnere AI (HuggingFace)...")
    try:
        result = run_engineer_cycle(
            snapshots=[snapshot],
            created_at_lap=created_at_lap,
            session_key="test_ai_layer_live",
            registry=registry,
            llm_client=client,
        )
    except EngineerCycleError as exc:
        print(f"ERRORE: il modello non ha risposto in modo valido: {exc}")
        raise

    print("Risultato:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("Scrivo l'analisi su Supabase (engineer_analyses)...")
    for driver_result in result["results"]:
        analysis = driver_result["analysis"]
        grounding_checks = driver_result["grounding_checks"]

        cited_deltas = []
        for cited, check in zip(analysis.get("cited_deltas", []), grounding_checks):
            cited_deltas.append({
                "metric_name": cited["metric_name"],
                "stated_value": cited["stated_value"],
                "grounding_passed": check["passed"],
            })

        writer.insert("engineer_analyses", {
            "driver": analysis["driver"],
            "created_at_lap": result["created_at_lap"],
            "summary": analysis["summary"],
            "doing_well": analysis.get("doing_well", []),
            "mistakes": analysis.get("mistakes", []),
            "cited_deltas": cited_deltas,
            "gap_trend_prediction": analysis.get("gap_trend_prediction"),
            "tire_cliff_prediction": analysis.get("tire_cliff_prediction"),
        })

    print("Fatto. Controlla il feed muretto sulla dashboard.")


if __name__ == "__main__":
    main()
