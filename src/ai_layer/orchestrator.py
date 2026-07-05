"""
run_engineer_cycle e' il punto di ingresso unico per un ciclo del capo
ingegnere: costruisce il prompt, chiama il modello, valida lo schema,
esegue il grounding check su ogni numero citato e registra le previsioni
verificabili (gap_trend, tire_cliff) nel ClaimRegistry.
"""
from __future__ import annotations

import json

from evaluation.registry import ClaimRegistry
from evaluation.claims import GroundingCheck

from .client import LLMClient
from .prompt import build_messages
from .schema import DriverAnalysis, EngineerCycleOutput


class EngineerCycleError(Exception):
    """L'output del modello non e' JSON valido o non rispetta lo schema."""


def _resolve_metric(metric_name: str, snapshot: dict) -> float | None:
    """Traduce 'sector:S2' nel valore corrispondente in snapshot['sector_deltas'].
    Ritorna None se il metric_name non e' riconosciuto: in quel caso il
    grounding check non puo' essere eseguito e va segnalato, non ignorato."""
    if metric_name.startswith("sector:"):
        sector_name = metric_name.split(":", 1)[1]
        return snapshot.get("sector_deltas", {}).get(sector_name)
    return None


def _process_driver_analysis(
    analysis: DriverAnalysis,
    snapshot: dict,
    session_key: str,
    created_at_lap: int,
    registry: ClaimRegistry,
) -> list[GroundingCheck]:
    grounding_results = []

    for cited in analysis.cited_deltas:
        computed_value = _resolve_metric(cited.metric_name, snapshot)
        if computed_value is None:
            # Metrica non risolvibile: non possiamo verificarla, lo trattiamo
            # come fallimento di grounding per non far passare numeri "orfani"
            computed_value = float("nan")
        check = registry.log_grounding_check(
            session_key=session_key,
            driver=analysis.driver,
            metric_name=cited.metric_name,
            stated_value=cited.stated_value,
            computed_value=computed_value if computed_value == computed_value else 0.0,
            created_at_lap=created_at_lap,
        )
        if computed_value != computed_value:  # era NaN: forziamo il fallimento
            check.score["passed"] = False
            check.score["note"] = "metrica non risolvibile nello snapshot"
        grounding_results.append(check)

    if analysis.gap_trend_prediction:
        pred = analysis.gap_trend_prediction
        registry.log_gap_trend(
            session_key=session_key,
            driver=analysis.driver,
            rival=pred.rival,
            predicted_delta_seconds=pred.predicted_delta_seconds,
            horizon_laps=pred.horizon_laps,
            created_at_lap=created_at_lap,
            assumptions=pred.assumptions,
        )

    if analysis.tire_cliff_prediction:
        pred = analysis.tire_cliff_prediction
        registry.log_tire_cliff(
            session_key=session_key,
            driver=analysis.driver,
            compound=pred.compound,
            stint_number=pred.stint_number,
            predicted_cliff_lap_min=pred.predicted_cliff_lap_min,
            predicted_cliff_lap_max=pred.predicted_cliff_lap_max,
            created_at_lap=created_at_lap,
        )

    return grounding_results


def run_engineer_cycle(
    snapshots: list[dict],
    created_at_lap: int,
    session_key: str,
    registry: ClaimRegistry,
    llm_client: LLMClient,
) -> dict:
    """
    Ritorna un dict con l'output validato del modello piu' i risultati di
    grounding per ogni pilota, pronto per essere scritto su Supabase e
    mostrato in dashboard.
    """
    messages = build_messages(snapshots, created_at_lap)
    raw_response = llm_client.generate(messages)

    try:
        parsed_json = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise EngineerCycleError(f"Risposta non e' JSON valido: {exc}") from exc

    try:
        output = EngineerCycleOutput.model_validate(parsed_json)
    except Exception as exc:  # pydantic ValidationError
        raise EngineerCycleError(f"Risposta non rispetta lo schema atteso: {exc}") from exc

    snapshots_by_driver = {s["driver"]: s for s in snapshots}

    results = []
    for analysis in output.analyses:
        snapshot = snapshots_by_driver.get(analysis.driver, {})
        grounding = _process_driver_analysis(
            analysis, snapshot, session_key, created_at_lap, registry
        )
        results.append(
            {
                "analysis": analysis.model_dump(),
                "grounding_checks": [g.score for g in grounding],
                "any_grounding_failed": any(not g.score["passed"] for g in grounding),
            }
        )

    return {"created_at_lap": created_at_lap, "results": results}
