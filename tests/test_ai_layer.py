import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_layer.client import MockLLMClient
from ai_layer.orchestrator import EngineerCycleError, run_engineer_cycle
from evaluation.claims import ClaimStatus
from evaluation.registry import ClaimRegistry
from evaluation.storage import LocalJSONClaimStorage


@pytest.fixture
def registry(tmp_path):
    storage = LocalJSONClaimStorage(tmp_path / "claims.json")
    return ClaimRegistry(storage)


SNAPSHOTS = [
    {
        "driver": "VER",
        "rival": "NOR",
        "gap": {"current_gap_seconds": 1.6, "slope_seconds_per_lap": -0.1, "laps_used": 5},
        "tire_degradation": {"degradation_seconds_per_lap": 0.08, "laps_used": 4},
        "estimated_laps_to_cliff": 8,
        "sector_deltas": {"S1": 0.05, "S2": 0.30, "S3": -0.02},
        "worst_sector": {"sector": "S2", "delta_seconds": 0.30},
    }
]


def _canned_response(stated_s2_value: float) -> str:
    return json.dumps(
        {
            "created_at_lap": 20,
            "analyses": [
                {
                    "driver": "VER",
                    "summary": "Passo solido, margine in T2.",
                    "doing_well": ["Ottima trazione in uscita curva 1"],
                    "mistakes": ["Perde tempo in settore 2"],
                    "cited_deltas": [{"metric_name": "sector:S2", "stated_value": stated_s2_value}],
                    "gap_trend_prediction": {
                        "rival": "NOR",
                        "predicted_delta_seconds": -0.3,
                        "horizon_laps": 5,
                        "assumptions": ["no_pit", "no_sc"],
                    },
                    "tire_cliff_prediction": {
                        "compound": "MEDIUM",
                        "stint_number": 1,
                        "predicted_cliff_lap_min": 26,
                        "predicted_cliff_lap_max": 30,
                    },
                }
            ],
        }
    )


def test_honest_response_passes_grounding(registry):
    client = MockLLMClient(canned_response=_canned_response(stated_s2_value=0.30))  # coincide con lo snapshot
    result = run_engineer_cycle(SNAPSHOTS, created_at_lap=20, session_key="s", registry=registry, llm_client=client)

    driver_result = result["results"][0]
    assert driver_result["any_grounding_failed"] is False

    pending = registry.pending_claims("s", claim_type="gap_trend")
    assert len(pending) == 1
    assert pending[0].predicted_delta_seconds == -0.3


def test_hallucinated_number_fails_grounding(registry):
    client = MockLLMClient(canned_response=_canned_response(stated_s2_value=0.90))  # inventato, snapshot dice 0.30
    result = run_engineer_cycle(SNAPSHOTS, created_at_lap=20, session_key="s", registry=registry, llm_client=client)

    driver_result = result["results"][0]
    assert driver_result["any_grounding_failed"] is True

    grounding_claims = registry.storage.list(session_key="s", claim_type="grounding_check")
    assert grounding_claims[0].status == ClaimStatus.SCORED
    assert grounding_claims[0].score["passed"] is False


def test_invalid_json_raises_clear_error(registry):
    client = MockLLMClient(canned_response="questo non e' json")
    with pytest.raises(EngineerCycleError):
        run_engineer_cycle(SNAPSHOTS, created_at_lap=20, session_key="s", registry=registry, llm_client=client)


def test_unresolvable_metric_forces_grounding_failure(registry):
    response = json.dumps(
        {
            "created_at_lap": 20,
            "analyses": [
                {
                    "driver": "VER",
                    "summary": "test",
                    "cited_deltas": [{"metric_name": "sector:INESISTENTE", "stated_value": 1.0}],
                }
            ],
        }
    )
    client = MockLLMClient(canned_response=response)
    result = run_engineer_cycle(SNAPSHOTS, created_at_lap=20, session_key="s", registry=registry, llm_client=client)
    assert result["results"][0]["any_grounding_failed"] is True
