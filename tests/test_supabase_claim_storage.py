import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.claims import ClaimStatus, GapTrendClaim
from evaluation.storage import SupabaseClaimStorage


@pytest.fixture
def storage():
    return SupabaseClaimStorage(url="https://fake.supabase.co", service_role_key="fake-key")


@pytest.fixture
def sample_claim():
    return GapTrendClaim(
        session_key="s", driver="VER", rival="NOR",
        predicted_delta_seconds=-0.3, horizon_laps=5, created_at_lap=10,
    )


def test_save_posts_correct_row(storage, sample_claim):
    with patch("evaluation.storage.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=201)
        storage.save(sample_claim)

    args, kwargs = mock_post.call_args
    assert args[0] == "https://fake.supabase.co/rest/v1/claims"
    assert kwargs["json"]["session_key"] == "s"
    assert kwargs["json"]["driver"] == "VER"
    assert kwargs["json"]["claim_type"] == "gap_trend"
    assert kwargs["json"]["data"]["rival"] == "NOR"


def test_save_raises_on_error_status(storage, sample_claim):
    with patch("evaluation.storage.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=400, text="errore")
        with pytest.raises(RuntimeError):
            storage.save(sample_claim)


def test_get_reconstructs_claim(storage, sample_claim):
    fake_row = {"data": sample_claim.model_dump(mode="json")}
    with patch("evaluation.storage.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, json=lambda: [fake_row])
        result = storage.get(sample_claim.id)

    assert result.driver == "VER"
    assert result.predicted_delta_seconds == -0.3


def test_get_returns_none_if_not_found(storage):
    with patch("evaluation.storage.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, json=lambda: [])
        result = storage.get("id-inesistente")

    assert result is None


def test_update_sends_patch_with_filter(storage, sample_claim):
    sample_claim.status = ClaimStatus.VOIDED
    with patch("evaluation.storage.requests.patch") as mock_patch:
        mock_patch.return_value = MagicMock(status_code=204)
        storage.update(sample_claim)

    args, kwargs = mock_patch.call_args
    assert kwargs["params"]["id"] == f"eq.{sample_claim.id}"
    assert kwargs["json"]["status"] == "voided"


def test_list_applies_filters(storage, sample_claim):
    fake_row = {"data": sample_claim.model_dump(mode="json")}
    with patch("evaluation.storage.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, json=lambda: [fake_row])
        results = storage.list(session_key="s", claim_type="gap_trend")

    args, kwargs = mock_get.call_args
    assert kwargs["params"]["session_key"] == "eq.s"
    assert kwargs["params"]["claim_type"] == "eq.gap_trend"
    assert len(results) == 1
    assert results[0].driver == "VER"
