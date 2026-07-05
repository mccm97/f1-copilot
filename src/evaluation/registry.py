"""
ClaimRegistry: l'unico punto da cui l'AI insight layer e il signal engine
passano per registrare previsioni verificabili e controlli di grounding.
Nasconde lo storage concreto dietro l'interfaccia ClaimStorage.
"""
from __future__ import annotations

from .claims import (
    BaseClaim,
    ClaimStatus,
    GapTrendClaim,
    GroundingCheck,
    InvalidatingEvent,
    TireCliffClaim,
)
from .storage import ClaimStorage


class ClaimRegistry:
    def __init__(self, storage: ClaimStorage):
        self.storage = storage

    # --- registrazione previsioni ---

    def log_gap_trend(
        self,
        session_key: str,
        driver: str,
        rival: str,
        predicted_delta_seconds: float,
        horizon_laps: int,
        created_at_lap: int,
        assumptions: list[str] | None = None,
    ) -> GapTrendClaim:
        claim = GapTrendClaim(
            session_key=session_key,
            driver=driver,
            rival=rival,
            predicted_delta_seconds=predicted_delta_seconds,
            horizon_laps=horizon_laps,
            created_at_lap=created_at_lap,
            assumptions=assumptions or [],
        )
        self.storage.save(claim)
        return claim

    def log_tire_cliff(
        self,
        session_key: str,
        driver: str,
        compound: str,
        stint_number: int,
        predicted_cliff_lap_min: int,
        predicted_cliff_lap_max: int,
        created_at_lap: int,
    ) -> TireCliffClaim:
        claim = TireCliffClaim(
            session_key=session_key,
            driver=driver,
            compound=compound,
            stint_number=stint_number,
            predicted_cliff_lap_min=predicted_cliff_lap_min,
            predicted_cliff_lap_max=predicted_cliff_lap_max,
            created_at_lap=created_at_lap,
        )
        self.storage.save(claim)
        return claim

    def log_grounding_check(
        self,
        session_key: str,
        driver: str,
        metric_name: str,
        stated_value: float,
        computed_value: float,
        created_at_lap: int,
        tolerance: float = 0.02,
    ) -> GroundingCheck:
        # Il grounding check si autoverifica subito: non resta mai "pending"
        claim = GroundingCheck(
            session_key=session_key,
            driver=driver,
            metric_name=metric_name,
            stated_value=stated_value,
            computed_value=computed_value,
            created_at_lap=created_at_lap,
            tolerance=tolerance,
        )
        passed = abs(stated_value - computed_value) <= tolerance
        claim.status = ClaimStatus.SCORED
        claim.score = {
            "passed": passed,
            "abs_error": abs(stated_value - computed_value),
        }
        self.storage.save(claim)
        return claim

    # --- gestione ciclo di vita ---

    def void_claim(self, claim_id: str, reason: InvalidatingEvent) -> BaseClaim:
        claim = self.storage.get(claim_id)
        if claim is None:
            raise KeyError(f"Claim {claim_id} non trovato")
        claim.status = ClaimStatus.VOIDED
        claim.voided_reason = reason
        self.storage.update(claim)
        return claim

    def pending_claims(
        self, session_key: str, claim_type: str | None = None
    ) -> list[BaseClaim]:
        return self.storage.list(
            session_key=session_key,
            status=ClaimStatus.PENDING,
            claim_type=claim_type,
        )

    def mark_scored(self, claim_id: str, score: dict) -> BaseClaim:
        claim = self.storage.get(claim_id)
        if claim is None:
            raise KeyError(f"Claim {claim_id} non trovato")
        claim.status = ClaimStatus.SCORED
        claim.score = score
        self.storage.update(claim)
        return claim
