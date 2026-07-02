"""
Modelli dei "claim": ogni previsione o affermazione del capo ingegnere AI
che deve poter essere verificata (o esplicitamente annullata) contro i dati reali.

Tre tipi, con logiche di verifica diverse:
- GapTrendClaim: previsione quantitativa sull'evoluzione del gap da un rivale
- TireCliffClaim: previsione del giro in cui arriva il calo prestazionale gomme
- GroundingCheck: non e' una previsione futura, e' un controllo immediato che il
  numero citato dall'AI in un consiglio coincida con quello calcolato dal signal engine
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class ClaimStatus(str, Enum):
    PENDING = "pending"          # in attesa che arrivi l'orizzonte per verificare
    VOIDED = "voided"            # annullato per evento esterno (SC, pit, bandiera...)
    SCORED = "scored"            # verificato, ha un punteggio


class InvalidatingEvent(str, Enum):
    """Eventi che rendono una previsione non verificabile in modo equo."""
    SAFETY_CAR = "safety_car"
    RED_FLAG = "red_flag"
    UNPLANNED_PIT = "unplanned_pit"
    SPIN_OR_OFF_TRACK = "spin_or_off_track"
    DNF = "dnf"
    RAIN_ONSET = "rain_onset"


class BaseClaim(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_key: str  # es. "2026_monza_race", "2026_monza_fp2"
    driver: str  # es. "VER"
    claim_type: Literal["gap_trend", "tire_cliff", "grounding_check"]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_at_lap: int
    status: ClaimStatus = ClaimStatus.PENDING
    voided_reason: InvalidatingEvent | None = None
    score: dict | None = None  # popolato dallo scoring, struttura specifica per tipo


class GapTrendClaim(BaseClaim):
    claim_type: Literal["gap_trend"] = "gap_trend"
    rival: str
    predicted_delta_seconds: float  # negativo = si avvicina, positivo = si allontana
    horizon_laps: int
    assumptions: list[str] = Field(default_factory=list)  # es. ["no_pit", "no_sc"]


class TireCliffClaim(BaseClaim):
    claim_type: Literal["tire_cliff"] = "tire_cliff"
    compound: str  # SOFT / MEDIUM / HARD / INTER / WET
    stint_number: int
    predicted_cliff_lap_min: int
    predicted_cliff_lap_max: int


class GroundingCheck(BaseClaim):
    claim_type: Literal["grounding_check"] = "grounding_check"
    metric_name: str  # es. "delta_frenata_T4_vs_rivale"
    stated_value: float  # quello che l'AI ha scritto nel commento
    computed_value: float  # quello calcolato dal signal engine nello stesso momento
    tolerance: float = 0.02  # tolleranza assoluta prima di considerarlo un'allucinazione
