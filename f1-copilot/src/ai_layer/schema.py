"""
Schema dell'output del capo ingegnere. Il modello DEVE produrre JSON che
valida contro EngineerCycleOutput: se non valida, il ciclo fallisce in modo
esplicito invece di far passare testo libero non verificabile a valle.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class CitedDelta(BaseModel):
    """Un numero che il modello cita a supporto di un commento (es. un delta
    di settore). metric_name segue la convenzione 'sector:<nome>' cosi' il
    grounding check sa dove andarlo a verificare nello snapshot."""
    metric_name: str
    stated_value: float


class GapTrendPrediction(BaseModel):
    rival: str
    predicted_delta_seconds: float
    horizon_laps: int
    assumptions: list[str] = Field(default_factory=lambda: ["no_pit", "no_sc"])


class TireCliffPrediction(BaseModel):
    compound: str
    stint_number: int
    predicted_cliff_lap_min: int
    predicted_cliff_lap_max: int


class DriverAnalysis(BaseModel):
    driver: str
    summary: str
    doing_well: list[str] = Field(default_factory=list)
    mistakes: list[str] = Field(default_factory=list)
    cited_deltas: list[CitedDelta] = Field(default_factory=list)
    gap_trend_prediction: GapTrendPrediction | None = None
    tire_cliff_prediction: TireCliffPrediction | None = None


class EngineerCycleOutput(BaseModel):
    created_at_lap: int
    analyses: list[DriverAnalysis]
