"""
Interfaccia di storage per i claim, con un'implementazione locale su file JSON
per sviluppo e test. Quando sara' il momento, si aggiunge SupabaseClaimStorage
con la stessa interfaccia, senza toccare il resto del codice (registry.py e
scoring.py parlano solo con ClaimStorage, mai con l'implementazione concreta).
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from .claims import BaseClaim, ClaimStatus


class ClaimStorage(ABC):
    @abstractmethod
    def save(self, claim: BaseClaim) -> None: ...

    @abstractmethod
    def get(self, claim_id: str) -> BaseClaim | None: ...

    @abstractmethod
    def update(self, claim: BaseClaim) -> None: ...

    @abstractmethod
    def list(
        self,
        session_key: str | None = None,
        driver: str | None = None,
        status: ClaimStatus | None = None,
        claim_type: str | None = None,
    ) -> list[BaseClaim]: ...


# Mappa claim_type -> classe concreta, serve per deserializzare correttamente
def _claim_class_for_type(claim_type: str):
    from .claims import GapTrendClaim, GroundingCheck, TireCliffClaim

    return {
        "gap_trend": GapTrendClaim,
        "tire_cliff": TireCliffClaim,
        "grounding_check": GroundingCheck,
    }[claim_type]


class LocalJSONClaimStorage(ClaimStorage):
    """Storage su singolo file JSON. Solo per sviluppo locale e test:
    non e' pensato per concorrenza o volumi grandi."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def _read_all(self) -> list[dict]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write_all(self, records: list[dict]) -> None:
        self.path.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")

    def save(self, claim: BaseClaim) -> None:
        records = self._read_all()
        records.append(json.loads(claim.model_dump_json()))
        self._write_all(records)

    def get(self, claim_id: str) -> BaseClaim | None:
        for record in self._read_all():
            if record["id"] == claim_id:
                cls = _claim_class_for_type(record["claim_type"])
                return cls.model_validate(record)
        return None

    def update(self, claim: BaseClaim) -> None:
        records = self._read_all()
        for i, record in enumerate(records):
            if record["id"] == claim.id:
                records[i] = json.loads(claim.model_dump_json())
                self._write_all(records)
                return
        raise KeyError(f"Claim {claim.id} non trovato, impossibile aggiornare")

    def list(
        self,
        session_key: str | None = None,
        driver: str | None = None,
        status: ClaimStatus | None = None,
        claim_type: str | None = None,
    ) -> list[BaseClaim]:
        results = []
        for record in self._read_all():
            if session_key and record["session_key"] != session_key:
                continue
            if driver and record["driver"] != driver:
                continue
            if status and record["status"] != status.value:
                continue
            if claim_type and record["claim_type"] != claim_type:
                continue
            cls = _claim_class_for_type(record["claim_type"])
            results.append(cls.model_validate(record))
        return results
