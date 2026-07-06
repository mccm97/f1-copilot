"""
Interfaccia di storage per i claim, con un'implementazione locale su file JSON
per sviluppo e test. Quando sara' il momento, si aggiunge SupabaseClaimStorage
con la stessa interfaccia, senza toccare il resto del codice (registry.py e
scoring.py parlano solo con ClaimStorage, mai con l'implementazione concreta).
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

import requests

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


class SupabaseClaimStorage(ClaimStorage):
    """
    Storage persistente su Supabase: stessa interfaccia di LocalJSONClaimStorage,
    cosi' ClaimRegistry e tutto il resto del codice non cambiano. Necessario
    perche' il worker automatico (che gira come processo separato ad ogni
    ciclo, senza stato condiviso) possa ritrovare i claim dei cicli precedenti.

    Serve la tabella SQL 'claims' (vedi sql/create_claims_table.sql) e la
    service_role key, mai l'anon key: qui si scrive e si aggiornano claim,
    non e' un accesso di sola lettura come quello del frontend.
    """

    def __init__(self, url: str | None = None, service_role_key: str | None = None):
        self.url = (url or os.environ["SUPABASE_URL"]).rstrip("/")
        self.key = service_role_key or os.environ["SUPABASE_SERVICE_ROLE_KEY"]

    def _headers(self, prefer: str = "return=representation") -> dict:
        return {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Prefer": prefer,
        }

    def save(self, claim: BaseClaim) -> None:
        row = {
            "id": claim.id,
            "session_key": claim.session_key,
            "driver": claim.driver,
            "claim_type": claim.claim_type,
            "status": claim.status.value,
            "created_at_lap": claim.created_at_lap,
            "data": json.loads(claim.model_dump_json()),
        }
        resp = requests.post(
            f"{self.url}/rest/v1/claims",
            headers=self._headers(prefer="return=minimal"),
            json=row,
            timeout=20,
        )
        if resp.status_code >= 300:
            raise RuntimeError(f"Salvataggio claim fallito ({resp.status_code}): {resp.text}")

    def get(self, claim_id: str) -> BaseClaim | None:
        resp = requests.get(
            f"{self.url}/rest/v1/claims",
            headers=self._headers(),
            params={"id": f"eq.{claim_id}", "select": "data"},
            timeout=20,
        )
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            return None
        record = rows[0]["data"]
        cls = _claim_class_for_type(record["claim_type"])
        return cls.model_validate(record)

    def update(self, claim: BaseClaim) -> None:
        row = {
            "status": claim.status.value,
            "data": json.loads(claim.model_dump_json()),
        }
        resp = requests.patch(
            f"{self.url}/rest/v1/claims",
            headers=self._headers(prefer="return=minimal"),
            params={"id": f"eq.{claim.id}"},
            json=row,
            timeout=20,
        )
        if resp.status_code >= 300:
            raise RuntimeError(f"Aggiornamento claim fallito ({resp.status_code}): {resp.text}")

    def list(
        self,
        session_key: str | None = None,
        driver: str | None = None,
        status: ClaimStatus | None = None,
        claim_type: str | None = None,
    ) -> list[BaseClaim]:
        params = {"select": "data"}
        if session_key:
            params["session_key"] = f"eq.{session_key}"
        if driver:
            params["driver"] = f"eq.{driver}"
        if status:
            params["status"] = f"eq.{status.value}"
        if claim_type:
            params["claim_type"] = f"eq.{claim_type}"

        resp = requests.get(
            f"{self.url}/rest/v1/claims", headers=self._headers(), params=params, timeout=20
        )
        resp.raise_for_status()

        results = []
        for row in resp.json():
            record = row["data"]
            cls = _claim_class_for_type(record["claim_type"])
            results.append(cls.model_validate(record))
        return results
