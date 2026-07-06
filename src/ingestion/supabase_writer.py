"""
Scrive righe in Supabase via REST, usando la service_role key (permessi di
scrittura). Questa chiave non deve MAI finire nel frontend: qui vive solo
lato backend/script, passata come variabile d'ambiente.
"""
from __future__ import annotations

import os

import requests


class SupabaseWriter:
    def __init__(self, url: str | None = None, service_role_key: str | None = None):
        self.url = (url or os.environ["SUPABASE_URL"]).rstrip("/")
        self.key = service_role_key or os.environ["SUPABASE_SERVICE_ROLE_KEY"]

    def _headers(self) -> dict:
        return {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }

    def insert(self, table: str, row: dict) -> None:
        resp = requests.post(
            f"{self.url}/rest/v1/{table}",
            headers=self._headers(),
            json=row,
            timeout=20,
        )
        if resp.status_code >= 300:
            raise RuntimeError(f"Scrittura su {table} fallita ({resp.status_code}): {resp.text}")

    def select_latest(self, table: str, order_by: str, limit: int = 1) -> list[dict]:
        resp = requests.get(
            f"{self.url}/rest/v1/{table}",
            headers=self._headers(),
            params={"select": "*", "order": f"{order_by}.desc", "limit": limit},
            timeout=20,
        )
        if resp.status_code >= 300:
            raise RuntimeError(f"Lettura da {table} fallita ({resp.status_code}): {resp.text}")
        return resp.json()

    def write_engineer_analysis(self, driver_result: dict, created_at_lap: int) -> None:
        """Scrive una riga in engineer_analyses fondendo l'esito del grounding
        check dentro cited_deltas, cosi' il frontend puo' mostrare i badge
        senza dover ricalcolare nulla."""
        analysis = driver_result["analysis"]
        grounding_checks = driver_result["grounding_checks"]

        cited_deltas = [
            {
                "metric_name": cited["metric_name"],
                "stated_value": cited["stated_value"],
                "grounding_passed": check["passed"],
            }
            for cited, check in zip(analysis.get("cited_deltas", []), grounding_checks)
        ]

        self.insert("engineer_analyses", {
            "driver": analysis["driver"],
            "created_at_lap": created_at_lap,
            "summary": analysis["summary"],
            "doing_well": analysis.get("doing_well", []),
            "mistakes": analysis.get("mistakes", []),
            "cited_deltas": cited_deltas,
            "gap_trend_prediction": analysis.get("gap_trend_prediction"),
            "tire_cliff_prediction": analysis.get("tire_cliff_prediction"),
        })
