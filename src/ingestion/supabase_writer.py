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
