"""
Interfaccia LLMClient astratta, cosi' l'orchestratore non dipende da quale
provider genera il testo. MockLLMClient serve per i test (nessuna chiamata
di rete). HuggingFaceLLMClient e' un punto di partenza per l'uso reale:
verifica endpoint/nome modello sulla documentazione HuggingFace corrente
prima di usarlo, i router "Inference Providers" cambiano nel tempo.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def generate(self, messages: list[dict]) -> str: ...


class MockLLMClient(LLMClient):
    """Usato nei test: ritorna sempre la stessa risposta pre-costruita,
    zero rete, zero costi, cosi' possiamo testare parsing e grounding
    senza dipendere da un servizio esterno."""

    def __init__(self, canned_response: str):
        self.canned_response = canned_response

    def generate(self, messages: list[dict]) -> str:
        return self.canned_response


class HuggingFaceLLMClient(LLMClient):
    """
    Implementazione verso l'Inference Providers router di HuggingFace
    (compatibile OpenAI chat completions). Il token usato DEVE essere
    "fine-grained" con il permesso "Make calls to Inference Providers"
    attivo, altrimenti le richieste falliscono per permessi insufficienti.

    Il modello di default va verificato prima dell'uso: il catalogo dei
    modelli disponibili via router cambia spesso. Controlla modelli e prezzi
    aggiornati su huggingface.co/playground e usa il suffisso ":cheapest"
    per il piu' economico disponibile, cosi' non serve inseguire nomi fissi.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-8B:cheapest",
        api_token: str | None = None,
        base_url: str = "https://router.huggingface.co/v1/chat/completions",
        timeout_seconds: int = 30,
    ):
        self.model = model
        self.api_token = api_token or os.environ.get("HF_API_TOKEN")
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    def generate(self, messages: list[dict]) -> str:
        import requests

        if not self.api_token:
            raise RuntimeError("HF_API_TOKEN non impostato: serve un token gratuito HuggingFace")

        response = requests.post(
            self.base_url,
            headers={"Authorization": f"Bearer {self.api_token}"},
            json={"model": self.model, "messages": messages},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
