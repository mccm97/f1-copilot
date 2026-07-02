# F1 copilot — Evaluation Layer

Primo modulo del sistema: il registro dei claim (previsioni verificabili) e la
logica di scoring che confronta le previsioni del "capo ingegnere AI" con i
dati reali di fine sessione.

## Struttura

- `src/evaluation/claims.py` — modelli pydantic dei tre tipi di claim
  (gap_trend, tire_cliff, grounding_check)
- `src/evaluation/storage.py` — interfaccia di storage astratta +
  implementazione locale JSON per sviluppo/test (Supabase si aggiunge dopo
  con la stessa interfaccia, senza toccare il resto del codice)
- `src/evaluation/registry.py` — punto di ingresso unico per loggare claim e
  gestirne il ciclo di vita (pending → voided / scored)
- `src/evaluation/scoring.py` — funzioni deterministiche di scoring,
  incluso il rilevamento del "cliff" gomme via changepoint detection lineare

## Come si usa

```python
from evaluation.storage import LocalJSONClaimStorage
from evaluation.registry import ClaimRegistry

storage = LocalJSONClaimStorage("data/claims.json")
registry = ClaimRegistry(storage)

claim = registry.log_gap_trend(
    session_key="2026_monza_race",
    driver="VER", rival="NOR",
    predicted_delta_seconds=-0.3, horizon_laps=5, created_at_lap=10,
    assumptions=["no_pit", "no_sc"],
)
```

## Test

```
pip install -r requirements.txt
python -m pytest tests/ -v
```

## Prossimi passi

1. Signal Engine — calcolo dei segnali reali (gap, degrado, settori) da cui
   nascono i claim e con cui si fa lo scoring
2. AI insight layer — prompt del capo ingegnere, batchato su tutti i piloti
3. Frontend — dashboard live (Lovable + Supabase)
