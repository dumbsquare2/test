# StockCheck API (GPT Action MVP)

FastAPI backend for stock lookup and historical price endpoints designed to be connected to GPT Actions.

## Endpoints
- `GET /search`: search by ticker/company name in a market listing.
- `GET /quote`: latest close, previous close, and daily change.
- `GET /history`: OHLCV history for date range.
- `GET /health`: health check.

## Run
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

OpenAPI schema:
- `http://localhost:8000/openapi.json`
