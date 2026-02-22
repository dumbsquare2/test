from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import FinanceDataReader as fdr
from cachetools import TTLCache
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

app = FastAPI(
    title="StockCheck API",
    version="0.1.0",
    description="Price/listing endpoints for a GPT Action. Educational / informational only.",
)

LISTING_CACHE = TTLCache(maxsize=16, ttl=60 * 60)
PRICE_CACHE = TTLCache(maxsize=512, ttl=60)

Market = Literal["KRX", "NASDAQ", "NYSE", "S&P500"]


class TickerHit(BaseModel):
    market: str
    symbol: str
    name: str
    sector: Optional[str] = None


class QuoteResponse(BaseModel):
    market: str
    symbol: str
    asof: str
    close: float
    prev_close: float
    change: float
    change_pct: float


class OHLCVPoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


class HistoryResponse(BaseModel):
    market: str
    symbol: str
    start: str
    end: str
    points: List[OHLCVPoint]


def _get_listing(market: Market) -> pd.DataFrame:
    key = f"listing:{market}"
    if key in LISTING_CACHE:
        return LISTING_CACHE[key]

    try:
        df = fdr.StockListing(market)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch listing for {market}: {exc}",
        ) from exc

    rename_map: Dict[str, str] = {}
    for col in df.columns:
        lower = col.lower()
        if lower in ["symbol", "code", "ticker"]:
            rename_map[col] = "Symbol"
        if lower in ["name", "company", "companyname", "korname"]:
            rename_map[col] = "Name"
        if lower in ["sector", "industry"]:
            rename_map[col] = "Sector"

    if rename_map:
        df = df.rename(columns=rename_map)

    LISTING_CACHE[key] = df
    return df


def _read_prices(symbol: str, start: str, end: str) -> pd.DataFrame:
    cache_key = f"price:{symbol}:{start}:{end}"
    if cache_key in PRICE_CACHE:
        return PRICE_CACHE[cache_key]

    try:
        df = fdr.DataReader(symbol, start, end)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch prices for {symbol}: {exc}",
        ) from exc

    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No price data for symbol={symbol}")

    need = {"Open", "High", "Low", "Close"}
    if not need.issubset(set(df.columns)):
        raise HTTPException(status_code=502, detail=f"Unexpected price columns: {list(df.columns)}")

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    PRICE_CACHE[cache_key] = df
    return df


@app.get("/search", response_model=List[TickerHit])
def search_ticker(
    q: str = Query(..., min_length=1, description="Search keyword: ticker or company name"),
    market: Market = Query("KRX", description="Market universe"),
    limit: int = Query(10, ge=1, le=50),
):
    df = _get_listing(market)

    if "Symbol" not in df.columns or "Name" not in df.columns:
        raise HTTPException(
            status_code=502,
            detail=f"Listing for {market} does not contain Symbol/Name columns.",
        )

    q_norm = q.strip().lower()
    hits = df[
        df["Symbol"].astype(str).str.lower().str.contains(q_norm, na=False)
        | df["Name"].astype(str).str.lower().str.contains(q_norm, na=False)
    ].head(limit)

    out: List[TickerHit] = []
    for _, row in hits.iterrows():
        out.append(
            TickerHit(
                market=market,
                symbol=str(row["Symbol"]),
                name=str(row["Name"]),
                sector=(
                    str(row["Sector"])
                    if "Sector" in row and not pd.isna(row["Sector"])
                    else None
                ),
            )
        )

    return out


@app.get("/quote", response_model=QuoteResponse)
def quote(
    symbol: str = Query(..., description="Ticker/symbol"),
    market: Market = Query("KRX", description="Market context"),
):
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=21)

    df = _read_prices(symbol, start_dt.isoformat(), end_dt.isoformat()).dropna()
    if len(df) < 2:
        raise HTTPException(status_code=404, detail="Not enough data points to compute change.")

    last = df.iloc[-1]
    prev = df.iloc[-2]

    close = float(last["Close"])
    prev_close = float(prev["Close"])
    change = close - prev_close
    change_pct = (change / prev_close) * 100.0 if prev_close != 0 else float("nan")

    return QuoteResponse(
        market=market,
        symbol=symbol,
        asof=df.index[-1].date().isoformat(),
        close=close,
        prev_close=prev_close,
        change=change,
        change_pct=float(change_pct),
    )


@app.get("/history", response_model=HistoryResponse)
def history(
    symbol: str = Query(..., description="Ticker/symbol"),
    market: Market = Query("KRX"),
    start: str = Query(..., description="YYYY-MM-DD"),
    end: str = Query(..., description="YYYY-MM-DD"),
    limit: int = Query(400, ge=10, le=2000),
):
    df = _read_prices(symbol, start, end).dropna().tail(limit)

    points: List[OHLCVPoint] = []
    for idx, row in df.iterrows():
        points.append(
            OHLCVPoint(
                date=idx.date().isoformat(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=(
                    float(row["Volume"])
                    if "Volume" in df.columns and not pd.isna(row.get("Volume", np.nan))
                    else None
                ),
            )
        )

    return HistoryResponse(
        market=market,
        symbol=symbol,
        start=start,
        end=end,
        points=points,
    )


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "time": datetime.utcnow().isoformat() + "Z"}
