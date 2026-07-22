"""Tests for check-stocks.py."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "check-stocks.py"
SPEC = importlib.util.spec_from_file_location("check_stocks", MODULE_PATH)
check_stocks = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = check_stocks
SPEC.loader.exec_module(check_stocks)


def build_price_frame(closes, start="2024-01-01"):
    dates = pd.date_range(start=start, periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": closes,
            "High": [value * 1.01 for value in closes],
            "Low": [value * 0.99 for value in closes],
            "Close": closes,
            "Volume": [1_000_000] * len(closes),
        }
    )


def test_load_positions_from_csv_with_blank_column(tmp_path):
    csv_file = tmp_path / "positions.csv"
    csv_file.write_text(
        "ticker,entry price,entry date,share size,stop loss,,trade style\n"
        "AAPL,100,2025-01-10,10,95,,SWING-TRADE\n",
        encoding="utf-8",
    )

    positions = check_stocks.load_positions_from_csv(csv_file)

    assert len(positions) == 1
    assert positions[0].ticker == "AAPL"
    assert positions[0].entry_price == 100.0
    assert positions[0].share_size == 10.0
    assert positions[0].stop_loss == 95.0
    assert positions[0].trade_style == "SWING-TRADE"


def test_load_cached_price_data_uses_latest_file(tmp_path):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()

    older = cache_root / "AAPL_prices_old.pkl"
    newer = cache_root / "AAPL_prices_new.pkl"

    old_df = build_price_frame([100 + index for index in range(210)])
    new_df = build_price_frame([200 + index for index in range(210)])

    old_df.to_pickle(older)
    new_df.to_pickle(newer)

    older.touch()
    newer.touch()

    loaded, cache_file = check_stocks.load_cached_price_data("AAPL", cache_root)

    assert cache_file == newer
    assert loaded["Close"].iloc[-1] == new_df["Close"].iloc[-1]


def test_analyze_position_emits_sell_when_stop_is_hit(tmp_path):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()

    build_price_frame([100.0] * 205).to_pickle(cache_root / "AAPL_prices.pkl")
    build_price_frame([500.0] * 205).to_pickle(cache_root / "SPY_prices.pkl")

    position = check_stocks.StockPosition(
        ticker="AAPL",
        entry_price=110.0,
        entry_date=datetime(2025, 1, 10),
        share_size=5.0,
        stop_loss=105.0,
        trade_style="SWING-TRADE",
    )

    benchmark = check_stocks.load_cached_benchmark(cache_root)
    result = check_stocks.analyze_position(position, benchmark, cache_root)

    assert result["status"] == "ok"
    assert result["action"] == "SELL NOW"
    assert result["current_price"] == 100.0


def test_analyze_position_recommends_raising_stop_in_uptrend(tmp_path):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()

    closes = [100 + (index * 0.4) for index in range(260)]
    spy_closes = [400 + (index * 0.1) for index in range(260)]
    build_price_frame(closes).to_pickle(cache_root / "AAPL_prices.pkl")
    build_price_frame(spy_closes).to_pickle(cache_root / "SPY_prices.pkl")

    position = check_stocks.StockPosition(
        ticker="AAPL",
        entry_price=120.0,
        entry_date=datetime(2025, 1, 10),
        share_size=5.0,
        stop_loss=115.0,
        trade_style="SWING-TRADE",
    )

    benchmark = check_stocks.load_cached_benchmark(cache_root)
    result = check_stocks.analyze_position(position, benchmark, cache_root)

    assert result["status"] == "ok"
    assert result["recommended_stop"] > 115.0
    assert result["action"] in {"RAISE SL", "RAISE SL / CONSIDER PARTIAL"}
