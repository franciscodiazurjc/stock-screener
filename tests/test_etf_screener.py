"""Unit tests for ETF screening modules.

Covers:
- ETFUniverseFetcher (cache, fallback, filtering)
- ETFFetcher (cache, info parsing, price history)
- score_etf_fundamentals (expense ratio, AUM, yield scoring)
- score_etf_buy_signal (technical + ETF quality scoring)
- score_etf_sell_signal (breakdown scoring)
"""

import pickle
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.etf_fetcher import ETFFetcher, _classify_asset_class, score_etf_fundamentals
from src.data.etf_universe_fetcher import ETFUniverseFetcher, POPULAR_ETFS
from src.screening.etf_signal_engine import (
    score_etf_buy_signal,
    score_etf_sell_signal,
    format_etf_signal_output,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_cache(tmp_path):
    d = tmp_path / "etf_cache"
    d.mkdir()
    return str(d)


@pytest.fixture
def etf_fetcher(temp_cache):
    return ETFFetcher(cache_dir=temp_cache, cache_expiry_hours=24, max_retries=1, retry_delay=0)


@pytest.fixture
def universe_fetcher(temp_cache):
    return ETFUniverseFetcher(cache_dir=temp_cache)


def _make_price_df(n: int = 250, base: float = 100.0, trend: float = 0.05) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame with an uptrend (DatetimeIndex)."""
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="B")
    prices = base * np.cumprod(1 + np.random.normal(trend / 252, 0.01, n))
    df = pd.DataFrame(
        {
            "Open": prices * 0.99,
            "High": prices * 1.01,
            "Low": prices * 0.98,
            "Close": prices,
            "Volume": np.random.randint(100_000, 1_000_000, n).astype(float),
        },
        index=dates,
    )
    return df


def _make_phase2_info(price: float) -> dict:
    """Return a synthetic phase_info dict for a Stage 2 stock."""
    sma_50 = price * 0.92
    sma_200 = price * 0.78
    return {
        "phase": 2,
        "sma_50": sma_50,
        "sma_150": price * 0.85,
        "sma_200": sma_200,
        "slope_50": 0.06,
        "slope_200": 0.04,
        "distance_from_50sma": ((price - sma_50) / sma_50) * 100,
        "distance_from_200sma": ((price - sma_200) / sma_200) * 100,
        "week_52_high": price * 1.02,
        "week_52_low": price * 0.60,
    }


def _make_rs_series(n: int = 250, slope: float = 0.0003) -> pd.Series:
    """Create a synthetic RS series with a gentle upward slope."""
    np.random.seed(7)
    vals = 100.0 + slope * np.arange(n) + np.random.normal(0, 0.5, n)
    return pd.Series(vals)


# ---------------------------------------------------------------------------
# ETFUniverseFetcher tests
# ---------------------------------------------------------------------------

class TestETFUniverseFetcher:
    """Tests for ETFUniverseFetcher."""

    def test_popular_etfs_not_empty(self, universe_fetcher):
        """get_popular_etfs() must return a non-empty sorted list."""
        etfs = universe_fetcher.get_popular_etfs()
        assert isinstance(etfs, list)
        assert len(etfs) > 0
        assert etfs == sorted(etfs), "List must be sorted"

    def test_popular_etfs_contains_key_tickers(self, universe_fetcher):
        """Key benchmark ETFs must always be in the curated list."""
        popular = universe_fetcher.get_popular_etfs()
        for ticker in ("SPY", "QQQ", "IWM", "GLD", "TLT"):
            assert ticker in popular, f"{ticker} missing from popular ETF list"

    def test_fetch_universe_fallback_on_network_failure(self, universe_fetcher):
        """If all live sources fail the fetcher falls back to POPULAR_ETFS."""
        with patch.object(universe_fetcher, "_fetch_nasdaq_etf_list", return_value=pd.DataFrame()), \
             patch.object(universe_fetcher, "_fetch_other_listed_etfs", return_value=pd.DataFrame()):
            result = universe_fetcher.fetch_universe(force_refresh=True)

        assert isinstance(result, list)
        assert len(result) > 0
        # Should contain at least some popular ETFs
        assert any(t in result for t in ("SPY", "QQQ"))

    def test_fetch_universe_uses_cache(self, universe_fetcher, tmp_path):
        """Second call within 24 h must hit the cache."""
        # Pre-populate cache
        cache_data = {
            "symbols": ["SPY", "QQQ", "IWM"],
            "fetch_date": datetime.now().isoformat(),
            "count": 3,
            "source": "test",
        }
        with open(universe_fetcher.cache_file, "wb") as fh:
            pickle.dump(cache_data, fh)

        with patch.object(universe_fetcher, "_fetch_nasdaq_etf_list") as mock_fetch:
            result = universe_fetcher.fetch_universe(force_refresh=False)

        mock_fetch.assert_not_called()
        assert result == ["SPY", "QQQ", "IWM"]

    def test_get_universe_info_no_cache(self, universe_fetcher):
        info = universe_fetcher.get_universe_info()
        assert info["cached"] is False
        assert info["count"] == len(POPULAR_ETFS)

    def test_filter_etfs_removes_special_chars(self, universe_fetcher):
        df = pd.DataFrame({"symbol": ["SPY", "QQQ", "SPY.W", "X$Y"], "name": ["A", "B", "C", "D"]})
        filtered = universe_fetcher._filter_etfs(df)
        assert "SPY.W" not in filtered["symbol"].values
        assert "X$Y" not in filtered["symbol"].values
        assert "SPY" in filtered["symbol"].values

    def test_filter_etfs_removes_single_char(self, universe_fetcher):
        df = pd.DataFrame({"symbol": ["X", "QQ", "SPY"], "name": ["A", "B", "C"]})
        filtered = universe_fetcher._filter_etfs(df)
        assert "X" not in filtered["symbol"].values


# ---------------------------------------------------------------------------
# ETFFetcher tests
# ---------------------------------------------------------------------------

class TestETFFetcher:
    """Tests for ETFFetcher."""

    def test_cache_path_prefix(self, etf_fetcher):
        """Cache keys must include the 'etf_' prefix to avoid collision."""
        p = etf_fetcher._cache_path("SPY", "info")
        assert "etf_SPY" in str(p)

    def test_cache_valid_fresh(self, etf_fetcher, tmp_path):
        """A freshly written file should be considered valid."""
        path = Path(etf_fetcher.cache_dir) / "test.pkl"
        path.write_bytes(b"")
        assert etf_fetcher._cache_valid(path) is True

    def test_cache_valid_expired(self, etf_fetcher):
        """A file older than cache_expiry_hours should be invalid."""
        path = Path(etf_fetcher.cache_dir) / "old.pkl"
        path.write_bytes(b"")
        # Make the file appear 48 h old
        old_ts = (datetime.now() - timedelta(hours=48)).timestamp()
        import os
        os.utime(str(path), (old_ts, old_ts))
        assert etf_fetcher._cache_valid(path) is False

    def test_fetch_etf_info_success(self, etf_fetcher):
        """fetch_etf_info should parse yfinance info dict and return expected keys."""
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "longName": "SPDR S&P 500 ETF Trust",
            "category": "Large Blend",
            "annualReportExpenseRatio": 0.0003,
            "totalAssets": 400_000_000_000,
            "navPrice": 450.0,
            "yield": 0.015,
            "currentPrice": 451.0,
            "fiftyTwoWeekHigh": 480.0,
            "fiftyTwoWeekLow": 380.0,
            "beta3Year": 1.0,
        }

        with patch.object(etf_fetcher, "_fetch_ticker", return_value=mock_ticker):
            result = etf_fetcher.fetch_etf_info("SPY")

        assert result["ticker"] == "SPY"
        assert result["name"] == "SPDR S&P 500 ETF Trust"
        assert result["expense_ratio"] == 0.0003
        assert result["aum"] == 400_000_000_000
        assert result["yield_pct"] == 0.015
        assert "fetch_date" in result

    def test_fetch_etf_info_returns_empty_on_failure(self, etf_fetcher):
        """Returns empty dict when yfinance raises an exception."""
        with patch.object(etf_fetcher, "_fetch_ticker", return_value=None):
            result = etf_fetcher.fetch_etf_info("INVALID_ETF")

        assert result == {}

    def test_fetch_etf_info_uses_cache(self, etf_fetcher):
        """Second call within expiry window must use the cache."""
        cached = {"ticker": "SPY", "name": "SPDR", "expense_ratio": 0.0003, "fetch_date": "2024-01-01"}
        cache_path = etf_fetcher._cache_path("SPY", "info")
        etf_fetcher._save_cache(cached, cache_path)

        with patch.object(etf_fetcher, "_fetch_ticker") as mock_t:
            result = etf_fetcher.fetch_etf_info("SPY")

        mock_t.assert_not_called()
        assert result["ticker"] == "SPY"

    def test_fetch_price_history_success(self, etf_fetcher):
        """fetch_price_history should return a DataFrame with DatetimeIndex."""
        price_df = _make_price_df(260)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = price_df

        with patch.object(etf_fetcher, "_fetch_ticker", return_value=mock_ticker):
            result = etf_fetcher.fetch_price_history("QQQ", period="2y")

        assert not result.empty
        assert isinstance(result.index, pd.DatetimeIndex)
        assert "Close" in result.columns

    def test_fetch_price_history_empty_on_failure(self, etf_fetcher):
        with patch.object(etf_fetcher, "_fetch_ticker", return_value=None):
            result = etf_fetcher.fetch_price_history("INVALID", period="1y")
        assert result.empty

    def test_clear_cache_removes_files(self, etf_fetcher):
        """clear_cache should delete all etf_* cache files."""
        for name in ("etf_SPY_info.pkl", "etf_QQQ_info.pkl"):
            p = Path(etf_fetcher.cache_dir) / name
            p.write_bytes(b"")

        etf_fetcher.clear_cache()

        for name in ("etf_SPY_info.pkl", "etf_QQQ_info.pkl"):
            assert not (Path(etf_fetcher.cache_dir) / name).exists()


# ---------------------------------------------------------------------------
# _classify_asset_class tests
# ---------------------------------------------------------------------------

class TestClassifyAssetClass:
    def test_bond_etf(self):
        info = {"category": "Long-Term Bond", "longName": "iShares 20+ Year Treasury Bond ETF"}
        assert _classify_asset_class(info) == "Fixed Income"

    def test_commodity_etf(self):
        info = {"category": "Commodities Gold", "longName": "SPDR Gold Shares"}
        assert _classify_asset_class(info) == "Commodity"

    def test_reit_etf(self):
        info = {"category": "Real Estate", "longName": "Vanguard Real Estate ETF"}
        assert _classify_asset_class(info) == "Real Estate"

    def test_equity_etf(self):
        info = {"category": "Large Blend", "longName": "SPDR S&P 500 ETF Trust"}
        assert _classify_asset_class(info) == "Equity"

    def test_unknown(self):
        info = {"category": "", "longName": ""}
        assert _classify_asset_class(info) == "Unknown"


# ---------------------------------------------------------------------------
# score_etf_fundamentals tests
# ---------------------------------------------------------------------------

class TestScoreETFFundamentals:
    """Tests for the ETF-specific fundamental scoring function."""

    def test_ultra_low_cost_high_aum(self):
        """An ultra-low-cost mega-cap ETF should score near the maximum (40)."""
        info = {
            "expense_ratio": 0.0003,   # 0.03 % — index fund level
            "aum": 400_000_000_000,    # $400B
            "yield_pct": 0.015,        # 1.5%
        }
        score, reasons = score_etf_fundamentals(info)
        assert score >= 35, f"Expected >= 35, got {score}"
        assert isinstance(reasons, list) and len(reasons) > 0

    def test_high_cost_small_aum(self):
        """An expensive, small ETF should score near 0."""
        info = {
            "expense_ratio": 0.015,   # 1.5% — very expensive
            "aum": 10_000_000,        # $10M — small
            "yield_pct": 0.0,
        }
        score, reasons = score_etf_fundamentals(info)
        assert score <= 10, f"Expected <= 10, got {score}"

    def test_missing_data_neutral(self):
        """Missing fields should yield a neutral score (~20)."""
        score, reasons = score_etf_fundamentals({})
        # With all fields missing, should get neutral (half of max ~20)
        assert 15 <= score <= 25, f"Expected neutral 15–25, got {score}"

    def test_score_in_range(self):
        """Score must always be between 0 and 40."""
        for expense, aum, y in [
            (0.0001, 500e9, 0.03),
            (0.02, 1e6, 0.0),
            (None, None, None),
        ]:
            info = {"expense_ratio": expense, "aum": aum, "yield_pct": y}
            score, _ = score_etf_fundamentals(info)
            assert 0 <= score <= 40, f"Score out of range: {score}"

    def test_expense_ratio_contribution(self):
        """Lower expense ratio should yield a higher score."""
        cheap = score_etf_fundamentals({"expense_ratio": 0.0003, "aum": None, "yield_pct": None})[0]
        expensive = score_etf_fundamentals({"expense_ratio": 0.01, "aum": None, "yield_pct": None})[0]
        assert cheap > expensive

    def test_aum_contribution(self):
        """Higher AUM should yield a higher score."""
        large = score_etf_fundamentals({"expense_ratio": None, "aum": 100e9, "yield_pct": None})[0]
        small = score_etf_fundamentals({"expense_ratio": None, "aum": 5e6, "yield_pct": None})[0]
        assert large > small

    def test_high_yield_bonus(self):
        """A high distribution yield should increase the score."""
        high_yield = score_etf_fundamentals({"expense_ratio": None, "aum": None, "yield_pct": 0.05})[0]
        no_yield = score_etf_fundamentals({"expense_ratio": None, "aum": None, "yield_pct": 0.0})[0]
        assert high_yield > no_yield


# ---------------------------------------------------------------------------
# score_etf_buy_signal tests
# ---------------------------------------------------------------------------

class TestScoreETFBuySignal:
    """Tests for the ETF buy signal scorer."""

    def _base_signal(self, etf_info=None):
        """Generate a base Phase 2 buy signal."""
        price_df = _make_price_df(260, base=100.0, trend=0.05)
        current_price = float(price_df["Close"].iloc[-1])
        phase_info = _make_phase2_info(current_price)
        rs_series = _make_rs_series(260, slope=0.0003)

        return score_etf_buy_signal(
            ticker="QQQ",
            price_data=price_df,
            current_price=current_price,
            phase_info=phase_info,
            rs_series=rs_series,
            etf_info=etf_info,
        )

    def test_phase2_passes_minervini(self):
        """A healthy Stage 2 ETF should at least enter the scoring logic."""
        sig = self._base_signal()
        assert "score" in sig
        assert "is_buy" in sig
        assert sig["phase"] == 2

    def test_non_phase2_rejected(self):
        """Non-Phase-2 ETFs must be rejected immediately."""
        price_df = _make_price_df(260)
        current_price = float(price_df["Close"].iloc[-1])
        phase_info = _make_phase2_info(current_price)
        phase_info["phase"] = 3  # Force Phase 3
        rs_series = _make_rs_series(260)

        sig = score_etf_buy_signal("SPY", price_df, current_price, phase_info, rs_series)
        assert sig["is_buy"] is False
        assert sig["score"] == 0

    def test_score_in_valid_range(self):
        """Score must be in [0, 125]."""
        sig = self._base_signal()
        assert 0 <= sig["score"] <= 125

    def test_etf_info_improves_score(self):
        """Providing high-quality ETF metadata should improve the score."""
        sig_no_info = self._base_signal(etf_info=None)

        good_etf_info = {
            "expense_ratio": 0.0003,
            "aum": 400_000_000_000,
            "yield_pct": 0.015,
        }
        sig_with_info = self._base_signal(etf_info=good_etf_info)

        # A low-cost mega-cap ETF should score higher than neutral
        assert sig_with_info["score"] >= sig_no_info["score"]

    def test_reasons_list_non_empty(self):
        """Reasons must be a non-empty list of strings."""
        sig = self._base_signal()
        if sig["score"] > 0:
            assert isinstance(sig["reasons"], list)
            assert len(sig["reasons"]) > 0

    def test_stop_loss_below_current_price(self):
        """Stop loss must always be below the current price."""
        sig = self._base_signal()
        if sig.get("stop_loss"):
            price_df = _make_price_df(260)
            current_price = float(price_df["Close"].iloc[-1])
            assert sig["stop_loss"] < current_price

    def test_output_keys_present(self):
        """Result dict must contain all expected keys."""
        expected_keys = {
            "ticker", "is_buy", "score", "phase", "stop_loss",
            "risk_reward_ratio", "entry_quality", "reasons", "details",
        }
        sig = self._base_signal()
        assert expected_keys.issubset(set(sig.keys()))


# ---------------------------------------------------------------------------
# score_etf_sell_signal tests
# ---------------------------------------------------------------------------

class TestScoreETFSellSignal:
    """Tests for the ETF sell signal scorer."""

    def _make_phase4_sell_signal(self):
        """Generate a Phase 4 (downtrend) sell signal."""
        price_df = _make_price_df(260, base=100.0, trend=-0.05)
        current_price = float(price_df["Close"].iloc[-1])

        sma_50 = current_price * 1.10   # Price below 50 SMA
        sma_200 = current_price * 1.20  # Price below 200 SMA

        phase_info = {
            "phase": 4,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "slope_50": -0.05,
            "slope_200": -0.02,
            "distance_from_50sma": ((current_price - sma_50) / sma_50) * 100,
        }
        rs_series = _make_rs_series(260, slope=-0.0004)

        return score_etf_sell_signal("TLT", price_df, current_price, phase_info, rs_series)

    def test_phase4_generates_sell(self):
        """A clear Phase 4 breakdown should be flagged as a sell."""
        sig = self._make_phase4_sell_signal()
        assert sig["is_sell"] == True  # noqa: E712 — numpy.bool_ vs bool
        assert sig["score"] > 0

    def test_phase2_not_a_sell(self):
        """An ETF in Phase 2 should not be flagged as a sell."""
        price_df = _make_price_df(260)
        current_price = float(price_df["Close"].iloc[-1])
        phase_info = _make_phase2_info(current_price)
        rs_series = _make_rs_series(260)

        sig = score_etf_sell_signal("QQQ", price_df, current_price, phase_info, rs_series)
        assert sig["is_sell"] is False

    def test_sell_score_in_valid_range(self):
        """Sell score must be in [0, 110]."""
        sig = self._make_phase4_sell_signal()
        assert 0 <= sig["score"] <= 110

    def test_sell_keys_present(self):
        """Sell result must contain all expected keys."""
        expected = {"ticker", "is_sell", "score", "phase", "breakdown_level", "severity", "reasons", "details"}
        sig = self._make_phase4_sell_signal()
        assert expected.issubset(set(sig.keys()))

    def test_severity_classification(self):
        """Severity should be one of the known values."""
        sig = self._make_phase4_sell_signal()
        assert sig["severity"] in ("critical", "high", "moderate")


# ---------------------------------------------------------------------------
# format_etf_signal_output tests
# ---------------------------------------------------------------------------

class TestFormatETFSignalOutput:
    def test_buy_format(self):
        sig = {
            "ticker": "QQQ",
            "is_buy": True,
            "is_sell": False,
            "score": 85.0,
            "phase": 2,
            "breakout_price": 420.0,
            "stop_loss": 395.0,
            "risk_reward_ratio": 3.2,
            "entry_quality": "Good",
            "reasons": ["Strong Stage 2", "Good RS"],
            "details": {},
        }
        output = format_etf_signal_output(sig)
        assert "BUY" in output
        assert "QQQ" in output
        assert "85.0" in output

    def test_sell_format(self):
        sig = {
            "ticker": "TLT",
            "is_buy": False,
            "is_sell": True,
            "score": 72.0,
            "phase": 4,
            "breakdown_level": 85.0,
            "severity": "high",
            "reasons": ["Stage 4 downtrend"],
            "details": {},
        }
        output = format_etf_signal_output(sig)
        assert "SELL" in output
        assert "TLT" in output
        assert "HIGH" in output
