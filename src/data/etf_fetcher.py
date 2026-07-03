"""ETF data fetching module for retrieving ETF-specific metrics from Yahoo Finance.

ETFs differ from individual stocks in that they lack traditional fundamental
metrics (EPS, revenue, P/E ratio, etc.).  This module fetches the ETF-relevant
attributes that *are* available:

- **Expense ratio** — the annual management fee charged to holders (lower = better).
- **AUM** (total assets) — assets under management; higher = more liquid.
- **Category / asset class** — broad classification (e.g. "Large Blend", "Technology").
- **NAV** — Net Asset Value per share.
- **Distribution yield** — dividend/interest yield paid to holders.
- **52-week high / low** — price range for technical context.
- **Trailing 3-month return** — short-term performance proxy.

The fetcher reuses the caching infrastructure already established by
:class:`~src.data.fetcher.YahooFinanceFetcher`.
"""

import logging
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ETFFetcher:
    """Fetches ETF-specific data from Yahoo Finance with caching and retry logic.

    Mirrors the interface of :class:`~src.data.fetcher.YahooFinanceFetcher` but
    targets ETF-specific fields.  Price-history fetching is *inherited* from the
    same yfinance calls, so this class can be used as a drop-in replacement when
    the underlying instrument is an ETF.

    Attributes:
        cache_dir: Path to the directory used for caching.
        cache_expiry_hours: Hours before a cached entry is considered stale.
        max_retries: Maximum API retry attempts per ticker.
        retry_delay: Seconds to wait between retries.

    Example::

        fetcher = ETFFetcher()
        info = fetcher.fetch_etf_info("QQQ")
        print(info["expense_ratio"], info["aum"], info["category"])
        prices = fetcher.fetch_price_history("QQQ", period="2y")
    """

    def __init__(
        self,
        cache_dir: str = "./data/cache",
        cache_expiry_hours: int = 24,
        max_retries: int = 3,
        retry_delay: int = 2,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry_hours = cache_expiry_hours
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        logger.info(f"ETFFetcher initialized (cache: {cache_dir})")

    # ------------------------------------------------------------------
    # Cache helpers (identical to YahooFinanceFetcher)
    # ------------------------------------------------------------------

    def _cache_path(self, ticker: str, data_type: str) -> Path:
        return self.cache_dir / f"etf_{ticker}_{data_type}.pkl"

    def _cache_valid(self, path: Path) -> bool:
        if not path.exists():
            return False
        modified = datetime.fromtimestamp(path.stat().st_mtime)
        return modified > datetime.now() - timedelta(hours=self.cache_expiry_hours)

    def _load_cache(self, path: Path):
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)
        except Exception as exc:
            logger.warning(f"Cache load failed ({path.name}): {exc}")
            return None

    def _save_cache(self, data, path: Path) -> None:
        try:
            with open(path, "wb") as fh:
                pickle.dump(data, fh)
        except Exception as exc:
            logger.warning(f"Cache save failed ({path.name}): {exc}")

    def _fetch_ticker(self, ticker: str) -> Optional[yf.Ticker]:
        """Return a yfinance Ticker object with retry logic."""
        for attempt in range(self.max_retries):
            try:
                t = yf.Ticker(ticker)
                _ = t.info  # Validate the ticker is accessible
                return t
            except Exception as exc:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} for {ticker}: {exc}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        logger.error(f"Failed to fetch {ticker} after {self.max_retries} attempts")
        return None

    # ------------------------------------------------------------------
    # ETF info
    # ------------------------------------------------------------------

    def fetch_etf_info(self, ticker: str) -> Dict[str, any]:
        """Fetch ETF-specific metadata for *ticker*.

        Fields returned:

        ============== ==================================================
        Key            Description
        ============== ==================================================
        ticker         Ticker symbol (uppercased)
        name           Long name / fund name
        category       Morningstar-style category (e.g. "Large Growth")
        asset_class    Broad class: "Equity", "Fixed Income", "Commodity",
                       "Real Estate", "Multi-Asset", or "Unknown"
        expense_ratio  Annual expense ratio as a decimal (e.g. 0.0003)
        aum            Total net assets (AUM) in USD
        nav            Net Asset Value per share
        yield_pct      Trailing 12-month distribution yield (0–1 decimal)
        current_price  Most recent closing price
        week_52_high   52-week high price
        week_52_low    52-week low price
        beta_3y        3-year beta vs market (may be None)
        fetch_date     ISO timestamp of when the data was fetched
        ============== ==================================================

        Args:
            ticker: ETF ticker symbol (e.g. ``"SPY"``).

        Returns:
            Dictionary with ETF metrics.  Returns an empty dict on failure.
        """
        cache_path = self._cache_path(ticker, "info")
        if self._cache_valid(cache_path):
            cached = self._load_cache(cache_path)
            if cached is not None:
                return cached

        logger.info(f"Fetching ETF info for {ticker}")
        t = self._fetch_ticker(ticker)
        if t is None:
            logger.error(f"Could not fetch ETF info for {ticker}")
            return {}

        try:
            info = t.info

            # Expense ratio — yfinance exposes this under different keys depending
            # on the ETF provider; we try all known variants.
            expense_ratio = (
                info.get("annualReportExpenseRatio")
                or info.get("totalExpenseRatio")
                or info.get("expenseRatio")
            )

            category = info.get("category", "Unknown")
            asset_class = _classify_asset_class(info)

            result: Dict[str, any] = {
                "ticker": ticker.upper(),
                "name": info.get("longName") or info.get("shortName", ticker),
                "category": category,
                "asset_class": asset_class,
                "expense_ratio": expense_ratio,
                "aum": info.get("totalAssets"),
                "nav": info.get("navPrice") or info.get("regularMarketPrice"),
                "yield_pct": info.get("yield"),
                "current_price": (
                    info.get("currentPrice")
                    or info.get("regularMarketPrice")
                    or info.get("navPrice")
                ),
                "week_52_high": info.get("fiftyTwoWeekHigh"),
                "week_52_low": info.get("fiftyTwoWeekLow"),
                "beta_3y": info.get("beta3Year"),
                "fetch_date": datetime.now().isoformat(),
            }

            missing = [k for k, v in result.items() if v is None and k not in ("ticker", "fetch_date")]
            if missing:
                logger.warning(f"{ticker}: missing fields: {', '.join(missing)}")

            self._save_cache(result, cache_path)
            logger.info(f"ETF info fetched for {ticker}")
            return result

        except Exception as exc:
            logger.error(f"Error parsing ETF info for {ticker}: {exc}")
            return {}

    def fetch_price_history(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV price history for an ETF.

        Identical in behaviour to :meth:`YahooFinanceFetcher.fetch_price_history`;
        the ETF-specific prefix on the cache key prevents collisions.

        Args:
            ticker: ETF ticker symbol.
            period: History window (default ``"2y"``).
            interval: Data frequency (default ``"1d"``).

        Returns:
            DataFrame with a DatetimeIndex and columns
            ``Open, High, Low, Close, Volume``.
            Returns an empty DataFrame on failure.
        """
        cache_path = self._cache_path(ticker, f"prices_{period}_{interval}")
        if self._cache_valid(cache_path):
            cached = self._load_cache(cache_path)
            if cached is not None and isinstance(cached, pd.DataFrame):
                return cached

        logger.info(f"Fetching price history for ETF {ticker} (period={period})")
        t = self._fetch_ticker(ticker)
        if t is None:
            return pd.DataFrame()

        try:
            hist = t.history(period=period, interval=interval)
            if hist.empty:
                logger.warning(f"No price data for {ticker}")
                return pd.DataFrame()

            hist.columns = [c.capitalize() for c in hist.columns]

            if not isinstance(hist.index, pd.DatetimeIndex):
                logger.warning(f"{ticker}: unexpected index type {type(hist.index)}")
                return pd.DataFrame()

            cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in hist.columns]
            hist = hist[cols]

            self._save_cache(hist, cache_path)
            logger.info(f"Price history fetched for {ticker}: {len(hist)} days")
            return hist

        except Exception as exc:
            logger.error(f"Error fetching price history for {ticker}: {exc}")
            return pd.DataFrame()

    def fetch_multiple(
        self,
        tickers: List[str],
        period: str = "2y",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch ETF info and price history for multiple tickers.

        Args:
            tickers: List of ETF ticker symbols.
            period: History window passed to :meth:`fetch_price_history`.

        Returns:
            ``(info_df, prices_df)`` where *info_df* has one row per ETF and
            *prices_df* is the concatenated price history with a ``ticker`` column.
        """
        logger.info(f"Fetching data for {len(tickers)} ETFs")
        all_info: List[Dict] = []
        all_prices: List[pd.DataFrame] = []

        for ticker in tickers:
            info = self.fetch_etf_info(ticker)
            if info:
                all_info.append(info)

            prices = self.fetch_price_history(ticker, period=period)
            if not prices.empty:
                prices = prices.copy()
                prices["ticker"] = ticker
                all_prices.append(prices)

        info_df = pd.DataFrame(all_info) if all_info else pd.DataFrame()
        prices_df = (
            pd.concat(all_prices, ignore_index=True) if all_prices else pd.DataFrame()
        )

        logger.info(
            f"Fetched {len(all_info)}/{len(tickers)} ETF infos, "
            f"{len(all_prices)}/{len(tickers)} price histories"
        )
        return info_df, prices_df

    def clear_cache(self, ticker: Optional[str] = None) -> None:
        """Clear cached ETF data.

        Args:
            ticker: If given, clears cache only for that ticker.
                    If ``None``, clears all ETF cache files.
        """
        pattern = f"etf_{ticker}_*.pkl" if ticker else "etf_*.pkl"
        removed = 0
        for f in self.cache_dir.glob(pattern):
            try:
                f.unlink()
                removed += 1
            except Exception as exc:
                logger.warning(f"Could not remove {f}: {exc}")
        logger.info(f"Cleared {removed} ETF cache file(s)")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _classify_asset_class(info: Dict) -> str:
    """Infer the broad asset class of an ETF from yfinance *info*.

    Args:
        info: Raw dict returned by ``yf.Ticker.info``.

    Returns:
        One of ``"Equity"``, ``"Fixed Income"``, ``"Commodity"``,
        ``"Real Estate"``, ``"Multi-Asset"``, ``"Currency"``, or ``"Unknown"``.
    """
    category = (info.get("category") or "").lower()
    name = (info.get("longName") or info.get("shortName") or "").lower()

    keywords_map = [
        (["bond", "fixed income", "treasury", "credit", "debt", "income", "maturity"], "Fixed Income"),
        (["gold", "silver", "commodity", "oil", "natural gas", "metal", "energy commodity"], "Commodity"),
        (["real estate", "reit"], "Real Estate"),
        (["currency", "forex", "fx "], "Currency"),
        (["multi-asset", "balanced", "allocation", "target date"], "Multi-Asset"),
        (["equity", "stock", "large", "small", "mid", "growth", "value", "blend",
          "sector", "technology", "health", "financial", "consumer",
          "industrial", "energy", "material", "utility", "emerging market",
          "international", "global"], "Equity"),
    ]

    combined = f"{category} {name}"
    for keywords, asset_class in keywords_map:
        if any(kw in combined for kw in keywords):
            return asset_class

    return "Unknown"


def score_etf_fundamentals(etf_info: Dict[str, any]) -> Tuple[float, List[str]]:
    """Score an ETF based on its non-technical attributes.

    This replaces the stock-specific fundamental scoring (P/E, EPS growth, etc.)
    with ETF-relevant criteria.

    Scoring components (total 40 points):

    ==================  ========  =======================================================
    Component           Max pts   Rationale
    ==================  ========  =======================================================
    Expense ratio       15 pts    Lower fees compound significantly over time.
    AUM / liquidity     15 pts    Larger AUM = better NAV tracking + tighter spreads.
    Distribution yield  10 pts    Income generation; relevant but secondary to cost.
    ==================  ========  =======================================================

    Args:
        etf_info: Dict returned by :meth:`ETFFetcher.fetch_etf_info`.

    Returns:
        ``(score, reasons)`` where *score* is in ``[0, 40]`` and *reasons* is a
        list of human-readable strings explaining the score.
    """
    score = 0.0
    reasons: List[str] = []

    # ------------------------------------------------------------------ #
    # 1. Expense ratio (15 points)                                         #
    #    < 0.10% (10 bps) → 15 pts  (index / passive ETF)                #
    #    < 0.20% (20 bps) → ~12 pts                                       #
    #    < 0.50% (50 bps) → ~8 pts                                        #
    #    > 1.00% (100 bps) → 0 pts  (expensive active / leveraged)       #
    # ------------------------------------------------------------------ #
    expense_ratio = etf_info.get("expense_ratio")
    if expense_ratio is not None and expense_ratio >= 0:
        pct = expense_ratio * 100  # convert to percentage points

        if pct <= 0.10:
            er_score = 15.0
        elif pct <= 1.00:
            # Linear: 0.10% → 15 pts, 1.00% → 0 pts
            er_score = 15.0 * (1.0 - (pct - 0.10) / 0.90)
        else:
            er_score = 0.0

        score += er_score

        if pct <= 0.10:
            reasons.append(f"🟢 Expense ratio: {pct:.2f}% (ultra-low cost)")
        elif pct <= 0.25:
            reasons.append(f"🟢 Expense ratio: {pct:.2f}% (low cost)")
        elif pct <= 0.50:
            reasons.append(f"🟡 Expense ratio: {pct:.2f}% (moderate)")
        else:
            reasons.append(f"🔴 Expense ratio: {pct:.2f}% (high)")
    else:
        # Unknown — neutral (half points)
        score += 7.5
        reasons.append("Expense ratio: unknown (neutral)")

    # ------------------------------------------------------------------ #
    # 2. AUM / Liquidity (15 points)                                      #
    #    >= $50B  → 15 pts                                                #
    #    >= $10B  → ~12 pts                                               #
    #    >= $1B   → ~7.5 pts                                              #
    #    >= $100M → ~3 pts                                                #
    #    <  $100M → 0 pts  (thinly traded / tracking risk)               #
    # ------------------------------------------------------------------ #
    aum = etf_info.get("aum")
    if aum is not None and aum > 0:
        aum_b = aum / 1e9  # in billions

        if aum_b >= 50:
            aum_score = 15.0
        elif aum_b >= 0.1:
            # Log-linear: $100M → 0 pts, $50B → 15 pts
            import math
            log_min = math.log10(0.1)
            log_max = math.log10(50)
            aum_score = 15.0 * (math.log10(aum_b) - log_min) / (log_max - log_min)
            aum_score = max(0.0, min(15.0, aum_score))
        else:
            aum_score = 0.0

        score += aum_score

        if aum_b >= 50:
            reasons.append(f"🟢 AUM: ${aum_b:.1f}B (mega-cap, highly liquid)")
        elif aum_b >= 10:
            reasons.append(f"🟢 AUM: ${aum_b:.1f}B (large, liquid)")
        elif aum_b >= 1:
            reasons.append(f"🟡 AUM: ${aum_b:.1f}B (medium)")
        else:
            reasons.append(f"🔴 AUM: ${aum * 1e-6:.0f}M (small, possible tracking risk)")
    else:
        score += 7.5  # Neutral
        reasons.append("AUM: unknown (neutral)")

    # ------------------------------------------------------------------ #
    # 3. Distribution yield (10 points)                                   #
    #    >= 4%  → 10 pts                                                  #
    #    >= 2%  → 7  pts                                                  #
    #    >= 0.5%→ 4  pts                                                  #
    #    0%     → 2  pts  (growth ETF — not penalised)                    #
    # ------------------------------------------------------------------ #
    yield_pct = etf_info.get("yield_pct")
    if yield_pct is not None:
        y = yield_pct * 100  # convert 0.04 → 4.0

        if y >= 4.0:
            yield_score = 10.0
        elif y >= 0.5:
            # Linear: 0.5% → 3 pts, 4% → 10 pts
            yield_score = 3.0 + (y - 0.5) / 3.5 * 7.0
        elif y > 0:
            yield_score = 2.0
        else:
            yield_score = 2.0  # Growth ETF — no yield is fine

        score += yield_score

        if y >= 3.0:
            reasons.append(f"🟢 Yield: {y:.2f}% (income-generating)")
        elif y >= 1.0:
            reasons.append(f"🟡 Yield: {y:.2f}%")
        elif y > 0:
            reasons.append(f"Yield: {y:.2f}% (low income)")
        else:
            reasons.append("Yield: N/A (growth-oriented)")
    else:
        score += 5.0  # Neutral
        reasons.append("Yield: unknown (neutral)")

    score = round(min(max(score, 0.0), 40.0), 2)
    return score, reasons
