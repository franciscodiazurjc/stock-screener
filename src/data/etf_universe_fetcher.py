"""Fetch and maintain the universe of US-listed ETFs.

This module fetches the complete list of US-traded ETFs from multiple sources
and maintains a daily-updated universe for ETF screening.  The implementation
mirrors the design of :mod:`src.data.universe_fetcher` but targets ETFs instead
of individual stocks.
"""

import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Curated list of the most liquid and widely-followed US-traded ETFs.
# This list is used as a reliable fallback when the live NASDAQ feed is not
# reachable.  It covers the most common asset classes:
#   - Broad US equity (S&P 500, total market, Russell 2000 …)
#   - Sector equity (XLK, XLF, XLE …)
#   - International equity
#   - Fixed income
#   - Commodities
#   - Volatility / alternative
# ---------------------------------------------------------------------------
POPULAR_ETFS: List[str] = [
    # Broad US equity
    "SPY", "IVV", "VOO", "VTI", "QQQ", "IWM", "DIA", "MDY", "IJH", "IJR",
    "VUG", "VTV", "SCHB", "SCHX", "ITOT",
    # Sector ETFs
    "XLK", "XLF", "XLV", "XLE", "XLI", "XLU", "XLP", "XLY", "XLB", "XLRE",
    "VGT", "VFH", "VHT", "VDE", "VIS", "VPU", "VDC", "VCR", "VAW", "KBWB",
    "IBB", "XBI", "SOXX", "SMH", "HACK", "FINX", "ARKK", "ARKW", "ARKG",
    "ARKF", "ARKQ",
    # International equity
    "EFA", "EEM", "VEA", "VWO", "IEFA", "IEMG", "ACWI", "VXUS",
    "EWJ", "EWZ", "EWG", "EWU", "EWC", "INDA", "FXI", "MCHI",
    "EZU", "RSX", "GXC", "KWEB",
    # Fixed income
    "AGG", "BND", "LQD", "HYG", "JNK", "TLT", "IEF", "SHY", "TIP",
    "VCIT", "VCSH", "BSV", "BIV", "BLV", "MBB", "EMB",
    # Commodities
    "GLD", "IAU", "SLV", "USO", "UNG", "DBC", "PDBC", "CORN", "WEAT",
    "COPX", "GDX", "GDXJ",
    # Real estate
    "VNQ", "SCHH", "IYR", "REM",
    # Factor / Smart beta
    "MTUM", "VLUE", "QUAL", "SIZE", "USMV", "SPLV", "DGRO", "VIG", "SDY",
    "DVY", "NOBL",
    # Leveraged / Inverse (commonly screened)
    "TQQQ", "SQQQ", "UPRO", "SPXU", "SPXL", "TECL",
    # Thematic
    "ICLN", "TAN", "CIBR", "ROBO", "BOTZ", "CLOUD", "WCLD", "GAMR",
    "HERO", "ESPO", "SNSR",
    # Short-duration / Money market alternative
    "BIL", "SGOV", "JPST", "NEAR",
]


class ETFUniverseFetcher:
    """Fetches and maintains the universe of US-listed ETFs.

    Attempts to fetch live ETF tickers from the NASDAQ FTP feeds (same source
    used by :class:`~src.data.universe_fetcher.USStockUniverseFetcher`).  If the
    live feed is unavailable the class falls back to the curated
    :data:`POPULAR_ETFS` list, ensuring the screener always has a working
    universe.

    The fetched universe is cached on disk for 24 hours to avoid hammering the
    data source on repeated runs.

    Example::

        fetcher = ETFUniverseFetcher()
        tickers = fetcher.fetch_universe()
        print(f"ETF universe: {len(tickers)} tickers")
    """

    def __init__(self, cache_dir: str = "./data/cache"):
        """Initialize the ETF universe fetcher.

        Args:
            cache_dir: Directory for caching universe data.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "etf_universe.pkl"
        logger.info("ETFUniverseFetcher initialized")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_nasdaq_etf_list(self) -> pd.DataFrame:
        """Fetch ETF tickers from the NASDAQ ETF screener endpoint.

        Returns:
            DataFrame with columns ``symbol`` and ``name``.
        """
        try:
            headers = {"User-Agent": "Mozilla/5.0 (ETF Screener Bot)"}
            url = (
                "https://api.nasdaq.com/api/screener/etf"
                "?tableonly=true&limit=5000&exchange=ALL"
            )
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()
            data = resp.json()

            rows = data.get("data", {}).get("data", {}).get("rows", [])
            if not rows:
                logger.warning("NASDAQ ETF API returned empty rows")
                return pd.DataFrame()

            df = pd.DataFrame(rows)
            if "symbol" not in df.columns:
                logger.warning("NASDAQ ETF API response missing 'symbol' column")
                return pd.DataFrame()

            df = df[["symbol", "companyName"]].rename(columns={"companyName": "name"})
            df = df[df["symbol"].notna() & (df["symbol"] != "")]
            logger.info(f"Fetched {len(df)} ETFs from NASDAQ API")
            return df

        except Exception as exc:
            logger.error(f"Error fetching NASDAQ ETF list: {exc}")
            return pd.DataFrame()

    def _fetch_other_listed_etfs(self) -> pd.DataFrame:
        """Extract ETFs from the NASDAQ FTP *otherlisted* feed.

        The ``otherlisted.txt`` feed contains NYSE/AMEX securities, many of
        which are ETFs.  We identify ETFs by the presence of keywords such as
        "ETF", "FUND", "INDEX", "TRUST" in the security name.

        Returns:
            DataFrame with columns ``symbol`` and ``name``.
        """
        try:
            url = "ftp://ftp.nasdaqtrader.com/symboldirectory/otherlisted.txt"
            df = pd.read_csv(url, sep="|")
            df = df[df["ACT Symbol"].notna()]
            df = df[df["Test Issue"] == "N"]
            df = df[["ACT Symbol", "Security Name"]].copy()
            df.columns = ["symbol", "name"]

            etf_keywords = ["ETF", "FUND", "INDEX", "TRUST", "PORTFOLIO", "SHARES"]
            name_upper = df["name"].str.upper()
            mask = pd.Series(False, index=df.index)
            for kw in etf_keywords:
                mask = mask | name_upper.str.contains(kw, na=False)
            df = df[mask]

            logger.info(f"Extracted {len(df)} ETF candidates from NASDAQ otherlisted")
            return df
        except Exception as exc:
            logger.error(f"Error fetching otherlisted ETFs: {exc}")
            return pd.DataFrame()

    def _filter_etfs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove obviously invalid tickers.

        Args:
            df: DataFrame with ``symbol`` and ``name`` columns.

        Returns:
            Filtered DataFrame.
        """
        if df.empty:
            return df

        initial = len(df)

        # Only allow alphanumeric tickers (no $, ^, ., -, etc.)
        df = df[~df["symbol"].str.contains(r"[^A-Z0-9]", regex=True, na=False)]

        # Length: ETF tickers are 2–6 characters (some leveraged have 5-6)
        df = df[df["symbol"].str.len().between(2, 6)]

        # Drop duplicates
        df = df.drop_duplicates(subset=["symbol"])

        logger.info(f"ETF filter: {initial} → {len(df)} kept")
        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_universe(self, force_refresh: bool = False) -> List[str]:
        """Fetch the complete universe of US-listed ETFs.

        Tries live sources first; falls back to the curated :data:`POPULAR_ETFS`
        list if all live sources fail.  Results are cached for 24 hours.

        Args:
            force_refresh: Force a fresh fetch even if the cache is recent.

        Returns:
            List of ETF ticker symbols (e.g. ``["SPY", "QQQ", …]``).
        """
        # --- Cache check ---
        if not force_refresh and self.cache_file.exists():
            age = datetime.now() - datetime.fromtimestamp(
                self.cache_file.stat().st_mtime
            )
            if age < timedelta(days=1):
                logger.info("Loading ETF universe from cache")
                with open(self.cache_file, "rb") as fh:
                    cached = pickle.load(fh)
                logger.info(f"Loaded {len(cached['symbols'])} ETFs from cache")
                return cached["symbols"]

        logger.info("Fetching fresh ETF universe...")

        # --- Try live NASDAQ API first ---
        nasdaq_df = self._fetch_nasdaq_etf_list()

        # --- Try otherlisted FTP as secondary source ---
        other_df = self._fetch_other_listed_etfs()

        # --- Combine and filter ---
        frames = [df for df in [nasdaq_df, other_df] if not df.empty]

        if frames:
            all_etfs = pd.concat(frames, ignore_index=True)
            all_etfs = self._filter_etfs(all_etfs)
            symbols = all_etfs["symbol"].tolist()
            source = "live"
        else:
            # Fallback to curated list
            logger.warning("All live sources failed — using curated ETF list")
            symbols = list(POPULAR_ETFS)
            source = "fallback"

        if not symbols:
            logger.error("Empty ETF universe — using curated list")
            symbols = list(POPULAR_ETFS)
            source = "fallback"

        symbols = sorted(set(symbols))

        # --- Persist to cache ---
        cache_data = {
            "symbols": symbols,
            "fetch_date": datetime.now().isoformat(),
            "count": len(symbols),
            "source": source,
        }
        try:
            with open(self.cache_file, "wb") as fh:
                pickle.dump(cache_data, fh)
            logger.info(f"Cached {len(symbols)} ETF tickers (source: {source})")
        except Exception as exc:
            logger.warning(f"Could not write ETF universe cache: {exc}")

        return symbols

    def get_popular_etfs(self) -> List[str]:
        """Return the curated list of popular/liquid ETFs.

        This is always available without network access and is useful for quick
        testing or when you only want to screen well-known ETFs.

        Returns:
            Sorted list of popular ETF ticker symbols.
        """
        return sorted(POPULAR_ETFS)

    def get_universe_info(self) -> Dict:
        """Return metadata about the cached ETF universe.

        Returns:
            Dict with keys: ``cached``, ``count``, ``fetch_date``,
            ``cache_age_hours``, ``source``.
        """
        if not self.cache_file.exists():
            return {"cached": False, "count": len(POPULAR_ETFS), "source": "fallback"}

        with open(self.cache_file, "rb") as fh:
            cached = pickle.load(fh)

        age = datetime.now() - datetime.fromtimestamp(
            self.cache_file.stat().st_mtime
        )
        return {
            "cached": True,
            "count": cached["count"],
            "fetch_date": cached["fetch_date"],
            "cache_age_hours": age.total_seconds() / 3600,
            "source": cached.get("source", "unknown"),
        }
