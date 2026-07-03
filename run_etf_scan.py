#!/usr/bin/env python3
"""Full ETF market scanner.

This script applies the same Minervini/Weinstein Stage 2 methodology used by
``run_optimized_scan.py`` to the ETF universe.  The only difference is that
the fundamental-scoring block uses ETF-specific quality metrics (expense ratio,
AUM, yield) instead of company financials.

Usage::

    python run_etf_scan.py
    python run_etf_scan.py --popular-only          # Curated list (~110 ETFs, fast)
    python run_etf_scan.py --workers 5             # More parallel workers
    python run_etf_scan.py --test-mode             # First 50 ETFs only
    python run_etf_scan.py --min-aum 1000000000   # Only ETFs with AUM > $1B

Expected runtime (full universe):
    ~5–15 minutes (ETF universe is much smaller than stocks: 1,000–3,000 tickers)
"""

import argparse
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from src.data.etf_universe_fetcher import ETFUniverseFetcher
from src.data.etf_fetcher import ETFFetcher
from src.screening.etf_signal_engine import score_etf_buy_signal, score_etf_sell_signal
from src.screening.phase_indicators import (
    classify_phase,
    calculate_relative_strength,
    detect_vcp_pattern,
)
from src.screening.benchmark import (
    analyze_spy_trend,
    calculate_market_breadth,
    format_benchmark_summary,
    should_generate_signals,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def save_report(
    results: Dict,
    buy_signals: List[Dict],
    sell_signals: List[Dict],
    spy_analysis: Dict,
    breadth: Dict,
    output_dir: str = "./data/etf_scans",
) -> Path:
    """Persist the scan results to a timestamped text file.

    Also writes the report to ``latest_etf_scan.txt`` for easy access.

    Args:
        results: Batch processing statistics dict.
        buy_signals: Sorted buy signal list.
        sell_signals: Sorted sell signal list.
        spy_analysis: SPY trend analysis dict.
        breadth: Market breadth dict.
        output_dir: Directory to write reports into.

    Returns:
        Path to the timestamped report file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_str = datetime.now().strftime("%Y-%m-%d")

    output = []
    output.append("=" * 80)
    output.append("ETF MARKET SCAN")
    output.append(f"Scan Date: {date_str}")
    output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("=" * 80)
    output.append("")

    # Stats
    output.append("SCANNING STATISTICS")
    output.append("-" * 80)
    output.append(f"Total ETFs Processed: {results['total_processed']:,}")
    output.append(f"Analyzed (≥200 days data): {results['total_analyzed']:,}")
    output.append(f"Processing Time: {results['processing_time_seconds'] / 60:.1f} minutes")
    output.append(f"Actual TPS: {results['actual_tps']:.2f}")

    error_rate = results.get("error_rate", 0) * 100
    error_emoji = "🟢" if error_rate < 1 else ("🟡" if error_rate < 5 else "🔴")
    output.append(f"{error_emoji} Error Rate: {error_rate:.2f}%")
    output.append(f"{'🟢' if buy_signals else ''} Buy Signals: {len(buy_signals)}")
    output.append(f"{'🔴' if sell_signals else ''} Sell Signals: {len(sell_signals)}")
    output.append("")

    # Benchmark
    output.append(format_benchmark_summary(spy_analysis, breadth))
    output.append("")

    # Buy signals
    output.append("=" * 80)
    output.append(f"🟢 ETF BUY SIGNALS (Score ≥ 60) — {len(buy_signals)} Total")
    output.append("=" * 80)
    output.append("")

    if buy_signals:
        for i, sig in enumerate(buy_signals[:50], 1):
            sc = sig["score"]
            emoji = "⭐" if sc >= 90 else "🟢" if sc >= 70 else "🟡"
            output.append(f"\n{'#' * 80}")
            output.append(f"{emoji} BUY #{i}: {sig['ticker']} | Score: {sc}/125")
            output.append(f"{'#' * 80}")
            output.append(f"Phase: {sig['phase']}")
            output.append(f"Entry Quality: {sig.get('entry_quality', 'N/A')}")

            if sig.get("stop_loss"):
                rr = sig.get("risk_reward_ratio", 0)
                rr_emoji = "🟢" if rr >= 3 else "🟡"
                output.append(f"Stop Loss: ${sig['stop_loss']:.2f}")
                if rr:
                    output.append(f"{rr_emoji} R/R: {rr:.1f}:1")

            if sig.get("breakout_price"):
                output.append(f"Breakout: ${sig['breakout_price']:.2f}")

            details = sig.get("details", {})
            if "rs_slope" in details:
                rs = details["rs_slope"]
                rs_emoji = "🟢" if rs > 0.1 else ("🟡" if rs > 0 else "🔴")
                output.append(f"{rs_emoji} RS Slope: {rs:.3f}")
            if "volume_ratio" in details:
                vr = details["volume_ratio"]
                vr_emoji = "🟢" if vr > 1.5 else ("🟡" if vr > 1.0 else "🔴")
                output.append(f"{vr_emoji} Volume Ratio: {vr:.1f}x")

            # ETF-specific info from etf_info embedded in signal
            etf_info = sig.get("etf_info", {})
            if etf_info:
                if etf_info.get("expense_ratio") is not None:
                    er_pct = etf_info["expense_ratio"] * 100
                    output.append(f"Expense Ratio: {er_pct:.2f}%")
                if etf_info.get("aum"):
                    aum_b = etf_info["aum"] / 1e9
                    output.append(f"AUM: ${aum_b:.1f}B")
                if etf_info.get("category"):
                    output.append(f"Category: {etf_info['category']}")

            output.append("\nKey Reasons:")
            for reason in sig["reasons"][:7]:
                output.append(f"  • {reason}")

        if len(buy_signals) > 50:
            output.append(f"\n{'=' * 80}")
            output.append(f"ADDITIONAL BUYS ({len(buy_signals) - 50} more)")
            output.append(f"{'=' * 80}")
            remaining = [s["ticker"] for s in buy_signals[50:]]
            for i in range(0, len(remaining), 10):
                output.append(", ".join(remaining[i:i + 10]))
    else:
        output.append("✗ NO ETF BUY SIGNALS TODAY")

    # Sell signals
    output.append(f"\n\n{'=' * 80}")
    output.append(f"🔴 ETF SELL SIGNALS (Score ≥ 60) — {len(sell_signals)} Total")
    output.append(f"{'=' * 80}")
    output.append("")

    if sell_signals:
        for i, sig in enumerate(sell_signals[:30], 1):
            sc = sig["score"]
            severity = sig["severity"]
            sev_emoji = "🚨" if severity == "critical" else ("🔴" if severity == "high" else "🟡")
            output.append(f"\n{'#' * 80}")
            output.append(f"{sev_emoji} SELL #{i}: {sig['ticker']} | Score: {sc}/110 | {severity.upper()}")
            output.append(f"{'#' * 80}")
            output.append(f"Phase: {sig['phase']}")
            if sig.get("breakdown_level"):
                output.append(f"Breakdown: ${sig['breakdown_level']:.2f}")
            details = sig.get("details", {})
            if "rs_slope" in details:
                output.append(f"RS Slope: {details['rs_slope']:.3f}")
            output.append("\nSell Reasons:")
            for reason in sig["reasons"][:5]:
                output.append(f"  • {reason}")

        if len(sell_signals) > 30:
            remaining = [s["ticker"] for s in sell_signals[30:]]
            output.append(f"\nADDITIONAL SELLS ({len(sell_signals) - 30} more)")
            for i in range(0, len(remaining), 10):
                output.append(", ".join(remaining[i:i + 10]))
    else:
        output.append("✗ NO ETF SELL SIGNALS TODAY")

    output.append(f"\n\n{'=' * 80}")
    output.append("END OF ETF SCAN")
    output.append(f"{'=' * 80}\n")

    report_text = "\n".join(output)

    filepath = Path(output_dir) / f"etf_scan_{timestamp}.txt"
    latest_path = Path(output_dir) / "latest_etf_scan.txt"

    for path in (filepath, latest_path):
        with open(path, "w") as fh:
            fh.write(report_text)

    logger.info(f"ETF scan report saved: {filepath}")
    print(report_text)
    return filepath


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

class ETFBatchProcessor:
    """Parallel batch processor for ETF screening.

    A lightweight version of :class:`OptimizedBatchProcessor` tailored for
    ETFs.  Key differences:

    * Fetches ETF metadata (expense ratio, AUM, etc.) in addition to price data.
    * Skips the stock-specific fundamentals fetcher.
    * Applies a minimum-AUM filter to avoid very thinly traded ETFs.
    """

    def __init__(
        self,
        cache_dir: str = "./data/cache",
        results_dir: str = "./data/etf_scans",
        max_workers: int = 3,
        rate_limit_delay: float = 0.5,
    ):
        self.fetcher = ETFFetcher(cache_dir=cache_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay

        self.spy_data: Optional[pd.DataFrame] = None
        self.spy_price: Optional[float] = None

        self.total_requests = 0
        self.error_count = 0
        self.filtered_count = 0
        self.filter_reasons: Dict[str, int] = {}

        self._rate_lock = threading.Lock()
        self._last_request = 0.0

        logger.info(
            f"ETFBatchProcessor: {max_workers} workers, "
            f"{rate_limit_delay}s delay (~{max_workers / rate_limit_delay:.1f} TPS)"
        )

    def _wait_rate_limit(self) -> None:
        with self._rate_lock:
            elapsed = time.time() - self._last_request
            if elapsed < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - elapsed)
            self._last_request = time.time()

    def fetch_spy_data(self) -> bool:
        """Fetch SPY data for RS calculations."""
        try:
            logger.info("Fetching SPY data…")
            hist = self.fetcher.fetch_price_history("SPY", period="2y")
            if hist.empty or not isinstance(hist.index, pd.DatetimeIndex):
                logger.error("SPY data invalid")
                return False
            self.spy_data = hist
            self.spy_price = float(hist["Close"].iloc[-1])
            logger.info(f"SPY: {len(hist)} days, ${self.spy_price:.2f}")
            return True
        except Exception as exc:
            logger.error(f"SPY fetch failed: {exc}")
            return False

    def analyze_single_etf(
        self,
        ticker: str,
        min_price: float,
        min_volume: int,
        min_aum: float,
    ) -> Optional[Dict]:
        """Analyse one ETF.

        Args:
            ticker: ETF ticker.
            min_price: Minimum price filter.
            min_volume: Minimum average daily volume filter.
            min_aum: Minimum AUM in USD (default $100M).

        Returns:
            Analysis dict or ``None`` if the ETF does not pass filters.
        """
        try:
            self._wait_rate_limit()
            self.total_requests += 1

            # Price history (2 years for technical analysis)
            price_data = self.fetcher.fetch_price_history(ticker, period="2y")

            if price_data.empty or len(price_data) < 200:
                self.filtered_count += 1
                self.filter_reasons["insufficient_data"] = (
                    self.filter_reasons.get("insufficient_data", 0) + 1
                )
                return None

            current_price = float(price_data["Close"].iloc[-1])

            if current_price < min_price:
                self.filtered_count += 1
                self.filter_reasons["price_too_low"] = (
                    self.filter_reasons.get("price_too_low", 0) + 1
                )
                return None

            if "Volume" in price_data.columns:
                avg_vol = float(price_data["Volume"].iloc[-20:].mean())
                if avg_vol < min_volume:
                    self.filtered_count += 1
                    self.filter_reasons["low_volume"] = (
                        self.filter_reasons.get("low_volume", 0) + 1
                    )
                    return None
            else:
                avg_vol = 0.0

            # ETF metadata (expense ratio, AUM, category …)
            etf_info = self.fetcher.fetch_etf_info(ticker)

            # AUM filter
            aum = etf_info.get("aum") if etf_info else None
            if aum is not None and aum < min_aum:
                self.filtered_count += 1
                self.filter_reasons["aum_too_small"] = (
                    self.filter_reasons.get("aum_too_small", 0) + 1
                )
                return None

            # Phase classification
            phase_info = classify_phase(price_data, current_price)
            phase = phase_info["phase"]

            if phase not in [1, 2, 3, 4]:
                self.filtered_count += 1
                self.filter_reasons["invalid_phase"] = (
                    self.filter_reasons.get("invalid_phase", 0) + 1
                )
                return None

            # Relative strength vs SPY
            rs_series = calculate_relative_strength(
                price_data["Close"],
                self.spy_data["Close"],
                period=63,
            )

            # VCP detection (Phase 1/2 only)
            vcp_data: Dict = {}
            if phase in [1, 2]:
                vcp_data = detect_vcp_pattern(price_data, current_price, phase_info)

            return {
                "ticker": ticker,
                "price_data": price_data,
                "current_price": current_price,
                "phase_info": phase_info,
                "rs_series": rs_series,
                "vcp_data": vcp_data,
                "etf_info": etf_info,
                "avg_volume": avg_vol,
            }

        except Exception as exc:
            logger.error(f"Error analysing {ticker}: {exc}")
            self.error_count += 1
            return None

    def process_batch(
        self,
        tickers: List[str],
        min_price: float = 5.0,
        min_volume: int = 50_000,
        min_aum: float = 100_000_000,  # $100M
    ) -> Dict:
        """Process all ETFs in parallel.

        Args:
            tickers: List of ETF ticker symbols.
            min_price: Minimum price filter (default $5).
            min_volume: Minimum average daily volume (default 50,000).
            min_aum: Minimum AUM in USD (default $100M).

        Returns:
            Dict with ``analyses``, ``phase_results``, ``total_processed``,
            ``total_analyzed``, ``error_rate``, ``processing_time_seconds``,
            ``actual_tps``.
        """
        logger.info(f"Processing {len(tickers)} ETFs with {self.max_workers} workers…")
        start = time.time()

        analyses: List[Dict] = []
        phase_results: List[Dict] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(
                    self.analyze_single_etf,
                    ticker,
                    min_price,
                    min_volume,
                    min_aum,
                ): ticker
                for ticker in tickers
            }

            completed = 0
            for future in as_completed(futures):
                completed += 1
                if completed % 50 == 0:
                    elapsed = time.time() - start
                    logger.info(
                        f"Progress: {completed}/{len(tickers)} "
                        f"({completed / max(elapsed, 1):.1f} TPS)"
                    )

                result = future.result()
                if result is None:
                    continue

                analyses.append(result)
                phase_results.append({
                    "ticker": result["ticker"],
                    "phase": result["phase_info"]["phase"],
                })

        elapsed = time.time() - start
        actual_tps = self.total_requests / max(elapsed, 1)

        logger.info(
            f"Batch complete: {len(analyses)} analysed, "
            f"{self.filtered_count} filtered, {self.error_count} errors"
        )
        logger.info(f"Filter breakdown: {self.filter_reasons}")

        return {
            "analyses": analyses,
            "phase_results": phase_results,
            "total_processed": len(tickers),
            "total_analyzed": len(analyses),
            "error_rate": self.error_count / max(self.total_requests, 1),
            "processing_time_seconds": elapsed,
            "actual_tps": actual_tps,
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point for the ETF scanner."""
    parser = argparse.ArgumentParser(description="ETF Market Scanner")
    parser.add_argument("--workers", type=int, default=3, help="Parallel workers (default: 3)")
    parser.add_argument("--delay", type=float, default=0.5, help="Per-request delay in seconds (default: 0.5)")
    parser.add_argument("--popular-only", action="store_true", help="Screen only the curated popular ETF list (~110 ETFs)")
    parser.add_argument("--test-mode", action="store_true", help="Process first 50 ETFs only")
    parser.add_argument("--min-price", type=float, default=5.0, help="Min price filter (default: $5)")
    parser.add_argument("--min-volume", type=int, default=50_000, help="Min avg daily volume (default: 50,000)")
    parser.add_argument("--min-aum", type=float, default=100_000_000, help="Min AUM in USD (default: $100M)")
    parser.add_argument("--output-dir", type=str, default="./data/etf_scans", help="Output directory")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ETF MARKET SCANNER — STARTING")
    logger.info("=" * 60)

    # --- Fetch ETF universe ---
    universe_fetcher = ETFUniverseFetcher()

    if args.popular_only:
        tickers = universe_fetcher.get_popular_etfs()
        logger.info(f"Popular ETF list: {len(tickers)} tickers")
    else:
        logger.info("Fetching full ETF universe…")
        tickers = universe_fetcher.fetch_universe()
        if not tickers:
            logger.error("Failed to fetch ETF universe — aborting")
            sys.exit(1)
        logger.info(f"ETF universe: {len(tickers)} tickers")

    if args.test_mode:
        tickers = tickers[:50]
        logger.info(f"TEST MODE: {len(tickers)} ETFs")

    # --- Processor ---
    processor = ETFBatchProcessor(
        max_workers=args.workers,
        rate_limit_delay=args.delay,
    )

    # --- SPY data ---
    if not processor.fetch_spy_data():
        logger.error("Cannot continue without SPY data")
        sys.exit(1)

    # --- Process ---
    results = processor.process_batch(
        tickers,
        min_price=args.min_price,
        min_volume=args.min_volume,
        min_aum=args.min_aum,
    )

    # --- Market context ---
    spy_analysis = analyze_spy_trend(processor.spy_data, processor.spy_price)
    breadth = calculate_market_breadth(results["phase_results"])
    signal_rec = should_generate_signals(spy_analysis, breadth)

    # --- Generate signals ---
    buy_signals: List[Dict] = []
    sell_signals: List[Dict] = []

    if signal_rec["should_generate_buys"]:
        for analysis in results["analyses"]:
            if analysis["phase_info"]["phase"] == 2:
                sig = score_etf_buy_signal(
                    ticker=analysis["ticker"],
                    price_data=analysis["price_data"],
                    current_price=analysis["current_price"],
                    phase_info=analysis["phase_info"],
                    rs_series=analysis["rs_series"],
                    etf_info=analysis.get("etf_info"),
                    vcp_data=analysis.get("vcp_data"),
                )
                if sig["is_buy"]:
                    sig["etf_info"] = analysis.get("etf_info", {})
                    buy_signals.append(sig)
    else:
        logger.info("Market conditions unfavourable — skipping buy signals")
        for reason in signal_rec.get("reasons", []):
            logger.info(f"  • {reason}")

    if signal_rec["should_generate_sells"]:
        for analysis in results["analyses"]:
            if analysis["phase_info"]["phase"] in [3, 4]:
                sig = score_etf_sell_signal(
                    ticker=analysis["ticker"],
                    price_data=analysis["price_data"],
                    current_price=analysis["current_price"],
                    phase_info=analysis["phase_info"],
                    rs_series=analysis["rs_series"],
                )
                if sig["is_sell"]:
                    sell_signals.append(sig)

    buy_signals = sorted(buy_signals, key=lambda x: x["score"], reverse=True)
    sell_signals = sorted(sell_signals, key=lambda x: x["score"], reverse=True)

    logger.info(f"Buy signals: {len(buy_signals)}")
    logger.info(f"Sell signals: {len(sell_signals)}")

    # --- Report ---
    save_report(results, buy_signals, sell_signals, spy_analysis, breadth, args.output_dir)

    logger.info("=" * 60)
    logger.info("ETF SCAN COMPLETE")
    logger.info(f"Time: {results['processing_time_seconds'] / 60:.1f} minutes")
    logger.info(f"Buy signals: {len(buy_signals)}")
    logger.info(f"Sell signals: {len(sell_signals)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as exc:
        logger.error(f"Fatal error: {exc}", exc_info=True)
        sys.exit(1)
