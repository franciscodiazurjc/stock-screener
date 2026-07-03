#!/usr/bin/env python3
"""Check the current status of a single ticker.

Analyzes a ticker and reports:
- Current phase (1-4)
- Buy / sell signal with score
- Breakout price, Stop Loss, Take Profit, Risk/Reward ratio
- Minervini Trend Template criteria
- Relative Strength vs SPY
- Volume analysis
- VCP pattern detection
- Key reasons for signal

Usage:
    python check_ticker.py AAPL
    python check_ticker.py NVDA --use-fmp
    python check_ticker.py TSLA --verbose
"""

import argparse
import logging
import sys
from datetime import datetime

import yfinance as yf

from src.data.fetcher import YahooFinanceFetcher
from src.data.fundamentals_fetcher import (
    fetch_quarterly_financials,
    analyze_fundamentals_for_signal,
)
from src.data.enhanced_fundamentals import EnhancedFundamentalsFetcher
from src.screening.phase_indicators import (
    classify_phase,
    calculate_relative_strength,
    calculate_sma,
    detect_vcp_pattern,
)
from src.screening.signal_engine import score_buy_signal, score_sell_signal

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _phase_label(phase: int) -> str:
    labels = {
        0: "Insufficient Data",
        1: "Phase 1 – Base Building / Accumulation",
        2: "Phase 2 – Uptrend / Breakout",
        3: "Phase 3 – Distribution / Top",
        4: "Phase 4 – Downtrend",
    }
    return labels.get(phase, f"Phase {phase}")


def _yn(value: bool) -> str:
    return "✅ YES" if value else "❌ NO"


def _fmt_price(price) -> str:
    if price is None:
        return "N/A"
    return f"${price:.2f}"


def _fmt_pct(value) -> str:
    if value is None:
        return "N/A"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}%"


def _signal_bar(score: float, max_score: int) -> str:
    """Visual score bar."""
    pct = score / max_score
    filled = int(pct * 20)
    bar = "█" * filled + "░" * (20 - filled)
    return f"[{bar}] {score:.0f}/{max_score}"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_ticker(ticker: str, use_fmp: bool = False, verbose: bool = False) -> None:
    """Run full analysis on a single ticker and print results."""

    ticker = ticker.upper().strip()
    print(f"\n{'='*70}")
    print(f"  TICKER ANALYSIS: {ticker}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    fetcher = YahooFinanceFetcher(cache_dir="./data/cache")

    # ------------------------------------------------------------------
    # 1. Fetch price data
    # ------------------------------------------------------------------
    print("⏳  Fetching price data …")
    price_data = fetcher.fetch_price_history(ticker, period='2y')

    if price_data is None or price_data.empty:
        print(f"❌  ERROR: No price data found for {ticker}. "
              "Check the ticker symbol and try again.")
        sys.exit(1)

    if len(price_data) < 200:
        print(f"⚠️   Only {len(price_data)} trading days of data available "
              "(need 200+). Results may be incomplete.")

    current_price = float(price_data['Close'].iloc[-1])
    prev_close = float(price_data['Close'].iloc[-2]) if len(price_data) > 1 else current_price
    day_change = ((current_price - prev_close) / prev_close) * 100

    print(f"\n📈  CURRENT PRICE: {_fmt_price(current_price)}  "
          f"({_fmt_pct(day_change)} today)\n")

    # ------------------------------------------------------------------
    # 2. Fetch SPY benchmark data
    # ------------------------------------------------------------------
    print("⏳  Fetching SPY benchmark …")
    spy_data = fetcher.fetch_price_history('SPY', period='2y')

    if spy_data is None or spy_data.empty:
        print("⚠️   Cannot fetch SPY data – Relative Strength will be unavailable.")
        rs_series = None
    else:
        rs_series = calculate_relative_strength(
            price_data['Close'], spy_data['Close'], period=63
        )

    # ------------------------------------------------------------------
    # 3. Phase classification
    # ------------------------------------------------------------------
    print("⏳  Classifying market phase …")
    phase_info = classify_phase(price_data, current_price)
    phase = phase_info['phase']

    print(f"\n{'─'*70}")
    print(f"  MARKET PHASE")
    print(f"{'─'*70}")
    print(f"  {_phase_label(phase)}  (confidence: {phase_info.get('confidence', 0):.0f}%)")

    sma_50 = phase_info.get('sma_50', 0)
    sma_150 = phase_info.get('sma_150', 0)
    sma_200 = phase_info.get('sma_200', 0)
    dist_50 = phase_info.get('distance_from_50sma', 0)
    dist_200 = phase_info.get('distance_from_200sma', 0)
    w52h = phase_info.get('week_52_high', 0)
    w52l = phase_info.get('week_52_low', 0)
    dist_52h = ((current_price - w52h) / w52h * 100) if w52h > 0 else 0

    print(f"\n  SMAs :")
    print(f"    50 SMA  : {_fmt_price(sma_50)}   ({_fmt_pct(dist_50)} from price)")
    print(f"   150 SMA  : {_fmt_price(sma_150)}")
    print(f"   200 SMA  : {_fmt_price(sma_200)}   ({_fmt_pct(dist_200)} from price)")
    print(f"\n  52-Week  : High {_fmt_price(w52h)} / Low {_fmt_price(w52l)}")
    print(f"             {_fmt_pct(dist_52h)} from 52-week high")

    if verbose:
        print(f"\n  Phase reasons:")
        for r in phase_info.get('reasons', []):
            print(f"    • {r}")

    # ------------------------------------------------------------------
    # 4. Minervini Trend Template
    # ------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print(f"  MINERVINI TREND TEMPLATE (SEPA)")
    print(f"{'─'*70}")

    sma_200_series = calculate_sma(price_data['Close'], 200)
    from src.screening.phase_indicators import validate_minervini_trend_template
    minervini = validate_minervini_trend_template(current_price, phase_info, sma_200_series)

    criteria_passed = minervini['criteria_passed']
    criteria_total = minervini['criteria_total']
    passes = minervini['passes_template']

    status = "✅ PASSES" if passes else "❌ FAILS"
    print(f"  {status}  ({criteria_passed}/{criteria_total} criteria)")

    crit = minervini.get('criteria_details', {})
    crit_map = [
        ('price_above_150_200',     'Price > 150 SMA & 200 SMA'),
        ('sma_150_above_200',       '150 SMA > 200 SMA'),
        ('sma_200_rising',          '200 SMA trending up (1 month)'),
        ('sma_50_above_150',        '50 SMA > 150 SMA'),
        ('price_above_50',          'Price > 50 SMA'),
        ('price_30pct_above_52w_low', 'Price ≥ 30% above 52-week low'),
        ('price_near_52w_high',     'Price within 25% of 52-week high'),
        ('confirmed_stage_2',       'Confirmed Stage 2 uptrend'),
    ]
    for key, label in crit_map:
        val = crit.get(key, False)
        icon = "  ✅" if val else "  ❌"
        # Add extra detail for distance criteria
        if key == 'price_30pct_above_52w_low' and 'distance_from_52w_low_pct' in crit:
            label += f"  ({crit['distance_from_52w_low_pct']:.1f}% actual)"
        if key == 'price_near_52w_high' and 'distance_from_52w_high_pct' in crit:
            label += f"  ({crit['distance_from_52w_high_pct']:.1f}% from high)"
        print(f"{icon}  {label}")

    # ------------------------------------------------------------------
    # 5. VCP Pattern
    # ------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print(f"  VCP PATTERN (Volatility Contraction Pattern)")
    print(f"{'─'*70}")

    vcp_data = detect_vcp_pattern(price_data, current_price, phase_info)
    is_vcp = vcp_data.get('is_vcp', False)
    vcp_quality = vcp_data.get('vcp_quality', 0)
    contractions = vcp_data.get('contraction_count', 0)

    if is_vcp:
        print(f"  ✅ VCP DETECTED  (quality: {vcp_quality:.0f}/100, "
              f"{contractions} contractions)")
        print(f"     {vcp_data.get('pattern_details', '')}")
    elif contractions > 0:
        print(f"  🟡 Partial pattern: {contractions} contraction(s) detected "
              f"(quality: {vcp_quality:.0f}/100)")
        print(f"     {vcp_data.get('pattern_details', '')}")
    else:
        print("  ❌ No VCP pattern detected")

    # ------------------------------------------------------------------
    # 6. Relative Strength
    # ------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print(f"  RELATIVE STRENGTH vs SPY")
    print(f"{'─'*70}")

    if rs_series is not None and len(rs_series) >= 20 and not rs_series.isna().all():
        from src.screening.phase_indicators import calculate_rs_slope
        rs_slope = calculate_rs_slope(rs_series, 20)
        if rs_slope > 0.10:
            rs_label = "🟢 STRONG (outperforming SPY)"
        elif rs_slope > 0.03:
            rs_label = "🟢 Positive"
        elif rs_slope > -0.03:
            rs_label = "🟡 Neutral"
        elif rs_slope > -0.10:
            rs_label = "🟠 Weak"
        else:
            rs_label = "🔴 DECLINING (underperforming SPY)"
        print(f"  RS Slope (20d): {rs_slope:.4f}  →  {rs_label}")
    else:
        print("  RS data unavailable")

    # ------------------------------------------------------------------
    # 7. Fundamentals
    # ------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print(f"  FUNDAMENTALS")
    print(f"{'─'*70}")

    print("⏳  Fetching fundamentals …")
    quarterly_data = fetch_quarterly_financials(ticker)
    fundamentals = analyze_fundamentals_for_signal(quarterly_data) if quarterly_data else None

    enhanced_fetcher = EnhancedFundamentalsFetcher()
    snapshot = enhanced_fetcher.create_snapshot(
        ticker,
        quarterly_data=quarterly_data or {},
        use_fmp=use_fmp,
    )
    if snapshot:
        print(snapshot)
    else:
        print("  No fundamental data available.")

    # ------------------------------------------------------------------
    # 8. Buy Signal Analysis
    # ------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print(f"  BUY SIGNAL ANALYSIS")
    print(f"{'─'*70}")

    if rs_series is None:
        import pandas as pd
        rs_series_for_signal = pd.Series(dtype=float)
    else:
        rs_series_for_signal = rs_series

    buy_signal = score_buy_signal(
        ticker=ticker,
        price_data=price_data,
        current_price=current_price,
        phase_info=phase_info,
        rs_series=rs_series_for_signal,
        fundamentals=fundamentals,
        vcp_data=vcp_data,
    )

    is_buy = buy_signal.get('is_buy', False)
    buy_score = buy_signal.get('score', 0)

    if is_buy:
        print(f"\n  🟢 BUY SIGNAL ACTIVE")
        print(f"  Score: {_signal_bar(buy_score, 125)}")
    else:
        print(f"\n  ❌ No buy signal")
        print(f"  Score: {_signal_bar(buy_score, 125)}")
        reason = buy_signal.get('reason', '')
        if reason:
            print(f"  Reason: {reason}")

    # Key trading levels (always show even without signal)
    details = buy_signal.get('details', {})
    stop_loss = buy_signal.get('stop_loss') or details.get('stop_loss')
    breakout_price = buy_signal.get('breakout_price')
    rr_ratio = buy_signal.get('risk_reward_ratio') or details.get('risk_reward_ratio', 0)
    reward_target = details.get('reward_target')

    print(f"\n  ── Trading Levels ──")
    print(f"  Current Price : {_fmt_price(current_price)}")
    print(f"  Breakout Level: {_fmt_price(breakout_price)}")
    print(f"  Stop Loss     : {_fmt_price(stop_loss)}", end="")
    if stop_loss and stop_loss > 0:
        risk_pct = ((current_price - stop_loss) / current_price) * 100
        print(f"  ({_fmt_pct(-risk_pct)} risk)", end="")
    print()
    print(f"  Take Profit   : {_fmt_price(reward_target)}", end="")
    if reward_target and reward_target > current_price:
        reward_pct = ((reward_target - current_price) / current_price) * 100
        print(f"  ({_fmt_pct(reward_pct)} upside)", end="")
    print()
    print(f"  Risk/Reward   : {rr_ratio:.1f}:1" if rr_ratio else "  Risk/Reward   : N/A")

    entry_quality = buy_signal.get('entry_quality', 'Unknown')
    eq_icon = "🟢" if entry_quality == 'Good' else ("🟡" if entry_quality == 'Extended' else "🔴")
    print(f"  Entry Quality : {eq_icon} {entry_quality}")

    if buy_signal.get('reasons'):
        print(f"\n  ── Buy Signal Reasons ──")
        for r in buy_signal['reasons']:
            print(f"    • {r}")

    if verbose and details:
        print(f"\n  ── Scoring Breakdown ──")
        score_keys = ['trend_score', 'fundamental_score', 'volume_score',
                      'rs_score', 'rr_score', 'entry_score', 'vcp_bonus']
        for k in score_keys:
            if k in details:
                print(f"    {k:25s}: {details[k]:.1f}")

    # ------------------------------------------------------------------
    # 9. Sell Signal Analysis
    # ------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print(f"  SELL SIGNAL ANALYSIS")
    print(f"{'─'*70}")

    sell_signal = score_sell_signal(
        ticker=ticker,
        price_data=price_data,
        current_price=current_price,
        phase_info=phase_info,
        rs_series=rs_series_for_signal,
        fundamentals=fundamentals,
    )

    is_sell = sell_signal.get('is_sell', False)
    sell_score = sell_signal.get('score', 0)
    severity = sell_signal.get('severity', 'none')

    if is_sell:
        sev_icon = "🚨" if severity == 'critical' else ("🔴" if severity == 'high' else "🟡")
        print(f"\n  {sev_icon} SELL SIGNAL ACTIVE  (severity: {severity.upper()})")
        print(f"  Score: {_signal_bar(sell_score, 100)}")
        breakdown = sell_signal.get('breakdown_level')
        if breakdown:
            print(f"  Breakdown level: {_fmt_price(breakdown)}")
        if sell_signal.get('reasons'):
            print(f"\n  ── Sell Signal Reasons ──")
            for r in sell_signal['reasons']:
                print(f"    • {r}")
    else:
        print(f"\n  ✅ No sell signal  (score: {sell_score:.0f}/100)")
        reason = sell_signal.get('reason', '')
        if reason:
            print(f"  Reason: {reason}")

    # ------------------------------------------------------------------
    # 10. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  SUMMARY: {ticker}")
    print(f"{'='*70}")
    print(f"  Price  : {_fmt_price(current_price)}  ({_fmt_pct(day_change)} today)")
    print(f"  Phase  : {_phase_label(phase)}")
    print(f"  Minervini Template: {criteria_passed}/8 criteria  ({status})")
    if is_buy:
        print(f"  Signal : 🟢 BUY  (score {buy_score:.0f}/125)")
        print(f"           Breakout: {_fmt_price(breakout_price)}")
        print(f"           Stop Loss: {_fmt_price(stop_loss)}")
        print(f"           Take Profit: {_fmt_price(reward_target)}")
        print(f"           R/R: {rr_ratio:.1f}:1")
    elif is_sell:
        sev_icon = "🚨" if severity == 'critical' else ("🔴" if severity == 'high' else "🟡")
        print(f"  Signal : {sev_icon} SELL  (score {sell_score:.0f}/100, {severity.upper()})")
        breakdown = sell_signal.get('breakdown_level')
        if breakdown:
            print(f"           Breakdown level: {_fmt_price(breakdown)}")
    else:
        print(f"  Signal : ⏳ NO SIGNAL  (buy score {buy_score:.0f}/125, "
              f"sell score {sell_score:.0f}/100)")
    print(f"{'='*70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Check the current status of a single ticker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_ticker.py AAPL
  python check_ticker.py NVDA --use-fmp
  python check_ticker.py TSLA --verbose
        """,
    )
    parser.add_argument('ticker', help='Stock ticker symbol (e.g. AAPL, NVDA)')
    parser.add_argument(
        '--use-fmp',
        action='store_true',
        help='Use FMP API for enhanced fundamental data (requires FMP_API_KEY env var)',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show additional detail (phase reasons, score breakdown, etc.)',
    )

    args = parser.parse_args()

    try:
        analyze_ticker(
            ticker=args.ticker,
            use_fmp=args.use_fmp,
            verbose=args.verbose,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n❌  Fatal error: {exc}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
