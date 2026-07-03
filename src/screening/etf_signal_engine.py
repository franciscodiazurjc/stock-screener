"""ETF-adapted signal scoring engine.

This module adapts the stock-oriented signal engine
(:mod:`src.screening.signal_engine`) for ETFs.  The core technical methodology
(Weinstein / Minervini Stage 2 trend template, phase classification, relative
strength, VCP patterns, volume analysis, risk/reward) remains **identical** —
these concepts apply to any liquid, exchange-traded instrument.

The only material change is in the **fundamental component**: ETFs lack
traditional company financials (EPS, revenue, inventory), so that 40-point
block is replaced by ETF-specific quality metrics:

=====================  =========  ==========================================
Component              Max pts    Description
=====================  =========  ==========================================
Expense ratio          15 pts     Lower ongoing cost compounds over time
AUM / liquidity        15 pts     Larger AUM → tighter spreads, less slippage
Distribution yield     10 pts     Income generation (secondary criterion)
=====================  =========  ==========================================

Total maximum score: **125** (identical to the stock screener), ensuring that
the same thresholds (≥ 60 for a buy signal, ≥ 70 for high-confidence) apply
without any re-tuning.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .phase_indicators import (
    calculate_volume_ratio,
    calculate_rs_slope,
    detect_volatility_contraction,
    detect_breakout,
    validate_minervini_trend_template,
    calculate_sma,
)
from src.data.etf_fetcher import score_etf_fundamentals

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers (reused from the stock signal engine)
# ---------------------------------------------------------------------------

def _calculate_stop_loss(
    price_data: pd.DataFrame,
    current_price: float,
    phase_info: Dict,
    phase: int,
) -> float:
    """Calculate a stop-loss level for an ETF position.

    Identical logic to the stock version: Stage 2 uses the 50-SMA or recent
    swing low (tighter), Stage 1 uses the 30-day base low.

    Args:
        price_data: OHLCV DataFrame.
        current_price: Current ETF price.
        phase_info: Phase classification dict (from :func:`classify_phase`).
        phase: Numeric phase (1 or 2).

    Returns:
        Stop-loss price.
    """
    sma_50 = phase_info.get("sma_50", 0)

    if phase == 2:
        recent_low = price_data["Low"].iloc[-10:].min() if len(price_data) >= 10 else price_data["Low"].min()
        swing_low_stop = recent_low * 0.995
        sma_stop = sma_50 * 0.99 if sma_50 > 0 else swing_low_stop
        stop_loss = max(swing_low_stop, sma_stop)

        risk_pct = (current_price - stop_loss) / current_price
        if risk_pct < 0.03:
            stop_loss = current_price * 0.97
        elif risk_pct > 0.10:
            stop_loss = current_price * 0.90
    else:
        base_low = price_data["Low"].iloc[-30:].min() if len(price_data) >= 30 else price_data["Low"].min()
        stop_loss = base_low * 0.99
        risk_pct = (current_price - stop_loss) / current_price
        if risk_pct > 0.10:
            stop_loss = current_price * 0.90

    return stop_loss


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_etf_buy_signal(
    ticker: str,
    price_data: pd.DataFrame,
    current_price: float,
    phase_info: Dict,
    rs_series: pd.Series,
    etf_info: Optional[Dict] = None,
    vcp_data: Optional[Dict] = None,
) -> Dict[str, any]:
    """Score a buy signal for an ETF.

    Mirrors :func:`src.screening.signal_engine.score_buy_signal` exactly but
    substitutes the 40-point stock-fundamental block with ETF quality metrics.

    Scoring components (total max 125):

    ==========================  ========  =================================
    Component                   Max pts   Notes
    ==========================  ========  =================================
    Trend structure / Stage 2    40 pts   Identical to stock screener
    ETF quality fundamentals     40 pts   Expense ratio + AUM + yield
    Volume behaviour             10 pts   Identical to stock screener
    Relative strength vs SPY     10 pts   Identical to stock screener
    Risk / Reward                15 pts   Identical to stock screener
    Entry quality                 5 pts   Identical to stock screener
    VCP bonus                    +5 pts   Identical to stock screener
    ==========================  ========  =================================

    Threshold: score ≥ 60 for a buy signal.

    Args:
        ticker: ETF ticker symbol.
        price_data: OHLCV DataFrame (DatetimeIndex, at least 200 rows).
        current_price: Most recent closing price.
        phase_info: Dict returned by :func:`classify_phase`.
        rs_series: Relative-strength series vs SPY.
        etf_info: Optional dict from :meth:`ETFFetcher.fetch_etf_info`.
        vcp_data: Optional VCP pattern dict from :func:`detect_vcp_pattern`.

    Returns:
        Dict with keys: ``ticker``, ``is_buy``, ``score``, ``phase``,
        ``stop_loss``, ``risk_reward_ratio``, ``entry_quality``,
        ``reasons``, ``details``.
    """
    phase = phase_info["phase"]

    # MINERVINI REQUIREMENT: Only Phase 2 (confirmed Stage 2 uptrend)
    if phase != 2:
        return {
            "ticker": ticker,
            "is_buy": False,
            "score": 0,
            "reason": (
                f"Not in Phase 2 (currently Phase {phase}) — "
                "Minervini requires confirmed uptrend"
            ),
            "details": {},
        }

    # Validate Minervini Trend Template (SEPA)
    sma_200 = calculate_sma(price_data["Close"], 200)
    minervini = validate_minervini_trend_template(current_price, phase_info, sma_200)

    if not minervini["passes_template"]:
        return {
            "ticker": ticker,
            "is_buy": False,
            "score": 0,
            "reason": (
                f"Fails Minervini Trend Template "
                f"({minervini['criteria_passed']}/8 criteria passed)"
            ),
            "details": {"minervini": minervini},
        }

    score = 0
    details: Dict = {}
    reasons: List[str] = []

    # ------------------------------------------------------------------ #
    # 1. TREND STRUCTURE / STAGE QUALITY (40 points)                     #
    # ------------------------------------------------------------------ #
    trend_score = 0

    sma_50 = phase_info.get("sma_50", 0)
    slope_50 = phase_info.get("slope_50", 0)
    slope_200 = phase_info.get("slope_200", 0)
    distance_50 = phase_info.get("distance_from_50sma", 0)
    distance_200 = phase_info.get("distance_from_200sma", 0)

    # A) Stage 2 quality (30 pts)
    stage2_quality = 0
    distance_component = min(15, max(0,
        (distance_50 / 15.0 * 10) + (distance_200 / 20.0 * 5)
    ))
    stage2_quality += distance_component

    if distance_50 >= 10:
        reasons.append(f"Strong Stage 2: {distance_50:.1f}% above 50 SMA")
    elif distance_50 >= 3:
        reasons.append(f"Good Stage 2: {distance_50:.1f}% above 50 SMA")
    elif distance_50 >= 0:
        reasons.append(f"Weak Stage 2: {distance_50:.1f}% above 50 SMA")
    else:
        reasons.append(f"Very weak Stage 2: {distance_50:.1f}% from 50 SMA")

    slope_component = min(15, max(0,
        (slope_50 / 0.08 * 10) + (slope_200 / 0.05 * 5)
    ))
    stage2_quality += slope_component

    if slope_50 > 0.05:
        reasons.append(f"SMAs rising strongly (50:{slope_50:.3f}, 200:{slope_200:.3f})")
    elif slope_50 > 0.02:
        reasons.append("SMAs rising moderately")
    elif slope_50 > 0:
        reasons.append("SMAs rising weakly")
    else:
        reasons.append("⚠ SMAs flat or declining")

    trend_score += stage2_quality

    # B) Breakout detection (10 pts)
    breakout_info = detect_breakout(price_data, current_price, phase_info, vcp_data)
    if breakout_info["is_breakout"]:
        trend_score += 10
        breakout_type = breakout_info["breakout_type"]
        vol_confirmed = breakout_info.get("volume_confirmed", False)
        reasons.append(
            f"{'🟢' if vol_confirmed else '🟡'} {breakout_type} "
            f"({'volume confirmed' if vol_confirmed else 'low volume'})"
        )
        details["breakout"] = breakout_info

    # C) Over-extension penalty
    if distance_50 > 30:
        trend_score -= 10
        reasons.append(f"⚠ Over-extended: {distance_50:.1f}% above 50 SMA")
    elif distance_50 > 20:
        trend_score -= 5
        reasons.append("Moderately extended above 50 SMA")

    score += min(trend_score, 40)
    details["trend_score"] = min(trend_score, 40)

    # ------------------------------------------------------------------ #
    # 2. ETF QUALITY FUNDAMENTALS (40 points)                             #
    # ------------------------------------------------------------------ #
    if etf_info:
        fundamental_score, etf_reasons = score_etf_fundamentals(etf_info)
        reasons.extend(etf_reasons)
    else:
        # No ETF metadata — neutral (half points)
        fundamental_score = 20.0
        reasons.append("ETF metadata unavailable (neutral)")

    details["fundamental_score"] = round(fundamental_score, 2)
    score += fundamental_score

    # ------------------------------------------------------------------ #
    # 3. VOLUME BEHAVIOUR (10 points)                                     #
    # ------------------------------------------------------------------ #
    volume_score = 0

    if "Volume" in price_data.columns and len(price_data) >= 30:
        recent_prices = price_data["Close"].iloc[-6:]
        recent_volume = price_data["Volume"].iloc[-5:]
        avg_volume = price_data["Volume"].iloc[-30:-5].mean()

        up_days = down_days = 0
        vol_up = vol_down = 0

        for i in range(1, len(recent_prices)):
            change = recent_prices.iloc[i] - recent_prices.iloc[i - 1]
            vol = recent_volume.iloc[i - 1]
            if change > 0:
                up_days += 1
                vol_up += vol
            else:
                down_days += 1
                vol_down += vol

        avg_vol_up = (vol_up / up_days) if up_days > 0 else 0
        avg_vol_down = (vol_down / down_days) if down_days > 0 else 0
        vol_ratio = (avg_vol_up / avg_vol_down) if avg_vol_down > 0 else 1.0

        volume_score = min(10, max(0, 5 + (vol_ratio - 1.0) * 10))

        if vol_ratio >= 1.3:
            reasons.append(
                f"✓ Volume heavier on up days "
                f"({avg_vol_up/1e6:.1f}M vs {avg_vol_down/1e6:.1f}M, ratio {vol_ratio:.2f})"
            )
        elif vol_ratio >= 1.1:
            reasons.append(f"Volume slightly heavier on up days (ratio {vol_ratio:.2f})")
        elif vol_ratio >= 0.9:
            reasons.append(f"Volume pattern neutral (ratio {vol_ratio:.2f})")
        else:
            reasons.append(f"⚠ Volume heavier on down days (ratio {vol_ratio:.2f} — distribution)")

        details["volume_ratio"] = round(vol_ratio, 2)
        details["volume_score"] = volume_score
    else:
        volume_score = 5
        details["volume_score"] = volume_score

    score += volume_score

    # ------------------------------------------------------------------ #
    # 4. RELATIVE STRENGTH vs SPY (10 points)                            #
    # ------------------------------------------------------------------ #
    rs_score = 0

    if len(rs_series) >= 20 and not rs_series.isna().all():
        rs_slope = calculate_rs_slope(rs_series, 20)
        details["rs_slope"] = round(rs_slope, 3)
        rs_score = min(10, max(0, 5 + (rs_slope * 16.67)))

        if rs_slope > 0.10:
            reasons.append(f"✓ Strong RS: {rs_slope:.3f} (outperforming SPY)")
        elif rs_slope > 0.03:
            reasons.append(f"Positive RS: {rs_slope:.3f}")
        elif rs_slope > -0.03:
            reasons.append(f"Neutral RS: {rs_slope:.3f}")
        elif rs_slope > -0.10:
            reasons.append(f"Weak RS: {rs_slope:.3f}")
        else:
            reasons.append(f"⚠ Declining RS: {rs_slope:.3f} (underperforming SPY)")
    else:
        details["rs_slope"] = None
        rs_score = 5

    score += rs_score
    details["rs_score"] = round(rs_score, 2)

    # ------------------------------------------------------------------ #
    # 5. STOP LOSS (not scored; risk management)                          #
    # ------------------------------------------------------------------ #
    stop_loss = _calculate_stop_loss(price_data, current_price, phase_info, phase)
    details["stop_loss"] = round(stop_loss, 2)

    # ------------------------------------------------------------------ #
    # 6. RISK / REWARD RATIO (15 points)                                  #
    # ------------------------------------------------------------------ #
    rr_score = 0
    risk_amount = current_price - stop_loss if stop_loss else 0

    if phase == 2:
        reward_target = current_price * 1.30
    else:
        reward_target = (
            breakout_info["breakout_level"] * 1.25
            if breakout_info.get("is_breakout")
            else sma_50 * 1.25
        )

    reward_amount = reward_target - current_price

    if risk_amount > 0:
        rr_ratio = reward_amount / risk_amount
        if rr_ratio < 2.0:
            rr_score = 0
        else:
            rr_score = min(15, ((rr_ratio - 2.0) * 6) + 3)

        details["risk_reward_ratio"] = round(rr_ratio, 2)
        details["risk_amount"] = round(risk_amount, 2)
        details["reward_amount"] = round(reward_amount, 2)
        details["reward_target"] = round(reward_target, 2)

        if rr_ratio >= 5.0:
            reasons.append(f"🟢 Outstanding R/R: {rr_ratio:.1f}:1 (${reward_amount:.2f} upside)")
        elif rr_ratio >= 4.0:
            reasons.append(f"🟢 Excellent R/R: {rr_ratio:.1f}:1")
        elif rr_ratio >= 3.0:
            reasons.append(f"🟢 Good R/R: {rr_ratio:.1f}:1")
        elif rr_ratio >= 2.0:
            reasons.append(f"🟡 Acceptable R/R: {rr_ratio:.1f}:1")
        else:
            reasons.append(f"🔴 Poor R/R: {rr_ratio:.1f}:1 (need 2:1+)")
    else:
        details["risk_reward_ratio"] = 0
        rr_score = 0

    score += rr_score
    details["rr_score"] = round(rr_score, 2)

    # ------------------------------------------------------------------ #
    # 7. ENTRY QUALITY (5 points)                                         #
    # ------------------------------------------------------------------ #
    entry_score = 0
    week_52_high = phase_info.get("week_52_high", current_price)
    distance_from_52w_high = (
        ((current_price - week_52_high) / week_52_high * 100) if week_52_high > 0 else -100
    )

    if phase == 2:
        if distance_from_52w_high >= -5:
            high_proximity_score = 3
            reasons.append(
                f"🟢 At 52W high: {abs(distance_from_52w_high):.1f}% from high (pivot zone)"
            )
        elif distance_from_52w_high >= -15:
            high_proximity_score = 3 - ((abs(distance_from_52w_high) - 5) / 10.0)
            reasons.append(f"🟢 Near 52W high: {abs(distance_from_52w_high):.1f}% from high")
        elif distance_from_52w_high >= -25:
            high_proximity_score = 2 - ((abs(distance_from_52w_high) - 15) / 10.0)
            reasons.append(f"🟡 Within 25% of 52W high: {abs(distance_from_52w_high):.1f}% from high")
        else:
            high_proximity_score = 0
            reasons.append(
                f"🔴 Far from 52W high: {abs(distance_from_52w_high):.1f}% below"
            )

        entry_score += high_proximity_score

        if distance_50 > 0 and distance_50 <= 20:
            sma_score = 2 - (distance_50 / 20.0)
        elif distance_50 > 20:
            sma_score = max(0, 1 - ((distance_50 - 20) / 15.0))
        else:
            sma_score = 0

        entry_score += sma_score
    else:
        ideal_position = 1.0
        deviation = abs(distance_50 - ideal_position)
        entry_score += max(0, 2 - (deviation / 6.0) * 2)

    score += entry_score
    details["entry_score"] = round(entry_score, 2)

    # ------------------------------------------------------------------ #
    # 8. VCP PATTERN BONUS (+5 points)                                    #
    # ------------------------------------------------------------------ #
    vcp_bonus = 0

    if vcp_data and vcp_data.get("is_vcp"):
        vcp_quality = vcp_data.get("vcp_quality", 0)
        if vcp_quality >= 80:
            vcp_bonus = 5
            reasons.append(
                f"⭐ VCP pattern: {vcp_data.get('pattern_details', 'N/A')} "
                f"(quality: {vcp_quality:.0f}/100)"
            )
        elif vcp_quality >= 60:
            vcp_bonus = 3
            reasons.append(
                f"🟢 VCP pattern: {vcp_data.get('pattern_details', 'N/A')} "
                f"(quality: {vcp_quality:.0f}/100)"
            )
        else:
            vcp_bonus = 1
            reasons.append(
                f"🟡 VCP pattern: {vcp_data.get('pattern_details', 'N/A')} "
                f"(quality: {vcp_quality:.0f}/100)"
            )

        details["vcp_data"] = {
            "quality": vcp_quality,
            "contractions": vcp_data.get("contraction_count", 0),
            "pattern": vcp_data.get("pattern_details", ""),
            "base_length_weeks": vcp_data.get("base_length_weeks", 0),
            "volume_ratio": vcp_data.get("breakout_volume_ratio", 0),
        }
    elif vcp_data and vcp_data.get("contraction_count", 0) > 0:
        reasons.append(f"🟡 Partial pattern: {vcp_data.get('pattern_details', 'N/A')}")

    score += vcp_bonus
    details["vcp_bonus"] = round(vcp_bonus, 2)
    details["minervini_template"] = minervini

    # ------------------------------------------------------------------ #
    # Final score (max 125; threshold 60 for buy)                        #
    # ------------------------------------------------------------------ #
    final_score = max(0, min(score, 125))
    is_buy = final_score >= 60

    return {
        "ticker": ticker,
        "is_buy": is_buy,
        "score": round(final_score, 1),
        "phase": phase,
        "minervini_template_score": minervini["template_score"],
        "minervini_criteria_passed": minervini["criteria_passed"],
        "breakout_price": breakout_info.get("breakout_level") if breakout_info["is_breakout"] else None,
        "stop_loss": round(stop_loss, 2) if stop_loss else None,
        "risk_reward_ratio": details.get("risk_reward_ratio", 0),
        "entry_quality": "Good" if entry_score >= 3 else "Extended" if entry_score >= 1.5 else "Poor",
        "reasons": reasons,
        "details": details,
    }


def score_etf_sell_signal(
    ticker: str,
    price_data: pd.DataFrame,
    current_price: float,
    phase_info: Dict,
    rs_series: pd.Series,
    previous_phase: Optional[int] = None,
) -> Dict[str, any]:
    """Score a sell signal for an ETF position.

    Mirrors :func:`src.screening.signal_engine.score_sell_signal`.  ETFs
    follow the same Stage 3/4 breakdown mechanics as individual stocks — the
    absence of company-specific fundamental deterioration is acceptable because
    the underlying index / asset class trend is fully captured by price action.

    Scoring components (max 110):

    ==========================  ========  ===================================
    Component                   Max pts   Notes
    ==========================  ========  ===================================
    Breakdown structure          60 pts   Phase 3/4 transition severity
    Volume on decline            30 pts   Confirming distribution
    RS weakness vs SPY           10 pts   Underperformance acceleration
    ==========================  ========  ===================================

    Threshold: score ≥ 60 for a sell signal.

    Args:
        ticker: ETF ticker.
        price_data: OHLCV DataFrame.
        current_price: Current price.
        phase_info: Phase classification dict.
        rs_series: RS series vs SPY.
        previous_phase: Previous phase (if tracked).

    Returns:
        Dict with ``ticker``, ``is_sell``, ``score``, ``phase``,
        ``breakdown_level``, ``severity``, ``reasons``, ``details``.
    """
    phase = phase_info["phase"]
    score = 0
    reasons: List[str] = []
    details: Dict = {}

    if phase not in [3, 4]:
        return {
            "ticker": ticker,
            "is_sell": False,
            "score": 0,
            "reason": f"Not in Phase 3/4 (currently Phase {phase})",
            "details": {},
        }

    sma_50 = phase_info.get("sma_50", 0)
    sma_200 = phase_info.get("sma_200", 0)
    distance_50 = phase_info.get("distance_from_50sma", 0)
    slope_50 = phase_info.get("slope_50", 0)
    slope_200 = phase_info.get("slope_200", 0)

    # ------------------------------------------------------------------ #
    # 1. BREAKDOWN STRUCTURE (60 points)                                  #
    # ------------------------------------------------------------------ #
    breakdown_score = 0
    breakdown_level = None

    # Phase severity
    if phase == 4:
        breakdown_score += 30
        reasons.append("🔴 Stage 4 downtrend — confirmed distribution")
    elif phase == 3:
        breakdown_score += 15
        reasons.append("🟡 Stage 3 topping — distribution in progress")

    # Price below SMAs
    if current_price < sma_50 and current_price < sma_200:
        breakdown_score += 15
        reasons.append(f"Below both 50 SMA (${sma_50:.2f}) and 200 SMA (${sma_200:.2f})")
        breakdown_level = sma_200
    elif current_price < sma_200:
        breakdown_score += 10
        reasons.append(f"Below 200 SMA (${sma_200:.2f})")
        breakdown_level = sma_200
    elif current_price < sma_50:
        breakdown_score += 5
        reasons.append(f"Below 50 SMA (${sma_50:.2f})")
        breakdown_level = sma_50

    # Declining SMAs
    if slope_50 < -0.03 and slope_200 < -0.01:
        breakdown_score += 10
        reasons.append(f"Both SMAs declining (50: {slope_50:.3f}, 200: {slope_200:.3f})")
    elif slope_50 < -0.03:
        breakdown_score += 5
        reasons.append(f"50 SMA declining sharply ({slope_50:.3f})")

    # Phase transition from 2
    if previous_phase == 2 and phase in [3, 4]:
        breakdown_score += 5
        reasons.append("Stage 2 → Stage 3/4 transition (recent breakdown)")

    score += min(breakdown_score, 60)
    details["breakdown_score"] = min(breakdown_score, 60)

    # ------------------------------------------------------------------ #
    # 2. VOLUME ON DECLINE (30 points)                                    #
    # ------------------------------------------------------------------ #
    volume_score = 0

    if "Volume" in price_data.columns and len(price_data) >= 30:
        recent_prices = price_data["Close"].iloc[-6:]
        recent_volume = price_data["Volume"].iloc[-5:]
        avg_volume = price_data["Volume"].iloc[-30:-5].mean()

        vol_up = vol_down = 0
        up_days = down_days = 0

        for i in range(1, len(recent_prices)):
            change = recent_prices.iloc[i] - recent_prices.iloc[i - 1]
            vol = recent_volume.iloc[i - 1]
            if change > 0:
                up_days += 1
                vol_up += vol
            else:
                down_days += 1
                vol_down += vol

        avg_vol_up = (vol_up / up_days) if up_days > 0 else 0
        avg_vol_down = (vol_down / down_days) if down_days > 0 else 0

        if avg_vol_down > 0:
            down_up_ratio = avg_vol_down / avg_vol_up if avg_vol_up > 0 else 2.0
            # Higher down/up ratio → more distribution → higher sell score
            volume_score = min(30, max(0, 15 + (down_up_ratio - 1.0) * 15))

            if down_up_ratio >= 1.5:
                reasons.append(
                    f"🔴 Heavy distribution: volume {down_up_ratio:.1f}x heavier on down days"
                )
            elif down_up_ratio >= 1.2:
                reasons.append(f"Volume heavier on down days (ratio {down_up_ratio:.2f})")
            else:
                reasons.append(f"Volume pattern mixed (ratio {down_up_ratio:.2f})")

            details["volume_ratio"] = round(down_up_ratio, 2)
    else:
        volume_score = 15  # Neutral

    score += volume_score
    details["volume_score"] = volume_score

    # ------------------------------------------------------------------ #
    # 3. RS WEAKNESS (10 points)                                          #
    # ------------------------------------------------------------------ #
    rs_score = 0

    if len(rs_series) >= 20 and not rs_series.isna().all():
        rs_slope = calculate_rs_slope(rs_series, 20)
        details["rs_slope"] = round(rs_slope, 3)

        # Negative RS → higher sell score
        # rs_slope = -0.30 → 10 pts;  0 → 5 pts;  +0.30 → 0 pts
        rs_score = min(10, max(0, 5 - (rs_slope * 16.67)))

        if rs_slope < -0.10:
            reasons.append(f"🔴 RS rolling over: {rs_slope:.3f} (underperforming SPY)")
        elif rs_slope < 0:
            reasons.append(f"RS weakening: {rs_slope:.3f}")
        else:
            reasons.append(f"RS still positive: {rs_slope:.3f} (watch for rollover)")

    score += rs_score
    details["rs_score"] = round(rs_score, 2)

    # ------------------------------------------------------------------ #
    # Severity classification                                             #
    # ------------------------------------------------------------------ #
    if score >= 80:
        severity = "critical"
    elif score >= 60:
        severity = "high"
    else:
        severity = "moderate"

    final_score = max(0, min(score, 110))
    is_sell = final_score >= 60

    return {
        "ticker": ticker,
        "is_sell": is_sell,
        "score": round(final_score, 1),
        "phase": phase,
        "breakdown_level": round(breakdown_level, 2) if breakdown_level else None,
        "severity": severity,
        "reasons": reasons,
        "details": details,
    }


def format_etf_signal_output(signal: Dict) -> str:
    """Format a buy/sell signal dict into a human-readable string.

    Args:
        signal: Output of :func:`score_etf_buy_signal` or
                :func:`score_etf_sell_signal`.

    Returns:
        Multi-line formatted string.
    """
    lines = []
    ticker = signal.get("ticker", "N/A")
    score = signal.get("score", 0)
    is_buy = signal.get("is_buy", False)
    is_sell = signal.get("is_sell", False)

    if is_buy:
        lines.append(f"{'#'*60}")
        lines.append(f"BUY {ticker} | Score: {score}/125")
        lines.append(f"{'#'*60}")
        lines.append(f"Phase: {signal.get('phase', 'N/A')}")
        if signal.get("breakout_price"):
            lines.append(f"Breakout: ${signal['breakout_price']:.2f}")
        if signal.get("stop_loss"):
            lines.append(f"Stop Loss: ${signal['stop_loss']:.2f}")
        rr = signal.get("risk_reward_ratio", 0)
        if rr:
            lines.append(f"R/R: {rr:.1f}:1")
    elif is_sell:
        lines.append(f"{'#'*60}")
        lines.append(f"SELL {ticker} | Score: {score}/110 | {signal.get('severity', '').upper()}")
        lines.append(f"{'#'*60}")
        lines.append(f"Phase: {signal.get('phase', 'N/A')}")
        if signal.get("breakdown_level"):
            lines.append(f"Breakdown: ${signal['breakdown_level']:.2f}")

    lines.append("\nReasons:")
    for reason in signal.get("reasons", []):
        lines.append(f"  • {reason}")

    return "\n".join(lines)
