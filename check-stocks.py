#!/usr/bin/env python3
"""Check owned stocks from a CSV file using only local cached price data."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.screening.phase_indicators import classify_phase, calculate_relative_strength
from src.screening.signal_engine import score_sell_signal


REPO_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = REPO_ROOT / "data" / "cache"


@dataclass
class StockPosition:
    ticker: str
    entry_price: float
    entry_date: datetime
    share_size: float
    stop_loss: float
    trade_style: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze owned stocks from CSV using only the latest local cache."
    )
    parser.add_argument("csv_file", help="CSV file with stock positions")
    return parser.parse_args()


def normalize_trade_style(value: str) -> str:
    normalized = value.strip().replace("_", "-").upper()
    if normalized not in {"SWING-TRADE", "LONG-TERM"}:
        raise ValueError(
            f"Invalid trade style '{value}'. Use SWING-TRADE or LONG-TERM."
        )
    return normalized


def parse_float(value: str, field_name: str) -> float:
    cleaned = value.strip().replace("$", "").replace(",", "")
    if not cleaned:
        raise ValueError(f"Missing value for {field_name}")
    return float(cleaned)


def parse_date(value: str) -> datetime:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("Missing entry date")

    for date_format in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(cleaned, date_format)
        except ValueError:
            continue

    raise ValueError(
        f"Invalid entry date '{value}'. Use YYYY-MM-DD, YYYY/MM/DD, or MM/DD/YYYY."
    )


def _row_value(row: Dict[str, str], *candidates: str) -> str:
    normalized_row = {
        key.strip().lower().replace("_", " ").replace("-", " "): (value or "").strip()
        for key, value in row.items()
        if key is not None
    }
    for candidate in candidates:
        normalized_candidate = candidate.strip().lower().replace("_", " ").replace("-", " ")
        if normalized_candidate in normalized_row and normalized_row[normalized_candidate]:
            return normalized_row[normalized_candidate]
    return ""


def load_positions_from_csv(csv_path: Path) -> List[StockPosition]:
    positions: List[StockPosition] = []

    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames:
            has_named_headers = any(
                "ticker" in (field or "").strip().lower() for field in reader.fieldnames
            )
        else:
            has_named_headers = False

        if has_named_headers:
            for row_number, row in enumerate(reader, start=2):
                if not any((value or "").strip() for value in row.values()):
                    continue

                positions.append(
                    StockPosition(
                        ticker=_row_value(row, "ticker").upper(),
                        entry_price=parse_float(
                            _row_value(row, "entry price", "entry"), "entry price"
                        ),
                        entry_date=parse_date(
                            _row_value(row, "entry date", "date")
                        ),
                        share_size=parse_float(
                            _row_value(row, "share size", "shares", "quantity"),
                            "share size",
                        ),
                        stop_loss=parse_float(
                            _row_value(row, "stop loss", "stop"), "stop loss"
                        ),
                        trade_style=normalize_trade_style(
                            _row_value(row, "trade style", "style", "position type")
                        ),
                    )
                )
            return positions

    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        for row_number, row in enumerate(reader, start=1):
            values = [cell.strip() for cell in row]
            if not any(values):
                continue
            if row_number == 1 and values[0].strip().lower() == "ticker":
                continue

            non_empty = [value for value in values if value]
            if len(non_empty) < 6:
                raise ValueError(
                    f"Row {row_number} must contain at least 6 non-empty values."
                )

            positions.append(
                StockPosition(
                    ticker=non_empty[0].upper(),
                    entry_price=parse_float(non_empty[1], "entry price"),
                    entry_date=parse_date(non_empty[2]),
                    share_size=parse_float(non_empty[3], "share size"),
                    stop_loss=parse_float(non_empty[4], "stop loss"),
                    trade_style=normalize_trade_style(non_empty[-1]),
                )
            )

    return positions


def normalize_price_data(price_data: pd.DataFrame) -> pd.DataFrame:
    if price_data.empty:
        return pd.DataFrame()

    data = price_data.copy()
    rename_map = {
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    data = data.rename(columns=rename_map)

    if "Date" not in data.columns:
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.reset_index()
            first_col = data.columns[0]
            if first_col != "Date":
                data = data.rename(columns={first_col: "Date"})
        else:
            return pd.DataFrame()

    required_columns = {"Date", "Open", "High", "Low", "Close"}
    if not required_columns.issubset(data.columns):
        return pd.DataFrame()

    if "Volume" not in data.columns:
        data["Volume"] = 0.0

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    data = data.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    data = data.set_index("Date")

    return data[["Open", "High", "Low", "Close", "Volume"]]


def find_latest_cache_file(ticker: str, cache_root: Path = CACHE_ROOT) -> Optional[Path]:
    patterns = [
        f"{ticker.upper()}_prices*.pkl",
        f"price_history/{ticker.upper()}_prices*.pkl",
        f"**/{ticker.upper()}_prices*.pkl",
    ]

    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(cache_root.glob(pattern))

    files = [path for path in matches if path.is_file()]
    if not files:
        return None

    return max(files, key=lambda path: path.stat().st_mtime)


def load_cached_price_data(
    ticker: str, cache_root: Path = CACHE_ROOT
) -> Tuple[pd.DataFrame, Optional[Path]]:
    cache_file = find_latest_cache_file(ticker, cache_root)
    if cache_file is None:
        return pd.DataFrame(), None

    try:
        price_data = pd.read_pickle(cache_file)
    except Exception:
        return pd.DataFrame(), cache_file

    return normalize_price_data(price_data), cache_file


def load_cached_benchmark(cache_root: Path = CACHE_ROOT) -> pd.DataFrame:
    benchmark, _ = load_cached_price_data("SPY", cache_root)
    return benchmark


def calculate_previous_phase(price_data: pd.DataFrame) -> Optional[int]:
    if len(price_data) < 220:
        return None

    previous_slice = price_data.iloc[:-20]
    previous_price = float(previous_slice["Close"].iloc[-1])
    previous_phase = classify_phase(previous_slice, previous_price)
    phase_value = previous_phase.get("phase", 0)
    return phase_value if phase_value else None


def calculate_recommended_stop(
    entry_price: float,
    current_price: float,
    current_stop: float,
    phase_info: Dict[str, object],
    price_data: pd.DataFrame,
) -> Tuple[float, str]:
    gain_pct = ((current_price - entry_price) / entry_price) * 100
    sma_50 = float(phase_info.get("sma_50", 0) or 0)
    recent_low = float(price_data["Low"].iloc[-10:].min())

    recommended_stop = current_stop
    rationale = "Keep current stop."

    if gain_pct < 5:
        technical_floor = max(current_stop, recent_low * 0.995)
        recommended_stop = round(min(current_price * 0.97, technical_floor), 2)
        rationale = "Small gain; keep a loose stop near the recent swing low."
    else:
        locked_profit_pct = min(gain_pct - 3, gain_pct * 0.5)
        profit_based_stop = entry_price * (1 + locked_profit_pct / 100)

        sma_based_stop = 0.0
        if sma_50 > 0 and sma_50 < current_price:
            sma_buffer_pct = max(0.5, 1.5 - (gain_pct / 50))
            sma_based_stop = sma_50 * (1 - sma_buffer_pct / 100)

        chosen_stop = max(current_stop, profit_based_stop, sma_based_stop, recent_low * 0.995)
        recommended_stop = round(min(chosen_stop, current_price * 0.995), 2)
        rationale = "Raise stop to lock in gains while respecting the 50 SMA and recent lows."

    if recommended_stop <= 0:
        recommended_stop = round(current_stop, 2)

    return recommended_stop, rationale


def determine_action(
    trade_style: str,
    current_price: float,
    stop_loss: float,
    recommended_stop: float,
    current_gain_pct: float,
    phase: int,
    sell_signal: Dict[str, object],
) -> Tuple[str, str]:
    if current_price <= stop_loss:
        return "SELL NOW", "Current price is already at or below the active stop loss."

    if bool(sell_signal.get("is_sell")) or phase == 4:
        return "SELL NOW", "Cached technicals show a sell condition or a Phase 4 downtrend."

    if phase == 3 and current_gain_pct > 0:
        return "SELL PARTIAL / TIGHTEN SL", "Phase 3 distribution suggests reducing risk and tightening the stop."

    if recommended_stop > stop_loss + 0.01:
        if current_gain_pct >= 15:
            return "RAISE SL / CONSIDER PARTIAL", "Large open gain allows a tighter stop and optional partial profit-taking."
        return "RAISE SL", "Trend is still constructive and the stop can be raised."

    if trade_style == "LONG-TERM" and current_gain_pct >= 0 and phase in {1, 2}:
        return "MAINTAIN", "Long-term position remains technically intact in cached data."

    if phase == 2 and current_gain_pct >= 0:
        return "MAINTAIN", "Uptrend is still intact; keep managing the existing position."

    return "HOLD / REVIEW", "No automatic sell or stop adjustment trigger was found."


def analyze_position(
    position: StockPosition,
    benchmark_data: pd.DataFrame,
    cache_root: Path = CACHE_ROOT,
) -> Dict[str, object]:
    price_data, cache_file = load_cached_price_data(position.ticker, cache_root)
    if price_data.empty:
        return {
            "ticker": position.ticker,
            "status": "missing_cache",
            "message": f"No usable local cached price data found for {position.ticker}.",
        }

    current_price = float(price_data["Close"].iloc[-1])
    phase_info = classify_phase(price_data, current_price)
    phase = int(phase_info.get("phase", 0) or 0)
    previous_phase = calculate_previous_phase(price_data)

    rs_series = pd.Series(dtype=float)
    if not benchmark_data.empty:
        rs_series = calculate_relative_strength(price_data["Close"], benchmark_data["Close"])

    sell_signal = score_sell_signal(
        ticker=position.ticker,
        price_data=price_data,
        current_price=current_price,
        phase_info=phase_info,
        rs_series=rs_series,
        previous_phase=previous_phase,
    )

    recommended_stop, stop_reason = calculate_recommended_stop(
        entry_price=position.entry_price,
        current_price=current_price,
        current_stop=position.stop_loss,
        phase_info=phase_info,
        price_data=price_data,
    )

    gain_pct = round(((current_price - position.entry_price) / position.entry_price) * 100, 2)
    pnl_dollars = round((current_price - position.entry_price) * position.share_size, 2)
    stop_gap_pct = round(((current_price - recommended_stop) / current_price) * 100, 2)
    action, action_reason = determine_action(
        trade_style=position.trade_style,
        current_price=current_price,
        stop_loss=position.stop_loss,
        recommended_stop=recommended_stop,
        current_gain_pct=gain_pct,
        phase=phase,
        sell_signal=sell_signal,
    )

    return {
        "ticker": position.ticker,
        "status": "ok",
        "trade_style": position.trade_style,
        "cache_file": str(cache_file) if cache_file else "N/A",
        "cache_timestamp": datetime.fromtimestamp(cache_file.stat().st_mtime).isoformat()
        if cache_file
        else "N/A",
        "entry_price": position.entry_price,
        "entry_date": position.entry_date.strftime("%Y-%m-%d"),
        "share_size": position.share_size,
        "current_stop": round(position.stop_loss, 2),
        "recommended_stop": recommended_stop,
        "current_price": round(current_price, 2),
        "gain_pct": gain_pct,
        "pnl_dollars": pnl_dollars,
        "phase": phase,
        "phase_name": phase_info.get("phase_name", "Unknown"),
        "sell_score": sell_signal.get("score", 0),
        "action": action,
        "action_reason": action_reason,
        "stop_reason": stop_reason,
        "stop_gap_pct": stop_gap_pct,
        "phase_reasons": phase_info.get("reasons", []),
        "sell_reasons": sell_signal.get("reasons", []),
    }


def format_report(results: Iterable[Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append("=" * 100)
    lines.append("CHECK STOCKS REPORT (LOCAL CACHE ONLY)")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 100)

    for result in results:
        lines.append("")
        lines.append("-" * 100)
        lines.append(str(result["ticker"]))
        lines.append("-" * 100)

        if result["status"] != "ok":
            lines.append(f"STATUS: {result['status'].upper()}")
            lines.append(str(result["message"]))
            continue

        lines.append(
            "Entry ${entry_price:.2f} on {entry_date} | Current ${current_price:.2f} | "
            "Shares {share_size:g} | P/L {gain_pct:+.2f}% (${pnl_dollars:+.2f})".format(**result)
        )
        lines.append(
            "Style: {trade_style} | Phase {phase} ({phase_name}) | Sell score: {sell_score}".format(
                **result
            )
        )
        lines.append(
            "Current SL ${current_stop:.2f} -> Recommended SL ${recommended_stop:.2f} "
            "({stop_gap_pct:.2f}% below current price)".format(**result)
        )
        lines.append(f"ACTION: {result['action']}")
        lines.append(f"WHY: {result['action_reason']}")
        lines.append(f"STOP LOGIC: {result['stop_reason']}")
        lines.append(f"CACHE: {result['cache_file']}")
        lines.append(f"CACHE UPDATED: {result['cache_timestamp']}")

        phase_reasons = result.get("phase_reasons") or []
        if phase_reasons:
            lines.append("PHASE SIGNALS:")
            for reason in phase_reasons[:4]:
                lines.append(f"  - {reason}")

        sell_reasons = result.get("sell_reasons") or []
        if sell_reasons:
            lines.append("SELL SIGNALS:")
            for reason in sell_reasons[:4]:
                lines.append(f"  - {reason}")

    lines.append("")
    lines.append("=" * 100)
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_file).expanduser().resolve()

    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    try:
        positions = load_positions_from_csv(csv_path)
    except Exception as exc:
        print(f"Failed to read CSV: {exc}", file=sys.stderr)
        return 1

    if not positions:
        print("No stock rows found in CSV.", file=sys.stderr)
        return 1

    benchmark_data = load_cached_benchmark()
    results = [analyze_position(position, benchmark_data) for position in positions]
    print(format_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
