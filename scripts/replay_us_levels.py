#!/usr/bin/env python3
"""Replay frozen US swing zones on completed local daily bars.

A touch is inclusive OHLC overlap from the next session. A breakout is a
close strictly beyond the far zone edge, even on a gap without a touch.
A reaction is a later close at least 1 baseline ATR beyond the near edge
on the favorable side, before a breakout. Reaction eligibility requires
an observed touch and at least one following bar. No intrabar ordering.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.trading_calendar import get_latest_completed_us_session, get_trading_dates
from src.services.us_technical import completed_bars, technical_levels


RULES = {"60x3": (60, 3), "20x2": (20, 2), "30x2": (30, 2)}
HORIZONS = (5, 10, 20)
MIN_HISTORY = 60


def distance(close, level, atr):
    return level, abs(close - level), 100 * abs(close - level) / close, abs(close - level) / atr


def observe(future, zone, side, atr, stop):
    outcome = {"touch_day": None, "reaction_day": None, "break_day": None, "stop_day": None}
    if stop is not None:
        outcome["stop_day"] = next((i for i, bar in enumerate(future, 1) if bar.close < stop), None)
    if zone is None:
        return {**outcome, "reaction_eligible": None, "touched": None, "reacted": None, "broken": None}

    support = side == "support"
    near = zone["high"] if support else zone["low"]
    far = zone["low"] if support else zone["high"]
    for i, bar in enumerate(future, 1):
        if outcome["touch_day"] is None and bar.low <= zone["high"] and bar.high >= zone["low"]:
            outcome["touch_day"] = i
        beyond = bar.close < far if support else bar.close > far
        if outcome["break_day"] is None and beyond:
            outcome["break_day"] = i
        favorable = bar.close >= near + atr if support else bar.close <= near - atr
        touch = outcome["touch_day"]
        if touch is not None and touch < i and outcome["break_day"] is None and favorable:
            if outcome["reaction_day"] is None:
                outcome["reaction_day"] = i
    touch = outcome["touch_day"]
    eligible = touch is not None and touch < len(future)
    return {
        **outcome, "reaction_eligible": eligible, "touched": touch is not None,
        "reacted": outcome["reaction_day"] is not None if eligible else None,
        "broken": outcome["break_day"] is not None,
    }


def replay(frame, sessions):
    records = []
    for name, (lookback, confirm) in RULES.items():
        options = {
            "US_LEVEL_LOOKBACK_BARS": str(lookback), "US_SWING_CONFIRM_BARS": str(confirm),
            "US_LEVEL_ZONE_ATR": "0.25", "US_STOP_BUFFER_ATR": "0.5",
        }
        with patch.dict("os.environ", options):
            for code, ticker in frame.groupby("code"):
                positions = pd.Series(sessions.get_indexer(pd.to_datetime(ticker["date"])), index=ticker.index)
                new_segment = positions.diff().ne(1) | ticker["data_source"].ne(ticker["data_source"].shift())
                for _, segment in ticker.groupby(new_segment.cumsum()):
                    segment = segment.reset_index(drop=True)
                    bars = list(segment.itertuples(index=False))
                    for i in range(MIN_HISTORY - 1, len(bars)):
                        bar = bars[i]
                        window = segment.iloc[:i + 1]
                        window = window[window["date"] >= bar.date - timedelta(days=180)]
                        levels = technical_levels(window, None, bar.date)
                        atr = levels["atr14"]
                        stop = levels["stop_loss"]
                        for side in ("support", "resistance"):
                            zone = levels[side]
                            row = {
                                "ticker": code, "basis_date": bar.date.isoformat(), "rule": name, "side": side,
                                "source": bar.data_source, "currency": "USD", "basis_close": bar.close,
                                "history_bars": len(window), "atr14": atr, "present": zone is not None,
                                **dict.fromkeys(("low", "high", "pivot_date", "confirmed_date", "pivot_price",
                                    "nearest_boundary", "boundary_distance_usd", "boundary_distance_pct",
                                    "boundary_distance_atr", "distance_usd", "distance_pct", "distance_atr",
                                    "stop_loss", "stop_distance_usd", "stop_distance_pct", "stop_distance_atr")),
                                "zone_position": "missing",
                            }
                            if zone:
                                inside = zone["low"] <= bar.close <= zone["high"]
                                boundary = min((zone["low"], zone["high"]), key=lambda price: abs(bar.close - price))
                                _, usd, pct, atr_dist = distance(bar.close, boundary, atr)
                                row.update(zone)
                                row.update(
                                    nearest_boundary=boundary, zone_position="inside" if inside else "outside",
                                    boundary_distance_usd=usd, boundary_distance_pct=pct, boundary_distance_atr=atr_dist,
                                    distance_usd=0 if inside else usd, distance_pct=0 if inside else pct,
                                    distance_atr=0 if inside else atr_dist,
                                )
                            if stop is not None and side == "support":
                                price, usd, pct, atr_dist = distance(bar.close, stop, atr)
                                row.update(stop_loss=price, stop_distance_usd=usd, stop_distance_pct=pct, stop_distance_atr=atr_dist)
                            for horizon in HORIZONS:
                                next_bars = bars[i + 1:i + horizon + 1]
                                record = {**row, "horizon": horizon, "forward_bars": len(next_bars)}
                                if len(next_bars) == horizon:
                                    record.update(observe(next_bars, zone, side, atr, stop if side == "support" else None))
                                records.append(record)
    return pd.DataFrame.from_records(records)


def summarize(data):
    measured = data[data["phase"].isin(["train", "holdout"])]
    grouped = measured.groupby(["phase", "rule", "horizon", "side"])
    per_ticker = measured.groupby(["phase", "rule", "horizon", "side", "ticker"])

    def statistics(group):
        present = group[group["present"]]
        touched = present[present["touched"].eq(True)]
        eligible = present[present["reaction_eligible"].eq(True)]
        levels = len(present)
        output = {
            "bases": len(group), "levels": levels, "coverage": levels / len(group),
            "missing_rate": 1 - levels / len(group),
            "touch_n": len(touched), "touch_rate": len(touched) / levels if levels else None,
            "touch_over_all_bases": len(touched) / len(group),
            "reaction_eligible_n": len(eligible),
            "reaction_n": int(eligible["reacted"].eq(True).sum()),
            "reaction_rate": eligible["reacted"].eq(True).mean() if len(eligible) else None,
            "break_n": int(present["broken"].eq(True).sum()),
            "break_rate": present["broken"].eq(True).mean() if levels else None,
            "break_after_or_on_touch_n": int(touched["break_day"].ge(touched["touch_day"]).sum()),
        }
        output["break_after_or_on_touch_rate"] = (
            output["break_after_or_on_touch_n"] / len(touched) if len(touched) else None
        )
        for col in ("distance_pct", "distance_atr", "stop_distance_pct", "stop_distance_atr"):
            for label, q in (("p25", 0.25), ("p50", 0.5), ("p75", 0.75)):
                output[f"{col}_{label}"] = group[col].quantile(q)
        return output

    rows = []
    for keys, group in grouped:
        rows.append(dict(zip(("phase", "rule", "horizon", "side"), keys), ticker="ALL", **statistics(group)))
    for keys, group in per_ticker:
        rows.append(dict(zip(("phase", "rule", "horizon", "side", "ticker"), keys), **statistics(group)))
    return pd.DataFrame(rows)


def paired_table(data):
    data = data[data["phase"].isin(["train", "holdout"]) & data["present"]]
    keys = ["phase", "ticker", "basis_date", "horizon", "side"]
    base = data[data["rule"].eq("60x3")]
    rows = []
    for rule in ("20x2", "30x2"):
        paired = data[data["rule"].eq(rule)].merge(base, on=keys, suffixes=("_short", "_base"))
        paired = paired[paired["distance_atr_short"] < paired["distance_atr_base"] - 1e-9]
        for (phase, horizon, side), group in paired.groupby(["phase", "horizon", "side"]):
            row = {"phase": phase, "rule": rule, "horizon": horizon, "side": side, "closer_n": len(group)}
            row["distance_atr_delta_p50"] = (group["distance_atr_short"] - group["distance_atr_base"]).median()
            for event in ("touched", "broken"):
                for suffix in ("short", "base"):
                    row[f"{event}_{suffix}_rate"] = group[f"{event}_{suffix}"].eq(True).mean()
            rows.append(row)
    return pd.DataFrame(rows)


def threshold_table(data):
    # Tune the proposed alert against the incumbent rule. Use only 10-day
    # touch as the target and exclude missing zones from near/far rates.
    data = data[data["rule"].eq("60x3") & data["horizon"].eq(10)]
    rows = []
    for phase in ("train", "holdout"):
        sample = data[data["phase"].eq(phase)]
        for threshold in (0.5, 1.0, 1.5, 2.0):
            near = sample[sample["distance_atr"].le(threshold)]
            far = sample[sample["distance_atr"].gt(threshold)]
            rows.append({
                "phase": phase, "threshold_atr": threshold, "bases": len(sample),
                "near_n": len(near), "far_n": len(far),
                "near_touch_10": near["touched"].eq(True).mean() if len(near) else None,
                "far_touch_10": far["touched"].eq(True).mean() if len(far) else None,
                "no_near_or_missing_rate": 1 - len(near) / len(sample) if len(sample) else None,
            })
    table = pd.DataFrame(rows)
    train = table[table["phase"].eq("train") & table["near_n"].ge(20) & table["far_n"].ge(20)].copy()
    train["lift"] = train["near_touch_10"] - train["far_touch_10"]
    best = train.sort_values(["lift", "threshold_atr"], ascending=[False, True]).head(1)
    table["selected_on_train"] = table["threshold_atr"].isin(best["threshold_atr"])
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/stock_analysis.db"))
    parser.add_argument("--tickers", required=True, help="Comma-separated US tickers")
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--train-end", type=date.fromisoformat, required=True)
    parser.add_argument("--holdout-start", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    end = get_latest_completed_us_session()
    if end is None or not args.start < args.train_end < args.holdout_start < args.end <= end:
        parser.error("Dates must be ordered, and --end must be a completed US session or earlier.")
    sessions = get_trading_dates("us", args.start, args.end)
    if sessions is None or sum(args.train_end < day < args.holdout_start for day in sessions.date) < max(HORIZONS):
        parser.error("A trading calendar and at least 20 embargo sessions are required.")
    codes = list(dict.fromkeys(args.tickers.upper().replace(" ", "").split(",")))
    if not all(codes):
        parser.error("Ticker list contains an empty symbol.")
    if not args.db.is_file():
        parser.error("Local database not found.")
    placeholders = ",".join("?" for _ in codes)
    wal = args.db.with_name(args.db.name + "-wal")
    if wal.exists() and wal.stat().st_size:
        parser.error("Use a checkpointed local SQLite snapshot when a nonempty WAL is present.")
    connection = sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        raw = pd.read_sql_query(
            f"""SELECT code, date, open, high, low, close, data_source FROM stock_daily
            WHERE code IN ({placeholders}) AND date BETWEEN ? AND ? ORDER BY code, date, updated_at, id""",
            connection, params=[*codes, args.start.isoformat(), args.end.isoformat()],
        )
    finally:
        connection.close()
    frames = [
        completed_bars(ticker, args.end, sessions)
        for _, ticker in raw.dropna(subset=["data_source"]).groupby("code")
    ]
    if not frames or not any(not frame.empty for frame in frames):
        parser.error("No valid completed bars found for the requested tickers.")
    frame = pd.concat(frames, ignore_index=True)
    details = replay(frame, sessions)
    if details.empty:
        parser.error("No contiguous same-source segment has the required 60 bars.")

    # The longest horizon fixes the same analysis cohort for all horizons.
    # Preserve embargo and right-censored observations in the daily CSV.
    complete = details[details["horizon"].eq(max(HORIZONS)) & details["forward_bars"].eq(max(HORIZONS))]
    complete_keys = set(zip(complete["ticker"], complete["basis_date"]))
    details["phase"] = [
        "right_censored" if (ticker, day) not in complete_keys
        else "train" if day <= args.train_end.isoformat()
        else "holdout" if day >= args.holdout_start.isoformat()
        else "embargo"
        for ticker, day in zip(details["ticker"], details["basis_date"])
    ]
    if not {"train", "holdout"}.issubset(set(details["phase"])):
        parser.error("Both analysis phases need complete observations.")
    args.output.mkdir(parents=True, exist_ok=True)
    details.sort_values(["basis_date", "ticker", "rule", "side", "horizon"]).to_csv(
        args.output / "daily.csv", index=False, encoding="utf-8-sig",
    )
    summarize(details).to_csv(args.output / "summary.csv", index=False, encoding="utf-8-sig")
    paired_table(details).to_csv(args.output / "paired.csv", index=False, encoding="utf-8-sig")
    threshold_table(details).to_csv(args.output / "thresholds.csv", index=False, encoding="utf-8-sig")
    counts = details.drop_duplicates(["ticker", "basis_date"]).groupby(["ticker", "phase"]).size().unstack(fill_value=0)
    print(counts.reindex(codes, fill_value=0).to_string())
    print(f"Wrote {len(details)} rule/side/horizon rows to {args.output}")


if __name__ == "__main__":
    main()
