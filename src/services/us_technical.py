"""Completed-session US technical levels and their report presentation."""

import os
import re

import pandas as pd


def completed_bars(frame, end_date, sessions):
    if frame is None or frame.empty or sessions is None:
        return pd.DataFrame()
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    prices = frame[["open", "high", "low", "close"]]
    valid = (
        ((prices > 0) & (prices < float("inf"))).all(axis=1)
        & (frame["high"] >= prices[["open", "low", "close"]].max(axis=1))
        & (frame["low"] <= prices[["open", "close"]].min(axis=1))
    )
    return (
        frame.loc[valid & frame["date"].le(end_date) & frame["date"].isin(sessions.date)]
        .sort_values("date", kind="stable")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )


def technical_levels(frame, trend, target_date):
    left = int(os.getenv("US_SWING_CONFIRM_BARS", "3"))
    lookback = int(os.getenv("US_LEVEL_LOOKBACK_BARS", "60"))
    zone_atr = float(os.getenv("US_LEVEL_ZONE_ATR", "0.25"))
    stop_atr = float(os.getenv("US_STOP_BUFFER_ATR", "0.5"))
    if not (1 <= left <= 20 and 2 * left < lookback <= 120 and 0 < zone_atr <= 5 and 0 < stop_atr <= 5):
        raise ValueError("Invalid US technical level settings")
    basis = None if frame.empty else frame.iloc[-1]
    sources = (
        sorted(frame["data_source"].dropna().unique())
        if "data_source" in frame else []
    )
    result = {
        "basis_date": basis.date.isoformat() if basis is not None else None,
        "expected_date": target_date.isoformat() if target_date else None,
        "currency": "USD",
        "sources": sources,
        "samples": len(frame),
        "close": float(basis.close) if basis is not None else None,
        "ma5": trend.ma5 if trend and trend.ma5 > 0 else None,
        "ma10": trend.ma10 if trend and trend.ma10 > 0 else None,
        "ma20": trend.ma20 if trend and trend.ma20 > 0 else None,
        "ma60": trend.ma60 if trend else None,
        "atr14": None,
        "support": None,
        "resistance": None,
        "stop_loss": None,
        "trigger": "常规交易时段日线收盘低于止损线",
        "parameters": {"confirm_bars_each_side": left, "lookback": lookback, "zone_atr": zone_atr, "stop_atr": stop_atr},
        "historically_validated": False,
        "holder_cost": "unavailable",
    }
    # Levels require a current, comparable series. The moving averages still
    # describe the supplied series, whose provenance is available above.
    if basis is None or basis.date != target_date or len(frame) < 15 or len(sources) > 1:
        return result
    prior = frame["close"].shift()
    tr = pd.concat(
        [frame["high"] - frame["low"], (frame["high"] - prior).abs(), (frame["low"] - prior).abs()],
        axis=1,
    ).max(axis=1).iloc[1:]
    atr = float(tr.iloc[:14].mean())
    for value in tr.iloc[14:]:
        atr = (13 * atr + value) / 14
    result["atr14"] = atr
    if atr <= 0:
        return result
    closes = float(basis.close)
    window = frame.tail(lookback).reset_index(drop=True)
    width = zone_atr * atr
    for label, column, direction in (("support", "low", -1), ("resistance", "high", 1)):
        candidates = []
        for i in range(left, len(window) - left):
            center = float(window.at[i, column])
            neighbors = window[column].iloc[i-left:i+left+1].drop(i)
            is_pivot = ((center - neighbors) * direction > 0).all()
            if is_pivot and (center - closes) * direction > width and center > width:
                candidates.append((abs(center - closes), -i, center))
        if candidates:
            _, index, center = min(candidates)
            result[label] = {
                "low": round(center - width, 2),
                "high": round(center + width, 2),
                "pivot_date": window.at[-index, "date"].isoformat(),
                "confirmed_date": window.at[-index + left, "date"].isoformat(),
                "pivot_price": center,
            }
    support = result["support"]
    if support:
        stop = round(support["pivot_price"] - (zone_atr + stop_atr) * atr, 2)
        if 0 < stop < support["low"]:
            result["stop_loss"] = stop
    return result


def lock_us_levels(result, levels):
    if levels is None:
        return
    dashboard = result.dashboard or {}
    result.dashboard = dashboard
    dashboard["computed_us_levels"] = levels
    perspective = dashboard.get("data_perspective") or {}
    dashboard["data_perspective"] = perspective
    price = {"current_price": levels["close"]}
    perspective["price_position"] = price
    for key in ("ma5", "ma10", "ma20", "ma60"):
        value = levels[key]
        price[key] = round(value, 4) if value is not None else None
    for key in ("support", "resistance"):
        zone = levels[key]
        price[f"{key}_level"] = zone["pivot_price"] if zone else None
    battle = dashboard.get("battle_plan") or {}
    dashboard["battle_plan"] = battle
    # These model fields can restate conflicting levels or triggers.
    battle["sniper_points"] = {"stop_loss": levels["stop_loss"]}
    battle.pop("position_strategy", None)
    battle.pop("action_checklist", None)
    perspective.pop("chip_structure", None)
    # Publish the computed technical section in place of model price prose.
    for field in ("trend_analysis", "technical_analysis", "ma_analysis", "pattern_analysis", "buy_reason"):
        setattr(result, field, "")
    for field in ("analysis_summary", "key_points", "risk_warning", "short_term_outlook", "medium_term_outlook"):
        value = getattr(result, field, None)
        if value and re.search(r"\d", str(value)):
            setattr(result, field, "")
    core = dashboard.get("core_conclusion")
    if isinstance(core, dict):
        for key, value in list(core.items()):
            if re.search(r"\d", str(value)):
                core.pop(key)
    dashboard.pop("strategy_synthesis", None)
    result.raw_response = None


def format_us_levels(results, *, full):
    lines = []
    for result in results:
        levels = (result.dashboard or {}).get("computed_us_levels")
        if not levels:
            continue
        basis = levels["basis_date"] or "不可用"
        close = levels.get("close")
        atr = levels.get("atr14")

        def distance(delta):
            percent = f"{delta / close:.2%}"
            atr_multiple = f"{delta / atr:.2f} ATR" if atr and atr > 0 else "ATR 倍数不可用"
            return f"距基准收盘 {percent} / {atr_multiple}"

        def zone(key):
            bounds = levels[key]
            if bounds is None:
                return "暂无已确认区间"
            low, high = bounds["low"], bounds["high"]
            text = f"{low:.2f}–{high:.2f}"
            if not close or close <= 0:
                return f"{text}（距基准收盘不可用）"
            if low <= close <= high:
                return f"{text}（区间内，{distance(0)}）"
            return f"{text}（{distance(min(abs(close - low), abs(close - high)))}）"

        stop = levels["stop_loss"]
        trigger = "不可用"
        if stop is not None:
            stop_distance = distance(abs(close - stop)) if close and close > 0 else "距基准收盘不可用"
            trigger = f'{stop:.2f}（{stop_distance}），{levels["trigger"]}'
        reference = f" · 收盘 {close:.2f}" if close and close > 0 else ""
        lines.append(f'**{result.code} · USD · 基准 {basis}{reference}**')
        lines.append(f'支撑 {zone("support")}｜压力 {zone("resistance")}｜条件止损 {trigger}')
        if not full:
            continue
        metrics = "｜".join(
            f'{label} {levels[key]:.4f}' if levels[key] is not None else f'{label} 不可用'
            for label, key in (("MA20", "ma20"), ("MA60", "ma60"), ("Wilder ATR14", "atr14"))
        )
        lines.append(metrics)
        sources = ", ".join(levels["sources"]) or "不可用"
        lines.append(f'已完成日线 {levels["samples"]} 根｜来源 {sources}')
        params = levels["parameters"]
        lines.append(
            f'最近 {params["lookback"]} 根内按前后各 {params["confirm_bars_each_side"]} 根确认摆动，'
            f'区域为摆动价 ± {params["zone_atr"]} ATR。止损为支撑区下沿 − {params["stop_atr"]} ATR。'
            '参数尚未经历史表现验证。'
        )
        for key, label in (("support", "支撑"), ("resistance", "压力")):
            if levels[key]:
                lines.append(f'{label}摆动日 {levels[key]["pivot_date"]}，确认日 {levels[key]["confirmed_date"]}')
        for quote in levels.get("quotes", []):
            timestamp = quote.get("timestamp") or "时间未知"
            stale = "｜已陈旧" if quote.get("is_stale") else ""
            lines.append(f'报价 {quote["session"]} {quote["price"]:.2f} USD｜{timestamp}｜{quote["source"]}{stale}')
        profile = levels.get("trade_profile")
        if profile:
            dates = ", ".join(profile["dates"]) or "未知"
            prices = ", ".join(f'{row["price"]:.4f}'.rstrip("0").rstrip(".") for row in profile["peaks"])
            lines.append(f'长桥成交密集价档 {prices} USD｜交易日 {dates}｜统计时间 {profile["timestamp"] or "未知"}')
            lines.append('按买、卖和中性成交统计字段之和排序。价档统计字段单位和覆盖时段未确认，仅供辅助观察。')
        else:
            lines.append("长桥按价成交统计不可用。")
        lines.append("当前持有人成本分布不可用。")
    return "\n\n".join(lines)
