import json
import sqlite3
import tempfile
import unittest
from contextlib import ExitStack, closing
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from data_provider.realtime_types import RealtimeSource, UnifiedRealtimeQuote
from src.analyzer import GeminiAnalyzer
from src.config import Config
from src.core.pipeline import StockAnalysisPipeline
from src.core.trading_calendar import get_latest_completed_us_session, get_trading_dates
from src.enums import ReportType
from src.services.us_technical import technical_levels
from src.stock_analyzer import StockTrendAnalyzer
from src.storage import DatabaseManager


BASIS = date(2026, 11, 17)
PREMARKET = datetime(2026, 11, 18, 6, 19, tzinfo=timezone.utc)


def daily_fixture():
    days = get_trading_dates("us", date(2026, 7, 31), BASIS)[-77:]
    close = [550 + [0, 15, 30, 15, 0, -15, -30, -15][i % 8] for i in range(len(days))]
    close[-1] = 559.97
    return pd.DataFrame({
        "date": days, "open": close, "close": close,
        "high": [price + 5 for price in close], "low": [price - 5 for price in close],
        "volume": 1000,
    })


class USCompletedLevelsTests(unittest.TestCase):
    def test_session_uses_dst_early_close_and_holiday(self):
        self.assertEqual(get_latest_completed_us_session(PREMARKET), BASIS)
        self.assertEqual(
            get_latest_completed_us_session(datetime(2026, 11, 27, 17, 59, tzinfo=timezone.utc)),
            date(2026, 11, 25),
        )
        self.assertEqual(
            get_latest_completed_us_session(datetime(2026, 11, 27, 18, tzinfo=timezone.utc)),
            date(2026, 11, 27),
        )
        self.assertEqual(
            get_latest_completed_us_session(datetime(2026, 7, 6, 20, tzinfo=timezone.utc)),
            date(2026, 7, 6),
        )

    def test_wilder_atr_and_absent_pivots(self):
        frame = daily_fixture().tail(42).copy()
        frame["date"] = frame.date.dt.date
        frame["close"] = [100 + i for i in range(len(frame))]
        frame["open"] = frame.close
        frame["high"] = frame.close + 2
        frame["low"] = frame.close - 2
        # The last TR is 18, following fourteen-or-more TRs of 4.
        frame.loc[frame.index[-1], "high"] = frame.close.iloc[-1] + 16
        trend = StockTrendAnalyzer().analyze(frame, "LITE")
        self.assertIsNone(trend.ma60)
        self.assertIsNone(trend.to_dict()["ma60"])
        levels = technical_levels(frame, trend, BASIS)
        self.assertAlmostEqual(levels["atr14"], 5.0)
        self.assertIsNone(levels["support"])
        self.assertIsNone(levels["resistance"])
        self.assertIsNone(levels["stop_loss"])
        self.assertIsNone(technical_levels(pd.DataFrame(), None, BASIS)["atr14"])

    def test_normal_pipeline_refills_cache_and_locks_report_and_database(self):
        frame = daily_fixture()
        clipped = frame.tail(42)
        quote = UnifiedRealtimeQuote(
            code="LITE", source=RealtimeSource.LONGBRIDGE, price=559.97,
            provider_timestamp="2026-11-17T21:00:00+00:00", session="常规", currency="USD",
        )
        fetcher = MagicMock()
        fetcher.get_stock_name.return_value = "Lumentum"
        fetcher.get_daily_data.return_value = (frame, "LongbridgeFetcher")
        fetcher.get_realtime_quote.return_value = quote
        fetcher.get_chip_distribution.return_value = None
        fetcher.get_us_trade_profile.return_value = {
            "peaks": [{"price": 550.0}], "dates": ["2026-11-17"],
            "timestamp": "2026-11-17T21:00:00+00:00",
        }
        config = Config(report_summary_only=True, ntfy_url="https://ntfy.example/lite-audit")
        with ExitStack() as stack:
            path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            db_path = path / "analysis.sqlite"
            db = DatabaseManager(f"sqlite:///{db_path.as_posix()}")
            stack.callback(db._engine.dispose)
            db.save_daily_data(clipped, "LITE", "LongbridgeFetcher")
            model_contexts = []

            def analyze(context, **_):
                model_contexts.append(context)
                parser = object.__new__(GeminiAnalyzer)
                parser._get_runtime_config = lambda: config
                return parser._parse_response(json.dumps({
                    "analysis_summary": "Model suggested 9999 as stop",
                    "sentiment_score": 50, "operation_advice": "观望",
                    "trend_prediction": "震荡",
                    "dashboard": {
                        "data_perspective": {"price_position": {"ma20": 7777, "resistance_level": 8888}},
                        "battle_plan": {"sniper_points": {"stop_loss": 9999, "take_profit": 9998}},
                    },
                }), "LITE", "Lumentum")

            with (
                patch("src.core.pipeline.get_db", return_value=db),
                patch("src.core.pipeline.DataFetcherManager", return_value=fetcher),
                patch("src.core.pipeline.GeminiAnalyzer", return_value=MagicMock(analyze=analyze)),
                patch("src.notification.get_config", return_value=config),
                patch("src.core.pipeline.SearchService", return_value=MagicMock(is_available=False)),
                patch("src.core.pipeline.SocialSentimentService", return_value=MagicMock(is_available=False)),
                patch("src.core.pipeline.StockAnalysisPipeline._build_market_structure_context", return_value=None),
                patch("src.core.pipeline.StockAnalysisPipeline._load_persisted_intelligence_context", return_value=None),
                patch("src.notification_sender.ntfy_sender.requests.post") as post,
            ):
                post.return_value.status_code = 200
                pipeline = StockAnalysisPipeline(config=config, max_workers=1, daily_market_context_enabled=False)
                local = path / "report.md"

                def save_report(text):
                    local.write_text(text, encoding="utf-8")
                    return str(local)

                with patch.object(pipeline.notifier, "save_report_to_file", side_effect=save_report):
                    results = pipeline.run(["LITE"], current_time=PREMARKET)
                self.assertEqual(len(results), 1)
                fetcher.get_daily_data.assert_called_once()
                self.assertEqual(fetcher.get_daily_data.call_args.kwargs["end_date"], BASIS.isoformat())
                context = model_contexts[0]
                self.assertEqual(context["today"]["date"], BASIS.isoformat())
                levels = results[0].dashboard["computed_us_levels"]
                self.assertEqual(levels["samples"], 77)
                self.assertAlmostEqual(levels["ma20"], frame.close.tail(20).mean())
                self.assertAlmostEqual(levels["ma60"], frame.close.tail(60).mean())
                self.assertEqual(levels["quotes"][0]["timestamp"], quote.provider_timestamp)
                self.assertIsNotNone(levels["stop_loss"])
                with closing(sqlite3.connect(db_path)) as connection:
                    raw, stop = connection.execute("SELECT raw_result, stop_loss FROM analysis_history").fetchone()
                self.assertEqual(stop, levels["stop_loss"])
                self.assertEqual(json.loads(raw)["dashboard"]["computed_us_levels"], levels)
                post.assert_called_once()
                summary = post.call_args.kwargs["json"]["message"]
                full = local.read_text(encoding="utf-8")
                for report in (summary, full):
                    self.assertIn("LITE · USD · 基准 2026-11-17", report)
                    self.assertIn("常规交易时段日线收盘低于止损线", report)
                    self.assertNotIn("9999", report)
                    self.assertNotIn("8888", report)
                self.assertIn("Wilder ATR14", full)
                self.assertIn("长桥成交密集价档", full)
                self.assertNotIn("Wilder ATR14", summary)
