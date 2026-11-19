from datetime import date
from types import SimpleNamespace
import unittest

import pandas as pd

from scripts.replay_us_levels import observe, replay
from src.core.trading_calendar import get_trading_dates


class ReplayUSLevelsTests(unittest.TestCase):
    def test_reaction_requires_a_later_session_and_stops_after_breakout(self):
        zone = {"low": 90, "high": 100}
        touch = SimpleNamespace(low=100, high=116, close=114)
        rebound = SimpleNamespace(low=110, high=116, close=114)
        gap = SimpleNamespace(low=85, high=89, close=86)

        last_day = observe([touch], zone, "support", atr=10, stop=88)
        self.assertEqual(last_day["touch_day"], 1)
        self.assertIsNone(last_day["reacted"])

        reacted = observe([touch, rebound], zone, "support", atr=10, stop=88)
        self.assertEqual(reacted["reaction_day"], 2)

        broken = observe([gap, touch, rebound], zone, "support", atr=10, stop=88)
        self.assertEqual(broken["break_day"], 1)
        self.assertEqual(broken["stop_day"], 1)
        self.assertEqual(broken["touch_day"], 2)
        self.assertFalse(broken["reacted"])

    def test_basis_uses_only_available_bars_and_censors_short_horizons(self):
        sessions = get_trading_dates("us", date(2026, 5, 21), date(2026, 9, 15))[:65]
        close = pd.Series([100 + [0, 3, 6, 3, 0, -3, -6, -3][i % 8] for i in range(len(sessions))])
        frame = pd.DataFrame({
            "code": "LITE", "data_source": "LongbridgeFetcher", "date": sessions.date,
            "open": close, "high": close + 1, "low": close - 1, "close": close,
        })
        basis = sessions[59].date().isoformat()
        original = replay(frame, sessions)
        changed = frame.copy()
        changed.loc[60:, ["open", "high", "low", "close"]] += 200
        observed = replay(changed, sessions)
        columns = ["rule", "horizon", "side", "atr14", "low", "high", "confirmed_date", "distance_atr", "stop_loss"]
        first_basis = original[original["basis_date"].eq(basis)]
        pd.testing.assert_frame_equal(
            first_basis[columns].reset_index(drop=True),
            observed.loc[observed["basis_date"].eq(basis), columns].reset_index(drop=True),
        )
        self.assertTrue(first_basis["confirmed_date"].dropna().le(basis).all())
        self.assertTrue(first_basis.loc[first_basis["horizon"].eq(5), "forward_bars"].eq(5).all())
        self.assertTrue(first_basis.loc[first_basis["horizon"].eq(10), "touched"].isna().all())

        # A provider change resets history and stops forward labels crossing the boundary.
        changed = frame.copy()
        changed.loc[62:, "data_source"] = "YfinanceFetcher"
        split = replay(changed, sessions)
        self.assertEqual(set(split["source"]), {"LongbridgeFetcher"})
        self.assertTrue(split["forward_bars"].le(2).all())
        self.assertNotIn("touched", split.columns)
