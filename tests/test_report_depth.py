# -*- coding: utf-8 -*-
"""Report-tier depth tests: trade-metrics math and the JSON-validity fix
(profit factor / win-loss ratios returned float('inf') with zero losses,
which json.dump emits as `Infinity` - invalid JSON that strict parsers
reject, corrupting exported reports)."""

import json
import tempfile

import pytest

from trading.report.report_generator import ReportGenerator


@pytest.fixture()
def rg():
    return ReportGenerator(output_dir=tempfile.mkdtemp())


class TestTradeMetrics:
    def test_hand_checked_metrics(self, rg):
        trades = [{"pnl": p} for p in (120, 80, -40, 200, -60, 90, 150, -30)]
        m = rg._calculate_trade_metrics({"trades": trades})
        assert m.total_trades == 8
        assert m.winning_trades == 5 and m.losing_trades == 3
        assert m.win_rate == pytest.approx(0.625)
        assert m.total_pnl == pytest.approx(510.0)
        assert m.profit_factor == pytest.approx(640.0 / 130.0)

    def test_zero_losses_is_json_safe(self, rg):
        m = rg._calculate_trade_metrics(
            {"trades": [{"pnl": 100.0}, {"pnl": 50.0}]}
        )
        assert m.profit_factor == 999.0
        # Must round-trip as STRICT JSON (no Infinity extension).
        payload = json.dumps({"pf": m.profit_factor})
        assert json.loads(payload)["pf"] == 999.0

    def test_all_losses(self, rg):
        m = rg._calculate_trade_metrics(
            {"trades": [{"pnl": -10.0}, {"pnl": -5.0}]}
        )
        assert m.win_rate == 0.0
        assert m.profit_factor == 0.0

    def test_empty_trades(self, rg):
        m = rg._calculate_trade_metrics({"trades": []})
        assert m.total_trades == 0 and m.profit_factor == 0.0


class TestUnifiedReporterRatios:
    def test_no_inf_anywhere(self):
        import trading.report.unified_trade_reporter as U
        src = open(U.__file__).read()
        assert 'float("inf")' not in src, (
            "ratio metrics must stay JSON-safe (capped, not inf)"
        )
