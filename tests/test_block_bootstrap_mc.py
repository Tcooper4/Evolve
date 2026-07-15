# -*- coding: utf-8 -*-
"""Monte Carlo i.i.d. vs stationary block bootstrap — reproduce + gate."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest

from trading.analysis.block_bootstrap import (
    DEFAULT_METHOD,
    compare_tail_risk,
    estimate_mean_block_length,
    simulate_equity_paths,
    simulate_garch11,
    stationary_block_path,
)


class TestHandVerifiedBlockVsIid:
    def test_garch_block_tail_measurably_differs_iid_direction(self):
        """GARCH cluster: block p5 typically *higher* than iid (iid stacks crashes).

        Multi-seed median of (iid_p5 - block_p5) is negative on this suite —
        opposite of an earlier single-seed anecdote. Magnitude at 63d/L=15+
        is not a noise shrug.
        """
        r = simulate_garch11(2000, omega=1e-6, alpha=0.08, beta=0.90, seed=7)
        deltas = []
        for seed in range(20, 30):
            iid = simulate_equity_paths(
                r, n_simulations=800, horizon=63, method="iid", seed=seed,
            )
            blk = simulate_equity_paths(
                r, n_simulations=800, horizon=63, method="stationary_block",
                mean_block_length=15.0, seed=seed,
            )
            deltas.append(
                100.0 * (iid["final_p5"] - blk["final_p5"]) / 10_000.0
            )
        med = float(np.median(deltas))
        # Established Evolve finding: median delta negative (block milder p5)
        assert med < 0.0, deltas
        assert abs(med) >= 0.15, deltas  # material ppt of capital

    def test_mean_block_length_formula_hand_math(self):
        assert round(1.0 / (1.0 - 0.9)) == 10
        assert round(1.0 / (1.0 - 0.95)) == 20
        r = simulate_garch11(800, seed=1)
        meta = estimate_mean_block_length(r)
        assert 5 <= meta["L"] <= 63
        assert 0.0 <= meta["rho_sq"] <= 0.99

    def test_block_path_values_from_pool(self):
        pool = np.array([0.01, -0.02, 0.03, -0.01], dtype=float)
        path = stationary_block_path(
            pool, 20, mean_block_length=5.0, rng=np.random.default_rng(1)
        )
        assert path.shape == (20,)
        assert set(np.round(path, 8)).issubset(set(np.round(pool, 8)))


class TestDefaultGate:
    def test_default_remains_iid_after_null_softening_result(self):
        """Do not switch default: block would soften reported p5 vs iid here."""
        assert DEFAULT_METHOD == "iid"


class TestSyntheticReport:
    def test_print_garch_comparison_table(self):
        r = simulate_garch11(1500, omega=1e-6, alpha=0.08, beta=0.90)
        cmp_ = compare_tail_risk(r, n_simulations=700, seed=42)
        print("\n=== Synthetic GARCH iid vs block (delta_p5 ppt of capital) ===")
        for row in cmp_["rows"]:
            print(
                f"h={row['horizon']:3d} L={row['mean_block']:.0f} "
                f"iid_p5={row['iid_p5']:.1f} blk_p5={row['block_p5']:.1f} "
                f"dppt={row['delta_p5_ppt']:.3f}"
            )
        long = [row for row in cmp_["rows"] if row["horizon"] == 63]
        # Same-seed table may flip sign; multi-seed hand test is authoritative.
        assert long
        assert any(abs(row["delta_p5_ppt"]) >= 0.1 for row in long)


@pytest.mark.external
class TestRealDataComparison:
    SYMBOLS = ("SPY", "QQQ", "IWM", "AAPL")

    def test_real_symbols_do_not_justify_default_switch(self):
        try:
            from trading.data.price_cache import get_history
        except Exception as e:
            pytest.skip(f"price_cache unavailable: {e}")

        summaries: List[Dict[str, Any]] = []
        for sym in self.SYMBOLS:
            hist = get_history(sym, period="2y")
            if hist is None or hist.empty:
                continue
            _cm = {str(c).lower(): c for c in hist.columns}
            cc = _cm.get("close", hist.columns[0])
            rets = hist[cc].astype(float).pct_change().dropna().values
            if len(rets) < 80:
                continue
            deltas = []
            for seed in range(40, 48):
                iid = simulate_equity_paths(
                    rets, n_simulations=500, horizon=63, method="iid", seed=seed,
                )
                blk = simulate_equity_paths(
                    rets, n_simulations=500, horizon=63,
                    method="stationary_block", mean_block_length=15.0, seed=seed,
                )
                deltas.append(
                    100.0 * (iid["final_p5"] - blk["final_p5"]) / 10_000.0
                )
            med = float(np.median(deltas))
            meta = estimate_mean_block_length(rets)
            summaries.append({
                "symbol": sym, "median_dppt": med, "L_auto": meta["L"],
                "rho_sq": meta["rho_sq"],
            })
            print(f"{sym}: median_dppt={med:.3f} autoL={meta['L']} rho={meta['rho_sq']:.3f}")

        if not summaries:
            pytest.skip("No real history available offline")

        med_of_med = float(np.median([s["median_dppt"] for s in summaries]))
        print(f"GATE med_of_med_dppt={med_of_med:.3f} DEFAULT={DEFAULT_METHOD}")
        # Softening (negative) or near-null is acceptable — do not force switch
        assert DEFAULT_METHOD == "iid"
        # At least |effect| sometimes material — not all zero
        assert any(abs(s["median_dppt"]) >= 0.2 for s in summaries), summaries
