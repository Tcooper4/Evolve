# -*- coding: utf-8 -*-
"""Run real GNN purged OOS (Phase 3 research pass)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading.models.gnn_validation import run_gnn_oos_real  # noqa: E402


def main() -> int:
    report = run_gnn_oos_real(
        out_path=str(ROOT / "data" / "gnn_oos_real.json"),
        period="2y",
        epochs=15,
    )
    print(json.dumps({
        "success": report.get("success"),
        "baseline_da": (report.get("baseline") or {}).get("directional_accuracy"),
        "champion_da": (report.get("champion") or {}).get("directional_accuracy")
        if report.get("champion") else None,
        "delta_da": report.get("delta_da_vs_baseline"),
        "dsr": report.get("deflated_sharpe"),
        "n_trials": report.get("n_trials"),
        "recommend_live": report.get("recommend_live"),
        "would_clear_research_bar": report.get("would_clear_research_bar"),
        "note": report.get("note"),
        "error": report.get("error"),
    }, indent=2))
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
