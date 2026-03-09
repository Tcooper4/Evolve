import os
import sys
import traceback

import numpy as np
import pandas as pd
import yfinance as yf


def main() -> None:
    """Diagnostic script to verify strategy backtest math end-to-end."""
    sys.path.insert(0, os.getcwd())

    # Load AAPL history (1y) and strip timezone
    hist = yf.Ticker("AAPL").history(period="1y")
    if hist is None or hist.empty:
        print("[FAIL] yfinance returned no data for AAPL")
        sys.exit(1)

    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    print(f"Data: {len(hist)} rows, columns: {list(hist.columns)}")

    # Import backtest engine facade (EnhancedBacktester / BacktestEngine)
    try:
        from trading.backtesting import EnhancedBacktester, BacktestEngine  # type: ignore

        _ = EnhancedBacktester(hist)
        _ = BacktestEngine(hist)
        print("[OK] Backtesting engines imported")
    except Exception as e:
        print(f"[FAIL] Backtesting engine import: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Import Bollinger strategy
    try:
        from trading.strategies.bollinger_strategy import BollingerStrategy, BollingerConfig

        strategy = BollingerStrategy(BollingerConfig())
        print("[OK] BollingerStrategy imported")
    except Exception as e:
        print(f"[FAIL] BollingerStrategy import: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Run a minimal, UI-equivalent backtest (mirrors Quick Backtest math)
    try:
        data = hist.copy()
        data.columns = [c.lower() for c in data.columns]

        if "close" not in data.columns:
            raise RuntimeError("Expected 'close' column after normalization")
        if "volume" not in data.columns:
            data["volume"] = data.get("volume", 1_000_000)

        # Generate signals
        signals_df = strategy.generate_signals(data)
        if "signal" not in signals_df.columns:
            # Fallback: first column
            signals_df["signal"] = signals_df.iloc[:, 0]

        # Calculate returns and equity curve (same as Quick Backtest)
        data = data.copy()
        data["returns"] = data["close"].pct_change()
        data["strategy_returns"] = signals_df["signal"].shift(1) * data["returns"]

        initial_value = 100_000.0
        equity_curve = initial_value * (1 + data["strategy_returns"]).cumprod()
        equity_curve = equity_curve.fillna(initial_value)

        total_return = float(equity_curve.iloc[-1] / initial_value - 1.0)
        returns_series = data["strategy_returns"].dropna()
        if len(returns_series) > 1 and returns_series.std() > 0:
            sharpe = float(
                (returns_series.mean() / returns_series.std()) * np.sqrt(252.0)
            )
        else:
            sharpe = 0.0

        cumulative = (1 + data["strategy_returns"]).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max - 1.0)
        max_dd = float(drawdown.min()) if not drawdown.empty else 0.0

        trades = signals_df[signals_df["signal"] != 0]

        print("[OK] Backtest math complete")
        print(f"  total_return: {total_return:.4f}")
        print(f"  sharpe_ratio: {sharpe:.4f}")
        print(f"  max_drawdown: {max_dd:.4f}")
        print(f"  num_trades: {len(trades)}")
        print(f"  equity_curve length: {len(equity_curve)}")
        if len(equity_curve) > 2:
            start_val = float(equity_curve.iloc[0])
            end_val = float(equity_curve.iloc[-1])
            variance = "flat" if start_val == end_val else "OK"
            print(
                f"  equity_curve: {start_val:.2f} -> {end_val:.2f} (variance={variance})"
            )

    except Exception as e:
        print(f"[FAIL] Backtest run: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


