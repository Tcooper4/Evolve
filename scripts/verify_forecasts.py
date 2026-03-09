"""
Quick forecast verification — run this to confirm model outputs are in price space.
Usage: .\evolve_venv\Scripts\python.exe scripts/verify_forecasts.py
"""

import os
import sys

import numpy as np
import yfinance as yf

sys.path.insert(0, os.getcwd())


def main() -> int:
    hist = yf.Ticker("AAPL").history(period="6mo")
    if hist is None or hist.empty or "Close" not in hist.columns:
        print("[ERROR] Unable to fetch AAPL history from yfinance.")
        return 2

    last_price = float(hist["Close"].iloc[-1])
    lo = last_price * 0.85
    hi = last_price * 1.15
    print(f"\nAAPL last price: ${last_price:.2f}")
    print(f"Expected forecast range: ${lo:.2f} - ${hi:.2f}\n")

    from trading.models.forecast_router import ForecastRouter  # noqa: E402

    router = ForecastRouter()
    models_to_test = ["arima", "xgboost", "ridge", "catboost", "prophet"]

    for model_name in models_to_test:
        try:
            result = router.get_forecast(hist, model_name=model_name, horizon=7, run_walk_forward=False)
            forecast = np.array(result.get("forecast", []), dtype="float64")
            if len(forecast) > 0:
                status = (
                    "OK"
                    if (last_price * 0.85) < float(forecast[0]) < (last_price * 1.15)
                    else "WRONG"
                )
                print(
                    f"[{status}] {model_name}: "
                    f"selected={result.get('model')}, "
                    f"first={forecast[0]:.2f}, last={forecast[-1]:.2f}, "
                    f"already_denorm={result.get('already_denormalized')}"
                )
            else:
                print(f"[EMPTY] {model_name}: no forecast returned")
        except Exception as e:
            print(f"[ERROR] {model_name}: {e}")

    print("\nVerification complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

