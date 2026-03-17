from __future__ import annotations

import ast
from pathlib import Path


FILES = [
    "trading/models/lstm_model.py",
    "pages/4_Trade.py",
    "pages/5_Backtest.py",
    "pages/7_Settings.py",
]


def main() -> None:
    for f in FILES:
        path = Path(f)
        if not path.exists():
            print(f"NOT FOUND {f}")
            continue
        src = path.read_text(encoding="utf-8", errors="replace")
        try:
            ast.parse(src)
            print(f"OK {f}")
        except SyntaxError as e:
            print(f"FAIL {f} line {e.lineno}: {e}")


if __name__ == "__main__":
    main()

