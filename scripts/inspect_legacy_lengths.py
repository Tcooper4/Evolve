from __future__ import annotations

from pathlib import Path


def main() -> None:
    files = [
        "scripts/old_4_Trade_Execution.py",
        "scripts/old_5_Portfolio.py",
        "scripts/old_7_Performance.py",
        "scripts/old_6_Risk_Management.py",
        "scripts/old_3_Strategy_Testing.py",
        "scripts/old_10_Alerts.py",
        "scripts/old_11_Admin.py",
    ]

    for f in files:
        p = Path(f)
        if not p.exists():
            print(f"NOT FOUND: {f}")
            continue
        try:
            n = len(p.read_text(encoding="utf-8", errors="replace").splitlines())
            print(f"{n} lines: {f}")
        except Exception as e:
            print(f"ERROR reading {f}: {e}")


if __name__ == "__main__":
    main()

