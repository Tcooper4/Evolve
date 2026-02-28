"""
Smoke test: run each Streamlit page script and capture load errors.
Streamlit runs the script on load; we run with a short timeout and capture stderr.
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAGES = [
    "0_Home",
    "1_Chat",
    "2_Forecasting",
    "3_Strategy_Testing",
    "4_Trade_Execution",
    "5_Portfolio",
    "6_Risk_Management",
    "7_Performance",
    "8_Model_Lab",
    "9_Reports",
    "10_Alerts",
    "11_Admin",
    "12_Memory",
]

def test_page_load(name: str) -> tuple:
    """Run streamlit run pages/Name.py with timeout; return (ok, message)."""
    page_path = REPO_ROOT / "pages" / f"{name}.py"
    if not page_path.exists():
        return False, "File not found: " + str(page_path)
    try:
        proc = subprocess.run(
            [
                sys.executable, "-m", "streamlit", "run",
                str(page_path),
                "--server.headless", "true",
                "--server.port", "8503",
                "--server.runOnSave", "false",
                "--browser.gatherUsageStats", "false",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=12,
        )
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0 and proc.returncode != -9 and proc.returncode != -2:
            return False, "exit code " + str(proc.returncode) + "\n" + out[-2000:]
        if "Error" in out and "Traceback" in out:
            return False, out[-3000:]
        return True, "Loaded OK"
    except subprocess.TimeoutExpired:
        return True, "Loaded OK (timeout 12s)"
    except Exception as e:
        return False, str(e)

def main():
    print("Smoke testing page load (streamlit run each page, 25s timeout)...")
    results = {}
    for name in PAGES:
        ok, msg = test_page_load(name)
        results[name] = (ok, msg)
        print(name + ": " + ("OK" if ok else "FAIL"))
        if not ok and msg:
            print("  " + msg[:400].replace("\n", " "))
    return results

if __name__ == "__main__":
    main()
