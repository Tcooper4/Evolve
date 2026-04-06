import ast, subprocess, sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 30 Verification ===\n")
PASS, FAIL = [], []
def ok(msg):   PASS.append(msg); print(f"OK    {msg}")
def fail(msg): FAIL.append(msg); print(f"FAIL  {msg}")

# Fix 1 - scanner slider
t = open("pages/3_Scanner.py",
    encoding="utf-8", errors="replace").read()
if "scanner_min_ai_score" in t:
    ok("scanner_min_ai_score key present")
else:
    fail("scanner_min_ai_score key missing")

# Fix 2 - _news_color dropped
if "_news_color" in t and "_drop" in t:
    ok("_news_color drop block present")
else:
    fail("_news_color drop block missing")

# Fix 3 - Reddit keys
t2 = open(
    "trading/analysis/ai_score.py",
    encoding="utf-8", errors="replace"
).read()
if "REDDIT_CLIENT_ID" in t2:
    ok("REDDIT_CLIENT_ID in external keys")
else:
    fail("REDDIT_CLIENT_ID missing")
if "REDDIT_CLIENT_SECRET" in t2:
    ok("REDDIT_CLIENT_SECRET in external keys")
else:
    fail("REDDIT_CLIENT_SECRET missing")

# Fix 4 - Chat cache
t3 = open("pages/6_Chat.py",
    encoding="utf-8", errors="replace").read()
if "home_briefing_report" in t3:
    ok("Chat reads Dashboard cache key")
else:
    fail("Chat not reading Dashboard cache")

# Fix 5 - sidebar init
import os
sidebar_files = []
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs
               if d not in [
                   "evolve_venv", ".git",
                   "__pycache__", "_archive"
               ]]
    for f in files:
        if f.endswith(".py"):
            path = os.path.join(root, f)
            content = open(
                path, encoding="utf-8",
                errors="replace"
            ).read()
            if ("sidebar" in content.lower()
                    and "ticker" in content.lower()
                    and "session_state" in content
                    and "text_input" in content):
                sidebar_files.append(path)
if sidebar_files:
    ok(f"Sidebar ticker init found in: "
       f"{sidebar_files[0]}")
else:
    fail("Sidebar ticker session init not found")

# Fix 6 - LSTM column norm
t4 = open(
    "trading/models/lstm_model.py",
    encoding="utf-8", errors="replace"
).read()
if "feature_names_in_" in t4:
    ok("LSTM column normalization present")
else:
    fail("LSTM column normalization missing")

# Syntax checks
for fpath in [
    "pages/3_Scanner.py",
    "trading/analysis/ai_score.py",
    "pages/6_Chat.py",
    "trading/models/lstm_model.py",
]:
    try:
        ast.parse(open(
            fpath, encoding="utf-8",
            errors="replace"
        ).read())
        ok(f"Syntax valid: {fpath}")
    except SyntaxError as e:
        fail(f"Syntax error {fpath}: {e}")

print()
print("--- Smoke test ---")
result = subprocess.run(
    [python, "tests/model_smoke_test.py"],
    capture_output=True, text=True
)
print((result.stdout + result.stderr)[-1500:])
if result.returncode == 0:
    ok("Smoke test passed")
else:
    fail("Smoke test FAILED")

print(f"\n=== {len(PASS)} passed, "
      f"{len(FAIL)} failed ===")
if FAIL:
    for f in FAIL:
        print(f"  FAIL  {f}")
    sys.exit(1)
else:
    print("All checks passed. "
          "Ready to commit v4.1.7.")
    sys.exit(0)
