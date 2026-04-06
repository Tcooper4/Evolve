import ast, subprocess, sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 32 Verification ===\n")
PASS, FAIL = [], []
def ok(msg):   PASS.append(msg); print(f"OK    {msg}")
def fail(msg): FAIL.append(msg); print(f"FAIL  {msg}")

# Fix 1 - Reddit
t = open(
    "trading/data/social_sentiment.py",
    encoding="utf-8", errors="replace"
).read()
if "not rid and not rsec" in t:
    ok("Reddit early return present")
else:
    fail("Reddit early return missing")

# Fix 2 - trading.ui warning
t2 = open(
    "trading/ui/__init__.py",
    encoding="utf-8", errors="replace"
).read()
if "config.registry" not in t2:
    ok("trading.ui.config.registry "
       "import removed")
else:
    fail("trading.ui.config.registry "
         "still imported")

# Fix 3 - timezone
t3 = open(
    "agents/briefing/morning_briefing.py",
    encoding="utf-8", errors="replace"
).read()
t4 = open(
    "trading/data/price_cache.py",
    encoding="utf-8", errors="replace"
).read()
if "tz_localize(None)" in t3:
    ok("TZ fix in morning_briefing.py")
else:
    fail("TZ fix missing in "
         "morning_briefing.py")
if "tz_localize(None)" in t4:
    ok("TZ fix in price_cache.py")
else:
    fail("TZ fix missing in price_cache.py")

# Fix 4 - Prophet
t5 = open(
    "trading/models/prophet_model.py",
    encoding="utf-8", errors="replace"
).read()
if "_unavailable" in t5:
    ok("Prophet graceful failure present")
else:
    fail("Prophet graceful failure missing")

# Fix 5 - Layout
t6 = open(
    "pages/1_Dashboard.py",
    encoding="utf-8", errors="replace"
).read()
if "_render_watchlist" in t6:
    ok("Watchlist fragment present")
else:
    fail("Watchlist fragment missing")
if "it.get(\"url\"" in t6 or \
   "it.get('url'" in t6:
    ok("News URL links present")
else:
    fail("News URL links missing")

# Syntax checks
for fpath in [
    "trading/data/social_sentiment.py",
    "trading/ui/__init__.py",
    "agents/briefing/morning_briefing.py",
    "trading/data/price_cache.py",
    "trading/models/prophet_model.py",
    "pages/1_Dashboard.py",
]:
    try:
        ast.parse(open(fpath,
            encoding="utf-8",
            errors="replace").read())
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
    sys.exit(1)
else:
    print("All checks passed. "
          "Ready to commit v4.1.9.")
    sys.exit(0)
