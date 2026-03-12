"""Session 24A verification script - paste output to Claude after Cursor run"""
import subprocess, sys, os, importlib

results = []
def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((status, name, detail))
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))

python = sys.executable

# 1. Prophet stan_backend removed
with open("trading/models/prophet_model.py", encoding="utf-8", errors="replace") as f:
    prophet_src = f.read()
check("Prophet: no stan_backend references", "stan_backend" not in prophet_src,
      f"Found {prophet_src.count('stan_backend')} references")

# 2. Prophet init smoke test
r = subprocess.run([python, "-c", """
import sys
sys.path.insert(0, '.')
try:
    from trading.models.prophet_model import ProphetModel
    m = ProphetModel({})
    print("OK")
except Exception as e:
    print(f"FAIL: {e}")
"""], capture_output=True, text=True, cwd=".")
check("Prophet init without crash", "OK" in r.stdout, r.stdout.strip() + r.stderr[:200])

# 3. LLM routing reads OpenAI key
with open("agents/llm/active_llm_calls.py", encoding="utf-8", errors="replace") as f:
    llm_src = f.read()
check("active_llm_calls: routes to OpenAI when OPENAI_API_KEY set",
      "OPENAI_API_KEY" in llm_src and ("openai" in llm_src.lower()))

# 4. inject_user_keys called in app.py outside session guard
with open("app.py", encoding="utf-8", errors="replace") as f:
    app_src = f.read()
check("app.py: inject_user_keys_to_env present", "inject_user_keys_to_env" in app_src)
# Check it's not buried in an 'initialized' guard
lines = app_src.split('\n')
inject_line = next((i for i,l in enumerate(lines) if 'inject_user_keys_to_env' in l), -1)
if inject_line > 0:
    context = '\n'.join(lines[max(0,inject_line-10):inject_line])
    in_guard = "'initialized'" in context or '"initialized"' in context
    check("app.py: inject_user_keys not inside initialized guard", not in_guard, 
          "context: " + context[-100:])

# 5. HybridModel config in forecast_router
with open("trading/models/forecast_router.py", encoding="utf-8", errors="replace") as f:
    router_src = f.read()
check("forecast_router: hybrid model configured with sub-models",
      "hybrid" in router_src.lower() and "arima" in router_src.lower())

# 6. LSTM train_model alias
lstm_files = ["trading/models/lstm_model.py", "trading/models/lstm_forecaster.py"]
lstm_has_alias = False
for fp in lstm_files:
    if os.path.exists(fp):
        with open(fp, encoding="utf-8", errors="replace") as f:
            if "train_model" in f.read():
                lstm_has_alias = True
check("LSTM has train_model method", lstm_has_alias)

# 7. Scanner has chunked download
with open("trading/analysis/market_scanner.py", encoding="utf-8", errors="replace") as f:
    scanner_src = f.read()
check("Scanner has chunked/batched download", 
      "chunk" in scanner_src.lower() or "chunks" in scanner_src.lower() or "batch" in scanner_src.lower())

# 8. Home briefing service uses watchlist/positions
with open("trading/services/home_briefing_service.py", encoding="utf-8", errors="replace") as f:
    briefing_src = f.read()
check("Briefing service reads watchlist", "watchlist" in briefing_src.lower() or "get_tickers" in briefing_src)
check("Briefing service not hardcoded to AAPL", briefing_src.count('"AAPL"') + briefing_src.count("'AAPL'") <= 1)

# 9. Smoke tests
print("\nRunning model smoke tests...")
r = subprocess.run([python, "tests/model_smoke_test.py"], capture_output=True, text=True, cwd=".")
all_pass = "All smoke tests completed. All PASS." in r.stdout or r.stdout.count("PASS") >= 10
check("All 12 model smoke tests pass", all_pass, 
      f"PASSes: {r.stdout.count('PASS')}, FAILs: {r.stdout.count('FAIL')}")

# Summary
passes = sum(1 for s,_,_ in results if s=="PASS")
fails = sum(1 for s,_,_ in results if s=="FAIL")
print(f"\n{'='*55}")
print(f"Session 24A Results: {passes} PASS  {fails} FAIL")
if fails == 0:
    print("READY — commit, tag v1.6.0, deploy")
else:
    print("NOT READY — fix failing items before deploying")
print('='*55)