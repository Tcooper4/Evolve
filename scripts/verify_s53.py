from pathlib import Path

def read(p):
    try:
        return Path(p).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""

results = []

agent = read("agents/llm/agent.py")
results.append(("agent preference ingestion logged", "preference ingestion failed" in agent))
results.append(("agent _get_rich_context logged", "get_short_interest failed" in agent or "ForecastRouter failed" in agent))

llm = read("agents/llm/active_llm_calls.py")
results.append(("active_llm key injection logged", "key injection failed" in llm))

fr = read("trading/models/forecast_router.py")
results.append(("router price space guard logged", "price space guard failed" in fr))
results.append(("router price space _col_map", "_col_map" in fr))

lstm = read("trading/models/lstm_model.py")
results.append(("lstm _col_map in fallback", "_col_map" in lstm))
results.append(("lstm no print warnings", "print(" not in lstm))

cg = read("trading/models/confidence_generator.py")
results.append(("confidence_generator _col_map", "_col_map" in cg))

scan = read("pages/3_Scanner.py")
results.append(("scanner universe load logged", "Universe load failed" in scan or "universe load failed" in scan))

dash = read("pages/1_Dashboard.py")
results.append(("dashboard universe load logged", "Universe load failed" in dash or "universe load failed" in dash))

chat = read("pages/6_Chat.py")
results.append(("chat preference logged", "preference ingestion failed" in chat))
results.append(("chat import logged", "call_active_llm_chat not available" in chat))

garch = read("trading/models/garch_model.py")
results.append(("garch _col_map for last_close", "_col_map" in garch))

xgb = read("trading/models/xgboost_model.py")
results.append(("xgboost direct Close access fixed", 'data["Close"]' not in xgb or "_col_map" in xgb))

bm = read("trading/models/base_model.py")
results.append(("base_model datetime coercion logged", "datetime coercion failed" in bm))

print()
for name, passed in results:
    print(f"{'PASS' if passed else 'FAIL'}  {name}")
print()
