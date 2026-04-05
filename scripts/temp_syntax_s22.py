import ast

files = [
    "components/analyze_ai_score.py",
    "components/analyze_forecast.py",
    "components/tabs/tab_ai_model_selection.py",
    "components/tabs/tab_backtester.py",
    "agents/briefing/morning_briefing.py",
]
for f in files:
    try:
        ast.parse(open(f, encoding="utf-8", errors="replace").read())
        print(f"OK  {f}")
    except SyntaxError as e:
        print(f"FAIL  {f}: {e}")
