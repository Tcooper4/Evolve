"""
Evolve Full Codebase Map Generator — v2
Captures:
1. All live product files (excluding tests/scripts/archive)
2. Static imports (ast)
3. Barrel/re-export relationships (__init__.py)
4. Dynamic import hints (importlib, try/except imports)
5. Feature utilization — which features are called by which agents/pages
6. Reachability from UI AND from agents separately
7. Suggested connections not yet made
"""

import os, ast, json, re
from pathlib import Path
from collections import defaultdict

ROOT = Path(".")
EXCLUDE = {"evolve_venv", ".venv", "__pycache__", ".git", ".cache", "_archive", "node_modules"}
EXCLUDE_PREFIXES = ["tests/", "scripts/", ".cache/", "_archive/", "docs/future_features/"]
OLD_FILES = ["/old_", "verify_s", "export_codebase"]

# Layer classification
def get_layer(f):
    if f in ("app.py",) or f.startswith("pages/") or f.startswith("components/"): return "ui"
    if f.startswith("trading/models/"): return "model"
    if any(f.startswith(p) for p in [
        "trading/analysis/","trading/services/","trading/validation/",
        "trading/backtesting/","trading/strategies/","trading/risk/",
        "trading/analytics/","trading/options/","trading/commentary/",
        "trading/nlp/","trading/feature_engineering/","trading/recovery/",
        "trading/portfolio/","trading/optimization/","trading/integration/",
        "trading/execution/","trading/core/","trading/forecasting/",
        "trading/ensemble/","trading/evaluation/","trading/market/",
        "trading/signals/","trading/signal_score/","trading/meta_learning/",
        "trading/context_manager/","trading/knowledge_base/","trading/async_utils/",
        "trading/pipeline/","trading/scheduler","trading/system_resilience",
        "trading/automation_core","trading/live_market","trading/demo_live",
        "trading/test_live","trading/launch_live",
    ]): return "service"
    if f.startswith("trading/data/") or f.startswith("data/"): return "data"
    if f.startswith("trading/memory/") or f.startswith("trading/agents/") or f.startswith("agents/"): return "agent"
    if f.startswith("trading/utils/") or f.startswith("utils/"): return "util"
    if f.startswith("config/"): return "config"
    if f.startswith("trading/"): return "service"
    if f.startswith("nlp/"): return "service"
    if f.startswith("ui/"): return "ui"
    return "other"

def is_included(f):
    for p in EXCLUDE_PREFIXES:
        if f.startswith(p): return False
    for p in OLD_FILES:
        if p in f: return False
    if f.endswith("__init__.py"):
        try:
            lines = Path(f).read_text(encoding="utf-8", errors="replace").strip().splitlines()
            if len(lines) <= 5: return False
        except: return False
    return True

# Collect all files
all_files = []
for root, dirs, files in os.walk(ROOT):
    dirs[:] = [d for d in dirs if d not in EXCLUDE]
    for file in files:
        if not file.endswith(".py"): continue
        path = Path(root) / file
        rel = str(path).replace("\\", "/").lstrip("./")
        if rel.startswith("./"): rel = rel[2:]
        if is_included(rel):
            all_files.append(rel)

file_set = set(all_files)
file_to_id = {f: f.replace("/","_").replace(".py","").replace("-","_") for f in all_files}

print(f"Total files to map: {len(all_files)}")

# Parse each file
nodes = []
all_imports = {}  # file -> list of resolved target files

for fpath in all_files:
    try:
        content = Path(fpath).read_text(encoding="utf-8", errors="replace")
        lines_count = len(content.splitlines())
    except:
        lines_count = 0
        content = ""

    layer = get_layer(fpath)
    
    # AST parse for imports
    imports_found = []
    dynamic_imports = []
    is_barrel = fpath.endswith("__init__.py")
    
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    mod = node.module
                    # Convert to file path candidates
                    candidates = [
                        mod.replace(".", "/") + ".py",
                        mod.replace(".", "/") + "/__init__.py",
                    ]
                    for c in candidates:
                        if c in file_set:
                            imports_found.append(c)
                            break
                    # Check relative
                    if node.level and node.level > 0:
                        parent = "/".join(fpath.split("/")[:-node.level])
                        if node.module:
                            rel_path = parent + "/" + node.module.replace(".", "/") + ".py"
                            if rel_path in file_set:
                                imports_found.append(rel_path)
    except:
        pass
    
    # Check for dynamic imports (importlib, try/except import hints)
    dynamic_patterns = [
        r'importlib\.import_module\(["\']([^"\']+)["\']',
        r'__import__\(["\']([^"\']+)["\']',
    ]
    for pat in dynamic_patterns:
        for match in re.finditer(pat, content):
            mod = match.group(1)
            c = mod.replace(".", "/") + ".py"
            if c in file_set:
                dynamic_imports.append(c)
    
    # Check if this is a try/except import (graceful degradation)
    has_try_imports = bool(re.search(r'try:\s*\n\s*(?:from|import)', content))
    
    imports_found = list(set(imports_found))
    all_imports[fpath] = imports_found
    
    nodes.append({
        "id": file_to_id[fpath],
        "label": Path(fpath).stem,
        "file": fpath,
        "layer": layer,
        "lines": lines_count,
        "is_barrel": is_barrel,
        "has_try_imports": has_try_imports,
        "dynamic_imports": dynamic_imports,
        "imports_from": [file_to_id[i] for i in imports_found if i in file_to_id],
        "imports_from_files": imports_found,
    })

# Build reverse map
id_to_file = {v: k for k, v in file_to_id.items()}
nmap = {n["id"]: n for n in nodes}

# Build incoming counts
incoming = defaultdict(int)
for n in nodes:
    for imp in n["imports_from"]:
        incoming[imp] += 1
for n in nodes:
    n["incoming"] = incoming[n["id"]]

# Reachability from UI (strict - direct import chain)
ui_ids = {n["id"] for n in nodes if n["layer"] == "ui"}
strict_reachable = set()
queue = list(ui_ids)
while queue:
    nid = queue.pop()
    if nid in strict_reachable: continue
    strict_reachable.add(nid)
    nn = nmap.get(nid)
    if nn:
        for dep in nn["imports_from"]:
            if dep not in strict_reachable: queue.append(dep)

# Reachability including barrel re-exports
# Barrels make their imported modules "available" even if not directly called
barrel_ids = {n["id"] for n in nodes if n["is_barrel"]}
extended_reachable = set(strict_reachable)
# Add anything imported by a barrel that is itself reachable
changed = True
while changed:
    changed = False
    for nid in list(extended_reachable):
        nn = nmap.get(nid)
        if nn and nn["is_barrel"]:
            for dep in nn["imports_from"]:
                if dep not in extended_reachable:
                    extended_reachable.add(dep)
                    changed = True

# Reachability from agents specifically
agent_ids = {n["id"] for n in nodes if n["layer"] == "agent"}
agent_reachable = set()
queue = list(agent_ids)
while queue:
    nid = queue.pop()
    if nid in agent_reachable: continue
    agent_reachable.add(nid)
    nn = nmap.get(nid)
    if nn:
        for dep in nn["imports_from"]:
            if dep not in agent_reachable: queue.append(dep)

# Mark reachability on each node
for n in nodes:
    n["reachable_ui_strict"] = n["id"] in strict_reachable
    n["reachable_ui_barrel"] = n["id"] in extended_reachable
    n["reachable_agent"] = n["id"] in agent_reachable
    # Overall: reachable if any path reaches it
    n["reachable"] = n["reachable_ui_strict"] or n["reachable_ui_barrel"]
    # True orphan: not reachable by any path
    n["true_orphan"] = (
        not n["reachable_ui_strict"] and
        not n["reachable_ui_barrel"] and
        not n["reachable_agent"] and
        n["incoming"] == 0
    )

# Known feature categories for the feature map
FEATURE_TAGS = {
    "trading/models/forecast_router.py": "forecasting",
    "trading/models/lstm_model.py": "forecasting",
    "trading/models/xgboost_model.py": "forecasting",
    "trading/models/arima_model.py": "forecasting",
    "trading/models/ridge_model.py": "forecasting",
    "trading/models/prophet_model.py": "forecasting",
    "trading/models/tcn_model.py": "forecasting",
    "trading/models/catboost_model.py": "forecasting",
    "trading/models/ensemble_model.py": "forecasting",
    "trading/models/garch_model.py": "forecasting",
    "trading/analysis/ai_score.py": "scoring",
    "trading/analysis/market_scanner.py": "scanning",
    "trading/analysis/macro_factors.py": "scoring",
    "trading/analysis/chart_pattern_detector.py": "analysis",
    "trading/analysis/econometric_diagnostics.py": "analysis",
    "trading/analysis/ml_score_trainer.py": "scoring",
    "trading/data/social_sentiment.py": "sentiment",
    "trading/data/options_flow.py": "options",
    "trading/data/price_cache.py": "data",
    "trading/data/news_aggregator.py": "news",
    "agents/briefing/morning_briefing.py": "agent",
    "agents/llm/tool_executor.py": "agent",
    "agents/llm/agent.py": "agent",
    "trading/services/alert_checker.py": "alerts",
    "trading/services/recommendation_tracker.py": "tracking",
    "utils/risk_metrics.py": "risk",
    "trading/validation/walk_forward_utils.py": "validation",
    "trading/commentary/commentary_engine.py": "commentary",
    "trading/analytics/alpha_attribution_engine.py": "attribution",
    "trading/backtesting/enhanced_backtester.py": "backtesting",
    "trading/options/options_forecaster.py": "options",
    "trading/strategies/pairs_trading_engine.py": "strategy",
    "trading/strategies/strategy_comparison.py": "strategy",
    "trading/strategies/adaptive_selector.py": "strategy",
    "trading/feature_engineering/utils.py": "ml",
    "data/streaming_pipeline.py": "data",
    "trading/system_resilience.py": "infrastructure",
    "trading/recovery/disaster_recovery_manager.py": "infrastructure",
    "trading/optimization/optuna_optimizer.py": "optimization",
}

for n in nodes:
    n["feature_tag"] = FEATURE_TAGS.get(n["file"], "")

# Build edge list
edges = []
seen_edges = set()
for n in nodes:
    for imp_id in n["imports_from"]:
        key = (n["id"], imp_id)
        if key not in seen_edges and imp_id in nmap:
            seen_edges.add(key)
            target = nmap[imp_id]
            edge_type = "barrel" if n["is_barrel"] else "imports"
            edges.append({
                "from": n["id"],
                "to": imp_id,
                "type": edge_type
            })

# Summary stats
true_orphans = [n for n in nodes if n["true_orphan"]]
strict_live = [n for n in nodes if n["reachable_ui_strict"]]
barrel_live = [n for n in nodes if n["reachable_ui_barrel"] and not n["reachable_ui_strict"]]
agent_only = [n for n in nodes if n["reachable_agent"] and not n["reachable"]]

print(f"Strict UI reachable: {len(strict_live)}")
print(f"Barrel-extended reachable: {len(barrel_live)}")
print(f"Agent-reachable only: {len(agent_only)}")
print(f"True orphans (no connections at all): {len(true_orphans)}")
print(f"Total edges: {len(edges)}")

output = {
    "generated": "2026-04-03",
    "total": len(nodes),
    "strict_live": len(strict_live),
    "barrel_live": len(barrel_live),
    "agent_only": len(agent_only),
    "true_orphans": len(true_orphans),
    "edges_count": len(edges),
    "nodes": nodes,
    "edges": edges,
}

out_path = Path("scripts/full_codebase_map_v2.json")
out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
print(f"Saved to {out_path} ({out_path.stat().st_size//1024}KB)")
