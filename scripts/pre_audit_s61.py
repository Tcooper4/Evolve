# scripts/pre_audit_s61.py
# Run: .\evolve_venv\Scripts\python.exe scripts\pre_audit_s61.py

import subprocess, sys, os, re
from pathlib import Path

python = r".\evolve_venv\Scripts\python.exe"
OUT = Path("scripts/pre_audit_s61_results.txt")
lines = []

def w(s=""):
    lines.append(str(s))

def section(title):
    w()
    w("=" * 55)
    w(f"  {title}")
    w("=" * 55)

def run(cmd, timeout=15):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return (r.stdout + r.stderr).strip()

def grep(pattern, skip=("evolve_venv", "_archive", "__pycache__", ".git", ".venv", "venv")):
    matches = []
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in skip]
        for f in files:
            if not f.endswith(".py"):
                continue
            p = os.path.join(root, f)
            try:
                txt = open(p, encoding="utf-8", errors="replace").read()
                for i, line in enumerate(txt.splitlines(), 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        matches.append((p, i, line.strip()))
            except Exception:
                pass
    return matches


# 1. GNN MULTI-ASSET
section("1. GNN Multi-Asset / Supply Chain")

gnn_path = "trading/models/advanced/gnn/gnn_model.py"
if os.path.exists(gnn_path):
    txt = open(gnn_path, encoding="utf-8", errors="replace").read()
    w(f"gnn_model.py: EXISTS ({len(txt.splitlines())} lines)")
    for kw in ["multi_asset", "correlation", "supply_chain", "graph",
                "adjacency", "edge", "GNNForecaster", "class GNN"]:
        if kw.lower() in txt.lower():
            w(f"  contains: {kw}")
else:
    w(f"gnn_model.py: NOT FOUND")

gnn_ui = [(p, l, ln) for p, l, ln in grep(r"gnn|GNN")
          if "pages/" in p or "components/" in p]
w(f"\nGNN in pages/components ({len(gnn_ui)} matches):")
for p, l, ln in gnn_ui[:10]:
    w(f"  {p}:{l}  {ln[:80]}")


# 2. GEOPOLITICAL / MACRO EVENT RISK
section("2. Geopolitical / Macro Event Risk")

geo = grep(r"geopolit|catalyst.type|event.risk|macro.event|catalyst_type")
w(f"Catalyst/geopolitical code ({len(geo)} matches):")
for p, l, ln in geo[:10]:
    w(f"  {p}:{l}  {ln[:80]}")

macro_path = "trading/analysis/macro_factors.py"
if os.path.exists(macro_path):
    mt = open(macro_path, encoding="utf-8", errors="replace").read()
    w(f"\nmacro_factors.py: EXISTS ({len(mt.splitlines())} lines)")
    for kw in ["FRED", "yield", "spread", "high_yield", "geopolit", "event", "VIX"]:
        if kw.lower() in mt.lower():
            w(f"  contains: {kw}")
else:
    w(f"\nmacro_factors.py: NOT FOUND")


# 3. OPTIONS TERM STRUCTURE / SKEW
section("3. Options Term Structure / Skew")

skew = grep(r"term.structure|skew|put.call.ratio|vol.surface|implied_vol")
w(f"Skew/term structure ({len(skew)} matches):")
for p, l, ln in skew[:12]:
    w(f"  {p}:{l}  {ln[:80]}")

options_path = "trading/options/options_forecaster.py"
if os.path.exists(options_path):
    ot = open(options_path, encoding="utf-8", errors="replace").read()
    w(f"\noptions_forecaster.py: EXISTS ({len(ot.splitlines())} lines)")
    for kw in ["skew", "term_structure", "smile", "expiry", "put_call", "surface"]:
        if kw.lower() in ot.lower():
            w(f"  contains: {kw}")


# 4. EARNINGS REVISION BREADTH
section("4. Earnings Revision Breadth")

ern = grep(r"revision.breadth|eps.revision|earnings.revision|upward.revision")
w(f"Earnings revision code ({len(ern)} matches):")
for p, l, ln in ern[:8]:
    w(f"  {p}:{l}  {ln[:80]}")

w("\nEarnings-related files in trading/data/:")
for root, dirs, files in os.walk("trading/data"):
    dirs[:] = [d for d in dirs if "__pycache__" not in d]
    for f in files:
        if "earn" in f.lower():
            w(f"  {os.path.join(root, f)}")


# 5. GPU / CUDA
section("5. GPU / PyTorch CUDA")

w(run([python, "-c",
    "import torch; "
    "print('torch:', torch.__version__); "
    "print('CUDA available:', torch.cuda.is_available()); "
    "print('CUDA version:', torch.version.cuda); "
    "print('Device count:', torch.cuda.device_count()); "
    "print('Device:', torch.cuda.get_device_name(0) "
    "if torch.cuda.is_available() else 'N/A')"
]))

device_refs = grep(r"\.to\(device\)|\.cuda\(\)|torch\.device")
w(f"\nDevice/CUDA references ({len(device_refs)} matches):")
for p, l, ln in device_refs[:8]:
    w(f"  {p}:{l}  {ln[:80]}")


# 6. STREAMING SCANNER
section("6. Real-Time / Streaming Scanner")

scanner_path = "pages/13_Scanner.py"
if os.path.exists(scanner_path):
    st_txt = open(scanner_path, encoding="utf-8", errors="replace").read()
    w(f"13_Scanner.py: EXISTS ({len(st_txt.splitlines())} lines)")
    for kw in ["rerun", "sleep", "auto_refresh", "streaming",
                "websocket", "polling", "fragment", "st.rerun", "refresh"]:
        cnt = st_txt.lower().count(kw.lower())
        if cnt:
            w(f"  '{kw}': {cnt}x")
else:
    w("13_Scanner.py: NOT FOUND")

stream = grep(r"streaming|websocket|auto.refresh|live.scan|st\.rerun")
w(f"\nStreaming refs in pages/components ({len(stream)} matches):")
for p, l, ln in [(p, l, ln) for p, l, ln in stream
                  if "pages/" in p or "components/" in p][:10]:
    w(f"  {p}:{l}  {ln[:80]}")


# 7. NEURALFORECAST
section("7. NeuralForecast Bonus Models")

w(run([python, "-c",
    "try:\n"
    "    import neuralforecast; print('installed:', neuralforecast.__version__)\n"
    "except ImportError as e: print('NOT installed:', e)\n"
]))

nf = grep(r"neuralforecast|AutoFormer|Informer|NBEATS|PatchTST|NHiTS|TFT")
w(f"\nNeuralForecast references ({len(nf)} matches):")
for p, l, ln in nf[:12]:
    w(f"  {p}:{l}  {ln[:80]}")


# 8. RL TRAINER
section("8. RL Strategy Trainer")

w(run([python, "-c",
    "for pkg in ['gymnasium','stable_baselines3','gym']:\n"
    "    try:\n"
    "        m=__import__(pkg)\n"
    "        print(pkg,':', getattr(m,'__version__','installed'))\n"
    "    except ImportError:\n"
    "        print(pkg,': NOT installed')\n"
]))

rl = grep(r"gymnasium|stable.baselines|rl_trainer|RLStrategy|PPO.*rl|reinforce")
w(f"\nRL trainer references ({len(rl)} matches):")
for p, l, ln in rl[:10]:
    w(f"  {p}:{l}  {ln[:80]}")


# 9. DB CONNECTIONS
section("9. Database Connections")

db = grep(r"sqlite3\.connect|_get_conn\(\)|\.connect\(.*\.db")
w(f"sqlite3.connect / _get_conn calls ({len(db)} matches):")
for p, l, ln in db:
    w(f"  {p}:{l}  {ln[:80]}")

closes = grep(r"conn\.close\(\)|\.close\(\).*conn|_conn\.close")
w(f"\nExplicit conn.close() calls ({len(closes)} matches):")
for p, l, ln in closes[:15]:
    w(f"  {p}:{l}  {ln[:80]}")

with_db = grep(r"with sqlite3\.connect|with _get_conn|with get_conn")
w(f"\nContext manager DB usage ({len(with_db)} matches):")
for p, l, ln in with_db[:10]:
    w(f"  {p}:{l}  {ln[:80]}")

w("\nDB files found:")
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in ("evolve_venv", ".git", "__pycache__", "_archive")]
    for f in files:
        if f.endswith(".db"):
            p = os.path.join(root, f)
            w(f"  {p}  ({os.path.getsize(p):,} bytes)")


# WRITE FILE
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Audit written to: {OUT}")
print("Open that file and paste its contents back.")