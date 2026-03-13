"""
audit_shap.py — Check SHAP implementation status
Run: .\evolve_venv\Scripts\python.exe scripts\audit_shap.py > scripts\audit_shap.txt 2>&1
"""
import os

keywords = ['shap', 'SHAP', 'explainability', 'explainer', 'shap_values', 'force_plot', 'summary_plot']

for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in ['evolve_venv', '.venv', '.git', '__pycache__', '.cache']]
    for f in files:
        if not f.endswith('.py'):
            continue
        path = os.path.join(root, f)
        try:
            lines = open(path, encoding='utf-8', errors='replace').read().splitlines()
            hits = [(i+1, l) for i, l in enumerate(lines)
                    if any(k in l for k in keywords)]
            if hits:
                print(f"\n{'='*60}")
                print(path)
                print('='*60)
                for lineno, line in hits:
                    print(f"  {lineno:5}: {line.rstrip()}")
        except Exception as e:
            pass