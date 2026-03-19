from pathlib import Path

def read(p):
    try:
        return Path(p).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""

src = read("trading/models/lstm_model.py")
results = []
results.append(("_col_map_check in _prepare_data", "_col_map_check" in src))
results.append(("missing_cols uses .lower()", "col.lower() not in _col_map_check" in src))
results.append(("target index uses _tgt_lower", "_feat_lower" in src and "_tgt_idx" in src))
results.append(("no raw .index(target_column) call",
    'self.config["feature_columns"].index(\n                        self.config["target_column"]' not in src))

print()
for name, passed in results:
    print(f"{'PASS' if passed else 'FAIL'}  {name}")
print()
