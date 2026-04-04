# Evolve — `_archive/` manifest

**Version:** v3.39.0  
**Date:** 2026-04-04  

## Purpose

Unreachable or legacy modules were moved here **without deletion**, preserving history under `_archive/<original relative path>/`.

Additional archive passes completed in v3.30–v3.35: 60 additional files moved in second pass, 5 in third pass, 101 true orphans in final pass. Total archived: ~267 files.

## This round (safe archive)

- **Source:** `scripts/dead_code_audit.json` entries with verdict `ARCHIVE` (248 listed).
- **Moved:** 96 `.py` files (non-`__init__.py` only, and only when **no** live import reference was found in the repo scan).
- **Skipped — `__init__.py`:** 31 (package barrels left in place; audit often mislabels these).
- **Skipped — still referenced:** 121 (absolute `from … import` / `import …` or same-directory relative imports).

The original user target of **377** paths was not fully applied: the audit JSON only contained **248** `ARCHIVE` rows, and many of those remain in the tree because they are still imported. Further pruning needs a manual or deeper dependency pass.

## Verification

- `tests/model_smoke_test.py`: all models **PASS** after moves.
- `pages/*.py`: **AST parse OK** (8 files).

## Tooling

- `scripts/run_archive_dead_code.py` — repeatable safe move; uses `os.walk` with directory excludes so `evolve_venv` / `.venv` are not scanned.
