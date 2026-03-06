# Pre-deployment checks — Community Cloud / Linux

Results of final pre-deployment checks. Fix any **FAIL** before deploying.

---

## CHECK 1 — Windows-only imports (Linux-safe)

**Status: PASS**

- **Action:** Searched for `pywin32`, `winreg`, `winsound`, `msvcrt`, `ctypes.windll` in `*.py` under the project.
- **Result:** Matches only in `evolve_venv/Scripts/` (pywin32_postinstall.py, pywin32_testall.py). These are venv installer scripts, not application code.
- **Conclusion:** No project source files use Windows-only imports. Deploy uses a clean Linux environment and does not include the local venv, so nothing needs to be wrapped. No code changes required.

---

## CHECK 2 — Torch import and CUDA (CPU-only Linux)

**Status: PASS**

- **Action:** Checked all `import torch` / `from torch` usages and all CUDA usage.
- **Result:** Torch is imported inside `try/except ImportError` in the modules that use it (e.g. trading/utils/gpu_utils.py, trading/models/base_model.py, trading/models/lstm_model.py, etc.). Every use of CUDA (e.g. `.cuda()`, `device="cuda"`, `torch.device("cuda")`) is guarded by `torch.cuda.is_available()` (or equivalent) before using GPU.
- **Conclusion:** Safe for CPU-only Linux; no code assumes CUDA without checking.

---

## CHECK 3 — requirements_deploy.txt (no Windows/CUDA-specific deps)

**Status: PASS**

- **Action:** Inspected `requirements_deploy.txt` for Windows-only or CUDA-specific packages.
- **Result:**
  - No `pywin32`.
  - No `+cpu` or `+cu118` (or other) torch variants.
  - Contains `cryptography>=41.0.0` and `streamlit>=1.28.0`.
- **Conclusion:** requirements_deploy.txt is suitable for Community Cloud (plain torch, no Windows-only deps).

---

## CHECK 4 — app.py optional imports (start with missing optional packages)

**Status: PASS**

- **Action:** Listed top-level `import` / `from` in `app.py` and verified optional packages are not required at startup.
- **Result:** app.py only imports: atexit, logging, os, sys, warnings, pathlib, streamlit, config.user_store, components.onboarding; optional or internal imports (tensorflow, dotenv, config.logging_config, trading.*) are already inside try/except. None of fredapi, praw, textblob, pyttsx3, or sentence_transformers are imported in app.py.
- **Conclusion:** app.py can start cleanly when optional packages (fredapi, praw, textblob, pyttsx3, sentence_transformers) are missing.

---

## CHECK 5 — .streamlit/config.toml (server and theme)

**Status: PASS**

- **Action:** Ensure `.streamlit/config.toml` exists with required server and theme settings.
- **Result:** File exists. Updated to include:
  - `[server]`: `maxUploadSize = 50`, `headless = true`, `port = 8501` (plus existing address/CORS/XSRF).
  - `[theme]`: `base = "dark"` (plus existing color overrides).
- **Conclusion:** Config present and updated for deploy (headless, max upload size, dark theme).

---

## CHECK 6 — data/ and .cache/ created at startup

**Status: PASS**

- **Action:** Ensure `data/` and `.cache/` are created at runtime so the app works on an empty filesystem (e.g. Community Cloud).
- **Result:** In `config/user_store.py` (imported by app.py before any page logic), module-level code runs:
  - `os.makedirs("data", exist_ok=True)`
  - `os.makedirs(".cache", exist_ok=True)`
  So both directories are created as soon as the app loads config/user_store.
- **Conclusion:** data/ and .cache/ are created at startup; no manual pre-creation needed on deploy.

---

## Summary

| Check | Description                    | Result |
|-------|--------------------------------|--------|
| 1     | No Windows-only imports in app code | PASS   |
| 2     | Torch/CUDA safe for CPU-only  | PASS   |
| 3     | requirements_deploy.txt      | PASS   |
| 4     | app.py optional imports      | PASS   |
| 5     | .streamlit/config.toml       | PASS   |
| 6     | data/ and .cache/ at startup  | PASS   |

All checks **PASS**. Ready for deployment to Community Cloud / Linux.
