# Logger → logging final fix (pages/11_Admin.py)

## Every occurrence of "logger" (line number and full line, before changes)

| Line | Full line |
|------|-----------|
| 42 | `logger = logging.getLogger(__name__)` |
| 51 | `logger.warning("EnhancedSettings not found, using placeholder")` |
| 57 | `logger.warning("AgentRegistry not found, using placeholder")` |
| 66 | `logger.warning("SystemHealthMonitor not found, using placeholder")` |
| 72 | `logger.warning("SystemStatus not found, using placeholder")` |
| 132 | `logger.error(f"Error initializing EnhancedSettings: {e}")` |
| 141 | `logger.error(f"Error initializing AgentRegistry: {e}")` |
| 150 | `logger.error(f"Error initializing SystemHealthMonitor: {e}")` |
| 970 | `logger.warning(f"Connection check failed: {e}")` |
| 1023 | `logger.debug(f"Suppressed error during disconnect: {e}")` |
| 1034 | `logger.warning(f"Connection check failed: {e}")` |
| 1163 | `logger.debug(f"Suppressed error: {e}")  # Callbacks may not be registered if connection is lost` |
| 3089 | `if 'audit_logger' in st.session_state:` |
| 3090 | `audit_logger = st.session_state.audit_logger` |
| 3126 | `logger.warning(f"Connection check failed: {e}")` |
| 3135 | `st.warning("⚠️ Audit logger not available. Enable in app.py first.")` |
| 3830 | `logger.warning(f"Task Orchestrator not available: {e}")` |

## Which ones are inside try/except at module level (outside any function/class)

- **51** — inside `except ImportError:` for EnhancedSettings (module-level try 48–52)
- **57** — inside `except ImportError:` for AgentRegistry (module-level try 54–58)
- **66** — inside nested `except ImportError:` for SystemHealthMonitor (module-level try 60–67)
- **72** — inside `except ImportError:` for SystemStatus (module-level try 69–73)
- **132** — inside `except Exception as e:` for EnhancedSettings init (module-level try 126–133)
- **141** — inside `except Exception as e:` for AgentRegistry init (module-level try 135–142)
- **150** — inside `except Exception as e:` for SystemHealthMonitor init (module-level try 144–151)
- **970** — inside `except Exception as e:` in WebSocket connection check block (module-level flow, not inside any `def`)
- **1034** — inside `except Exception as e:` for Disconnect button handler (module-level flow, not inside any `def`)
- **1163** — inside `except Exception as e:` for WebSocket callback registration (module-level flow, not inside any `def`)
- **3126** — inside `except Exception as e:` in Audit Trail / Total Audit Entries block (module-level flow)
- **3830** — inside `except ImportError as e:` for Task Orchestrator (module-level try 3822–3831)

**Not module-level (left unchanged):**

- **42** — module-level assignment defining `logger`; not replaced
- **1023** — inside nested function `disconnect_ws()` (defined ~1016); not replaced
- **3089, 3090** — variable name `audit_logger`, not `logger.`; not replaced
- **3135** — string literal `"Audit logger"`; not replaced

## Line numbers changed (logger. → logging.)

Replaced `logger.` with `logging.` on these lines only (all module-level try/except as above):

**51, 57, 66, 72, 132, 141, 150, 970, 1034, 1163, 3126, 3830**

Compile check: `py -3.10 -m py_compile pages/11_Admin.py` — **exit code 0** (no errors).
