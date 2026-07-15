# Personalization boundary (standing design principle)

Evolve may personalize **how** advice is framed. It must not silently
narrow **what** a user is shown based on inferred behavior.

## In scope (explicit / stated only)

- Settings fields the user sets directly — e.g. `risk_tolerance`
  (`conservative` | `moderate` | `aggressive`), `allow_undefined_risk`,
  preferred DTE range — stored in `data/users.db` preferences via
  `config/user_store.py`.
- Using those stated prefs for:
  - Kelly **guided** size (conservative → quarter-Kelly with reason
    `stated_conservative`, distinct from sample-size triggers)
  - Options-structure **lead priority** (undefined-risk flagged /
    deprioritized, never deleted from view)
  - Chat **tone and framing** (downside-first for conservative)

See `trading/portfolio/risk_profile.py`.

## Out of scope (do not build)

**Implicit preference-learning from clicks, watchlist edits, chart
dwell time, which ideas were expanded, or other engagement history —
when used to shape *which* symbols, structures, or recommendations are
surfaced — is explicitly out of scope.**

Why: engagement-optimized ranking is a well-documented echo-chamber /
filter-bubble failure mode. It reinforces past behavior instead of
giving calibrated, independent research. That is the wrong objective for
a trading research tool.

## If engagement signals are ever used later

Any future engagement-based personalization must be limited to:

- **Tone / framing / communication style only**

It must **never**:

- Silently filter or rank-down symbols the user has not clicked
- Hide option structures, strategies, or research overlays based on
  past engagement
- Auto-set `risk_tolerance` from behavior

Stated Settings preferences remain the only source of truth for risk
appetite and structure comfort.

## Where this is enforced in prose today

| Location | Role |
|---|---|
| `trading/portfolio/risk_profile.py` | Canonical module docstring |
| `config/user_store.py` | Prefs are user JSON — not engagement models |
| `trading/services/chat_turn.py` | Framing block from stated prefs only |
| Settings UI copy | "never infers from clicks / watchlist" |

This document is the permanent cross-cutting note. Do not treat
"engagement learning for recommendations" as a natural next step after
the stated risk-profile work.
