# -*- coding: utf-8 -*-
"""Tour-seen state: mark-one-page isolation + reset-all (hand-verifiable)."""

from __future__ import annotations

import pytest

from trading.services.tour_state import (
    TOUR_PAGE_IDS,
    apply_mark_page_seen,
    apply_reset_tours,
    get_tours_seen_from_prefs,
    mark_page_seen,
    normalize_tours_seen,
    reset_tours_seen,
)


def test_page_ids_match_app_tsx():
    """Guard against inventing parallel naming vs App.tsx PAGES."""
    from pathlib import Path

    app = (
        Path(__file__).resolve().parents[1]
        / "web"
        / "frontend"
        / "src"
        / "App.tsx"
    ).read_text(encoding="utf-8", errors="replace")
    for pid in TOUR_PAGE_IDS:
        assert f'id: "{pid}"' in app, pid
    assert len(TOUR_PAGE_IDS) == 7


def test_mark_one_page_does_not_affect_others():
    seen = mark_page_seen({}, "dashboard")
    assert seen == {"dashboard": True}
    # Mark analyze — dashboard stays True; others still absent (not silently True)
    seen2 = mark_page_seen(seen, "analyze")
    assert seen2["dashboard"] is True
    assert seen2["analyze"] is True
    assert "scanner" not in seen2
    assert "portfolio" not in seen2
    assert "backtest" not in seen2
    assert "chat" not in seen2
    assert "settings" not in seen2


def test_skip_semantics_only_current_page():
    """Skipping dashboard must not mark analyze/GEX tour as seen."""
    prefs: dict = {"risk_tolerance": "moderate"}
    apply_mark_page_seen(prefs, "dashboard")
    tours = get_tours_seen_from_prefs(prefs)
    assert tours.get("dashboard") is True
    assert tours.get("analyze") is not True
    # analyze still "unseen" for tour firing
    assert "analyze" not in tours


def test_reset_clears_everything():
    prefs: dict = {}
    apply_mark_page_seen(prefs, "dashboard")
    apply_mark_page_seen(prefs, "chat")
    apply_mark_page_seen(prefs, "settings")
    assert len(get_tours_seen_from_prefs(prefs)) == 3
    cleared = apply_reset_tours(prefs)
    assert cleared == {}
    assert prefs["tours_seen"] == {}
    assert get_tours_seen_from_prefs(prefs) == {}
    assert reset_tours_seen() == {}


def test_unknown_page_rejected():
    with pytest.raises(ValueError, match="Unknown tour page_id"):
        mark_page_seen({}, "not-a-page")


def test_normalize_ignores_junk_keys():
    raw = {
        "dashboard": True,
        "bogus": True,
        "analyze": "yes",
        "scanner": 0,
    }
    out = normalize_tours_seen(raw)
    assert out == {"dashboard": True, "analyze": True, "scanner": False}
    assert "bogus" not in out


def test_api_mark_and_reset_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("EVOLVE_AUTH_SECRET", "t")
    import config.user_store as US
    import trading.auth.accounts as A
    import trading.portfolio.paper_portfolio as PP

    monkeypatch.setattr(A, "DB_PATH", tmp_path / "a.db")
    monkeypatch.setattr(PP, "DB_PATH", tmp_path / "pp.db")
    monkeypatch.setattr(US, "USER_DB_PATH", tmp_path / "users.db")
    A.create_user("thomas", "hunter2secure")

    from fastapi.testclient import TestClient

    from web.backend.main import app

    c = TestClient(app)
    tok = c.post(
        "/api/auth/token",
        data={"username": "thomas", "password": "hunter2secure"},
    ).json()["access_token"]
    H = {"Authorization": f"Bearer {tok}"}

    r0 = c.get("/api/settings/tours", headers=H)
    assert r0.status_code == 200
    assert r0.json()["tours_seen"] == {}

    r1 = c.post(
        "/api/settings/tours/seen",
        headers=H,
        json={"page_id": "dashboard"},
    )
    assert r1.status_code == 200
    body1 = r1.json()
    assert body1["success"] is True
    assert body1["tours_seen"] == {"dashboard": True}

    r2 = c.post(
        "/api/settings/tours/seen",
        headers=H,
        json={"page_id": "analyze"},
    )
    assert r2.json()["tours_seen"]["dashboard"] is True
    assert r2.json()["tours_seen"]["analyze"] is True
    assert "scanner" not in r2.json()["tours_seen"]

    # Skip-only-current: dashboard mark left analyze unset until explicit
    # (already verified above). Prefs merge must not wipe tours_seen.
    c.post("/api/settings/prefs", headers=H, json={"scoring_style": "balanced"})
    r3 = c.get("/api/settings/tours", headers=H)
    assert r3.json()["tours_seen"]["dashboard"] is True
    assert r3.json()["tours_seen"]["analyze"] is True

    r4 = c.post("/api/settings/tours/reset", headers=H, json={})
    assert r4.json()["success"] is True
    assert r4.json()["tours_seen"] == {}
    assert c.get("/api/settings/tours", headers=H).json()["tours_seen"] == {}


def test_api_rejects_unknown_page(monkeypatch, tmp_path):
    monkeypatch.setenv("EVOLVE_AUTH_SECRET", "t")
    import config.user_store as US
    import trading.auth.accounts as A
    import trading.portfolio.paper_portfolio as PP

    monkeypatch.setattr(A, "DB_PATH", tmp_path / "a.db")
    monkeypatch.setattr(PP, "DB_PATH", tmp_path / "pp.db")
    monkeypatch.setattr(US, "USER_DB_PATH", tmp_path / "users.db")
    A.create_user("thomas", "hunter2secure")

    from fastapi.testclient import TestClient

    from web.backend.main import app

    c = TestClient(app)
    tok = c.post(
        "/api/auth/token",
        data={"username": "thomas", "password": "hunter2secure"},
    ).json()["access_token"]
    H = {"Authorization": f"Bearer {tok}"}
    r = c.post(
        "/api/settings/tours/seen",
        headers=H,
        json={"page_id": "nope"},
    )
    assert r.status_code == 200
    assert r.json()["success"] is False
    assert "Unknown" in (r.json().get("error") or "")
