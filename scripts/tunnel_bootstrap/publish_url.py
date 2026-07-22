# -*- coding: utf-8 -*-
"""Publish current trycloudflare URL to the stable bootstrap Worker."""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict


def load_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path} — run scripts/setup-tunnel-bootstrap.ps1 first."
        )
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        return json.load(f)


def publish_url(
    *,
    tunnel_url: str,
    bootstrap_url: str,
    update_secret: str,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    base = bootstrap_url.rstrip("/")
    endpoint = f"{base}/update"
    payload = json.dumps({"url": tunnel_url.rstrip("/")}).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Evolve-Secret": update_secret,
            "User-Agent": "EvolveBootstrap/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body) if body else {"ok": True}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Bootstrap publish failed HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Bootstrap publish failed: {e}") from e


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("Usage: publish_url.py <tunnel-url>", file=sys.stderr)
        return 2
    tunnel_url = args[0].strip()
    root = Path(__file__).resolve().parents[2]
    cfg_path = root / "data" / "tunnel_bootstrap.json"
    cfg = load_config(cfg_path)
    bootstrap_url = str(cfg.get("bootstrap_url") or "").strip()
    secret = str(cfg.get("update_secret") or "").strip()
    if not bootstrap_url or not secret:
        print("bootstrap_url and update_secret required in tunnel_bootstrap.json", file=sys.stderr)
        return 1
    out = publish_url(
        tunnel_url=tunnel_url,
        bootstrap_url=bootstrap_url,
        update_secret=secret,
    )
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
