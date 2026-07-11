#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Account management CLI for Evolve's multi-user live mode.

Usage:
    python scripts/manage_users.py add <username> [--name "Display"] [--email a@b.c] [--admin]
    python scripts/manage_users.py list
    python scripts/manage_users.py passwd <username>
    python scripts/manage_users.py deactivate <username>
    python scripts/manage_users.py activate <username>

Passwords are prompted interactively (never on the command line, never
echoed) and stored only as bcrypt hashes.
"""

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading.auth import accounts  # noqa: E402


def _prompt_password() -> str:
    pw = getpass.getpass("Password (min 8 chars): ")
    if pw != getpass.getpass("Confirm password: "):
        print("Passwords do not match.", file=sys.stderr)
        sys.exit(1)
    return pw


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add")
    a.add_argument("username")
    a.add_argument("--name", default=None)
    a.add_argument("--email", default=None)
    a.add_argument("--admin", action="store_true")

    for cmd in ("passwd", "deactivate", "activate"):
        s = sub.add_parser(cmd)
        s.add_argument("username")
    sub.add_parser("list")

    args = p.parse_args()
    try:
        if args.cmd == "add":
            accounts.create_user(
                args.username, _prompt_password(), display_name=args.name,
                email=args.email, role="admin" if args.admin else "user",
            )
            print(f"Created {args.username}.")
        elif args.cmd == "passwd":
            accounts.set_password(args.username, _prompt_password())
            print(f"Password updated for {args.username}.")
        elif args.cmd == "deactivate":
            accounts.set_active(args.username, False)
            print(f"Deactivated {args.username}.")
        elif args.cmd == "activate":
            accounts.set_active(args.username, True)
            print(f"Activated {args.username}.")
        elif args.cmd == "list":
            for u in accounts.list_users():
                flag = "" if u["active"] else "  [DEACTIVATED]"
                print(f"{u['username']:<20} {u['role']:<6} "
                      f"last_login={u['last_login'] or 'never'}{flag}")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
