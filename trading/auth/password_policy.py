# -*- coding: utf-8 -*-
"""Signup password policy for invite-gated account creation.

Bar (stated, enforced on POST /api/auth/signup only — CLI/bootstrap
``create_user`` still accepts ≥8 chars for first-admin tooling):

* At least 10 characters
* At least one uppercase letter (A–Z)
* At least one lowercase letter (a–z)
* At least one digit (0–9)

Errors are specific so the user knows what to fix — never a vague
"invalid password".
"""

from __future__ import annotations


MIN_SIGNUP_PASSWORD_LEN = 10


def validate_signup_password(password: str) -> None:
    """Raise ValueError with a clear reason if ``password`` is too weak."""
    if password is None:
        raise ValueError("Password is required")
    if password != password.strip():
        raise ValueError("Password must not start or end with whitespace")
    if len(password) < MIN_SIGNUP_PASSWORD_LEN:
        raise ValueError(
            f"Password must be at least {MIN_SIGNUP_PASSWORD_LEN} characters"
        )
    if not any(c.isupper() for c in password):
        raise ValueError("Password must include at least one uppercase letter")
    if not any(c.islower() for c in password):
        raise ValueError("Password must include at least one lowercase letter")
    if not any(c.isdigit() for c in password):
        raise ValueError("Password must include at least one digit")


__all__ = ["MIN_SIGNUP_PASSWORD_LEN", "validate_signup_password"]
