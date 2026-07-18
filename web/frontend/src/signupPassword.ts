/**
 * Client-side mirror of trading/auth/password_policy.py for signup UX.
 * Server still enforces the same rules on POST /api/auth/signup.
 */

export const MIN_SIGNUP_PASSWORD_LEN = 10;

/** Return a human-readable failure reason, or null if the password passes. */
export function validateSignupPassword(password: string): string | null {
  if (password == null || password === "") {
    return "Password is required";
  }
  if (password !== password.trim()) {
    return "Password must not start or end with whitespace";
  }
  if (password.length < MIN_SIGNUP_PASSWORD_LEN) {
    return `Password must be at least ${MIN_SIGNUP_PASSWORD_LEN} characters`;
  }
  if (![...password].some((c) => c >= "A" && c <= "Z")) {
    return "Password must include at least one uppercase letter";
  }
  if (![...password].some((c) => c >= "a" && c <= "z")) {
    return "Password must include at least one lowercase letter";
  }
  if (![...password].some((c) => c >= "0" && c <= "9")) {
    return "Password must include at least one digit";
  }
  return null;
}

/** Hand-typeable invite display: XXXX-XXXX-XXXX (12 alnum chars). */
export function formatInviteInput(raw: string): string {
  const body = (raw || "")
    .toUpperCase()
    .replace(/[^A-Z0-9]/g, "")
    .slice(0, 12);
  const parts: string[] = [];
  for (let i = 0; i < body.length; i += 4) {
    parts.push(body.slice(i, i + 4));
  }
  return parts.join("-");
}
