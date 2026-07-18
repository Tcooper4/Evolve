import { useState } from "react";
import { signup } from "./api";
import {
  formatInviteInput,
  MIN_SIGNUP_PASSWORD_LEN,
  validateSignupPassword,
} from "./signupPassword";

export default function Signup({
  onLogin,
  onBack,
}: {
  onLogin: (name: string) => void;
  onBack: () => void;
}) {
  const [invite, setInvite] = useState("");
  const [username, setUsername] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit() {
    if (!invite.trim() || !username.trim() || !password) return;
    const pwErr = validateSignupPassword(password);
    if (pwErr) {
      setError(pwErr);
      return;
    }
    if (password !== confirm) {
      setError("Passwords do not match");
      return;
    }
    setBusy(true);
    setError("");
    try {
      onLogin(
        await signup({
          username: username.trim(),
          password,
          invite_code: invite.trim(),
          display_name: displayName.trim() || undefined,
        }),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Signup failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login-wrap">
      <div className="card login-card fade-in">
        <div className="brand">
          <span className="dot" /> EVOLVE <small>terminal</small>
        </div>
        <div className="login-sub">
          Create an account with your invite code.
        </div>
        <input
          placeholder="Invite code (XXXX-XXXX-XXXX)"
          value={invite}
          onChange={(e) => setInvite(formatInviteInput(e.target.value))}
          autoFocus
          autoComplete="off"
          spellCheck={false}
        />
        <input
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          autoComplete="username"
        />
        <input
          placeholder="Display name (optional)"
          value={displayName}
          onChange={(e) => setDisplayName(e.target.value)}
          autoComplete="nickname"
        />
        <input
          placeholder="Password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          autoComplete="new-password"
        />
        <input
          placeholder="Confirm password"
          type="password"
          value={confirm}
          onChange={(e) => setConfirm(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && void submit()}
          autoComplete="new-password"
        />
        <div className="login-hint">
          Password: ≥{MIN_SIGNUP_PASSWORD_LEN} chars, upper, lower, and a digit.
        </div>
        {error && <div className="error">{error}</div>}
        <button className="primary" onClick={() => void submit()} disabled={busy}>
          {busy ? "Creating account…" : "Create account"}
        </button>
        <button type="button" className="ghost" onClick={onBack} disabled={busy}>
          Back to sign in
        </button>
      </div>
    </div>
  );
}
