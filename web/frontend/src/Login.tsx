import { useState } from "react";
import { login } from "./api";

export default function Login({
  onLogin,
  onCreateAccount,
}: {
  onLogin: (name: string) => void;
  onCreateAccount: () => void;
}) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit() {
    if (!username || !password) return;
    setBusy(true);
    setError("");
    try {
      onLogin(await login(username, password));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Login failed");
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
        <div className="login-sub">Your research workspace. Sign in to continue.</div>
        <input
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          autoFocus
          autoComplete="username"
        />
        <input
          placeholder="Password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && void submit()}
          autoComplete="current-password"
        />
        {error && <div className="error">{error}</div>}
        <button className="primary" onClick={() => void submit()} disabled={busy}>
          {busy ? "Signing in…" : "Sign in"}
        </button>
        <button
          type="button"
          className="ghost"
          onClick={onCreateAccount}
          disabled={busy}
        >
          Have an invite? Create account
        </button>
      </div>
    </div>
  );
}
