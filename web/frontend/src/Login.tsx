import { useState } from "react";
import { login } from "./api";

export default function Login({ onLogin }: { onLogin: (name: string) => void }) {
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
        />
        <input
          placeholder="Password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
        />
        {error && <div className="error">{error}</div>}
        <button className="primary" onClick={submit} disabled={busy}>
          {busy ? "Signing in…" : "Sign in"}
        </button>
      </div>
    </div>
  );
}
