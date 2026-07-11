import { useState } from "react";
import { login } from "./api";

export default function Login({ onLogin }: { onLogin: (name: string) => void }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit() {
    setBusy(true);
    setError("");
    try {
      const name = await login(username, password);
      onLogin(name);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Login failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login-wrap">
      <div className="card login-card">
        <div className="brand">EVOLVE</div>
        <div className="dim">Sign in to your workspace</div>
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
