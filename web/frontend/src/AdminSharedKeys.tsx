import { useCallback, useEffect, useState } from "react";
import { getSharedKeys, setSharedKeys } from "./api";

/**
 * Admin-only live toggle for operator env-key fallback.
 * Same defensive gating as AdminInvites: only renders if GET succeeds
 * (403 → hidden). No client-side role checks.
 */
export default function AdminSharedKeys({
  onToast,
}: {
  onToast?: (msg: string) => void;
}) {
  const [visible, setVisible] = useState(false);
  const [allowed, setAllowed] = useState(true);
  const [source, setSource] = useState<"env" | "persisted">("env");
  const [envDefault, setEnvDefault] = useState(true);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const refresh = useCallback(async () => {
    const snap = await getSharedKeys();
    if (snap === null) {
      setVisible(false);
      return;
    }
    setVisible(true);
    setAllowed(snap.allowed);
    setSource(snap.source === "persisted" ? "persisted" : "env");
    setEnvDefault(snap.env_default);
  }, []);

  useEffect(() => {
    void refresh().catch(() => setVisible(false));
  }, [refresh]);

  if (!visible) return null;

  async function toggle(next: boolean) {
    setBusy(true);
    setErr("");
    try {
      const snap = await setSharedKeys(next);
      setAllowed(snap.allowed);
      setSource(snap.source === "persisted" ? "persisted" : "env");
      setEnvDefault(snap.env_default);
      onToast?.(
        next
          ? "Shared operator keys ON — users without personal keys use the server env."
          : "Shared operator keys OFF — each user must supply their own keys.",
      );
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Could not update shared-keys policy");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div
      className="card card-pad"
      data-tour="settings-shared-keys"
      style={{ maxWidth: 560, marginBottom: 16 }}
    >
      <div className="rail-label" style={{ marginTop: 0 }}>Shared API keys</div>
      <p style={{ fontSize: 12.5, color: "var(--text-2)", margin: "0 0 12px", lineHeight: 1.45 }}>
        Admin only. When on, users without personal keys may fall back to the
        server&apos;s environment keys. When off, each account must save their
        own keys. Takes effect immediately — no restart.
      </p>
      <label
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          fontSize: 13.5,
          cursor: busy ? "wait" : "pointer",
        }}
      >
        <input
          type="checkbox"
          checked={allowed}
          disabled={busy}
          onChange={(e) => void toggle(e.target.checked)}
        />
        <span>
          {allowed ? "Shared keys enabled" : "Shared keys disabled"}
          <span className="dim" style={{ display: "block", fontSize: 12, marginTop: 2 }}>
            Source: {source === "persisted" ? "admin toggle (persisted)" : "env default"}
            {source === "env" ? ` (EVOLVE_SHARED_KEYS → ${envDefault ? "on" : "off"})` : ""}
          </span>
        </span>
      </label>
      {err && <div className="error" style={{ marginTop: 10 }}>{err}</div>}
    </div>
  );
}
