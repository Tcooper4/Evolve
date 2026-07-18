import { useCallback, useEffect, useState } from "react";
import {
  createInvite,
  listInvites,
  type InviteRow,
} from "./api";

const EXPIRY_OPTIONS = [
  { days: 7, label: "7 days" },
  { days: 14, label: "14 days" },
  { days: 30, label: "30 days" },
] as const;

export default function AdminInvites({
  onToast,
}: {
  onToast?: (msg: string) => void;
}) {
  const [visible, setVisible] = useState(false);
  const [invites, setInvites] = useState<InviteRow[]>([]);
  const [expiresDays, setExpiresDays] = useState(14);
  const [busy, setBusy] = useState(false);
  const [freshCode, setFreshCode] = useState<string | null>(null);
  const [err, setErr] = useState("");

  const refresh = useCallback(async () => {
    const rows = await listInvites();
    if (rows === null) {
      setVisible(false);
      return;
    }
    setVisible(true);
    setInvites(rows);
  }, []);

  useEffect(() => {
    void refresh().catch(() => setVisible(false));
  }, [refresh]);

  if (!visible) return null;

  async function generate() {
    setBusy(true);
    setErr("");
    setFreshCode(null);
    try {
      const inv = await createInvite(expiresDays);
      setFreshCode(inv.code);
      onToast?.(`Invite created: ${inv.code}`);
      await refresh();
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Could not create invite");
    } finally {
      setBusy(false);
    }
  }

  async function copyCode(code: string) {
    try {
      await navigator.clipboard.writeText(code);
      onToast?.("Invite code copied");
    } catch {
      onToast?.("Could not copy — select the code manually");
    }
  }

  return (
    <div className="card card-pad" data-tour="settings-invites" style={{ maxWidth: 560, marginBottom: 16 }}>
      <div className="rail-label" style={{ marginTop: 0 }}>Invite codes</div>
      <p style={{ fontSize: 12.5, color: "var(--text-2)", margin: "0 0 12px", lineHeight: 1.45 }}>
        Admin only. Each code creates one account — hand it to someone you trust.
        There is no public signup.
      </p>
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center", marginBottom: 12 }}>
        <select
          value={expiresDays}
          onChange={(e) => setExpiresDays(Number(e.target.value))}
          style={{ maxWidth: 140 }}
        >
          {EXPIRY_OPTIONS.map((o) => (
            <option key={o.days} value={o.days}>{o.label}</option>
          ))}
        </select>
        <button className="primary" disabled={busy} onClick={() => void generate()}>
          {busy ? "Generating…" : "Generate invite"}
        </button>
      </div>
      {freshCode && (
        <div
          style={{
            marginBottom: 12,
            padding: "10px 12px",
            borderRadius: 8,
            background: "var(--surface-2)",
            border: "1px solid var(--border-strong)",
            fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
            fontSize: 15,
            letterSpacing: 1,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 10,
          }}
        >
          <span>{freshCode}</span>
          <button type="button" className="ghost" onClick={() => void copyCode(freshCode)}>
            Copy
          </button>
        </div>
      )}
      {err && <div className="error" style={{ marginBottom: 10 }}>{err}</div>}
      {invites.length === 0 ? (
        <div className="dim" style={{ fontSize: 13 }}>No invites yet.</div>
      ) : (
        <div className="tbl" style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", fontSize: 12.5, borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ textAlign: "left", color: "var(--text-3)" }}>
                <th style={{ padding: "6px 4px" }}>Code</th>
                <th style={{ padding: "6px 4px" }}>Status</th>
                <th style={{ padding: "6px 4px" }}>Expires</th>
                <th style={{ padding: "6px 4px" }}>Used by</th>
              </tr>
            </thead>
            <tbody>
              {invites.map((row) => (
                <tr key={`${row.code}-${row.created_at}`} style={{ borderTop: "1px solid var(--border)" }}>
                  <td style={{ padding: "7px 4px", fontFamily: "ui-monospace, monospace" }}>
                    <button
                      type="button"
                      className="ghost"
                      style={{ padding: 0, font: "inherit" }}
                      onClick={() => void copyCode(row.code)}
                      title="Copy"
                    >
                      {row.code}
                    </button>
                  </td>
                  <td style={{ padding: "7px 4px" }}>{row.status}</td>
                  <td style={{ padding: "7px 4px" }}>{fmtWhen(row.expires_at)}</td>
                  <td style={{ padding: "7px 4px" }}>{row.used_by || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function fmtWhen(iso: string | null | undefined): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleDateString();
  } catch {
    return iso.slice(0, 10);
  }
}
