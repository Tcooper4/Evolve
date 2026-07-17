/**
 * Dismissible notice shown only when the app is served over an insecure
 * origin in production (not localhost). On such origins the browser silently
 * disables the service worker and install prompt, so PWA/offline features
 * won't appear — this explains why rather than leaving the user confused.
 *
 * Never shown on localhost (a secure context) or when the page is already
 * secure (https), so normal use and local dev are unaffected.
 */
import { useEffect, useState } from "react";

const DISMISS_KEY = "evolve_insecure_notice_dismissed";

function isLocalhost(host: string): boolean {
  return (
    host === "localhost" ||
    host === "127.0.0.1" ||
    host === "[::1]" ||
    host.endsWith(".localhost")
  );
}

export default function InsecureContextNotice() {
  const [show, setShow] = useState(false);

  useEffect(() => {
    try {
      if (sessionStorage.getItem(DISMISS_KEY) === "1") return;
      const secure = window.isSecureContext;
      const local = isLocalhost(window.location.hostname);
      // Only nag when genuinely insecure AND not localhost.
      if (!secure && !local) setShow(true);
    } catch {
      /* storage/blocked context — stay silent */
    }
  }, []);

  if (!show) return null;

  const dismiss = () => {
    try {
      sessionStorage.setItem(DISMISS_KEY, "1");
    } catch {
      /* ignore */
    }
    setShow(false);
  };

  return (
    <div
      role="status"
      style={{
        position: "fixed",
        left: 16,
        bottom: 16,
        zIndex: 10001,
        maxWidth: 340,
        padding: "12px 14px",
        borderRadius: 10,
        border: "1px solid var(--border-strong)",
        background: "var(--surface-2)",
        color: "var(--text)",
        boxShadow: "var(--shadow)",
        fontSize: 13,
        lineHeight: 1.45,
      }}
    >
      <div style={{ marginBottom: 10 }}>
        You're on an insecure (HTTP) connection. Installing Evolve as an app and
        offline support need HTTPS — ask your host to enable TLS.
      </div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <button type="button" className="ghost" onClick={dismiss}>
          Dismiss
        </button>
      </div>
    </div>
  );
}
