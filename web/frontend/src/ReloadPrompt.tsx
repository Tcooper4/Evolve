/**
 * Non-intrusive shell-update prompt for registerType: "autoUpdate".
 *
 * The new service worker may claim the page, but we never force-reload
 * mid-use. Instead we show "Update available — refresh to update" and let
 * the user apply the new static shell when ready.
 */
import { useEffect, useState } from "react";
import { useRegisterSW } from "virtual:pwa-register/react";

export default function ReloadPrompt() {
  const [show, setShow] = useState(false);

  // Register SW (autoUpdate). We intentionally ignore needRefresh from the
  // hook — autoUpdate activates via skipWaiting; we surface UX ourselves.
  useRegisterSW({
    immediate: true,
    onRegisteredSW(_swUrl, registration) {
      if (!registration) return;
      const hour = 60 * 60 * 1000;
      window.setInterval(() => {
        void registration.update();
      }, hour);
    },
  });

  useEffect(() => {
    if (!("serviceWorker" in navigator)) return;
    // First install also fires controllerchange — only prompt on updates.
    let hadController = Boolean(navigator.serviceWorker.controller);
    const onControllerChange = () => {
      if (hadController) setShow(true);
      hadController = true;
    };
    navigator.serviceWorker.addEventListener("controllerchange", onControllerChange);
    return () => {
      navigator.serviceWorker.removeEventListener("controllerchange", onControllerChange);
    };
  }, []);

  if (!show) return null;

  return (
    <div
      role="status"
      style={{
        position: "fixed",
        right: 16,
        bottom: 16,
        zIndex: 10001,
        maxWidth: 320,
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
        Update available — refresh to update.
      </div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <button
          type="button"
          className="primary"
          onClick={() => window.location.reload()}
        >
          Refresh
        </button>
        <button
          type="button"
          className="ghost"
          onClick={() => setShow(false)}
        >
          Later
        </button>
      </div>
    </div>
  );
}
