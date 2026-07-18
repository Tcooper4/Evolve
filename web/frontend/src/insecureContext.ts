/**
 * Pure helpers for the insecure-context banner (login / signup / Settings).
 * Extracted so the decision is unit-testable without a DOM.
 */

export function isLocalhost(host: string): boolean {
  return (
    host === "localhost" ||
    host === "127.0.0.1" ||
    host === "[::1]" ||
    host.endsWith(".localhost")
  );
}

/** True when credentials must not be typed (plain HTTP outside localhost). */
export function shouldShowInsecureNotice(opts: {
  isSecureContext: boolean;
  hostname: string;
}): boolean {
  if (opts.isSecureContext) return false;
  if (isLocalhost(opts.hostname)) return false;
  return true;
}
