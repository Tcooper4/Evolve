/**
 * insecureContext.selftest — login/signup HTTPS banner decision.
 */
import { shouldShowInsecureNotice, isLocalhost } from "./insecureContext.ts";

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}

assert(isLocalhost("localhost"), "localhost");
assert(isLocalhost("127.0.0.1"), "loopback");
assert(!isLocalhost("192.168.1.10"), "lan not localhost");

assert(
  !shouldShowInsecureNotice({ isSecureContext: true, hostname: "evolve.example" }),
  "https → no banner",
);
assert(
  !shouldShowInsecureNotice({ isSecureContext: true, hostname: "localhost" }),
  "secure localhost → no banner",
);
assert(
  !shouldShowInsecureNotice({ isSecureContext: false, hostname: "localhost" }),
  "http localhost exempt",
);
assert(
  shouldShowInsecureNotice({ isSecureContext: false, hostname: "192.168.1.10" }),
  "http LAN → banner (credentials at risk)",
);
assert(
  shouldShowInsecureNotice({ isSecureContext: false, hostname: "evolve.example.com" }),
  "http public host → banner",
);

console.log("insecureContext.selftest: PASS");
