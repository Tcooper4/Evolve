/**
 * signupPassword.selftest — client password bar + invite formatting.
 */
import {
  formatInviteInput,
  MIN_SIGNUP_PASSWORD_LEN,
  validateSignupPassword,
} from "./signupPassword.ts";

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}

assert(MIN_SIGNUP_PASSWORD_LEN === 10, "min len");

assert(validateSignupPassword("short") !== null, "too short");
assert(
  validateSignupPassword("alllowercase1")?.toLowerCase().includes("uppercase") ?? false,
  "needs upper",
);
assert(
  validateSignupPassword("ALLUPPERCASE1")?.toLowerCase().includes("lowercase") ?? false,
  "needs lower",
);
assert(
  validateSignupPassword("NoDigitsHere")?.toLowerCase().includes("digit") ?? false,
  "needs digit",
);
assert(
  validateSignupPassword(" Leading9Aa")?.toLowerCase().includes("whitespace") ?? false,
  "leading space",
);
assert(validateSignupPassword("CorrectHorse9") === null, "strong ok");

assert(formatInviteInput("abcd1234efgh") === "ABCD-1234-EFGH", "format hyphens");
assert(formatInviteInput("ab-cd 12-34 ef-ghxx") === "ABCD-1234-EFGH", "strip + truncate");
assert(formatInviteInput("xy") === "XY", "partial");

console.log("signupPassword.selftest: PASS");
