import ast, os, glob

src = open("pages/6_Risk_Management.py", encoding="utf-8", errors="replace").read()

# Parse check
try:
    ast.parse(src)
    print("[PASS] Risk_Management.py parses")
except SyntaxError as e:
    print(f"[FAIL] Risk_Management.py syntax error: {e}")

# Check for position sizing content
has_position = "position" in src.lower() or "Position" in src
has_fast_info = "fast_info" in src
has_cannot_fetch = "Cannot fetch price" in src

print(f"[INFO] Contains position sizing logic: {has_position}")
print(f"[INFO] Uses fast_info for price fetch: {has_fast_info}")
print(f"[INFO] Still has old 'Cannot fetch price' error message: {has_cannot_fetch}")

if has_fast_info:
    print("[PASS] Price fetch fix is present")
elif has_cannot_fetch:
    print("[FAIL] Old broken price fetch still present — needs fast_info fix")
else:
    print("[INFO] Cannot determine price fetch method — manual check needed")
