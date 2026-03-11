"""
Show exact context around SyntaxError in 8_Model_Lab.py
Run: .\evolve_venv\Scripts\python.exe scripts\inspect_model_lab.py
"""
import ast

with open("pages/8_Model_Lab.py", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

# Show lines 4700-4730
print("=== lines 4700-4730 ===")
for i in range(4699, min(4730, len(lines))):
    print(f"{i+1}: {lines[i].rstrip()}")

# Confirm error location
try:
    ast.parse("".join(lines))
    print("\nFile parses cleanly — no error")
except SyntaxError as e:
    print(f"\nSyntaxError at line {e.lineno}: {e.msg}")
    print(f"Text: {repr(e.text)}")
    # Show wider context around exact error line
    el = (e.lineno or 4717) - 1
    print(f"\n=== lines {el-15}-{el+5} ===")
    for i in range(max(0, el-15), min(len(lines), el+6)):
        marker = " <<<" if i == el else ""
        print(f"{i+1}: {lines[i].rstrip()}{marker}")