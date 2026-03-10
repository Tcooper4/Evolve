import os

pages = [
    "pages/4_Trade_Execution.py",
    "pages/3_Strategy_Testing.py", 
    "pages/5_Portfolio.py",
    "pages/8_Model_Lab.py"
]

for page in pages:
    if not os.path.exists(page):
        continue
    src = open(page, encoding="utf-8", errors="replace").read()
    has_position = "position size" in src.lower() or "position_size" in src.lower()
    has_cannot_fetch = "cannot fetch price" in src.lower()
    has_fast_info = "fast_info" in src
    if has_position or has_cannot_fetch:
        print(f"\n{'='*50}")
        print(f"FILE: {page}")
        print(f"  Has position sizing: {has_position}")
        print(f"  Has 'Cannot fetch price': {has_cannot_fetch}")
        print(f"  Uses fast_info: {has_fast_info}")
        # Print surrounding lines for context
        lines = src.split('\n')
        for i, line in enumerate(lines):
            if "cannot fetch price" in line.lower() or "fetch" in line.lower() and "price" in line.lower():
                start = max(0, i-3)
                end = min(len(lines), i+4)
                print(f"  --- lines {start+1}-{end} ---")
                for j in range(start, end):
                    print(f"  {j+1:4d}: {lines[j]}")
