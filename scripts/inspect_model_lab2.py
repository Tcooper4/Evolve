"""
Find the missing try: block above line 4718 in 8_Model_Lab.py
Run: .\evolve_venv\Scripts\python.exe scripts\inspect_model_lab2.py
"""
with open("pages/8_Model_Lab.py", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

# Show lines 4600-4725 to find the matching try:
print("=== lines 4600-4725 ===")
for i in range(4599, min(4725, len(lines))):
    print(f"{i+1}: {lines[i].rstrip()}")