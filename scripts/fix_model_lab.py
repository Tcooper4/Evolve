"""
Removes the malformed except block at lines 4717-4720 in 8_Model_Lab.py
Run: .\evolve_venv\Scripts\python.exe scripts\fix_model_lab.py
"""
import ast

fpath = "pages/8_Model_Lab.py"
with open(fpath, encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

# Lines 4717-4720 (0-indexed: 4716-4719) are:
#   4717:     except Exception as e:
#   4718:         st.caption(f"Model Innovation: {e}")
#   4719:                         import traceback
#   4720:                         st.code(traceback.format_exc())
# These form a broken duplicate except. Remove them.

target = [
    "    except Exception as e:\n",
    '        st.caption(f"Model Innovation: {e}")\n',
]

# Find the exact position — must be at lines 4716-4717 (0-indexed)
idx = 4716  # 0-indexed line 4717
if (lines[idx].rstrip() == "    except Exception as e:" and
        "st.caption" in lines[idx + 1] and
        "import traceback" in lines[idx + 2] and
        "st.code(traceback" in lines[idx + 3]):
    # Remove lines 4717-4720 (indices 4716-4719)
    del lines[4716:4720]
    print(f"[FIXED] Removed malformed except block (old lines 4717-4720)")
else:
    print("[WARN] Expected lines not found at 4716-4719, searching...")
    for i, line in enumerate(lines):
        if ('st.caption(f"Model Innovation: {e}")' in line and
                i > 4700):
            print(f"  Found at line {i+1}: {line.rstrip()}")
            # Remove this line and the except above it plus 2 lines below
            start = i - 1
            end = i + 3
            print(f"  Removing lines {start+1}-{end}:")
            for j in range(start, min(end, len(lines))):
                print(f"    {j+1}: {lines[j].rstrip()}")
            del lines[start:end]
            print(f"[FIXED] Removed malformed block")
            break

new_src = "".join(lines)

try:
    ast.parse(new_src)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(new_src)
    print(f"[OK] 8_Model_Lab.py parses cleanly and saved")
except SyntaxError as e:
    print(f"[ERROR] Still has SyntaxError at line {e.lineno}: {e.msg}")
    print(f"  text: {repr(e.text)}")
    el = (e.lineno or 1) - 1
    print(f"  Context:")
    for i in range(max(0, el-5), min(len(lines), el+5)):
        print(f"    {i+1}: {lines[i].rstrip()}")
    print("File NOT saved — manual fix required")