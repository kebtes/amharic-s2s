"""
Validates that Amharic_S2S_Baseline.ipynb has valid JSON and Python syntax in all code cells.
"""
import json
import ast

with open("Amharic_S2S_Baseline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
errors = 0

for idx, cell in enumerate(nb["cells"]):
    cell_type = cell["cell_type"]
    source = "".join(cell["source"])
    if cell_type == "code":
        # Remove Colab specific exclamation lines for AST parse check
        clean_lines = []
        for line in source.splitlines():
            if line.strip().startswith("!") or line.strip().startswith("%"):
                clean_lines.append("# " + line)
            else:
                clean_lines.append(line)
        clean_code = "\n".join(clean_lines)
        try:
            ast.parse(clean_code)
        except SyntaxError as e:
            print(f"Syntax error in code cell {idx}: {e}")
            errors += 1

if errors == 0:
    print("[PASS] All code cells passed Python AST validation successfully!")
else:
    print(f"[FAIL] Found {errors} syntax errors.")
