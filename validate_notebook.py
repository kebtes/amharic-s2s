"""
Validates that notebooks have valid JSON and Python syntax in all code cells.
"""
import json
import ast
import glob

notebooks = glob.glob("*.ipynb") + glob.glob("notebooks/*.ipynb")
print(f"Found {len(notebooks)} notebooks to validate: {notebooks}")

total_errors = 0
for nb_path in notebooks:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    print(f"\nValidating {nb_path} ({len(nb['cells'])} cells)...")
    errors = 0

    for idx, cell in enumerate(nb["cells"]):
        cell_type = cell["cell_type"]
        source = "".join(cell["source"])
        if cell_type == "code":
            clean_lines = []
            for line in source.splitlines():
                if line.strip().startswith("!") or line.strip().startswith("%"):
                    clean_lines.append("# " + line)
                elif line.strip().startswith("await "):
                    # Wrap top-level await for AST validation
                    clean_lines.append(f"async def _test(): {line}")
                else:
                    clean_lines.append(line)
            clean_code = "\n".join(clean_lines)
            try:
                ast.parse(clean_code)
            except SyntaxError as e:
                print(f"  [Syntax Error] Cell {idx} in {nb_path}: {e}")
                errors += 1

    if errors == 0:
        print(f"  [PASS] {nb_path} passed Python AST validation.")
    else:
        print(f"  [FAIL] {nb_path} had {errors} syntax errors.")
        total_errors += errors

if total_errors == 0:
    print("\n[ALL PASS] All notebooks passed validation successfully!")
else:
    print(f"\n[FAIL] Found {total_errors} total errors across notebooks.")

