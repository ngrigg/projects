import os
import json

def count_lines_of_code(directory='.'):
    total_line_count = 0
    file_line_counts = {}

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".py") and "checkpoint" not in file_path and "__init__" not in file_path:
                line_count = 0
                in_triple_quote = False

                if file.endswith(".py"):
                    with open(file_path, 'r') as f:
                        for line in f:
                            stripped_line = line.strip()
                            if '"""' in stripped_line or "'''" in stripped_line:
                                in_triple_quote = not in_triple_quote
                                continue
                            if not stripped_line.startswith("#") and stripped_line and not in_triple_quote:
                                line_count += 1

                elif file.endswith(".ipynb"):
                    with open(file_path, 'r') as f:
                        notebook = json.load(f)
                        for cell in notebook['cells']:
                            if cell['cell_type'] == 'code':
                                code_lines = cell['source']
                                for line in code_lines:
                                    stripped_line = line.strip()
                                    if '"""' in stripped_line or "'''" in stripped_line:
                                        in_triple_quote = not in_triple_quote
                                        continue
                                    if not stripped_line.startswith("#") and stripped_line and not in_triple_quote:
                                        line_count += 1

                file_line_counts[file] = line_count
                total_line_count += line_count

    return total_line_count, file_line_counts

total_lines, breakdown = count_lines_of_code()

print(f"Total lines of code (excluding comments): {total_lines}")
print("Breakdown by file:")
for file, count in breakdown.items():
    print(f"{file}: {count} lines")
