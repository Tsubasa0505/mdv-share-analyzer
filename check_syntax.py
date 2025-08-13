import ast
import sys

def check_syntax(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        print(f"OK: {filename} has valid Python syntax")
        return True
    except SyntaxError as e:
        print(f"ERROR: Syntax error in {filename}:")
        print(f"  Line {e.lineno}: {e.msg}")
        print(f"  {e.text}")
        return False
    except Exception as e:
        print(f"ERROR: Error reading {filename}: {e}")
        return False

if __name__ == "__main__":
    if check_syntax("app.py"):
        print("\nNo syntax errors found.")
    else:
        print("\nSyntax errors detected. Please fix them.")
        sys.exit(1)