import ast
from glob import glob
from pathlib import Path


def check_ast(node: ast.AST, path: str):
    ok = True
    if hasattr(node, "body"):
        for child in node.body:
            child_ok = check_ast(child, path)
            ok = ok and child_ok
    else:
        if isinstance(node, ast.Assert):
            print(f"assert detected in line {node.lineno} of {path}")
            ok = False
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            value = node.value
            if (
                hasattr(value, "func")
                and hasattr(value.func, "id")
                and value.func.id == "print"
            ):
                print(f"print detected in line {node.lineno} of {path}")
                ok = False
    return ok


def test_check_modules():
    paths = glob("../imod/**/*.py")
    ok = True
    for path in paths:
        with open(path) as f:
            content = f.read()
            try:
                tree = ast.parse(content)
                module_ok = check_ast(tree, path)
                ok = ok and module_ok
            except Exception as e:
                print(f"parsing error in {path}, with error: {e}.")
                ok = False
    assert ok
