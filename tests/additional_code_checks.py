import ast
from glob import glob
import sys


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


def check_modules(globpath: str):
    paths = glob(globpath)
    ok = True
    for path in paths:
        print("checking", path)
        with open(path) as f:
            tree = ast.parse(f.read())
        module_ok = check_ast(tree, path)
        ok = ok and module_ok
    if ok:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    globpath = sys.argv[1]
    print("checking", globpath)
    check_modules(globpath)
