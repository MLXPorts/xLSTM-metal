#!/usr/bin/env python
"""
Strict linter to flag Python numeric literals used in tensor math.

Rules (heuristic, AST-based):
- Disallow bare int/float constants in BinOp with suspected tensor vars.
- Disallow bare int/float constants passed to mx/torch math ops (add, mul, div, pow, etc.).

Allowed:
- Shape/axis/indices (we cannot reliably infer; this tool errs on the side of caution and prints context).

Exit code:
- 0 if no issues; 1 if any findings.
"""
import argparse
import ast
import pathlib
from typing import List, Tuple


MATH_FUNCS = {
    'add','subtract','multiply','divide','power','pow','tanh','sigmoid','gelu','erf','exp','log','maximum','minimum',
    'sin','cos','sqrt','rsqrt','relu','silu','softmax','matmul','einsum'
}


def is_numeric_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float))


def is_tensor_suspect(node: ast.AST) -> bool:
    # Heuristic: name like x, y, v, t and attributes starting with mx., torch.
    if isinstance(node, ast.Attribute):
        base = node.value
        if isinstance(base, ast.Name) and base.id in {'mx','torch','torchvision'}:
            return True
        return is_tensor_suspect(base)
    if isinstance(node, ast.Subscript):
        return is_tensor_suspect(node.value)
    if isinstance(node, ast.Call):
        return is_tensor_suspect(node.func)
    if isinstance(node, ast.BinOp):
        return is_tensor_suspect(node.left) or is_tensor_suspect(node.right)
    if isinstance(node, ast.Name):
        # We can't know type; treat variable names commonly used in compute as suspects
        return node.id in {'x','y','v','u','k','h','z','t','bias','weight','w','b'}
    return False


def visit_file(path: pathlib.Path) -> List[Tuple[int, str]]:
    src = path.read_text(encoding='utf-8')
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    findings: List[Tuple[int,str]] = []

    SHAPE_FUNCS = {'pad','reshape','transpose','permute','slice','take','stack','concatenate'}

    class V(ast.NodeVisitor):
        def __init__(self):
            self.stack = []

        def visit(self, node):
            self.stack.append(node)
            super().visit(node)
            self.stack.pop()
        def visit_BinOp(self, node: ast.BinOp):
            if is_numeric_constant(node.left) and is_tensor_suspect(node.right):
                if not self._inside_shape_call():
                    findings.append((node.lineno, 'Numeric literal on left in BinOp with tensor suspect'))
            if is_numeric_constant(node.right) and is_tensor_suspect(node.left):
                if not self._inside_shape_call():
                    findings.append((node.lineno, 'Numeric literal on right in BinOp with tensor suspect'))
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call):
            # flag mx.* or torch.* math funcs receiving numeric constants
            fn = node.func
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id in {'mx','torch'}:
                if fn.attr in MATH_FUNCS:
                    for arg in node.args:
                        if is_numeric_constant(arg):
                            findings.append((node.lineno, f'Numeric literal passed to {fn.value.id}.{fn.attr}'))
            self.generic_visit(node)

        def _inside_shape_call(self) -> bool:
            # If any ancestor is a Call to mx.<shape-func>, ignore
            for n in reversed(self.stack):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                    f = n.func
                    if isinstance(f.value, ast.Name) and f.value.id == 'mx' and f.attr in SHAPE_FUNCS:
                        return True
            return False

    V().visit(tree)
    return findings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+', help='Files or directories to scan')
    args = ap.parse_args()
    files: List[pathlib.Path] = []
    for p in args.paths:
        pa = pathlib.Path(p)
        if pa.is_dir():
            files.extend([x for x in pa.rglob('*.py')])
        elif pa.suffix == '.py':
            files.append(pa)
    total = 0
    for f in files:
        findings = visit_file(f)
        if findings:
            print(f'-- {f}')
            for ln, msg in findings:
                print(f'  L{ln}: {msg}')
            total += len(findings)
    if total:
        print(f'Found {total} potential numeric literal issues.')
        raise SystemExit(1)

if __name__ == '__main__':
    main()
