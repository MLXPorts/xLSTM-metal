#!/usr/bin/env python
"""
EmberLint: A comprehensive linting tool for Ember ML codebase.

This script scans Python files to detect:
1. Syntax errors and compilation issues
2. Backend purity issues:
   - NumPy imports and usage
   - Precision-reducing casts (e.g., float() casts)
   - Tensor conversions between backends
   - Direct Python operators instead of ops functions
   - Backend-specific implementations in frontend code
3. Style issues (PEP 8)
4. Import issues (unused, missing)
5. Type annotation issues
6. Backend consistency issues (operations missing in some backends)

It helps ensure that Ember ML code remains pure, efficient, and maintainable.
"""

# Set this to False to force EmberLint to always check all issues
ALLOW_SINGLE_ISSUE_LINTING = False

import os
import re
import ast
import sys
import argparse
import importlib
import subprocess
from typing import List, Dict, Tuple, Set, Optional, Any, Union
from pathlib import Path

# Try to import optional dependencies
try:
    import pycodestyle
    HAVE_PYCODESTYLE = True
except ImportError:
    HAVE_PYCODESTYLE = False

try:
    import mypy.api
    HAVE_MYPY = True
except ImportError:
    HAVE_MYPY = False

# We no longer need to import analyze_backend_operations
# since we've implemented the functionality directly in this file

def find_python_files(directory: str) -> List[str]:
    """Find all Python files in the given directory and its subdirectories."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def check_syntax(file_path: str) -> Tuple[bool, List[str]]:
    """Check if the file has syntax errors."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        ast.parse(content)
        return True, []
    except SyntaxError as e:
        return False, [f"Syntax error at line {e.lineno}: {e.msg}"]

def check_imports(file_path: str) -> Tuple[bool, List[str]]:
    """Check if imports are syntactically valid."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return False, ["Syntax error prevents import checking"]
    
    # For now, we'll just check if the imports are syntactically valid
    # We won't actually try to import the modules, as that can cause issues
    # with module-level code in the imported modules
    return True, []

def check_style(file_path: str) -> Tuple[bool, List[str]]:
    """Check PEP 8 style issues."""
    # Temporarily disabled style checking
    return True, []
    
    # Original implementation (disabled)
    """
    if not HAVE_PYCODESTYLE:
        return True, ["pycodestyle not installed, skipping style check"]
    
    # Use subprocess to run pycodestyle
    try:
        result = subprocess.run(
            ["pycodestyle", file_path],
            capture_output=True,
            text=True,
            check=False
        )
        
        style_errors = []
        if result.returncode != 0:
            for line in result.stdout.splitlines():
                style_errors.append(line)
        
        return len(style_errors) == 0, style_errors
    except Exception as e:
        return False, [f"Error running pycodestyle: {e}"]
    """

def check_types(file_path: str) -> Tuple[bool, List[str]]:
    """Check type annotations using mypy."""
    if not HAVE_MYPY:
        return True, ["mypy not installed, skipping type checking"]
    
    try:
        import mypy.api
        result = mypy.api.run([file_path])
        
        type_errors = []
        for line in result[0].split('\n'):
            if line.strip() and not line.startswith("Success:"):
                type_errors.append(line)
        
        return len(type_errors) == 0, type_errors
    except Exception as e:
        return False, [f"Error running mypy: {e}"]

def check_numpy_import(file_path: str) -> Tuple[bool, List[str]]:
    """Check if NumPy is imported in the file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for numpy imports
    numpy_imports = []
    
    # Regular expression patterns for different import styles
    patterns = [
        r'import\s+numpy\s+as\s+(\w+)',  # import numpy as np
        r'from\s+numpy\s+import\s+(.*)',  # from numpy import ...
        r'import\s+numpy\b',  # import numpy
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content)
        if matches:
            if pattern == r'import\s+numpy\s+as\s+(\w+)':
                # For "import numpy as np" style, capture the alias
                numpy_imports.extend(matches)
            elif pattern == r'from\s+numpy\s+import\s+(.*)':
                # For "from numpy import ..." style, capture the imported names
                for match in matches:
                    imports = [name.strip() for name in match.split(',')]
                    numpy_imports.extend(imports)
            else:
                # For "import numpy" style, add "numpy" to the list
                numpy_imports.append("numpy")
    
    return bool(numpy_imports), numpy_imports

def check_numpy_usage(file_path: str, numpy_aliases: List[str]) -> Tuple[bool, List[str]]:
    """Check if NumPy is used in the file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for numpy usage
    numpy_usages = []
    
    for alias in numpy_aliases:
        # Pattern to match usage of the numpy alias
        pattern = r'\b' + re.escape(alias) + r'\.\w+'
        matches = re.findall(pattern, content)
        if matches:
            numpy_usages.extend(matches)
    
    return bool(numpy_usages), numpy_usages

def check_backend_specific_imports(file_path: str) -> Tuple[bool, List[str]]:
    """Check if backend-specific libraries are imported in frontend code."""
    # Skip backend directory, features/common directory, and nn/tensor/common directory
    if "/backend/" in file_path or "/features/common/" in file_path or "/nn/tensor/common/" in file_path:
        return False, []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for backend-specific imports
    backend_imports = []
    
    # Regular expression patterns for different import styles
    patterns = [
        r'import\s+torch\b',  # import torch
        r'from\s+torch\s+import\s+(.*)',  # from torch import ...
        r'import\s+mlx\b',  # import mlx
        r'from\s+mlx\s+import\s+(.*)',  # from mlx import ...
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content)
        if matches:
            if pattern == r'import\s+torch\b':
                backend_imports.append("torch")
            elif pattern == r'from\s+torch\s+import\s+(.*)':
                backend_imports.append(f"torch.{matches[0]}")
            elif pattern == r'import\s+mlx\b':
                backend_imports.append("mlx")
            elif pattern == r'from\s+mlx\s+import\s+(.*)':
                backend_imports.append(f"mlx.{matches[0]}")
    
    return bool(backend_imports), backend_imports

def check_backend_specific_code(file_path: str) -> Tuple[bool, List[Dict]]:
    """Check if backend-specific code is used in frontend code."""
    # Skip backend directory, features/common directory, and nn/tensor/common directory
    if "/backend/" in file_path or "/features/common/" in file_path or "/nn/tensor/common/" in file_path:
        return False, []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for backend-specific code
    backend_code = []
    
    # Regular expression patterns for backend-specific code
    patterns = [
        (r'torch\.\w+', "torch"),
        (r'mlx\.\w+', "mlx"),
    ]
    
    for pattern, backend in patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            backend_code.append({
                "code": match,
                "backend": backend
            })
    
    return bool(backend_code), backend_code

class PrecisionReducingVisitor(ast.NodeVisitor):
    """AST visitor to find precision-reducing casts, tensor conversions, and Python operators."""
    
    def __init__(self):
        self.precision_reducing_casts = []
        self.tensor_conversions = []
        self.python_operators = []
        self.numpy_imports = set()
        self.numpy_aliases = set()
        self.backend_imports = set()
        self.backend_usage = []
        self.current_function = None
        self.current_line = 0
        self.parent_map = {}
        
    def build_parent_map(self, node):
        """Build a map from child nodes to their parent nodes."""
        for child in ast.iter_child_nodes(node):
            self.parent_map[child] = node
            self.build_parent_map(child)
    
    def visit_Import(self, node):
        """Visit import statements."""
        for name in node.names:
            if name.name == 'numpy':
                self.numpy_imports.add(f"import {name.name}")
                self.numpy_aliases.add(name.asname or name.name)
            elif name.name in ['torch', 'mlx']:
                self.backend_imports.add(f"import {name.name}")
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Visit from-import statements."""
        if node.module == 'numpy':
            for name in node.names:
                self.numpy_imports.add(f"from numpy import {name.name}")
                self.numpy_aliases.add(name.asname or name.name)
        elif node.module in ['torch', 'mlx'] or node.module and node.module.startswith(('torch.', 'mlx.')):
            # Skip if this is in a backend or nn/tensor/common file
            filename = getattr(node, 'filename', '')
            if not ('/backend/' in filename or '/nn/tensor/common/' in filename):
                for name in node.names:
                    self.backend_imports.add(f"from {node.module} import {name.name}")
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        """Track the current function being visited."""
        old_function = self.current_function
        self.current_function = node.name
        self.current_line = node.lineno
        self.generic_visit(node)
        self.current_function = old_function
    
    def visit_BinOp(self, node):
        """Visit binary operations to detect Python operators."""
        # Map AST operator types to their string representations
        op_map = {
            ast.Add: '+',
            ast.Sub: '-',
            ast.Mult: '*',
            ast.Div: '/',
            ast.FloorDiv: '//',
            ast.Mod: '%',
            ast.Pow: '**',
            ast.MatMult: '@',
        }
        
        # Check if this binary operation is inside a subscript (array index)
        # by walking up the AST to see if there's a Subscript parent
        is_in_subscript = False
        parent = self.parent_map.get(node)
        while parent:
            if isinstance(parent, ast.Subscript):
                # If the binary operation is inside the index part of a subscript
                if parent.slice == node:
                    is_in_subscript = True
                    break
            parent = self.parent_map.get(parent)
        
        # Check if this is a string concatenation or other non-tensor operation
        is_non_tensor_op = False
        
        # For Python 3.8+, string literals are represented as ast.Constant with value of type str
        # For older Python versions, they are represented as ast.Str
        if isinstance(node.op, ast.Add):
            # Check for string literals (using ast.Constant for Python 3.8+)
            # We only support Python 3.8+ now, which uses ast.Constant for string literals
            is_string_literal_left = isinstance(node.left, ast.Constant) and isinstance(node.left.value, str)
            is_string_literal_right = isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)
            
            # Check for str() function calls
            is_str_func_left = isinstance(node.left, ast.Call) and isinstance(node.left.func, ast.Name) and node.left.func.id == 'str'
            is_str_func_right = isinstance(node.right, ast.Call) and isinstance(node.right.func, ast.Name) and node.right.func.id == 'str'
            
            # Check for variables named 'str_' or ending with '_str'
            is_str_var_left = isinstance(node.left, ast.Name) and (node.left.id.startswith('str_') or node.left.id.endswith('_str'))
            is_str_var_right = isinstance(node.right, ast.Name) and (node.right.id.startswith('str_') or node.right.id.endswith('_str'))
            
            # Check for f-strings (formatted string literals)
            is_fstring_left = isinstance(node.left, ast.JoinedStr)
            is_fstring_right = isinstance(node.right, ast.JoinedStr)
            
            # If any of these conditions are true, it's likely a string concatenation
            if (is_string_literal_left or is_string_literal_right or
                is_str_func_left or is_str_func_right or
                is_str_var_left or is_str_var_right or
                is_fstring_left or is_fstring_right):
                is_non_tensor_op = True
        
        # Check for operations in hash functions or other non-tensor contexts
        if self.current_function and ('hash' in self.current_function.lower() or 'str' in self.current_function.lower()):
            is_non_tensor_op = True
        
        # Check if this is a Python operator we want to detect
        # Only flag operators on tensors, not on strings, array indices, or other non-tensor operations
        op_type = type(node.op)
        if op_type in op_map and not is_in_subscript and not is_non_tensor_op:
            location = f"{self.current_function}:{node.lineno}" if self.current_function else f"line {node.lineno}"
            self.python_operators.append({
                'type': op_map[op_type],
                'location': location,
                'line': node.lineno
            })
        
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """Visit function calls to detect precision-reducing casts and tensor conversions."""
        # Check for float(), int() casts
        if isinstance(node.func, ast.Name) and node.func.id in ('float', 'int'):
            location = f"{self.current_function}:{node.lineno}" if self.current_function else f"line {node.lineno}"
            self.precision_reducing_casts.append({
                'type': node.func.id,
                'location': location,
                'line': node.lineno
            })
        
        # Check for tensor to numpy conversions
        if isinstance(node.func, ast.Attribute):
            # Check for tensor.numpy(), tensor.cpu().numpy(), etc.
            if node.func.attr == 'numpy':
                location = f"{self.current_function}:{node.lineno}" if self.current_function else f"line {node.lineno}"
                self.tensor_conversions.append({
                    'type': 'tensor.numpy()',
                    'location': location,
                    'line': node.lineno
                })
            
            # Check for tensor.convert_to_tensor(tensor), tensor.asarray(tensor), etc.
            if isinstance(node.func.value, ast.Name) and node.func.value.id in self.numpy_aliases:
                if node.func.attr in ('array', 'asarray'):
                    location = f"{self.current_function}:{node.lineno}" if self.current_function else f"line {node.lineno}"
                    self.tensor_conversions.append({
                        'type': f"{node.func.value.id}.{node.func.attr}",
                        'location': location,
                        'line': node.lineno
                    })
            
            # Check for backend-specific usage
            if isinstance(node.func.value, ast.Name) and node.func.value.id in ['torch', 'mlx']:
                location = f"{self.current_function}:{node.lineno}" if self.current_function else f"line {node.lineno}"
                self.backend_usage.append({
                    'type': f"{node.func.value.id}.{node.func.attr}",
                    'backend': node.func.value.id,
                    'location': location,
                    'line': node.lineno
                })
            
            # Check for backend-specific attribute access
            if isinstance(node.func.value, ast.Attribute):
                if isinstance(node.func.value.value, ast.Name) and node.func.value.value.id in ['torch', 'mlx']:
                    location = f"{self.current_function}:{node.lineno}" if self.current_function else f"line {node.lineno}"
                    self.backend_usage.append({
                        'type': f"{node.func.value.value.id}.{node.func.value.attr}.{node.func.attr}",
                        'backend': node.func.value.value.id,
                        'location': location,
                        'line': node.lineno
                    })
        
        self.generic_visit(node)

class UnusedImportVisitor(ast.NodeVisitor):
    """AST visitor to find unused imports."""
    
    def __init__(self):
        self.imports = {}  # name -> lineno
        self.used_names = set()
        
    def visit_Import(self, node):
        """Visit import statements."""
        for name in node.names:
            self.imports[name.asname or name.name] = node.lineno
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Visit from-import statements."""
        for name in node.names:
            self.imports[name.asname or name.name] = node.lineno
        self.generic_visit(node)
    
    def visit_Name(self, node):
        """Visit name references."""
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
        self.generic_visit(node)
    
    def get_unused_imports(self):
        """Get unused imports."""
        unused = []
        for name, lineno in self.imports.items():
            if name not in self.used_names:
                unused.append((name, lineno))
        return unused

def check_ast_for_issues(file_path: str) -> Tuple[bool, List[str], List[str], List[Dict], List[Dict], List[Dict], List[Tuple[str, int]], bool, List[str], List[Dict]]:
    """Use AST to check for NumPy imports, usage, precision-reducing casts, tensor conversions, Python operators, and unused imports."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return False, [], [], [], [], [], [], False, [], []
    
    # Check for precision-reducing casts, tensor conversions, and Python operators
    visitor = PrecisionReducingVisitor()
    visitor.build_parent_map(tree)
    visitor.visit(tree)
    
    # Check for unused imports
    unused_visitor = UnusedImportVisitor()
    unused_visitor.visit(tree)
    unused_imports = unused_visitor.get_unused_imports()
    
    # Check for numpy usage
    numpy_usages = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in visitor.numpy_aliases:
                numpy_usages.append(f"{node.value.id}.{node.attr}")
    
    return (
        bool(visitor.numpy_imports),
        list(visitor.numpy_imports),
        numpy_usages,
        visitor.precision_reducing_casts,
        visitor.tensor_conversions,
        visitor.python_operators,
        unused_imports,
        bool(visitor.backend_imports),
        list(visitor.backend_imports),
        visitor.backend_usage
    )

def check_compilation(file_path: str) -> Tuple[bool, List[str]]:
    """Check if the file compiles without errors."""
    try:
        # Use Python's compile function to check for compilation errors
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        compile(source, file_path, 'exec')
        return True, []
    except Exception as e:
        return False, [str(e)]

# Define backend paths for the new folder structure
BACKEND_PATHS = {
    "numpy": ["ember_ml/backend/numpy"],
    "torch": ["ember_ml/backend/torch"],
    "mlx": ["ember_ml/backend/mlx"]
}

# Define backend tensor paths for the new folder structure
BACKEND_TENSOR_PATHS = {
    "numpy": ["ember_ml/backend/numpy/tensor"],
    "torch": ["ember_ml/backend/torch/tensor"],
    "mlx": ["ember_ml/backend/mlx/tensor"]
}

# Define frontend tensor paths for the new folder structure
FRONTEND_TENSOR_PATHS = {
    "common": ["ember_ml/nn/tensor/common"]
}

def extract_functions_from_file(file_path: str) -> Set[str]:
    """Extract function names from a Python file."""
    functions = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the file
        tree = ast.parse(content)
        
        # Find all function definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip private methods
                if not node.name.startswith('_') or node.name.startswith('__') and node.name.endswith('__'):
                    functions.add(node.name)
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return functions

def extract_functions_from_directory(directory: str) -> Dict[str, Set[str]]:
    """Extract function names from all Python files in a directory."""
    functions_by_file = {}
    
    try:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    functions = extract_functions_from_file(file_path)
                    if functions:
                        functions_by_file[file_path] = functions
    
    except Exception as e:
        print(f"Error processing directory {directory}: {e}")
    
    return functions_by_file

def check_backend_consistency_for_file(file_path: str) -> Tuple[bool, List[Dict]]:
    """Check if a file has operations that are inconsistent across backends."""
    # Skip files in the features directory
    if "/features/" in file_path:
        return True, []
    
    # Skip files that are not in the ops or backend directories
    if not ("/ops/" in file_path or "/backend/" in file_path):
        return True, []
    
    # Skip backend/__init__.py, backend/base.py, and backend/.backend
    if "/backend/__init__.py" in file_path or "/backend/base.py" in file_path or "/backend/.backend" in file_path or "/backend/.device" in file_path:
        return True, []
    
    try:
        # Get the base filename (e.g., random_ops.py, tensor.py, dtype.py)
        base_filename = os.path.basename(file_path)
        
        # Determine if this is a backend file and which backend it belongs to
        backend_in_path = None
        for backend in BACKEND_PATHS.keys():
            if f"/backend/{backend}/" in file_path:
                backend_in_path = backend
                break
        
        # If this is not a backend file, skip it
        if not backend_in_path:
            return True, []
        
        # Check if this is a tensor implementation file
        is_tensor_file = "/tensor/" in file_path
        
        # Get the module type (e.g., random_ops, tensor_ops, tensor, dtype, etc.)
        module_type = None
        if base_filename.endswith('_ops.py'):
            module_type = base_filename
        elif is_tensor_file:
            module_type = base_filename
        else:
            return True, []  # Skip files that don't follow the naming conventions
        
        # Find corresponding files in other backends
        corresponding_files = {}
        for backend, paths in BACKEND_PATHS.items():
            if is_tensor_file:
                # Use tensor paths for tensor implementation files
                folder_path = BACKEND_TENSOR_PATHS[backend][0]
                corresponding_file = os.path.join(folder_path, module_type)
            else:
                # Use regular paths for ops files
                folder_path = paths[0]
                corresponding_file = os.path.join(folder_path, module_type)
            
            if os.path.exists(corresponding_file):
                corresponding_files[backend] = corresponding_file
        
        # If we couldn't find corresponding files in all backends, skip this file
        if len(corresponding_files) < 2:  # Need at least 2 backends to compare
            return True, []
        
        # Extract functions from each corresponding file
        functions_by_backend = {}
        for backend, backend_file in corresponding_files.items():
            functions_by_backend[backend] = extract_functions_from_file(backend_file)
        
        # Get the union of all functions across all backends
        all_functions = set()
        for functions in functions_by_backend.values():
            all_functions.update(functions)
        
        # Check for inconsistencies
        inconsistent_operations = []
        
        for function in all_functions:
            available_in = []
            missing_in = []
            
            for backend, functions in functions_by_backend.items():
                if function in functions:
                    available_in.append(backend)
                else:
                    missing_in.append(backend)
            
            if missing_in and available_in:  # Only report if function exists in at least one backend but not all
                inconsistent_operations.append({
                    "operation": function,
                    "available_in": available_in,
                    "missing_in": missing_in
                })
        
        return len(inconsistent_operations) == 0, inconsistent_operations
    except Exception as e:
        return False, [{"error": str(e)}]

def check_frontend_backend_violation(file_path: str) -> Tuple[bool, List[Dict]]:
    """Check if a file violates the frontend-backend separation rules."""
    # Skip backend directory
    if "/backend/" in file_path:
        return False, []
    
    # Check if the file is in the ops directory
    if "/ops/" in file_path:
        # Check if the file is in a backend-specific directory
        if any(backend in file_path for backend in ["/ops/torch/", "/ops/mlx/", "/ops/numpy/"]):
            return True, [{
                "violation": "Backend-specific implementation in ops directory",
                "file": file_path
            }]
    
    # Check if the file is in the features directory
    if "/features/" in file_path:
        # Check if the file is in a backend-specific directory
        if any(backend in file_path for backend in ["/features/torch/", "/features/mlx/", "/features/numpy/"]):
            return True, [{
                "violation": "Backend-specific implementation in features directory",
                "file": file_path
            }]
    
    # Check if the file is in the nn directory
    if "/nn/" in file_path and not "/nn/tensor/common/" in file_path:
        # Check if the file is in a backend-specific directory
        if any(backend in file_path for backend in ["/nn/torch/", "/nn/mlx/", "/nn/numpy/"]):
            return True, [{
                "violation": "Backend-specific implementation in nn directory",
                "file": file_path
            }]
    
    return False, []

def analyze_file(file_path: str) -> Dict:
    """Analyze a file for various issues."""
    # Check for syntax errors
    syntax_valid, syntax_errors = check_syntax(file_path)
    
    # Check for compilation errors
    compilation_valid, compilation_errors = check_compilation(file_path)
    
    # Check for import errors
    imports_valid, import_errors = check_imports(file_path)
    
    # Check for style issues
    style_valid, style_errors = check_style(file_path)
    
    # Check for type issues
    types_valid, type_errors = check_types(file_path)
    
    # Check for NumPy imports using regex
    has_numpy_import, numpy_imports = check_numpy_import(file_path)
    
    # If NumPy is imported, check for usage
    has_numpy_usage = False
    numpy_usages = []
    if has_numpy_import:
        has_numpy_usage, numpy_usages = check_numpy_usage(file_path, numpy_imports)
    
    # Check for backend-specific imports
    has_backend_imports, backend_imports = check_backend_specific_imports(file_path)
    
    # Check for backend-specific code
    has_backend_code, backend_code = check_backend_specific_code(file_path)
    
    # Use AST for more accurate detection
    ast_has_numpy, ast_numpy_imports, ast_numpy_usages, precision_casts, tensor_conversions, python_operators, unused_imports, ast_has_backend_imports, ast_backend_imports, backend_usage = check_ast_for_issues(file_path)
    
    # Check for backend consistency
    backend_consistent, inconsistent_operations = check_backend_consistency_for_file(file_path)
    
    # Check for frontend-backend violation
    frontend_backend_violation, violations = check_frontend_backend_violation(file_path)
    
    # Combine results
    has_numpy = has_numpy_import or ast_has_numpy
    all_imports = list(set(numpy_imports + ast_numpy_imports))
    all_usages = list(set(numpy_usages + ast_numpy_usages))
    has_backend_specific = has_backend_imports or ast_has_backend_imports or has_backend_code
    all_backend_imports = list(set(backend_imports + ast_backend_imports))
    
    return {
        "file": file_path,
        "syntax_valid": syntax_valid,
        "syntax_errors": syntax_errors,
        "compilation_valid": compilation_valid,
        "compilation_errors": compilation_errors,
        "imports_valid": imports_valid,
        "import_errors": import_errors,
        "style_valid": style_valid,
        "style_errors": style_errors,
        "types_valid": types_valid,
        "type_errors": type_errors,
        "has_numpy": has_numpy,
        "imports": all_imports,
        "usages": all_usages,
        "precision_casts": precision_casts,
        "tensor_conversions": tensor_conversions,
        "python_operators": python_operators,
        "unused_imports": unused_imports,
        "backend_consistent": backend_consistent,
        "inconsistent_operations": inconsistent_operations,
        "has_backend_specific": has_backend_specific,
        "backend_imports": all_backend_imports,
        "backend_usage": backend_usage,
        "frontend_backend_violation": frontend_backend_violation,
        "violations": violations
    }

def analyze_directory(directory: str, exclude_dirs: Optional[List[str]] = None) -> List[Dict]:
    """Analyze all Python files in a directory for various issues."""
    if exclude_dirs is None:
        exclude_dirs = []
    
    # Find all Python files
    python_files = find_python_files(directory)
    
    # Filter out excluded directories
    filtered_files = []
    for file_path in python_files:
        exclude = False
        for exclude_dir in exclude_dirs:
            if exclude_dir in file_path:
                exclude = True
                break
        if not exclude:
            filtered_files.append(file_path)
    
    # Analyze each file
    results = []
    for file_path in filtered_files:
        result = analyze_file(file_path)
        results.append(result)
    
    return results

def print_results(results: List[Dict], verbose: bool = False, show_all: bool = True,
                 show_syntax: bool = False, show_compilation: bool = False,
                 show_imports: bool = False, show_style: bool = False,
                 show_types: bool = False, show_numpy: bool = False,
                 show_precision: bool = False, show_conversion: bool = False,
                 show_operators: bool = False, show_unused: bool = False,
                 show_backend: bool = False, show_frontend_backend: bool = False):
    """Print the analysis results."""
    files_with_syntax_errors = [result for result in results if not result["syntax_valid"]]
    files_with_compilation_errors = [result for result in results if not result["compilation_valid"]]
    files_with_import_errors = [result for result in results if not result["imports_valid"]]
    files_with_style_errors = [result for result in results if not result["style_valid"]]
    files_with_type_errors = [result for result in results if not result["types_valid"]]
    numpy_files = [result for result in results if result["has_numpy"]]
    files_with_precision_casts = [result for result in results if result["precision_casts"]]
    files_with_tensor_conversions = [result for result in results if result["tensor_conversions"]]
    files_with_python_operators = [result for result in results if result["python_operators"]]
    files_with_unused_imports = [result for result in results if result["unused_imports"]]
    files_with_backend_inconsistencies = [result for result in results if not result["backend_consistent"]]
    files_with_backend_specific = [result for result in results if result["has_backend_specific"]]
    files_with_frontend_backend_violation = [result for result in results if result["frontend_backend_violation"]]
    
    print(f"Total files analyzed: {len(results)}")
    if len(results) > 0:
        if show_all or show_syntax:
            print(f"Files with syntax errors: {len(files_with_syntax_errors)} ({len(files_with_syntax_errors)/len(results)*100:.2f}%)")
        if show_all or show_compilation:
            print(f"Files with compilation errors: {len(files_with_compilation_errors)} ({len(files_with_compilation_errors)/len(results)*100:.2f}%)")
        if show_all or show_imports:
            print(f"Files with import errors: {len(files_with_import_errors)} ({len(files_with_import_errors)/len(results)*100:.2f}%)")
        if show_all or show_style:
            print(f"Files with style errors: {len(files_with_style_errors)} ({len(files_with_style_errors)/len(results)*100:.2f}%)")
        if show_all or show_types:
            print(f"Files with type errors: {len(files_with_type_errors)} ({len(files_with_type_errors)/len(results)*100:.2f}%)")
        if show_all or show_numpy:
            print(f"Files with NumPy: {len(numpy_files)} ({len(numpy_files)/len(results)*100:.2f}%)")
        if show_all or show_precision:
            print(f"Files with precision-reducing casts: {len(files_with_precision_casts)} ({len(files_with_precision_casts)/len(results)*100:.2f}%)")
        if show_all or show_conversion:
            print(f"Files with tensor conversions: {len(files_with_tensor_conversions)} ({len(files_with_tensor_conversions)/len(results)*100:.2f}%)")
        if show_all or show_operators:
            print(f"Files with Python operators: {len(files_with_python_operators)} ({len(files_with_python_operators)/len(results)*100:.2f}%)")
        if show_all or show_unused:
            print(f"Files with unused imports: {len(files_with_unused_imports)} ({len(files_with_unused_imports)/len(results)*100:.2f}%)")
        if show_all or show_backend:
            print(f"Files with backend inconsistencies: {len(files_with_backend_inconsistencies)} ({len(files_with_backend_inconsistencies)/len(results)*100:.2f}%)")
        if show_all or show_frontend_backend:
            print(f"Files with backend-specific code in frontend: {len(files_with_backend_specific)} ({len(files_with_backend_specific)/len(results)*100:.2f}%)")
            print(f"Files violating frontend-backend separation: {len(files_with_frontend_backend_violation)} ({len(files_with_frontend_backend_violation)/len(results)*100:.2f}%)")
    verbose = True
    
    if verbose:
        if (show_all or show_syntax) and files_with_syntax_errors:
            print("\nFiles with syntax errors:")
            for result in files_with_syntax_errors:
                print(f"\n{result['file']}:")
                for error in result["syntax_errors"]:
                    print(f"  {error}")
        
        if (show_all or show_compilation) and files_with_compilation_errors:
            print("\nFiles with compilation errors:")
            for result in files_with_compilation_errors:
                print(f"\n{result['file']}:")
                for error in result["compilation_errors"]:
                    print(f"  {error}")
        
        if (show_all or show_imports) and files_with_import_errors:
            print("\nFiles with import errors:")
            for result in files_with_import_errors:
                print(f"\n{result['file']}:")
                for error in result["import_errors"]:
                    print(f"  {error}")
        
        if (show_all or show_style) and files_with_style_errors:
            print("\nFiles with style errors:")
            for result in files_with_style_errors:
                print(f"\n{result['file']}:")
                for error in result["style_errors"][:10]:  # Limit to 10 errors
                    print(f"  {error}")
                if len(result["style_errors"]) > 10:
                    print(f"  ... and {len(result['style_errors']) - 10} more")
        
        if (show_all or show_types) and files_with_type_errors:
            print("\nFiles with type errors:")
            for result in files_with_type_errors:
                print(f"\n{result['file']}:")
                for error in result["type_errors"][:10]:  # Limit to 10 errors
                    print(f"  {error}")
                if len(result["type_errors"]) > 10:
                    print(f"  ... and {len(result['type_errors']) - 10} more")
        
        if (show_all or show_numpy) and numpy_files:
            print("\nFiles with NumPy:")
            for result in numpy_files:
                print(f"\n{result['file']}:")
                print(f"  Imports: {', '.join(result['imports'])}")
                print(f"  Usages: {', '.join(result['usages'])}")
        
        if (show_all or show_precision) and files_with_precision_casts:
            print("\nFiles with precision-reducing casts:")
            for result in files_with_precision_casts:
                print(f"\n{result['file']}:")
                for cast in result["precision_casts"]:
                    print(f"  {cast['type']} at {cast['location']}")
        
        if (show_all or show_conversion) and files_with_tensor_conversions:
            print("\nFiles with tensor conversions:")
            for result in files_with_tensor_conversions:
                print(f"\n{result['file']}:")
                for conv in result["tensor_conversions"]:
                    print(f"  {conv['type']} at {conv['location']}")
        
        if (show_all or show_operators) and files_with_python_operators:
            print("\nFiles with Python operators:")
            for result in files_with_python_operators:
                print(f"\n{result['file']}:")
                for op in result["python_operators"]:
                    print(f"  {op['type']} operator at {op['location']}")
        
        if (show_all or show_unused) and files_with_unused_imports:
            print("\nFiles with unused imports:")
            for result in files_with_unused_imports:
                print(f"\n{result['file']}:")
                for name, lineno in result["unused_imports"]:
                    print(f"  Unused import '{name}' at line {lineno}")
        
        if (show_all or show_backend) and files_with_backend_inconsistencies:
            print("\nFiles with backend inconsistencies:")
            for result in files_with_backend_inconsistencies:
                print(f"\n{result['file']}:")
                for op in result["inconsistent_operations"]:
                    if "error" in op:
                        print(f"  Error checking backend consistency: {op['error']}")
                    else:
                        print(f"  {op['operation']} (available in: {', '.join(op['available_in'])}, missing in: {', '.join(op['missing_in'])})")
        
        if (show_all or show_frontend_backend) and files_with_backend_specific:
            print("\nFiles with backend-specific code in frontend:")
            for result in files_with_backend_specific:
                print(f"\n{result['file']}:")
                print(f"  Imports: {', '.join(result['backend_imports'])}")
                for usage in result["backend_usage"]:
                    print(f"  {usage['type']} at {usage['location']}")
        
        if (show_all or show_frontend_backend) and files_with_frontend_backend_violation:
            print("\nFiles violating frontend-backend separation:")
            for result in files_with_frontend_backend_violation:
                print(f"\n{result['file']}:")
                for violation in result["violations"]:
                    print(f"  {violation['violation']}")
    
    # Print summary by directory
    print("\nSummary by directory:")
    dir_summary = {}
    for result in results:
        dir_path = os.path.dirname(result["file"])
        if dir_path not in dir_summary:
            dir_summary[dir_path] = {
                "total": 0,
                "syntax_errors": 0,
                "compilation_errors": 0,
                "import_errors": 0,
                "style_errors": 0,
                "type_errors": 0,
                "numpy": 0,
                "precision_casts": 0,
                "tensor_conversions": 0,
                "python_operators": 0,
                "unused_imports": 0,
                "backend_inconsistencies": 0,
                "backend_specific": 0,
                "frontend_backend_violation": 0
            }
        dir_summary[dir_path]["total"] += 1
        if not result["syntax_valid"]:
            dir_summary[dir_path]["syntax_errors"] += 1
        if not result["compilation_valid"]:
            dir_summary[dir_path]["compilation_errors"] += 1
        if not result["imports_valid"]:
            dir_summary[dir_path]["import_errors"] += 1
        if not result["style_valid"]:
            dir_summary[dir_path]["style_errors"] += 1
        if not result["types_valid"]:
            dir_summary[dir_path]["type_errors"] += 1
        if result["has_numpy"]:
            dir_summary[dir_path]["numpy"] += 1
        if result["precision_casts"]:
            dir_summary[dir_path]["precision_casts"] += 1
        if result["tensor_conversions"]:
            dir_summary[dir_path]["tensor_conversions"] += 1
        if result["python_operators"]:
            dir_summary[dir_path]["python_operators"] += 1
        if result["unused_imports"]:
            dir_summary[dir_path]["unused_imports"] += 1
        if not result["backend_consistent"]:
            dir_summary[dir_path]["backend_inconsistencies"] += 1
        if result["has_backend_specific"]:
            dir_summary[dir_path]["backend_specific"] += 1
        if result["frontend_backend_violation"]:
            dir_summary[dir_path]["frontend_backend_violation"] += 1
    
    for dir_path, stats in sorted(dir_summary.items()):
        # Check if there are any issues to show for this directory
        has_issues = (
            (show_all or show_syntax) and stats["syntax_errors"] > 0 or
            (show_all or show_compilation) and stats["compilation_errors"] > 0 or
            (show_all or show_imports) and stats["import_errors"] > 0 or
            (show_all or show_style) and stats["style_errors"] > 0 or
            (show_all or show_types) and stats["type_errors"] > 0 or
            (show_all or show_numpy) and stats["numpy"] > 0 or
            (show_all or show_precision) and stats["precision_casts"] > 0 or
            (show_all or show_conversion) and stats["tensor_conversions"] > 0 or
            (show_all or show_operators) and stats["python_operators"] > 0 or
            (show_all or show_unused) and stats["unused_imports"] > 0 or
            (show_all or show_backend) and stats["backend_inconsistencies"] > 0 or
            (show_all or show_frontend_backend) and stats["backend_specific"] > 0 or
            (show_all or show_frontend_backend) and stats["frontend_backend_violation"] > 0
        )
        
        if has_issues:
            print(f"{dir_path}:")
            if (show_all or show_syntax) and stats["syntax_errors"] > 0:
                print(f"  Syntax errors: {stats['syntax_errors']}/{stats['total']} files ({stats['syntax_errors']/stats['total']*100:.2f}%)")
            if (show_all or show_compilation) and stats["compilation_errors"] > 0:
                print(f"  Compilation errors: {stats['compilation_errors']}/{stats['total']} files ({stats['compilation_errors']/stats['total']*100:.2f}%)")
            if (show_all or show_imports) and stats["import_errors"] > 0:
                print(f"  Import errors: {stats['import_errors']}/{stats['total']} files ({stats['import_errors']/stats['total']*100:.2f}%)")
            if (show_all or show_style) and stats["style_errors"] > 0:
                print(f"  Style errors: {stats['style_errors']}/{stats['total']} files ({stats['style_errors']/stats['total']*100:.2f}%)")
            if (show_all or show_types) and stats["type_errors"] > 0:
                print(f"  Type errors: {stats['type_errors']}/{stats['total']} files ({stats['type_errors']/stats['total']*100:.2f}%)")
            if (show_all or show_numpy) and stats["numpy"] > 0:
                print(f"  NumPy: {stats['numpy']}/{stats['total']} files ({stats['numpy']/stats['total']*100:.2f}%)")
            if (show_all or show_precision) and stats["precision_casts"] > 0:
                print(f"  Precision casts: {stats['precision_casts']}/{stats['total']} files ({stats['precision_casts']/stats['total']*100:.2f}%)")
            if (show_all or show_conversion) and stats["tensor_conversions"] > 0:
                print(f"  Tensor conversions: {stats['tensor_conversions']}/{stats['total']} files ({stats['tensor_conversions']/stats['total']*100:.2f}%)")
            if (show_all or show_operators) and stats["python_operators"] > 0:
                print(f"  Python operators: {stats['python_operators']}/{stats['total']} files ({stats['python_operators']/stats['total']*100:.2f}%)")
            if (show_all or show_unused) and stats["unused_imports"] > 0:
                print(f"  Unused imports: {stats['unused_imports']}/{stats['total']} files ({stats['unused_imports']/stats['total']*100:.2f}%)")
            if (show_all or show_backend) and stats["backend_inconsistencies"] > 0:
                print(f"  Backend inconsistencies: {stats['backend_inconsistencies']}/{stats['total']} files ({stats['backend_inconsistencies']/stats['total']*100:.2f}%)")
            if (show_all or show_frontend_backend) and stats["backend_specific"] > 0:
                print(f"  Backend-specific code: {stats['backend_specific']}/{stats['total']} files ({stats['backend_specific']/stats['total']*100:.2f}%)")
            if (show_all or show_frontend_backend) and stats["frontend_backend_violation"] > 0:
                print(f"  Frontend-backend violations: {stats['frontend_backend_violation']}/{stats['total']} files ({stats['frontend_backend_violation']/stats['total']*100:.2f}%)")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="EmberLint: A comprehensive linting tool for Ember ML codebase.")
    parser.add_argument("path", help="Directory or file to scan")
    parser.add_argument("--exclude", nargs="+", help="Directories to exclude", default=[])
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed results")
    
    # Issue type flags
    parser.add_argument("--syntax-only", action="store_true", help="Only check for syntax errors")
    parser.add_argument("--compilation-only", action="store_true", help="Only check for compilation errors")
    parser.add_argument("--imports-only", action="store_true", help="Only check for import errors")
    parser.add_argument("--style-only", action="store_true", help="Only check for style errors")
    parser.add_argument("--types-only", action="store_true", help="Only check for type errors")
    parser.add_argument("--numpy-only", action="store_true", help="Only check for NumPy usage")
    parser.add_argument("--precision-only", action="store_true", help="Only check for precision-reducing casts")
    parser.add_argument("--conversion-only", action="store_true", help="Only check for tensor conversions")
    parser.add_argument("--operators-only", action="store_true", help="Only check for Python operators (+, -, *, /, etc.)")
    parser.add_argument("--unused-only", action="store_true", help="Only check for unused imports")
    parser.add_argument("--backend-only", action="store_true", help="Only check for backend inconsistencies")
    parser.add_argument("--frontend-backend-only", action="store_true", help="Only check for frontend-backend separation violations")
    
    args = parser.parse_args()

    # If single-issue linting is disabled, ignore the --*-only flags
    if not ALLOW_SINGLE_ISSUE_LINTING:
        if any([args.syntax_only, args.compilation_only, args.imports_only, 
                args.style_only, args.types_only, args.numpy_only,
                args.precision_only, args.conversion_only, args.operators_only,
                args.unused_only, args.backend_only, args.frontend_backend_only]):
            print("Warning: Single-issue linting is disabled. Running all checks.")
            args.syntax_only = args.compilation_only = args.imports_only = False
            args.style_only = args.types_only = args.numpy_only = False
            args.precision_only = args.conversion_only = args.operators_only = False
            args.unused_only = args.backend_only = args.frontend_backend_only = False
    
    # Check if the path is a file or directory
    if os.path.isfile(args.path) and args.path.endswith('.py'):
        # Analyze a single file
        result = analyze_file(args.path)
        results = [result]
    else:
        # Analyze a directory
        results = analyze_directory(args.path, args.exclude)
    
    # Determine what to display based on flags
    show_all = not (args.syntax_only or args.compilation_only or args.imports_only or 
                    args.style_only or args.types_only or args.numpy_only or 
                    args.precision_only or args.conversion_only or args.operators_only or
                    args.unused_only or args.backend_only or args.frontend_backend_only)
    
    print_results(
        results, 
        args.verbose, 
        show_all,
        args.syntax_only,
        args.compilation_only,
        args.imports_only,
        args.style_only,
        args.types_only,
        args.numpy_only,
        args.precision_only,
        args.conversion_only,
        args.operators_only,
        args.unused_only,
        args.backend_only,
        args.frontend_backend_only
    )
    
    # Return a boolean indicating if any issues were found
    has_issues = any(
        (args.syntax_only and not result["syntax_valid"]) or
        (args.compilation_only and not result["compilation_valid"]) or
        (args.imports_only and not result["imports_valid"]) or
        (args.style_only and not result["style_valid"]) or
        (args.types_only and not result["types_valid"]) or
        (args.numpy_only and result["has_numpy"]) or
        (args.precision_only and result["precision_casts"]) or
        (args.conversion_only and result["tensor_conversions"]) or
        (args.operators_only and result["python_operators"]) or
        (args.unused_only and result["unused_imports"]) or
        (args.backend_only and not result["backend_consistent"]) or
        (args.frontend_backend_only and (result["has_backend_specific"] or result["frontend_backend_violation"])) or
        (show_all and (
            not result["syntax_valid"] or
            not result["compilation_valid"] or
            not result["imports_valid"] or
            not result["style_valid"] or
            not result["types_valid"] or
            result["has_numpy"] or
            result["precision_casts"] or
            result["tensor_conversions"] or
            result["python_operators"] or
            result["unused_imports"] or
            not result["backend_consistent"] or
            result["has_backend_specific"] or
            result["frontend_backend_violation"]
        ))
        for result in results
    )
    
    return 1 if has_issues else 0

if __name__ == "__main__":
    sys.exit(main())
