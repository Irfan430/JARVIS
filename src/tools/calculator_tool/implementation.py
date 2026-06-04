"""
Calculator Tool Implementation
===============================

Safe mathematical expression evaluation using AST parsing.
Supports basic arithmetic, advanced math functions, and constants.
No eval() usage — all expressions are parsed and validated safely.
"""

from __future__ import annotations

import ast
import cmath
import math
import operator
from typing import Any, Dict

from src.tools.base import BaseTool, ToolError


# Allowed binary operators
BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Allowed unary operators
UNARY_OPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Allowed math functions
MATH_FUNCTIONS = {
    "sqrt": math.sqrt,
    "cbrt": lambda x: x ** (1 / 3),
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "min": min,
    "max": max,
    "pow": pow,
}

# Allowed constants
MATH_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
    "nan": math.nan,
}


class _SafeEvaluator(ast.NodeVisitor):
    """
    Safe AST-based math expression evaluator.

    Walks the Python AST and evaluates only whitelisted operations,
    functions, and constants. Prevents arbitrary code execution.
    """

    def __init__(self):
        self.allowed_names = {**MATH_CONSTANTS}

    def evaluate(self, expression: str) -> float:
        """Parse and evaluate a math expression safely."""
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as e:
            raise ToolError(f"Invalid expression syntax: {e}")

        result = self.visit(tree.body)
        return result

    def visit_Expression(self, node: ast.Expression) -> float:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> float:
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ToolError(f"Unsupported constant type: {type(node.value).__name__}")

    def visit_Num(self, node: ast.Num) -> float:
        """Legacy Python < 3.8 numeric node support."""
        return node.n

    def visit_BinOp(self, node: ast.BinOp) -> float:
        op_type = type(node.op)
        if op_type not in BIN_OPS:
            raise ToolError(f"Unsupported binary operator: {op_type.__name__}")

        left = self.visit(node.left)
        right = self.visit(node.right)

        # Prevent division by zero
        if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ToolError("Division by zero")

        # Prevent extremely large exponents
        if op_type == ast.Pow and isinstance(right, (int, float)):
            if abs(right) > 1000:
                raise ToolError(f"Exponent too large: {right}")
            if left == 0 and right < 0:
                raise ToolError("Zero cannot be raised to a negative power")

        return BIN_OPS[op_type](left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        op_type = type(node.op)
        if op_type not in UNARY_OPS:
            raise ToolError(f"Unsupported unary operator: {op_type.__name__}")
        operand = self.visit(node.operand)
        return UNARY_OPS[op_type](operand)

    def visit_Call(self, node: ast.Call) -> float:
        # Get function name
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            raise ToolError("Attribute access is not allowed in math expressions")
        else:
            raise ToolError("Unsupported function call type")

        if func_name not in MATH_FUNCTIONS:
            raise ToolError(
                f"Unknown function: '{func_name}'. "
                f"Allowed: {sorted(MATH_FUNCTIONS.keys())}"
            )

        # Evaluate arguments
        args = [self.visit(arg) for arg in node.args]

        # Check argument count
        func = MATH_FUNCTIONS[func_name]
        try:
            result = func(*args)
        except TypeError as e:
            raise ToolError(f"Wrong arguments for {func_name}: {e}")
        except ValueError as e:
            raise ToolError(f"Math error in {func_name}: {e}")

        return result

    def visit_Name(self, node: ast.Name) -> float:
        if node.id in self.allowed_names:
            return self.allowed_names[node.id]
        raise ToolError(
            f"Unknown variable: '{node.id}'. "
            f"Available constants: {sorted(MATH_CONSTANTS.keys())}"
        )

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        raise ToolError("Subscript/slicing is not allowed in math expressions")

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        raise ToolError("Attribute access is not allowed in math expressions")

    def generic_visit(self, node: ast.AST) -> Any:
        raise ToolError(
            f"Unsupported expression type: {type(node).__name__}. "
            "Only math expressions are allowed."
        )


class CalculatorTool(BaseTool):
    """
    Evaluate mathematical expressions safely.

    Supports basic arithmetic (+, -, *, /, //, %, **) and advanced
    math functions (sin, cos, sqrt, log, etc.) with constants (pi, e).
    Uses AST parsing — no eval() or exec().
    """

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Evaluate mathematical expressions safely. Supports basic arithmetic "
            "(+, -, *, /, //, %, **) and advanced functions (sin, cos, sqrt, log, "
            "factorial, etc.) with constants (pi, e, tau). No code execution."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "Math expression to evaluate. Examples: '2 + 2', "
                        "'sqrt(144)', 'sin(pi/2)', 'log(100, 10)', '2**10'"
                    ),
                },
            },
            "required": ["expression"],
        }

    def validate_input(self, **kwargs) -> None:
        """Validate the expression parameter."""
        super().validate_input(**kwargs)
        expr = kwargs.get("expression", "")
        if not expr or not expr.strip():
            raise ToolError("Expression cannot be empty", tool_name=self.name)
        if len(expr) > 500:
            raise ToolError(
                f"Expression too long ({len(expr)} chars). Maximum: 500 characters.",
                tool_name=self.name,
            )

    async def _execute(self, **kwargs) -> Dict[str, Any]:
        """
        Evaluate a mathematical expression.

        Args:
            expression: The math expression string.

        Returns:
            Dict with 'expression', 'result', and 'result_type'.
        """
        expression = kwargs["expression"].strip()

        evaluator = _SafeEvaluator()
        result = evaluator.evaluate(expression)

        # Determine result type
        if isinstance(result, complex):
            result_type = "complex"
            result_value = {"real": result.real, "imag": result.imag}
        elif isinstance(result, float):
            result_type = "float"
            # Avoid -0.0
            result_value = 0.0 if result == 0.0 else result
        elif isinstance(result, int):
            result_type = "integer"
            result_value = result
        else:
            result_type = type(result).__name__
            result_value = result

        return {
            "expression": expression,
            "result": result_value,
            "result_type": result_type,
        }
