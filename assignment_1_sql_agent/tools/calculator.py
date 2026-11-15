"""Extra tool (bonus): a safe arithmetic calculator.

The model must not do arithmetic itself; percentages, differences and
growth are computed here. Only numbers, + - * / ** % and parentheses.
"""
import ast
import operator

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    raise ValueError("Only numbers and + - * / ** % ( ) are allowed")


def calculator_tool(expression):
    """Return {"result": number} or raise ValueError with a clear message."""
    text = str(expression or "").replace(",", "").strip()
    if not text or len(text) > 200:
        raise ValueError("Empty or too long expression")
    try:
        value = _eval(ast.parse(text, mode="eval"))
    except ZeroDivisionError as exc:
        raise ValueError("Division by zero") from exc
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression: {exc.msg}") from exc
    return {"result": round(value, 6) if isinstance(value, float) else value}
