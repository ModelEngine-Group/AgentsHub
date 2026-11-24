import ast

def analyze_python(code: str) -> dict:
    """只解析文本，不执行代码。最多接受 100KB UTF-8。"""
    if len(code.encode("utf-8")) > 100_000:
        return {"ok": False, "error": "代码超过100KB限制"}
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError) as exc:
        return {"ok": False, "error": str(exc), "line": getattr(exc, "lineno", None)}
    issues = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for default in node.args.defaults + [x for x in node.args.kw_defaults if x is not None]:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    issues.append({"line": node.lineno, "rule": "mutable-default", "message": "可变默认参数跨调用共享，请考虑 None"})
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            issues.append({"line": node.lineno, "rule": "bare-except", "message": "请捕获具体异常"})
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
            issues.append({"line": node.lineno, "rule": "dynamic-execution", "message": "动态执行需检查输入可信性"})
    return {"ok": True, "issues": issues, "executed": False, "scope": "语法和有限AST规则，不保证逻辑正确"}
