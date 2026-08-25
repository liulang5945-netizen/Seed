"""R6 一次性脚本：为 except 块内无 from 的 raise 补异常链（B904）。

用法: python scripts/fix_b904.py  （仓库根目录运行）
基于 AST 精确定位，仅在 except ... as VAR 处理器内的裸表达式 raise 末尾追加 `from VAR`。
"""

import ast
import sys
from pathlib import Path

TARGETS = [
    "api/routes_neuroplex.py",
    "api/routes_rag.py",
    "api/routes_chat.py",
    "api/routes_agent_workspace.py",
    "api/training/datasets.py",
    "api/routes_update.py",
    "api/routes_multimodal.py",
    "api/training/recommend.py",
    "api/routes_plugins.py",
    "api/routes_workflows.py",
    "api/training/resume.py",
    "api/routes_life.py",
    "api/routes_settings.py",
    "scripts/training/utils.py",
]


def collect_fixes(source: str):
    tree = ast.parse(source)
    fixes = []  # (end_lineno, exc_var)

    class _Visitor(ast.NodeVisitor):
        def __init__(self):
            self.handler_stack: list[str | None] = []

        def visit_ExceptHandler(self, node: ast.ExceptHandler):
            self.handler_stack.append(node.name)
            for child in ast.iter_child_nodes(node):
                self.visit(child)
            self.handler_stack.pop()

        def visit_Raise(self, node: ast.Raise):
            if node.exc is not None and node.cause is None:
                # 取最近的具名 except 处理器变量（未具名的层跳过）
                for var in reversed(self.handler_stack):
                    if var is not None:
                        fixes.append((node.end_lineno, var))
                        break
            self.generic_visit(node)

    _Visitor().visit(tree)
    return fixes


def main() -> int:
    total = 0
    for rel in TARGETS:
        path = Path(rel)
        if not path.exists():
            print(f"skip (missing): {rel}")
            continue
        source = path.read_text(encoding="utf-8")
        fixes = collect_fixes(source)
        if not fixes:
            continue
        lines = source.split("\n")
        # 从后往前改，避免行号漂移
        for lineno, var in sorted(fixes, reverse=True):
            idx = lineno - 1
            line = lines[idx]
            stripped = line.rstrip()
            # 处理行尾注释
            if "  #" in stripped:
                code_part, comment_part = stripped.split("  #", 1)
                lines[idx] = f"{code_part.rstrip()} from {var}  #{comment_part}"
            else:
                lines[idx] = f"{stripped} from {var}"
        path.write_text("\n".join(lines), encoding="utf-8")
        total += len(fixes)
        print(f"{rel}: +{len(fixes)} from-chains")
    print(f"TOTAL: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
