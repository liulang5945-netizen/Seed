# Plans 目录整理结果概览

执行时间：2026-08-24
操作性质：纯移动 / 重命名，无内容删除（可逆）

## 已完成的 7 项操作

| 原路径 | 目标路径 | 动作 |
|---|---|---|
| `archive/architecture/TAIJI_TRANSFORMER_SHELL_AUDIT_20260822.md` | `archive/audits/TAIJI_TRANSFORMER_SHELL_AUDIT_20260822.md` | 移动（审计归审计） |
| `active/NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md` | `archive/authored/NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md` | 移动（Legacy 底稿） |
| `active/BOOTSTRAP_CRITERIA.md` | `archive/authored/BOOTSTRAP_CRITERIA.md` | 移动（Legacy 底稿） |
| `active/DESIGN_PRINCIPLES.md` | `archive/authored/DESIGN_PRINCIPLES.md` | 移动（Legacy 底稿） |
| `active/TAIJI_VS_HUMAN_BRAIN_COMPARISON.md` | `archive/authored/TAIJI_VS_HUMAN_BRAIN_COMPARISON.md` | 移动（Legacy 底稿） |
| `archive/implementation/2026-07-27-side-channels-implementation.md` | `archive/implementation/SIDE_CHANNELS_IMPLEMENTATION.md` | 重命名（去日期前缀） |
| `archive/implementation/H1-H8-mechanism-fixes.md` | `archive/implementation/H1_H8_MECHANISM_FIXES.md` | 重命名（连字符→下划线） |

另：`archive/architecture/` 下的其余 4 份早期/废止设计稿统一移入新建的 `archive/architecture_design/`，原空目录已删除。

## 同步更新的链接（零断链）

- `plans/README.md` ×4：`NEUROPLEX`、`BOOTSTRAP` 链接改指 `archive/authored/`；`TAIJI0` 改指 `archive/architecture_design/`；`TAIJI_TRANSFORMER_SHELL_AUDIT` 改指 `archive/audits/`。
- `plans/active/BIO_INSPIRED_ARCHITECTURE_PLAN.md`：`BOOTSTRAP_CRITERIA` 链接改指 `../archive/authored/BOOTSTRAP_CRITERIA.md`。
- `plans/archive/architecture_design/COMPREHENSIVE_NEURON_ARCHITECTURE_PLAN.md`：`H1-H8-mechanism-fixes.md` 引用改指 `H1_H8_MECHANISM_FIXES.md`。

## 整理后结构

```
plans/
├── README.md
├── active/        # 5 份权威架构 + 2 份现行执行（roadmap / frontend checklist）
└── archive/
    ├── audits/            # 3 份审计（含 Transformer 壳审计）
    ├── architecture_design/ # 4 份早期/废止架构设计
    ├── authored/          # 4 份 Legacy 底稿 / 事实基线 / 对比
    ├── history/           # 3 份历史对话/事件/机制实验
    ├── reference/         # 训练参考
    └── implementation/    # 3 份实施记录（命名统一）
```

## 校验

- 用 grep 全量扫描旧路径/旧文件名引用，除本次的规划说明文档外，无任何文档残留失效链接。
- 相关 `.md` 文件均存在，相对链接目标可解析。

## 备注

- `plans/REORGANIZATION_PLAN.md`（本整理的方案草稿）已迁移至 `.workbuddy/artifacts/`，以保持 `plans/` 目录仅含项目文档。
- 命名规范：目录统一小写+下划线；文件统一 `UPPER_SNAKE`，保留语义日期后缀（如 `20260821`）以维持可追溯性。
