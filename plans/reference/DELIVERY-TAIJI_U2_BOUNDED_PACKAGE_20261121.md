# 交付说明：有界能力包（候选 v-20261121；零训练、零 check）

> 文档状态：[交付说明] — 2026-11-21（D-20261121-224）。**不签模型成型/final/C/D/J；零 check/final。**

## 1. 交付物身份

- **候选**：RouteB 30 单位图（6 群×5）＋两步注意力/位置注意力读出；三技能父 = 23.0b 联合训练包；
- **lineage**：22.8 RouteB 新主体 → 23.0b 三技能父 → 本会话读出升级（位置嵌入＋两步注意力，A1 批准为「单句内令牌序列」默认读出）；
- **身份/恢复**：胶囊恢复 diff ≤1e-10（多卡一致）；旧三技能保留（各卡实测损伤 0.0）。

## 2. 成立（有界、可复核）

| # | 能力 | 关键读数 | 回执/卡 |
|---|---|---|---|
| 1 | 单位机制/因果/生命周期 | 22.1–22.7 verified；删除损伤 0.41–0.69 | `RESULT-TAIJI_U2_22_7_*` 等 |
| 2 | 身份/胶囊/恢复 | diff ≤1e-10 | 多卡 |
| 3 | 三技能保留 | S1 0.98–0.99 / S2 0.86–0.88 / S3 1.0 | 23.0b/24.x |
| 4 | 中文词汇接地（封闭词表） | 0.327（控制 0.05–0.06） | D-201 |
| 5 | 中文语序（读出升级） | 单留出 1.0；8 色 10 模板 3 留出 **0.701**（平均池化 0.319） | D-205/220 |
| 6 | 外部历史可读（`external_history_select`） | 0.74–0.92 | D-213/217 |
| 7 | 读出工程的机制归因 | 同单位同任务仅换读出：中文语序 0.472→0.967 | D-205 |

## 3. 不成立（已按预注册出口结案）

| # | 项 | 关键证据 | 出口 |
|---|---|---|---|
| 1 | 单位状态跨步记忆 | z 在 t1 回落机会；a 仅 t1–t3 弱残留；H 不演化 | `state_decode_diagnostic`（D-217） |
| 2 | 可学习持久存储 | 写入门 0.333、有界键值槽 0.372 < mean 0.469 | `persistent_gate_partial`/`bounded_store_partial`（D-218/219） |
| 3 | 动作接地/闭环 | A2 replay 0.784<0.80；闭环未成立 | `action_prior_not_learnable`（D-215） |
| 4 | 结构生长收益 | 22.7 等 FLOPs 负、25.0 无压力、P3 30→48 无收益 | `structural_pressure_absent`（D-192） |
| 5 | 经验巩固/前缀维护 | 24.2 +0.010、24.3 无成本收益 | `consolidation_partial`/`prefix_maintenance_partial` |
| 6 | 自主选择/递归 | 22.6 negative；26/27 前置不满足 | D-22.6、D-188 |
| 7 | 开放中文/对话/生成 | 未测（A3c 待批） | 未运行 |

## 4. 复现（标准命令）

```powershell
cd E:\Seed
python -m ruff check --fix scripts
python -m black scripts
python scripts/training/audit_plan_single_mainline.py --json
# 各探针（示例，完整清单见各结果卡）：
python scripts/training/run_s2_readout_seq_20261120.py
python scripts/training/run_s2_state_decode_20261121.py
python scripts/training/run_s2_cn_vocab2b_20261121.py
```

- 回执路径：`reports/taiji_u2_*.json`（逐卡 CPU、seed、判据、verdict 均在文件内）；
- 卡路径：`plans/reference/RESULT-TAIJI_U2_*.md`；判词台账：`DECISION-LEDGER_20261109.md`（至 D-223）。

## 5. 成本与曝光

- 本阶段探针累计约 **7,000–8,000 CPU**；**零 check / 零 final**（check≤2、final≤1 未使用，final 未授权）；
- 包 22 历史超支按 B6 记**预算事故**（不追溯改帽）；25 主窗已释放（B2）。

## 6. 未立项的占位项（需 owner 裁定）

1. **A3c**（自然措辞中文，≤1,500 CPU、零 check）——预注册已冻结，待批；
2. **架构级持久状态 A/B**（≤1,500 CPU/方案、零 check）——预注册已冻结，待批；
3. 若两者均不批：本交付即最终状态，记忆轴按「外置历史可读、单位状态不承载」收束。

## 7. 明确非声明

不声称：模型成型、自主选择成立、长时记忆、动作闭环、结构生长收益、开放中文/对话/生成、真机迁移、final/C/D/J 通过、独立接管交付。零 check、零 final。
