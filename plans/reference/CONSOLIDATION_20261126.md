# Taiji / Seed 项目阶段收束与全局盘点（2026-11-26）

> [收束·阶段总账] — 2026-11-26（D-20261126-689）。总结 `≤N30 → N31 → N32 → N33 → N34 → N35 → N36 → N37` 有界主线全部收口成果、确认 seed 与 `check/final` 封存账本、可复用运行时/API 资产，以及当前正在运行的 **`N38.1.01` 开放自然中文 `mid_18m`（19.47M）24h CPU 预训练**。

## 1. 主线阶段终态总览

| 阶段 | 定位 | 终态 | 核心实证 / 交付物 |
|---|---|---|---|
| `≤N30` | 历史 RouteB / `S0` / `S1-RT` | **已冻结** | `taiji/u2_neural_units.py` 零改动封存 |
| `N31` | 机制资格（路由/泛化/生长） | **已结案（D-499）** | [CLOSEOUT-N31](CLOSEOUT-N31_20261124.md) |
| `N32` | 正式训练 T1（`R1` 正例） | **已收口（D-552）** | [CLOSEOUT-N32](CLOSEOUT-N32_20261125.md) |
| `N33` | 正式训练 T2（跨会话/生长/递归） | **已收口（D-609）** | [CLOSEOUT-N33](CLOSEOUT-N33_20261126.md) |
| `N34` | 长跑耐久与六轴综合确认 | **已收口（D-664）** | `N34.9.90`（6/6）＋ `N34.5.90`（100 会话/10 崩溃恢复，[CLOSEOUT-N34](CLOSEOUT-N34_20261126.md)） |
| `N35` | 模型级联合确认（方案 B1） | **已收口（D-682）** | `N35.1.91` 8/9 门 3/3 过 `confirm_partial`（开放中文标 `deferred_post_N35`，[CLOSEOUT-N35](CLOSEOUT-N35_20261126.md)） |
| `N36` | 有界交付与用户接管包 | **已收口（D-685）** | `D1..D5` 5/5 过 + 中文手册（[CLOSEOUT-N36](CLOSEOUT-N36_20261126.md) / [HANDBOOK-N36](HANDBOOK-N36-BOUNDED-TAKEOVER_20261126.md)） |
| `N37` | 有界运营状态机（`§15`） | **已收口（D-688）** | 8 态 FSM + 6 步影子升级 + 3 层变更分流，`O1..O6` 6/6 过（[CLOSEOUT-N37](CLOSEOUT-N37_20261126.md)） |
| **`N38`** | **开放中文统一认知底座（CPU 长训）** | **▶ 阶段 1 运行中（D-689）** | `mid_18m`（19.47M）`bfloat16` 24h 预训练（阶段 0 实测 `4,343 tok/s`，[MEMO-N38-STAGE0-PREFLIGHT](MEMO-N38-STAGE0-PREFLIGHT_20261126.md)） |

## 2. 能力边界（严格区分已证实与未签署）

- **已正式证实（有界范围 `bounded_closed_vocab_and_multimodule_v1`）**：
  - 封闭词表（43 token）跨回合中文连续指代（`N34.9.90` / `N34.5.90`）；
  - 两族合成多模态物理因果跨族迁移与障碍消融真实性（`N34.9.90`）；
  - 在线 Beta 可信度学习、学习式不确定性校准、状态相关生长决策、模型内递归发展（`N33.4` / `N34.9.90`）；
  - 百会话 `SessionStore` 持久化、sha256 审计链、崩溃恢复、`check/final` 封存守卫（`N34.5.90` / `N36`）；
  - 8 态运营状态机、日常影子升级闭环、三层变更分流与退役归档核验（`N37`）。
- **明确未签署 / 仍在攻关**：
  - 开放自然中文通用对话与长程指代（`N38` 正在通过 19.47M 统一底座预训练＋后续 SFT/`NCH.3` 微调攻关）；
  - 真机物理传感器/执行器闭环、24 小时现场试运行、`check / final` 读取（`check_reads = 0, final_reads = 0`）及 `24-S/25-S/29/final/C/D/J`。

## 3. 可复用工程与产品资产

- **认知与运行时内核**：`taiji/bounded_delivery_runtime.py`、`taiji/operational_state_machine.py`、`taiji/open_chinese_backbone.py`、`taiji/long_run_harness.py`、`taiji/session_store.py`；
- **CLI 工具**：`scripts/taiji_n36_takeover.py`、`scripts/taiji_n37_ops.py`、`scripts/training/n38_1_01_pretrain_open_zh.py`；
- **后端 API 与客户端**：`api/app.py`、`api/routes_taiji_ops.py`（`/api/taiji/ops/*` 与 `/taiji-ops` 实时看板）、`taiji-harness/`（Web 与桌面端）。
