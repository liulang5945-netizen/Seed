# 22.5.V.1 设计卡：硬选列直读（值侧冻结，只训选择与读出）

> 卡号：`22.5.V.1`｜日期：2026-11-17｜状态：**设计＋定价冻结完成**（smoke **36/36**、price **2.3906 CPU**、帽 **4.7812 CPU／9.5625 wall**、契约测试 **8 passed**），**停在 owner 训练授权**｜性质：P3.9 后继新范围，**不是 `unit-core-b-v1` 修订**（修订额度已 2/2 用满）。
> 上游：[值侧复核结果](RESULT-TAIJI_U2_VALUE_SIDE_REVIEW_20261117.md)（`mixed`；答案列头 3/3 达双门、attended 0/3、top-1 选择 0/3、painted 0.674–0.791）、[设计/预注册](DESIGN-TAIJI_U2_VALUE_SIDE_REVIEW_20261117.md)、[读路径收束卡](CLOSEOUT-TAIJI_U2_UNIT_CORE_B_READ_PATH_20261117.md)。
> 父：U.1r2 三枚训练后胶囊（`data/experiments/taiji_u2/u1r2_scale_supervision_20261116/seed*_final.bin`），route `B_position_temp`；**值侧冻结**，从父继承。

## 1 单问题

值侧复核显示：把查询色所在列直接拿出来读，位置类头 **3/3 达双门**；用注意力平均读，**0/3**；模型自己的 top-1 选择也不够准。本卡问：

> **只训"选择（W_q）＋读出（position_head）"、值侧完全冻结，硬选列直读能否达到读前提？**

这是单变量：从软注意力平均 → 硬选 top-1 列直读；值侧、`log_tau`、编码合同、数据、种子、读出类都不变。**不新增参数**。

## 2 机制

- `selected = argmax(attention)`（前向硬选）；
- 前向读出 `readout = position_head(projected[selected])`；
- 训练期给 `W_q` 一条直通梯度：`value = projected[selected] + (attention @ projected - (attention @ projected).detach())`，**前向仍是硬选**、反向经软注意力到 `W_q`；`position_head` 拿到硬选列的任务梯度；
- 注意力仍由 r2 的聚焦监督训练（48 条有标签 train，目标＝查询实体逐列颜色质量归一），信用只给 `W_q`（`log_tau` 冻结）；
- **可训张量仅 2 个**：`W_q`（64 标量）、`position_head`（32 标量）；其余全部 `requires_grad=False`；**新增参数 0**；
- route 改为 `B_hard_select`；父胶囊不能直接 `apply_capsule`（route 不同），runner 用**显式父继承**：读取父胶囊、校验父 route／参数名／形状，`load_state_dict` 后冻结值侧；父胶囊只读。

## 3 数据、种子与边界

- 数据根与 `data_digest` 与 U.1r2 相同（train 64／fit 32；48 条有标签 train 做聚焦监督）；**check／final／confirm 读取 0**、封存 0。
- 3 seed：2026111601／2026111602／2026111603；父胶囊逐 seed 对应。
- 训练：384 更新/seed，batch 1（逐例），float64；只训 `W_q`＋`position_head`。
- 预算：设计帽 600 CPU／1200 wall；先按 smoke＋批量计时测定实价，帽＝实测×2.0，再冻结；预计 ~10 CPU。

## 4 门（读数前冻结）

**值侧前置（来自冻结父，复核已测，作为证据 pin）**：答案列位置头 fit ≥0.80 且 train ≥0.90（复核 3/3 达门）；列颜色 painted 基线按 seed **0.7907／0.6860／0.6744**。

**主门（3 seed 全过才算过）**：
1. **选择门**：训练后 fit 上 `top-1 注意力列落在答案跨度` 的比例 **≥0.95**；
2. **读出门**：fit ≥0.80、train ≥0.90、fit distinct ≥3；
3. **控制**：RGB 置零 agreement <0.5、固定问题 agreement <0.5；
4. **干预**：`without_edges`／`permute_source`／`remove_history` 改变读出比例 ≥0.5，`cut_h` 逐位不变；
5. **颜色诊断（硬门，不降）**：列颜色 painted ≥ 基线 −0.05（阈值 **0.7407／0.6360／0.6244**）；
6. **恢复**：独立进程 capsule 恢复逐位一致＋篡改拒绝；
7. **成本**：实耗 ≤ 实测帽。

## 5 出口（互斥按序）

1. `budget_stop`——超帽。
2. `engineering_failure`——完成臂数少于 seed 数，或 smoke／recover／verify 有未覆盖项。
3. `selection_failed`——任一 seed 的 fit top-1-on-span <0.95。
4. `value_side_degraded`——任一 seed 的 painted < 基线 −0.05。
5. `read_failed`——选择与颜色都过，但读门/控制/干预不过。
6. `hard_readout_passed`——全部门过（**只签读前提**，不签父/能力/check②/22.6）。

## 6 能签与不能签

**能签**（若过）：硬选列直读在值侧冻结、只训 96 标量的条件下达到读前提；选择与读出是这条路径的充分修复；工程纪律（父继承、恢复、成本）。**不能签**：单位核心 B 架构整体可用/不可用；任何能力/父/check②/22.6；不得把"读前提通过"写成"自然中文任务可用"。

## 7 当前状态与唯一后继

**当前**：设计＋定价已冻结；**0 训练**、`training_authority=null`。**唯一后继**：实现模块＋runner → smoke（0 步）→ 精确定价 → 冻结契约 → 请 owner 训练授权；授权前 0 训练新增、0 能力、0 A 父；check② 不读、22.6 不解锁。

## 8 验证与定价（已完成，零训练）

- **模块**：[taiji/u2_unit_hard_selection_readout.py](../../taiji/u2_unit_hard_selection_readout.py) 的 `HardSelectionGraph`：`selected=argmax(attention)`、前向 `position_head(projected[selected])`、训练期直通软注意力梯度给 `W_q`；`route=B_hard_select`，父胶囊被 `apply_capsule` 响亮拒绝。
- **runner**：[run_v1_hard_selection_20261117.py](../../scripts/training/run_v1_hard_selection_20261117.py)，模式 draft／smoke／price／freeze／run／recover／verify；父继承用显式 `load_state_dict`＋父 route 校验，冻结除 `W_q`／`position_head` 外全部参数。
- **smoke** [报告](../../reports/taiji_u2_v1_hard_selection_smoke_20261117.json)：verdict `smoke_passed`、**36/36**、`uncovered=[]`；逐 seed 注意力／attended 与父逐位相同、`selected=argmax`、硬读出恒等、父胶囊响亮拒绝、可训恰两枚、state_dict 键与父相同、checkpoint 往返、两个可训参数有梯度、冻结参数无梯度。
- **定价** [报告](../../reports/taiji_u2_v1_hard_selection_price_20261117.json)：per_update **0.0012207**、per_eval **0.00048828**、per_intervention **0.00585938**；`measured_cpu = 3*(384*per_update + 288*per_eval + 32*per_intervention) =` **2.3906**；帽 **4.7812 CPU／9.5625 wall**（×2.0，设计帽 600／1200 内）。定价自身 3.0 CPU／68 个用后即弃更新。
- **契约** [manifest](../manifests/taiji_u2_v1_hard_selection_20261117.json)：`status=frozen`、`training_authority=null`、`criteria_sha256=00bc9cf904c8d7f6…`、13 枚 source pin、3 枚父胶囊摘要；`--run` 在授权前抛 `missing_training_authority`。
- **契约测试** [test_v1_hard_selection.py](../../tests/taiji_native/test_v1_hard_selection.py)：**8 passed**（契约冻结／帽、pin 与父摘要、父胶囊拒绝、可训恰两枚、硬读出恒等、run 授权守卫、价格与帽、smoke 全绿）。

**授权后执行顺序**：`--run` → `--recover` → `--verify`，按 §5 出口结案；授权字符串写进契约、**用完即止不外推到下一卡**。授权前 0 训练新增、0 能力、0 A 父；check② 不读、22.6 不解锁。
