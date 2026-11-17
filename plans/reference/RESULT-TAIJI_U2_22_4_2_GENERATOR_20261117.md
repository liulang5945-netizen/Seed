# 22.4.2.6 生成器入口、shape/schema 与微型恢复结果（A/B/C/D/J 全族通过）

> 卡号：`22.4.2.6`｜日期：2026-11-17｜判定：**`generator_entry_valid`**｜性质：**零训练工程/接口验收**，不是模型能力或父结果。
> 依据：[计划 §22.4.2.6](PLAN-TAIJI-CLOSED_LOOP_DEVELOPMENT_20261110.md)、[22.4.2.5 结果卡](RESULT-TAIJI_U2_22_4_2_INTERVENTIONS_20261117.md)、[世界规格](SPEC-TAIJI-U2_WORLD_22_4_2_20261112.md)、[22.4.2.3 回执](VALIDATION-TAIJI-U2_PHYSICAL_RUNTIME_22_4_2_20261112.md)。
> 纪律：**0 模型前向、0 optimizer step、0 项目数据读取、0 封存/check/final 读取**；实耗 **66.5312 CPU／9.891 wall**（帽 120／240）。
> 报告：[reports/taiji_u2_generator_22_4_2_20261117.json](../../reports/taiji_u2_generator_22_4_2_20261117.json)（26,351 字节）。

## 1 交付物

- **生成器入口**：[taiji/u2_task_generator.py](../../taiji/u2_task_generator.py) 的 `generate(family, seed)`；CLI 为 `python -m taiji.u2_task_generator --family {A,B,C,D,J} --seed 124 --out <capsule>`。
- **统一包络** `taiji-u2-task-sample-v1`：`world`／`executor`／`model_observation`／`legal_content`／`feedback_rights`／`task`／`provenance`／`envelope_digest`。
- **胶囊** `taiji-u2-task-sample-capsule-v1`：`atomic_save/atomic_load`＋源码 pin＋包络/world/observation/coordinate/outcome 摘要；可加载并逐位核对。
- **微型恢复**：同进程 save→load→rebuild 逐位一致；另做一步 `PhysicalExecutor` create→execute→snapshot→from_snapshot→query 回执（22.4.2.3 的独立进程执行器恢复仍有效，本卡不重复）。
- **测试**：[tests/taiji_native/test_u2_task_generator.py](../../tests/taiji_native/test_u2_task_generator.py) 2 passed（A 形状/胶囊/rebuild；C case/choice）。

## 2 shape 与 schema

| 层 | 键 | shape/dtype | 读数 |
|---|---|---|---|
| 模型观测 | `rgb` / `audio` / `proprio` | `[16,48]` / `[7,33]` / `[1,16]`，float32 | A/B/C/D/J 全同 |
| 传感坐标 | `rgb` / `audio` / `proprio` | `[16,8]` / `[7,8]` / `[1,8]`，float32 | 全同 |
| 执行结果 | `outcome` | `[6]`，float32 | 全同 |
| 合法内容 | `goal_raw` / `requests` / `model_inputs` / `observations` / `coordinates` / `outcome` / `action` | 摘要/hex | A/B/C/D/J 键集全同 |
| 反馈权利 | `observation`／`train_label`／`online_correction`／`help_reply`／`score_only` | kind→phase 表 | 全族逐字相同 |

**合法性控制**：A/B/C 的 `Case` 全部通过 `validate_case`；目标词只出现在 `task`（评分侧），不出现在 `legal_content`、goal 或 request；D/J 的 expected 也不在 `legal_content` 中。`allow_feedback` 正负控制：observation/perception 放行；train_label 在 check、score_only 在 perception、未提交预测的 online_correction 均拒绝；提交后的 help_reply 放行。

**执行器**：每族用初始 world 建一步 `PhysicalExecutor`，schema `taiji-u2-physical-executor-v1`、execute→query `applied`、snapshot→from_snapshot 逐位相等；5/5 通过。

## 3 读数与成本

| 族 | seed | 包络摘要 | 胶囊字节 | 生成 CPU | 重建 CPU | 执行器探针 CPU |
|---|---|---|---|---|---|---|
| A | 124 | `a654d34390d1…` | 277,373 | 1.9844 | 1.6406 | 0.5312 |
| B | 124 | `2ae539534dcd…` | 277,373 | 1.5312 | 1.5312 | 0.4375 |
| C | 124 | `e7b4c5eab37f…` | 104,670 | 0.2188 | 0.2188 | 0.5156 |
| D | 124 | `7316085c1205…` | 630,165 | 8.2812 | 8.9844 | 0.5312 |
| J | 124 | `6a829fd1eb0c…` | 642,045 | 4.1562 | 3.6875 | 0.5000 |

- **seed 扫**：A/B/C 各跑 124／125／126，包络摘要互异、problems 全 0；D/J 因成本只跑 seed 124 全族样张（22.4.3 分区再扩到 1,280 world）。
- **成本口径**：生成含世界推进与观测编码；`encoding_cpu` 低于计时分辨率（0.0），已并入生成；重建＝从 family+seed 重新生成并比摘要；保存/加载/执行器探针单独计时。总 **66.5312 CPU／9.891 wall**，帽 120／240。
- **胶囊**：A/B 277,373、C 104,670、D 630,165、J 642,045 字节（临时文件，不入库）；报告只存摘要。

## 4 范围与限制

- **能签**：生成器入口可对 A/B/C/D/J 产出统一包络；世界/执行器/模型原始观测 shape 与 schema 全族一致；合法内容与反馈权利全族一致；胶囊 save/load/rebuild 逐位一致；执行器一步恢复回执通过；真实成本已记录。
- **不能签**：模型能力/父/check②/22.6；22.4 七项交付仍缺第 2（真实任务族 recipe 及分区 digest）与第 7（`accepted_parent`）；本卡未生成 22.4.3 的 train/fit/check/final 分区，也未做 22.4.4 完整任务定价。
- **恢复范围**：本卡是**同进程**微型胶囊恢复；独立进程的世界/执行器/主体恢复由 22.4.2.3 负责，本卡不把同进程结果外推成独立进程结论。
- **模态边界**：A/B/C 的任务 case 使用冻结的字节输入；D/J 使用 12/18 个检查与合法帧摘要。`model_observation` 是执行器/世界侧张量包络，不等于模型已消费；本卡不签模型会看/会答。

## 5 唯一后继

**`22.4.3` 分区生成**：按计划生成 **train768／development-fit192／development-check192／final128（共 1,280 world）**，登记 digest 与曝光计数（全局 check ≤8，22.5=2／22.6=2／22.7=3／22.8=1；final 另按 R-SEAL-1 当次授权），复用本卡生成器入口与胶囊摘要。之后 22.4.4 完整定价 → 22.4 `accepted_parent`。在 22.4 准入前：0 训练新增、0 能力、0 父；check② 不读、22.6／22.7 不解锁。
