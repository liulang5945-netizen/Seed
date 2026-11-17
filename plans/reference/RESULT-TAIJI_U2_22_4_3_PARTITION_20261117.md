# 22.4.3 分区、数据 recipe 与曝光结果（1,280 world recipe，final 预留）

> 卡号：`22.4.3`｜日期：2026-11-17｜判定：**`partition_valid`**｜性质：**零训练分区/recipe 验收**，不是模型能力或父结果。
> 依据：[计划 §22.4.3](PLAN-TAIJI-CLOSED_LOOP_DEVELOPMENT_20261110.md)、[22.4.2.6 结果卡](RESULT-TAIJI_U2_22_4_2_GENERATOR_20261117.md)、[世界规格](SPEC-TAIJI-U2_WORLD_22_4_2_20261112.md)。
> 纪律：**0 模型前向、0 optimizer step、0 项目数据读取、0 check 读取、0 final 读取**；实耗 **28.6719 CPU／4.266 wall**（帽 200／400）。
> 报告：[reports/taiji_u2_partition_22_4_3_20261117.json](../../reports/taiji_u2_partition_22_4_3_20261117.json)（36,883 字节）。

## 1 分区

| split | world 数 | 用途 | 本卡生成内容 |
|---|---|---|---|
| `train` | 768 | 训练/发展 | recipe＋样张 |
| `development-fit` | 192 | 发展拟合 | recipe＋样张 |
| `development-check` | 192 | 固定 worker 评测 | **只给 seeds/digest，不生成/不读** |
| `final` | 128 | R-SEAL-1 封存 | **只预留 recipe/seal，不生成/不读** |
| **合计** | **1,280** | | **recipe 1,152；样张 3 world** |

- **seed 派生**：`base=20261110`、`stride=1009`、起点 `train=0／fit=768／check=960／final=1152`；每个 split 内 seed 唯一且互不重叠，并集 **1,280**。
- **world group**：一个 seed 承载 A/B/C/D/J 全族；`cross_split=false`，同 world 的关联轨迹/事实版本不跨 split。
- **fit 内部再拆**：`birth-fit=128`、`birth-check=64`，二者并集等于 fit、互不重叠；**birth-check 不是全局 check**，`global_check_used_for_birth=false`。
- **旧能力 A 独立题面**：A 在 train/fit/check/final 各自独立，`ability_A.independent_surface=true`；check/final 不被新任务挪用。
- **过时事实**：`stale_fact_policy="exclude_superseded_versions"`，纠正后旧版本不当作应保持的有效知识。

## 2 数据 recipe 与 digest

- **生成器**：`taiji/u2_task_generator.py:generate`（22.4.2.6 已验 A/B/C/D/J 统一包络、胶囊 save/load/rebuild 逐位一致）。
- **shape/schema**：观测 `rgb[16,48]／audio[7,33]／proprio[1,16]`、坐标 `[16,8]／[7,8]／[1,8]`、`outcome[6]`；反馈权利表全族逐字相同。
- **`data_recipe_digest`**：`a96cf26e2008256efeb9f5e223f4d5d3d7141a9daa79ada5ba6e14563d100644`
- **`partition_digest`**：`8f16e764911579c6e86af8950b6cd63c6cc72922d414c155ef592054134f10ea`
- 两者覆盖 seed 派生、split 计数/seed 摘要、birth 子拆、world group、shape/schema、反馈权利、A 独立题面、source pins、check 曝光与 final seal。

## 3 曝光台账

- 全局 check **max 8、used 0、remaining 8**；分配 **22.5=2／22.6=2／22.7=3／22.8=1**。
- 每次曝光＝完整候选 × 3 固定 seed × 该阶段全部臂；**首次访问即扣、失败也计、不可撤销**；只向继续开发流程释放冻结聚合结果，逐例标签不回流选择器。
- **本卡 `check_content_generated=false`、`check_labels_read=false`**；check 只登记 seeds/digest，不生成内容、不读标签。
- **final**：`authorization="R-SEAL-1"`、`status="reserved_not_generated"`、`generated=false`、`read=false`；仅记录 count/first/last/seeds_sha256/recipe_digest，不生成、不读取。

## 4 样张回执

| split | world | 族 | seed | 生成 CPU | 重建 CPU | 胶囊字节 | 形状/加载/重建 |
|---|---|---|---|---|---|---|---|
| train | 0 | A | 20261110 | 1.6094 | 1.8125 | 277,373 | ✓/✓/✓ |
| train | 0 | D | 20261110 | 7.0625 | 8.3750 | — | ✓/—/✓ |
| train | 1 | A | 20262119 | 2.0156 | — | — | ✓/—/— |
| development-fit | 0 | A | 21036022 | 1.8750 | — | 277,373 | ✓/✓/— |
| development-fit | 0 | J | 21036022 | 4.6094 | — | — | ✓/—/— |

- 3 个样张 world、A/D/J 三族；全部 shape 通过；train A 与 fit A 的胶囊 save→load 摘要一致；train A/D rebuild 逐位一致。
- **未物化全部 1,152 world**：`full_materialization=false`、`regenerable=true`；完整样本由 seed＋`generate` 重建（22.4.2.6 已对 A/B/C/D/J 逐位验证 rebuild）。这是 200 CPU 分区帽下的 recipe 级交付，不冒充"已生成 1,152 个胶囊"。
- check/final 无样张；report 不含 check 标签或 final 内容。

## 5 范围与限制

- **能签**：1,280 world 的 seed recipe、1,152 可生成 world 的 split/去重、fit 内 birth 子拆、A 独立题面、数据 recipe 与 partition digest、check 曝光台账初始化、final R-SEAL-1 预留守纪。
- **不能签**：模型能力/父/check②/22.6；22.4 七项交付仍缺第 4（完整任务定价）、第 5（逐卡正式 updates 数）与第 7（`accepted_parent`）；本卡未做 22.4.4 价格，也未读取 check/final。
- **口径**：本卡是 recipe 级分区，不是 1,152 个完整胶囊的物化回执；22.4.4/22.5 训练时应按 seed recipe 懒生成，并以本卡 digest 校验。

## 6 唯一后继

**`22.4.4` 完整任务定价**：按观测/编码、查询/表达、policy、单次学习、保存/重建、重放迁移、出生/成熟分项，先用 train/fit 独立定价片段测 CPU/wall/内存与重放长度曲线，覆盖生成器、所有臂、3 个固定 seed（20261110／20261111／20261112）与评估/保存/失败余量；价格超帽即 `pricing_over_budget`。之后写 `accepted_parent`，进入 22.5.0 A 父门。在 22.4 准入前：0 训练新增、0 能力、0 父；check② 不读、22.6／22.7 不解锁。
