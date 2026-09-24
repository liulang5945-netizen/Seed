# R2 T6 预注册：给 T5/对照接上 **CAP D+E 机器计分**（判读线先冻结）

前件：T5（读出只吃区 0，5M）已跑完，按其预注册记否定（结构保持不成立），但其 cue 的
**水平线**比 `t1` 高一个台阶。读法修正的目的从来不是可分离性本身，而是**答对** ——
所以先问"可分离的 cue 换没换来答对"，再决定要不要花 22 h 做逐笔归因。

## §1 被评分的 checkpoint（4 份，全部已有，不新训）

| 代号 | checkpoint | 是什么 |
|---|---|---|
| `t5` | `output/taiji_r2_t5/t5_r0_cue/checkpoint_5000000.pt` | **处理臂**：读出只吃区 0，5M |
| `t1` | `output/taiji_r2_t1t2/t1/checkpoint_5000000.pt` | **对照**：现行配方，同预算 5M |
| `abl` | `output/taiji_r2_t1t2/abl_fabric/checkpoint_5000000.pt` | **对照**：fabric 写入冻结，5M |
| `base` | `checkpoints/seed_beta.pt` | **对照**：产品基座，16M（已知 CAP D+E = **0/36**） |

## §2 仪器（**原样复用，不重写**）

* **M2 计分**：`eval_taiji_cap0_baseline.py::run_baseline(checkpoint=…, dimensions=("D","E"))`
  —— 与 M5_R2 主门同口径（产品链路、`relax_legacy_guard` 默认）。
* **M1 成句率（副读数）**：复用 `eval_taiji_r2_readout_retrain.py` 的同一套
  `m1_tasks / build_ngram_model / generate / well_formed / assert_criterion_discriminates`
  （逐 checkpoint 生成 + 回声控制），**不接 A/B/C 配对门**——那个门校验的是 A/B/C 战役的
  trainer/预算/前缀一致，我的四份 checkpoint 过不了也不该伪造元数据去骗它。
* **为什么绕开 `eval_taiji_r2_readout_retrain.py` 的主入口**：它的 `verify_pairing` 是
  A/B/C 战役的守卫（fail closed），本件的 checkpoint 不是那个 trainer 产的；
  **守卫不删、不放宽**，只是不进那个入口。

## §3 仪器自检（**先于一切读数**）

`base`（16M）的 CAP D+E 机器计分之和必须 **= 0**（已知基线 0/36）。
若 ≠ 0 ⇒ **仪器或口径有问题，全部读数作废**，本件只报这个矛盾。

## §4 判读线（**开跑前冻结**）

设 `S_x` = 模型 x 的 CAP D+E 机器计分之和（D、E 两维）。

* **自检**：`S_base == 0`，否则作废（见 §3）。
* **主判据**：`S_t5 ≥ 2` **且** `S_t5 > max(S_t1, S_abl, S_base)` ⇒
  记 **"可分离的 cue 换来了答对"（端到端信号成立）** ——
  下一步是把可分离读法推进到 16M（另立预注册），逐笔归因**降级为可选**。
* `S_t5 == 0` ⇒ 记 **"可分离的 cue 没换来答对"** —— 读法修正**不足以**解决答对问题，
  下一步才轮到"稳定性"归因（T3/T4 预定的逐笔分离，22 h）。
* `0 < S_t5 < 2` ⇒ 记 **"有信号但未过 2 题增量门槛"**（如实记下题数与所在维度），
  下一步由所有者在"加预算复核"与"稳定性归因"之间裁决。

## §5 副读数（只报不判）

M1 成句率（`well_formed_rate` + **回声控制**）、per-dimension tallies、
`identity`（git head / checkpoint sha / 评价集 sha —— 绑定执行身份）。

## §6 停止线

本件只回答"可分离的 cue 换没换来答对"。**逐笔归因（22 h）**、**把可分离读法推进到 16M**、
**接成句率之外的新评分维度**都**另立预注册**。
