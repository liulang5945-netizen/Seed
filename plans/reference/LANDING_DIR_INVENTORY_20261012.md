# 盘上落点清单（2026-10-12，四裁⑶「我只出清单不动手」的执行件）

**这份件只回答一件事**：`output/` 下现在有哪些未入库落点、各多大、最后一次被写是什么时候、
**有没有被仓内已跟踪文件引用**。删与不删由 owner 决定，本件一个字节都不删。

## 取法（可复算）

* 面：`git status --porcelain` 里 `?? output/` 的行，现读 **37** 枚（目录 35 个＋文件 2 枚）；
* 体积／文件数／最新修改时间：对每枚目录 `rglob('*')` 取文件求和，时间取该目录内最新一枚的 mtime；
* 引用面：`git ls-files tests scripts reports plans taiji seed api` 现读 **3939** 个已跟踪文件，按**裸目录名**做子串匹配（不是整路径——整路径会让同名不同目录的引用失明，而裸名的反向风险已由「逐个列出引用件」压住）；
* 合计 **2528.0 MB**；有引用 33 枚，无引用 4 枚。

**一条必须写在这里的读数抖动**：本件的计数是**移动面**。上一笔台账（㊵-736）写的是 36 枚，本件现读 37 枚——多出来的那枚（`output/seed_corpus_smoke_progress_exit.json`）是**整族回归自己写进去的**，不是有人新落了产物。⇒ 凡引用本件数字，必须带 `generated_utc`＝**2026-10-12T10:48Z**；不带时刻的落点计数一句话都不能说。

## 逐枚点名（按体积降序）

| 落点 | 文件数 | 体积 MB | 最新写入 | 被引用数 | 引用例 |
|---|---|---|---|---|---|
| `output/n5_10_arms/` | 21 | 265.199 | 2026-10-10T16:54Z | 9 | 05_TECH_DEBT_REGISTER.md、08_UPSTREAM_SYNC_PLAYBOOK.md、PLAN-N1-03_chain_and_roles_prereg_20261011.md |
| `output/n3a_armA/` | 3 | 229.189 | 2026-10-08T15:13Z | 12 | 03_CURRENT_EXECUTION.md、05_TECH_DEBT_REGISTER.md、08_UPSTREAM_SYNC_PLAYBOOK.md |
| `output/n3a_armB/` | 3 | 229.189 | 2026-10-08T16:18Z | 8 | 05_TECH_DEBT_REGISTER.md、08_UPSTREAM_SYNC_PLAYBOOK.md、09_NEXT_MAINLINE_PLAN.md |
| `output/n5_07/` | 27 | 227.314 | 2026-10-10T08:59Z | 27 | 05_TECH_DEBT_REGISTER.md、08_UPSTREAM_SYNC_PLAYBOOK.md、09_NEXT_MAINLINE_PLAN.md |
| `output/n3_04/` | 30 | 215.583 | 2026-10-08T07:36Z | 19 | 03_CURRENT_EXECUTION.md、05_TECH_DEBT_REGISTER.md、08_UPSTREAM_SYNC_PLAYBOOK.md |
| `output/n5_07_arms/` | 18 | 198.976 | 2026-10-10T12:07Z | 12 | 05_TECH_DEBT_REGISTER.md、08_UPSTREAM_SYNC_PLAYBOOK.md、09_NEXT_MAINLINE_PLAN.md |
| `output/n3_09_v2/` | 30 | 142.813 | 2026-10-08T10:56Z | 5 | 05_TECH_DEBT_REGISTER.md、08_UPSTREAM_SYNC_PLAYBOOK.md、PLAN-N3-09_ADJUDICATION_20261008.md |
| `output/n5_shadow_feed2/` | 8 | 85.591 | 2026-10-09T08:43Z | 0 | — |
| `output/n5d_rebase_control/` | 6 | 79.066 | 2026-10-10T07:12Z | 5 | 08_UPSTREAM_SYNC_PLAYBOOK.md、09_NEXT_MAINLINE_PLAN.md、PLAN-N5-04_ADJUDICATION_20261010.md |
| `output/n5d_rebase_treated/` | 6 | 79.066 | 2026-10-10T07:11Z | 6 | 05_TECH_DEBT_REGISTER.md、08_UPSTREAM_SYNC_PLAYBOOK.md、09_NEXT_MAINLINE_PLAN.md |
| `output/n5_shadow_feed3/` | 7 | 73.892 | 2026-10-09T08:56Z | 1 | 08_UPSTREAM_SYNC_PLAYBOOK.md |
| `output/n5_shadow_feed4/` | 7 | 73.892 | 2026-10-09T09:15Z | 1 | 08_UPSTREAM_SYNC_PLAYBOOK.md |
| `output/n5_shadow_feed5/` | 7 | 73.892 | 2026-10-09T09:43Z | 1 | 08_UPSTREAM_SYNC_PLAYBOOK.md |
| `output/n5_shadow_gate_off/` | 7 | 73.892 | 2026-10-09T10:15Z | 4 | PLAN-N5-04_retention_rebase_prereg_20261009.md、taiji_n5_pair_metric_source_20261009.json、taiji_n5_shadow_gate_adjudication_v2_20261009.json |
| `output/n5_shadow_gate_on/` | 7 | 73.892 | 2026-10-09T10:07Z | 7 | 08_UPSTREAM_SYNC_PLAYBOOK.md、PLAN-N5-02_shadow_utility_adjudication_prereg_20261009.md、PLAN-N5-04_retention_rebase_prereg_20261009.md |
| `output/n5_shadow_feed/` | 11 | 57.377 | 2026-10-09T08:36Z | 1 | 08_UPSTREAM_SYNC_PLAYBOOK.md |
| `output/n5_618_off/` | 6 | 49.304 | 2026-10-09T12:16Z | 3 | PLAN-N5-05_position_input_wiring_prereg_20261009.md、taiji_n5_pair_metric_source_618_20261009.json、taiji_n5_shadow_gate_adjudication_618_20261009.json |
| `output/n5_618_on/` | 6 | 49.304 | 2026-10-09T12:15Z | 2 | taiji_n5_pair_metric_source_618_20261009.json、taiji_n5_shadow_gate_adjudication_618_20261009.json |
| `output/n4_materials/` | 7 | 45.702 | 2026-10-09T06:16Z | 3 | 08_UPSTREAM_SYNC_PLAYBOOK.md、taiji_n4_addressing_n2c1_20261009.json、taiji_n4_addressing_n2c1_sparse_20261009.json |
| `output/tau_smoke_on/` | 5 | 32.672 | 2026-10-08T13:26Z | 1 | 08_UPSTREAM_SYNC_PLAYBOOK.md |
| `output/epi_smoke/` | 7 | 29.436 | 2026-10-09T05:55Z | 0 | — |
| `output/n3_04_canary_factory/` | 5 | 29.418 | 2026-10-08T07:15Z | 5 | 03_CURRENT_EXECUTION.md、08_UPSTREAM_SYNC_PLAYBOOK.md、09_NEXT_MAINLINE_PLAN.md |
| `output/n3_tau_circuit/` | 4 | 27.173 | 2026-10-08T14:05Z | 4 | 05_TECH_DEBT_REGISTER.md、08_UPSTREAM_SYNC_PLAYBOOK.md、taiji_n3_tau_per_chain_20261008.json |
| `output/n3_tau_beta/` | 4 | 26.807 | 2026-10-08T14:04Z | 4 | 05_TECH_DEBT_REGISTER.md、08_UPSTREAM_SYNC_PLAYBOOK.md、taiji_n3_tau_per_chain_20261008.json |
| `output/tau_smoke_off/` | 5 | 17.999 | 2026-10-08T13:26Z | 2 | 05_TECH_DEBT_REGISTER.md、08_UPSTREAM_SYNC_PLAYBOOK.md |
| `output/n3a_control_x2/` | 3 | 15.121 | 2026-10-09T01:50Z | 7 | 08_UPSTREAM_SYNC_PLAYBOOK.md、HANDOFF_N_MAINLINE_20261009.md、PLAN-N3-12_j_n3a_anchor_upgrade_prereg_20261008.md |
| `output/n3_10_smoke/` | 4 | 12.062 | 2026-10-08T11:13Z | 1 | 08_UPSTREAM_SYNC_PLAYBOOK.md |
| `output/n3_10_smoke_off/` | 4 | 12.062 | 2026-10-08T11:13Z | 0 | — |
| `output/n5_07_lane/` | 13 | 0.55 | 2026-10-10T12:22Z | 9 | 09_NEXT_MAINLINE_PLAN.md、PLAN-N5-07_ADJUDICATION_20261010.md、taiji_n5_07_control_pair_20261010.json |
| `output/n5_10_lane/` | 12 | 0.545 | 2026-10-11T05:45Z | 4 | taiji_n5_10_control_retention_pair_20261011.json、taiji_n5_10_full_adjudication_20261011.json、taiji_n5_10_noise_floor_retention_pair_20261011.json |
| `output/n5d_lane/` | 11 | 0.309 | 2026-10-10T07:56Z | 6 | 08_UPSTREAM_SYNC_PLAYBOOK.md、09_NEXT_MAINLINE_PLAN.md、taiji_n5d_rebase_control_pair_20261010.json |
| `output/n4_product_tier/` | 3 | 0.185 | 2026-10-09T11:04Z | 17 | 05_TECH_DEBT_REGISTER.md、08_UPSTREAM_SYNC_PLAYBOOK.md、PLAN-N4-07_angle_pair_upgrade_prereg_20261009.md |
| `output/n4_product_tier_a035/` | 4 | 0.184 | 2026-10-09T11:20Z | 2 | 08_UPSTREAM_SYNC_PLAYBOOK.md、taiji_n4_product_tier_a035_20261009.json |
| `output/n4_product_tier_a070/` | 4 | 0.184 | 2026-10-09T11:20Z | 1 | taiji_n4_product_tier_a070_20261009.json |
| `output/n4_product_tier_a105/` | 4 | 0.184 | 2026-10-09T11:28Z | 1 | taiji_n4_product_tier_a105_20261009.json |
| `output/ds-bot-0.1.2.tgz` | 0 | 0.0 | empty | 0 | — |
| `output/seed_corpus_smoke_progress_exit.json` | 0 | 0.0 | empty | 10 | 03_CURRENT_EXECUTION.md、05_TECH_DEBT_REGISTER.md、08_UPSTREAM_SYNC_PLAYBOOK.md |

## 无引用的那几枚（若将来要动，只有这几枚在候选里）

* `output/ds-bot-0.1.2.tgz`：0 个文件、0.0 MB、最新 empty。本件**不删**——四裁⑶ 的原话是「我只出清单不动手」。
* `output/epi_smoke/`：7 个文件、29.436 MB、最新 2026-10-09T05:55Z。本件**不删**——四裁⑶ 的原话是「我只出清单不动手」。
* `output/n3_10_smoke_off/`：4 个文件、12.062 MB、最新 2026-10-08T11:13Z。本件**不删**——四裁⑶ 的原话是「我只出清单不动手」。
* `output/n5_shadow_feed2/`：8 个文件、85.591 MB、最新 2026-10-09T08:43Z。本件**不删**——四裁⑶ 的原话是「我只出清单不动手」。

## 若将来要收束，判据该写成什么样（三条，先说清免得下次临场定标准）

* **引用判据**：裸名在 3939 枚已跟踪文件里零命中，**且**该目录内没有任何 `.pt` 被 `git check-ignore -v` 之外的路径硬编码引用——两条都要过，只过第一条就删会把「被仪器默认落点指着」的目录删掉（DEBT-G35／G50 那一族）；
* **时间判据**：最新 mtime 早于**最近一次引用它的判读件入库日**才算冷数据，否则它可能正被一次在跑的长跑重写；
* **半径判据**：删除动作属他人半径，只能由 owner 排期执行；助手侧的合格交付是这份清单＋一条「删前必查」的可复算命令。

## 与本件有关的两条现状

* `output/` 本身不被 `.gitignore` 忽略（忽略规则里只有 `*.pt` 与 `*.jsonl` 两类，`.json` 不忽略），所以这些目录会在 `git status` 里逐枚显形——这也是清单能做、且必须按裸名扫的原因（同一根因见 `runtime-credentials-gitignore-anchoring` 那条踩坑）。
* 长跑器把进度文件写进 `output/`（本件现读到的那枚 `*_progress_exit.json` 就是证据），⇒ **任何「整族跑期间」拍的落点快照都不许当终态引用**，跑完必须重取。

