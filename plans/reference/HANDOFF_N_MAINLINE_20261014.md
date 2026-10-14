# N 主线停靠交接（2026-10-14，㊵-834 落笔时点）

> 这一份是**会话尽头用的停靠件**：下一个会话读它就能接上，不需要读会话总结、不需要猜步骤。
> 判读本体在 08 台账 ㊵-833／㊵-834；完成度对照在 `N_MAINLINE_COMPLETION_AUDIT_20261011.md` §5 第 11 条。

## §1 此刻的机器事实（全部现取，可复算）

- **接手第一步：`git log -1 --format=%H` 现读 HEAD**，不要抄本件的任何哈希——本件每改一次就滞后一笔。
  写作时点的 ㊵-834 系列＝`99866c1e0`（㊵-833 判读格）→ `26802e9f6`（㊵-834 仪器升版＋十六裁＋本停靠件）→ `a17e9b29c`
  （㊵-834 ⑧ 状态列更正），三笔都已推送、`## main...origin/main` 同步。
- 树面上还有 **6 枚假脏**（`reports/taiji_n5_07_*_20261010.json`）：逐枚 `git show HEAD:<路径>` 与盘上**同字节**、
  mtime 全停在 10-10T09:21:17Z、`crlf=0` ⇒ 是 Windows 的 stat 脏，不是谁的在飞改动，**不要**把它们卷进提交。
- 整族基线（`python -m pytest tests/ -q`，跑在 `74a084e22` 那棵树，37 分 24 秒）＝**`2 failed, 3332 passed, 23 skipped, 1 xfailed`**；两枚红同源，落在引用普查的契约册，根因是那两枚被队首点名却未入库的读数件 ⇒ **已由 ㊵-833 结清**（引用普查 `verdict` 由 `violated` 翻成 `holds`、`test_n5_38` 复跑 8 passed）。
- 本行改动面＝一枚仪器（v2）＋一本契约册（10→13 枚）＋四份文档＋一枚新读数件；**本轮没有重跑整族回归**，所以下一格若要出版"整族面"必须在零并发的窗内重取一次，跨批合计不许发表。
- 重复臂这条线**已关闭**：两臂 `runner_commit`＝`fe39dfadef6e…`，批次 C＝`9299352d439b…`，按 `PLAN-N5-23` §2/§4 判为「第四臂」，地板不出版；那对臂花了 15.70 小时 CPU 换回的读数仍在库里（`reports/taiji_n5_23_*_20261014.json`），只许连着并排仪一起引。

## §2 下一格的次序（照做，不要重排）

1. 起草 `PLAN-N1-05`（DEBT-G142①＝逐步再激活的持久载体另案）：判据先冻、开关默认关、零训练取证在前；形状约束＝运行时属性，不做 `TaijiConfig` 字段（`taiji/model.py` 的 `restore()` 要求配置逐位相等）。
2. 复跑整族基线一次（零并发窗）：`python -m pytest tests/ -q`，判读只认汇总行；出版面须点名它跑在哪个 HEAD。
3. 等 owner 打完两臂 B／G 共 80 题 ⇒ 才做 DEBT-G143（给 `PLAN-N5-07` 追加"只有 G 的硬安全失败入判据、数必须为 0"的落地注）。
4. 我这侧还挂着的两格：DEBT-G115 修法③ 的迁移队列（㊵-829 现读剩 17 枚候选，跑器本体 3 枚禁改）与 #81「统一三台仪器的拒绝出口形状」。

## §3 命令处方（每条都带全部必需旗标；漏旗标是响亮错不是静默错）

- v2 地板仪：`python scripts/training/audit_taiji_n5_23_repeat_arm_noise_floor.py --batchc-treated output/n5_20_batchC_treated/A/run_report.json --batchc-control output/n5_20_batchC_control/A/run_report.json --repeat-treated output/n5_20_batchC2_treated/A/run_report.json --repeat-control output/n5_20_batchC2_control/A/run_report.json --parity-batchc reports/taiji_batch_c_arm_parity_20261013.json --parity-repeat reports/taiji_batch_c2_arm_parity_20261014.json --out-report <新名>`（真面现读 `rc=1`、`failing_verdicts=["runner_commit_equal_across_campaigns"]`，这是**正确**的判红）
- 契约册：`python -m pytest tests/taiji_native/test_n5_37_repeat_arm_noise_floor_contract.py tests/taiji_native/test_n5_38_ledger_cited_reports_contract.py tests/taiji_native/test_n5_18_arm_command_recipe_contract.py -q` ⇒ 27 passed（rc=0）
- 门族（本行逐条 rc 全 0）：`audit_taiji_ledger_marker_uniqueness.py`／`audit_taiji_debt_table_column_shape.py`／`audit_taiji_ledger_cited_reports_are_tracked.py`／`audit_taiji_tracked_report_line_endings.py`／`audit_taiji_prereg_exit_wording.py --doc plans/reference/PLAN-N5-23_…md`／`ruff check .`／`black --check .`，另加 `pytest tests/seed/test_project_identity.py`（4 passed）

## §4 行号勘误（升版右移，已入库文本不回改）

`audit_taiji_n5_23_repeat_arm_noise_floor.py`：`METRICS` `:39`→`:46`；`_same` 五枚比较键 `:158-166`→`:190-197`；CAP 逐维地板 `:185`→`:217`；新判据键在 `:230-234`，资格键名表在 `:302-303`。㊵-833 与 §6／DEBT-G141 行里写的旧号以本件为准。

## §5 待你（owner）的事

- 两臂 B／G 共 80 题打分：清单＝`plans/reference/CAP_B_G_HUMAN_REVIEW_C2_TREATED_20261014.md` 与 `…_CONTROL_…`（各 40 格 `verdict = ______`）；批次 C 那两枚 V3 清单各 40 格也仍全空。
- 无需再裁的：地板重跑（十六裁⑵甲＝不花）、S1b 路线（十六裁⑴甲＝开载体另案）、B／G 口径（十六裁⑶乙＝只 G 的硬安全失败入判据）。
