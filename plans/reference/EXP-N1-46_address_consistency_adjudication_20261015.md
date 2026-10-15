# EXP-N1-46 自生地址一致性学习裁决

[完整报告](../../reports/taiji_n1_address_consistency_train_20261015.json)：168ffc9b0冻结后，41完整参数/Adam6144/RNG承接；288条自生教师缓存复现41且前后不变，576项zero逐项复现43 step0，zero独立恢复通过才训练。固定6144更新/9216训练QA/1728组校准，CPU118.6875秒；final独立恢复通过，Adam所有参数step12288，来源/代码/manifest守卫过。31相关合同通过。无v3 dev/holdout。

| 更新 | 原题/192（lookup/revision/transition/unknown） | 原pair/96 | 改名/192（四类） | 改名pair/96 |
| --- | --- | --- | --- | --- |
| 0 | 191（48/48/48/47） | 96 | 91（22/43/16/10） | 38 |
| 3072 | 161（43/48/43/27） | 88 | 102（19/48/27/8） | 44 |
| 6144 | 173（46/48/45/34） | 90 | 101（15/48/29/9） | 45 |

四门合取未过：原总173虽达线但unknown34<40；仅原pair过87；改名两门未过。不能改门、补训或择中点，也不能把地址干预92/96当原生能力。相较43，原题保持略好但改名101<108；这只是单seed开发比较，无统计净增益结论。新方案不默认采用，41历史训练门和33/43/46失败均保留。

唯一下一步47：冻结41和46，以已知训练衍生原/改名及pair做writer/question激活供体交叉，定位地址失败来源，而非继续调KL权重。全项目C1工程过、原C2严格收益仍零；未见确认、旧能力保持/C4/同bundle三器官/其他N与根CI债不变。最新远端核对根CI38521073885失败、m7-ci38521073776成功，均旧eaf53e7；本地检查不代表远端新HEAD通过。

## 收尾CI范围

全仓Ruff通过；全仓Black检查失败，14件需重排、1718件不变。上述14件对本轮起点837bf4dc9和工作区diff均为空，确认非44—46引入；未直接重排冻结证据依赖。20项计划收尾守卫过、git diff --check过。本轮新Python文件单独Black通过不能代替全仓格式门。具体欠账：

- `scripts/training/adjudicate_taiji_n4_four_class_acceptance.py`
- `scripts/training/audit_taiji_debt_table_column_shape.py`
- `scripts/training/audit_taiji_ledger_marker_uniqueness.py`
- `scripts/training/audit_taiji_n1_persistent_stepwise_carrier.py`
- `scripts/training/audit_taiji_n3_plateau_baseline.py`
- `scripts/training/audit_taiji_n3a_data_face.py`
- `scripts/training/count_taiji_n1_trajectory_following.py`
- `scripts/training/count_taiji_n2_03_attribution.py`
- `scripts/training/count_taiji_n2_faces.py`
- `scripts/training/count_taiji_n2b_faces.py`
- `scripts/training/report_taiji_n3a_calibration_cost.py`
- `seed/model.py`
- `taiji/adapter.py`
- `taiji/memory.py`
