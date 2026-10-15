# EXP-N1-15 · 已批准三种子三臂小试裁决（2026-10-15）

结论：正式包完整但行为门未通过，不进入C4或默认采用。依据 [14方案](PLAN-N1-14_native_candidate_paired_pilot_20261015.md)、[15运行合同](SPEC-N1-15_paired_runner_execution_contract_20261015.md)、[完整原始报告](../../reports/taiji_n1_native_evidence_paired_20261015_final.json)。用户授权的低成本自主准入继续有效，不因负结果停在权限询问。

## 完整性与读数

冻结提交61d5ab03b；运行命令 `python scripts/training/run_taiji_n1_native_evidence_paired.py --execute-approved`。v2 manifest SHA e95a2e3be6a3a4abca92ea583cf35522545961280b575e5d5373857973d5e979，执行identity与readiness逐项相同；评价前后源码/配方复核无漂移。正式报告 SHA ebc6a59623d0805ba8aede2cccc93a1460ed381ac1d860cbb79d78f258a1a9dc。

3seed×3臂×384更新＝3456，全部9模型完成训练后才评价dev/holdout各32题，共576行完整原始bytes。18次zero/trained独立weights-only恢复全通过，对照是父进程存档前真实参数/optimizer/来源/RNG及首步logits，不是两个恢复副本相互印证。CPU81.796875秒<300，父与子计量完整；模型档在data/checkpoints独立包目录，不覆盖母档。

| seed | position holdout | 每来源末态holdout | 信用截断holdout |
| --- | --- | --- | --- |
| 11 | 0/32 | 0/32 | 0/32 |
| 12 | 0/32 | 0/32 | 0/32 |
| 13 | 0/32 | 0/32 | 0/32 |

每类holdout均0/8，两对照平均优势均0；每seed16/32、每类2/8、平均优势4/32的原门均未达，不重定阈值或选最好seed。dev仅seed11末态臂1/32，其余0。所有576行自然boundary，没有cap；错误不能归因于统一停止失败。末轮平均teacher-forced CE约1.43—1.90，较首轮约3.60—4.14下降，但不能据此宣称内容学会或泛化。原始自由回答集中在unon/unknon/eron等错串；没有执行训练集自由生成，当前不能断言是泛化而非基础拟合不足。

## 全项目与唯一下一步

C1载体通过，C2严格事实收益仍零；11—13接口/信用/恢复工程证据成立，14/15新候选小试没有形成能力增益。C3b不是成功通过、C3c/C4与同bundle三器官未完成；N2/N3/N5历史负结果及N4/N6、CI/评分欠账保留。主线仍证据→消费→信用→净增益，不退回已关闭C2回放增益盲试，不把参考可微图当完整Taiji。

唯一下一步：一次≤60秒CPU的train-only拟合诊断，对九个已冻模型各96条训练episode测teacher-forced与自由输出，0参数更新、0dev/holdout新增查询，先区分已见内容是否拟合与自由运行是否偏移。保留本小试结果，不按训练诊断改变原holdout门，不立即重跑配方或增大模型。诊断合同单独登记后自动执行。
