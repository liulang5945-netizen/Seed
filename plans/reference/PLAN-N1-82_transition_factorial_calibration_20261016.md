# 82 四条件学习固定校准（2026-10-16）

81完整：CPU8.703125秒，fresh seed13零态实际保存/独立进程恢复、父子两步四条件签名逐位一致，主轨迹0更新，4副本步16合成QA。81报告SHA `ac9e5cb95eb8beda998cc12b37d9036963c275083e7008d7f8018571056f3c21`；零态SHA `c8579e7e8ea360d21cd10ecc4f5b393a19e53306a8567a507edaa199f5e4485a`。

## 唯一配方与解释边界

仅fresh seed13 position，选该seed因为80已定位其组合失败，不算随机种子确认。加载81完整全新零态，保留其RNG、空Adam/历史；不是续训77失败终点。结构62056参数/五组Adam规则/学习率不变，沿77四个6144周期、同192父题、原始/改名交替与43改名流；每次transition更新按81等权四条件，其余目标逐位沿69。固定24576更新、55296QA、43008独立训练会话，0预训练继承，无中途选点/追加/重新初始化。CPU≤1800秒（30分钟，含源链检查、独立子进程恢复、保存与评分），单线程串行。0dev/v3holdout/v4/v5；v5继续封存。

## 前置与完整证据

先冻结实施与artifact-free合同并提交，确认81完整源链与零态字节不变；在正式更新前写自己的zero并独立恢复，副本两步使用81同一四条件目标。零/终点各父子两步，累计8副本步32合成QA，不进入主轨迹。终点所有41参数Adam步24576，全部43008观察会话完整恢复，下一两步参数/Adam/RNG/history与五组读数逐位一致。偏离停止保留partial checkpoint/完整已得损失读数，不自动重试。

五固定评分点0/6144/12288/18432/24576：原192、pair96、已知改名192、改名pair96，共576；另原/改名transition各48父题四条件各192，共384；每点960行、共4800行。每次原始条件须与同点评分行逐项复现，全部评分前后完整bundle中性。保留24576步全部loss/五组measurement，不只均值，checkpoint不裁历史。

## 冻结判读

原四训练充分性门不动：原题≥173/192且每类≥40/48、pair≥87/96、改名≥173/192且每类≥40/48、改名pair≥87/96。新增原名与已知改名的四条件联合正确各≥44/48，仅是避免轴内捷径的训练前置。六门合取；不足即失败，不自动补训。通过也只允许据结果规划fresh多seed/控制与未见准入，不能据已知训练变体宣称泛化或whole-project完成，不能消费v5。

## 预声明产物

`scripts/training/run_taiji_n1_transition_factorial_calibration.py`；`tests/taiji_native/test_native_transition_factorial_calibration_contract.py`；`reports/taiji_n1_transition_factorial_calibration_20261016.json`、`_attempt.json`；忽略目录`data/checkpoints/n1_transition_factorial_calibration_20261016/{zero,final,partial}.pt`；结果裁决`plans/reference/EXP-N1-82_transition_factorial_calibration_adjudication_20261016.md`。81工程结果直接封入既有81计划；77/79/80/81既有代码和原报告不改。完成后五计划同步并只推荐一个下一步。全项目保持/C4/三器官/N系列及全HEAD CI债仍欠。

## 固定中点运行记录

执行器提交733b51a31；同一会话8895/PID36128，zero实际保存及独立恢复后才开始更新。6144：77/0/76/0，CPU351.98秒；12288：134/37/128/38，CPU668.28秒（依次原/pair/改名/改名pair）。这些只是固定检查点，不选点、不提前判门，后续联合四条件和完整恢复以终点报告为准。执行器/学习器未修改。
