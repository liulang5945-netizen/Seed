# 80 冻结世界转向组合因素诊断（2026-10-16）

79完整，seed13出现方向轴内输出塌缩；这是相关模式，先检验真实因素使用再改结构/目标。77原门失败、78不准入、v5未消费，所有原门不变。

## 范围与冻结身份

只用77的四个固定final：seed11/12/13 position＋seed13 cut；各24576更新、完整Adam/RNG/history及已验证恢复，不迁移/训练。77报告SHA dbfecdfbebe46cdd1401f8946020e2d7415eda9ffa0be523a7659d0cbf6fe2c3；79报告SHA ab01c31d1ed6852dbe2a0c48ca3fefd826c2ff2b72e7503eacfe53af2b8ed336。核验原始源链、九分片/18恢复与各选中checkpoint字节及完整bundle fingerprint等于77已封存parent signature；不运行签名副本更新。

## 固定四条件

原v3训练48项transition，保持实体/问题/观察来源顺序，固定：original；turn_flip(left↔right)；direction_opposite(north↔south,east↔west)；both。方向对换保持原token字节长度，turn词长度变化为自然语义输入变化，不能据单一turn变化独自排除位置效应。完整2×2设计使单翻因素的正确答案均为原答案反向，两者翻则回到原答案；答案只作评分/既有teacher目标，不作推理输入或oracle地址。四模型共享同192变体，共768完整teacher+free序列，保持自然boundary/cap64。只是已知训练任务反事实诊断，不算新未见确认。

原始条件必须逐项复现77对应transition的raw/teacher/stop等读数，任何不符整批不判完整；不因已失败种子跳过强种子或cut。逐题临时会话清理，加载后和全条件结束full bundle(参数/Adam/RNG/全部历史)完全相同，不写训练checkpoint。

## 读数、预算与停止

独立CPU≤60秒，单线程串行，0参数/优化器/副本更新，0dev/v3holdout/v4/v5查询。保留所有768原始行、每条件分母48、八格分布、逐题single-flip输出变化与double-flip不变、四条件联合正确/48；单纯变化或double不变不等于正确组合，描述性诊断不新增能力成功门。出错/预算触顶保留partial计数/全部已得行，attempt独占，不自动重试。

## 产物与顺序

scripts/training/diagnose_taiji_n1_fresh_transition_factorial.py；tests/taiji_native/test_native_fresh_transition_factorial_contract.py；plans/manifests/n1_fresh_transition_factorial_20261016.json（192项固定LF，精确路径LF属性，不重抽/覆盖）；reports/taiji_n1_fresh_transition_factorial_20261016.json及对应_attempt.json，固定LF，完整来源/代码/CPU计数。先合成合同验证四条件代数/元数据不泄露/完整状态中性/原始复现/部分行/拒覆盖，静态检查与提交后才执行。原77/79文件和模型不改。

完成后只提出一个与因果因素使用结果相符、上限更高的机制或学习目标新预注册，绝不在80内追加训练。保持/C4/同bundle三器官与N系列/CI出口仍欠。
