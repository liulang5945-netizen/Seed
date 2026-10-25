# PLAN-N1-167：CC-2训练侧decoder与读出归因

日期：2026-10-25。状态：开工前冻结。用户要求本轮跨越更大：完整推进约定、实现、运行、归因、计划收束与提交，普通可逆修复不逐项询问。前置为[恢复v2完整结案](PLAN-R2-EVALUATION-RECOVERY-V2_execution_closeout_20261025.md)及用户确认48行判读；原dev面不重开。

## 0. VISION条款对账

| 继承条款 | 本包实际责任 |
|---|---|
| V-08/V-11/V-13/V-19；J1/VX3 | 比较真正字节消费者的条件概率，核对训练与生成路径，不用名称或格式指标替代内容能力 |
| V-20；J2/VX3/VX4 | 区分固定参数TF、逐token在线更新训练路径及prompt到plan的映射；本包不做参数学习 |
| V-33/V-35；N6/VX0 | 绑定checkpoint/代码/训练样本，恢复与不变检查，固定分母/资源/负结果出口 |

本包只诊断已有候选，不新增认知机制、训练配方或默认产品采用；不需要先完成完整自主性。完整判据见[SPEC-N1-167](SPEC-N1-167_train_decoder_attribution_20261025.md)。

## 1. 问题与固定对照

四端在原冻结decoder下均不可读，尚不能区分训练目标读出不匹配、只在真前缀下有效、plan预测失败、renderer本身未拟合，以及输入依赖过弱。

- 端点：parent_baseline、full_turn_1m、warm_start、fresh_same_capacity；沿已封存formal_001 readiness中的路径/SHA，用原只读loader恢复；不查询旧dev输出或更换候选。
- 共享样本：沿原train corpus顺序，取前4条包含汉字、target 48–384 UTF-8字节、prompt≤512字节、无非法C0控制字符的train episode；不足4条停止，不换过滤器。恢复仍使用完整train corpus确保原digest一致。
- 每端每题固定三种路径：configured（当前实际配置）、protected（仅既有predictive读出）、oracle_plan（仅在已有plan读出时注入该train题的目标plan到动态状态，参数不改）。共48计划单元，oracle不适用仍显式占位。
- 每单元：前128目标字节的TF原始/UTF-8掩码概率、top1/交叉熵/首错/相位分组，以及最多128字节自由生成。原configured生成与仪器化configured逐字节核对；另记录原`_target_pass(learn=False)`全目标读数。
- 每端每题增加一次固定的循环train-prompt替换TF，答案目标不变；报告与正确prompt的JS、top1一致率和NLL差。只作条件敏感度诊断，不外推整模型因果性。

## 2. 实现与执行

工具：`scripts/training/diagnose_taiji_n1_plan167_train_decoder.py`；合同测试：`tests/taiji_native/test_n1_plan167_train_decoder.py`。机器结果唯一位于`reports/taiji_n1_plan167_train_decoder_result_20261025.json`，partial/failure如实保留；模型/训练数据/checkpoint无写入。

按端点串行加载，Torch/BLAS单线程；每分支回到相同模型snapshot。任何状态型oracle注入仅发生在隔离副本并复原，禁止调用learn方法。每单元及端点结束核对模型digest，原checkpoint执行前后核对文件SHA；响应摘录仅来自train并标注路径/来源，oracle输出不能写成模型能力。

本次完整诊断上限CPU1800秒、wall2400秒，覆盖加载/恢复/全部固定单元/输出与失败；单列本包费用并披露既有训练/评价费用，不清旧账，不把旧dev资格作为资源。不得为取得好结果追加样本、beam搜索或训练。若触顶保留partial及缺项并结束该包。

## 3. 有限出口与后继

代码/合成检查、固定实测矩阵、原始输出/概率摘要、可接受或失败身份校验、实际费用、判据分类、计划与结案提交齐备即结束；负结果可完整结束。不能把四题诊断当L2/L3、泛化或三臂训练配方排序。

按SPEC分类给唯一后继：路径实现差异先修；否则根据TF/rollout、oracle/预测plan及protected对照选择一项必要的计算/学习主链改动。若不能区分，明确不可判，不用再加同质训练掩盖信息缺口。
