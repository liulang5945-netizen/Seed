# U2 22.3.4.3六次更新与两次故障回执

> 文档状态：[证据] — 2026-11-11。裁决`passed_six_updates_two_faults_only`。8次attempt全部消耗，真实optimizer.step为6次，后继22.3.5完成最终schema十二类合验。

## 1. 资格、配方与来源

沿用4bb053886的[产物预合同](../manifests/taiji_u2_22_3_4_3_precontract_20261111.json)和1c7196eae检查点预检。当前owner“下一步”承接已明确工程范围，授权引用`chat:user-next-step-22.3.4.3-20261111`写入manifest、run.json及ledger头行。

先冻结[配方设计](../manifests/taiji_u2_22_3_4_3_execution_design_20261111.json)，零step资格检查通过后冻结[执行manifest](../manifests/taiji_u2_22_3_4_3_execution_20261111.json)。run_id为`20261111T123154Z`，完整证据见[汇总](../../reports/taiji_u2_22_3_4_3_summary_20261111T123154Z.json)及其8份S12/worker引用。

F1采用float64感知预测及actor/critic结果损失，F2采用float32逐字原文重现并保留预测/结果目标。F2请求为逐字输出当前唯一记录的完整原文，目标为`红盒在左边。`及EOS。Adam lr=0.001、betas=(0.9,0.999)、eps=1e−8、weight_decay=0，其他值见冻结manifest。

梯度由父参数拥有的合成影子片段计算，数据、行为及RNG由recipe固定，主体事实历史保持指定保存点。按参数名把梯度复制到同值fork，在durable open及一次性claim后执行唯一step，再经已验迁移事务提交。本轮按工程更新/恢复范围签署。

## 2. 实际结果

| attempt | 状态 | optimizer.step | 独立worker |
|---|---|---:|---|
| f1-step1 | accepted | 1 | pass |
| f1-step2-orig | accepted | 1 | pass |
| f1-step2-recovery | accepted | 1 | pass |
| f2-step1 | accepted | 1 | pass |
| f2-step2-orig | accepted | 1 | pass |
| f2-step2-recovery | accepted | 1 | pass |
| fault-rejection | rejected | 0 | pass |
| fault-before-commit | exception | 0 | pass |

各组step2从同一step1.bin读入，恢复更新在独立进程执行。两侧状态、参数、optimizer moment、历史引用及下一输出/策略精确一致。每份accepted worker对191参数及其moment核验Adam方程，全部worker为0额外step。

冻结活性参数的step1最大变化约0.001，step2约0.001001，逐参数梯度/变化均记入S12。8个规范checkpoint全部读回，正式运行实际readback为24次。初始Adam为合法空lazy状态，真实step1产生非零moment，恢复step2继承该实际moment。

两次故障使用F1/step1的父及实际moment和既有具名参数扰动，父文件与内存指纹均保持原身份，两个attempt占额。原始检查点位于预合同规定的隔离目录，供22.3.5继续核验。

## 3. 正常入口和前置检查

| 入口 | 责任 |
|---|---|
| [账务模块](../../scripts/training/u2_attempt_ledger_22_3_4_3.py) | fsync、前置计费、哈希链/事件摘要、对账与尾修复证明 |
| [更新支持](../../scripts/training/u2_update_support_22_3_4_3.py) | 影子片段、参数归属梯度、moment映射、即时读回及step claim |
| [attempt执行](../../scripts/training/u2_update_executor_22_3_4_3.py) | 一次更新/迁移或故障及证据载荷 |
| [运行入口](../../scripts/training/run_taiji_u2_22_3_4_3.py) | 固定顺序、恢复step2进程、存档/S12/close |
| [S12 worker](../../scripts/training/verify_taiji_u2_s12_22_3_4_3.py) | 重算梯度、Adam方程、状态重建、下一分布及账务核对 |

[最终ledger预检](../../reports/taiji_u2_22_3_4_3_ledger_verification_20261111_v2.json)12项及[claim回归](../../tests/taiji_native/test_u2_engineering_ledger_22_3_4_3.py)6项通过，均无真实optimizer调用。首轮开发ledger记录用于费用留痕，最终资格由v2与执行manifest绑定。

每行record_sha256及prev_line_sha256、汇总整账SHA一起核验。实际step前的claim使用create-once及fsync，worker重启不能重复同一attempt。run_start+open为2完整行、1条attempt事件；8/8表示规范存档覆盖，全部读回次数另计。S12预留close序号后，worker再核对实写序号。Windows目录fsync登记unsupported，保存后立即读回作为执行门。

## 4. 条件、成本与剩余范围

正式父/恢复更新worker/S12 worker合计70.1875 CPU秒、91.437 wall秒，低于100/300帽。CPU、intra-op 1、interop 12，F1 float64、F2 float32，Python3.12.10、Torch2.13.0+cpu。各S12/worker记录完整现场条件。

配方零step预检另实测7.4375 CPU秒，ledger最终预检0.71875 CPU秒，6项pytest为5.57 wall秒且CPU未逐次采集。dry-run父落盘/读回费用保存在run根，正式总数覆盖--run及其worker。

本轮0正式训练、0项目语料/confirm读取。工程检查点承担恢复输入，任务学习收益、旧能力资格和架构净益按22.4/5验收。当前进入22.3.5，整卡接受父尚待最终schema合验。
