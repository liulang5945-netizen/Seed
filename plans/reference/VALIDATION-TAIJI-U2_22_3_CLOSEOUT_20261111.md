# U2 22.3完整恢复工程结案

> 文档状态：[证据] — 2026-11-11。裁决`passed_engineering_recovery_only`，22.3工程卡通过。交付CPU小profile初始化父，任务/学习/架构净益待22.4之后验证。

## 1. 接受身份

[接受父清单](../manifests/taiji_u2_22_3_accepted_parent_20261111.json)绑定源、schema、参数初值、34组最终核验、6次更新/2次故障、S12及其worker摘要。后继22.4以该清单的accepted_parent字段为输入。

父配方为U2Config.small、seed20261110、default dtype float32后转换至选中float32/64、CPU。参数191张量、主体40字段，runtime v3、原始链v2、主体v2。实例化时声明新的subject_id，optimizer由下一卡合同初始化。

| 精度 | 初始参数SHA256 |
|---|---|
| float64 | 6ec1c6db11373b5aba8676f25ef5a1bcd48f71430bc44dce9e228a8b833b3842 |
| float32 | b3e8a106ecdd34ad40f8983a7cb8c4c756097d0c089b013d1587f70a6397acfc |

该身份与本轮F1/F2 parent.bin的初始参数摘要逐一相符。工程step后的权重保留在隔离运行目录作为本轮核验输入，后继能力训练从明确声明的初始化及新任务合同开始。

## 2. 最终schema覆盖

[最终合同](../manifests/taiji_u2_22_3_closeout_contract_20261111.json)在读数前冻结，执行[核验入口](../../scripts/training/verify_taiji_u2_closeout_22_3.py)，原始结果见[结案报告](../../reports/taiji_u2_22_3_final_closure_20261111.json)。

| 状态族 | 最终核验 |
|---|---|
| S01–S03 | 初态、目标/记录、已采样未发出 |
| S04–S06 | 执行后未ack、pending切目标、未知结果 |
| S07–S09 | F编辑、UTF-8中间、完整实际输出 |
| S10–S11 | 重放/等待、两代出生 |
| S12 | 六个实际更新后的接受点、拒绝点、异常点 |

S01–S11每精度13个快照，S12取同一授权run的8个状态，共34组。控制与独立worker各自恢复初始状态，比较初始值/parameter/moment指纹、下一输出/策略，然后推进合法后继并测量预测/actor/critic及具名原文表达探针的191项梯度。探针的表达请求与唯一记录全文一致。

控制和worker使用由已验证snapshot物化的独立可控模拟器，实际run输入保持其来源身份。全部34组通过，191参数各比较34次，最大梯度差0，184参数至少在一个端点非零，未使用项继续单列。额外optimizer调用为0。

## 3. 更新与事务证据合拢

[真实更新回执](VALIDATION-TAIJI-U2_22_3_UPDATES_20261111.md)绑定run_id20261111T123154Z的8个open/8个close、6次Adam.step、8份S12/8份worker、8个规范存档及24次正式readback。恢复step2在独立进程执行，两组的原/恢复结果精确一致；accepted worker独立检查191项Adam方程，故障父/moment指纹保持其原身份。

接受清单另外核对闭包合同、更新汇总、ledger、每份S12/worker与8个checkpoint的冻结摘要。旧pending条件信用、有限窗、原始链、重编码及正常receive拆分对齐的报告均按其源码身份保留，最终schema通过由本回执单独签署。

本轮NDJSON账务及22.3 manifest命名空间通过目录级.gitattributes固定LF，Git blob与当前文件摘要逐一核对，避免Windows检出转换改变已冻结的原始字节身份。

## 4. 成本、条件和范围

最终父/worker合计90.671875 CPU秒、96.953 wall秒，低于120/300帽。CPU、intra-op1、interop12、float32/64，Python3.12.10、Torch2.13.0+cpu，OMP/MKL/OPENBLAS未设置。真实更新运行另用70.1875 CPU/91.437 wall，两个核验范围分别记账。

本结案0新增optimizer、0项目语料/confirm读取。既有开发pytest的CPU缺逐次采集，已在分卡回执登记，正式核验数字覆盖各自合同程序及worker。

实际覆盖为small profile的恢复、有限窗信用、参数重编码、真实更新与回滚。nominal profile的历史反向证据保留其原范围，相应规模的恢复若被22.4选中，需在同schema下重签。未建立U2 A能力父，接受清单的capability_qualifications为空，长期保持、实际自主效用及成熟生长按后继任务验收。

## 5. 唯一后继

进入22.4.0，先绑定接受父，冻结目标/输出请求、数据/反馈权利、模型可见元数据及合法可辨性。之后完成必要结构适配与三架构实际价格，按冻结预算准入22.5的A父获得、B学习与保持。长期23–28路线继续按各包实际父资格承接。
