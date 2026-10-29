# 项目阶段收束：N177 E实现、推理恢复与F0a标定

日期：2026-10-29，Asia/Shanghai。审计基线`85e7267ef`，当时main与origin/main一致、tracked工作树干净。本次收束代码/报告/活动计划及明确结束的测试残留，不表示长期目标、E全部训练工程门或F1能力验收完成。

## 1. 已完成资产与证据

| 资产 | 可保留的结论 | 证据入口 |
|---|---|---|
| 原N177.6 | 15臂与原数值评价完成，固定切片正向收益保留；终点独立恢复余证仍交E0 | [原结案](../../reports/taiji_n1_plan177_6_t1_closeout_manifest_20261029.json)、[E决策](PLAN-N1-177_E_state_consumption_and_credit_decision_20261029.md) |
| N176 | 原1360份固定输出已生成并登记消费，sealed生成不能再当未消费面；输出在本机按hash保留 | [结案记录](../../reports/taiji_n1_plan176_sealed_closeout_20261028.json) |
| E1/E2a | 状态依赖查询合同与独立课程实现；新sealed未物化 | [E1合同](SPEC-N1-177_E1_state_consumption_contract_20261029.md)、[课程报告](../../reports/taiji_n1_plan177_e1_course_20261029.json) |
| E2b | 持久特征存储、查询选择及读出接口形成；query-only不携带事实字节 | [已知实体](../../reports/taiji_n1_plan177_e1_state_path_known_entity_20261029.json)、[组件恢复](../../reports/taiji_n1_plan177_e1_value_feature_restore_20261029.json) |
| F0a | byte value encoder与颜色分类头在64条train事实上完成200次更新；训练准确1.0、copy-only 1.0、no-state 0.078125；zero/final独立恢复通过 | [F0a报告](../../reports/taiji_n1_plan177_f_value_encoder_f0_20261029.json) |
| E3推理恢复子门 | v11/v12两次parse/span/base-adapter身份及下一字节logits一致；零更新、未读新sealed | [v11](../../reports/taiji_n1_plan177_f_full_e3_20261029_v11.json)、[v12](../../reports/taiji_n1_plan177_f_full_e3_20261029_v12.json) |

F0a是实际训练侧标定，不是零训练；它没有执行完整producer/V4链，不能把其单位成本扩充为12-run完整候选报价。训练侧1.0不证明未见泛化或相对复制的机制净收益。

## 2. 失败、限制及不合并的结论

- 新实体“笔”的producer抽取缺口保留；已知“钥匙”fixture通过不替所有实体补分。
- 固定探针中mean、value-span mean、逐byte特征均rank 2/原型25%，8色不可分；这是对应producer与探针的负结果，不是所有未来表征必败。
- value-token fixture通过控制和恢复，但与copy-only一致，只是离散复制下界；不把它计作连续特征已学会。
- [E3 v6](../../reports/taiji_n1_plan177_f_full_e3_20261029_v6.json)的lineage gap保留。span按字节排序后v11/v12严格推理恢复通过，修复不追改旧回执。
- 现E3预检在no-grad下比较内容/来源/下一字节logits；bundle没有完整新F训练状态。它不覆盖E1要求的credit-on/stop、有效禁写组、候选optimizer/RNG/cursor往返，不能写“完整E3合同全部闭合”。
- 当前预检普通`comparison.passed`与strict lineage分开，CLI退出码仍按前者；以后准入应同时核strict字段，不能仅凭rc=0。

## 3. 尚未闭合的前置与唯一接力

| 责任 | 待交付 | 顺序 |
|---|---|---|
| E0历史补证 | 原15臂终点fresh-process parse/answer回执，B真实输入/有效mask与隔离证据 | **当前唯一队首：证据收束包先补E0**；按历史源码或披露漂移，不用现排序修复替旧包追签 |
| E2c/训练态E3余项 | 当前F候选的真实任务信用on/stop、禁写组与optimizer/RNG/cursor独立恢复 | 与E0同属F1准入补证；已有推理恢复资产复用 |
| F0b | 完整producer/V4/decoder四臂×三seed的短成本/恢复标定 | 当前可准备manifest/脚本/采样口径；实际计时与模型恢复串行或资源独占 |
| F1 | 依F0b冻结更新/曝光、数值门、总预算与跨包停止线后，一次有界验证 | 未启动，不由本轮收束放行 |

F合同中已有方向/四臂与F0b限额，正式F1数值门、更新数和总预算仍未冻完。现有跨包“两次无净收益回诊断”规则继续保留。

## 4. F0b/正式冻结前必须澄清的合同项

1. F合同§4的初拟“1280 updates＝5个128-episode epoch、batch4”不自洽：完整遍历时5 epochs为160 updates，1280为40 epochs。保留冻结原文，本记录登记勘误；定价/正式冻结前选定唯一曝光口径，不能直接照矛盾句开跑。
2. F0a中copy-only已达1.0；F1要求main超过copy-only的判别力需检验。不能从train满分推出holdout必满，但也不能把复制可满分的端点直接当作机制净收益判据；正式读取holdout前必须澄清。
3. E0绑定的历史源已与当前producer/runner有漂移。将来的CI类型修正也不替旧源生成恢复证据；源码/制品身份分别登记。
4. 冻结件的阶段状态提示已经滞后，尤其E1首行与F的“完整E3”提示；当前验收完成范围以03及本收束记录为准，冻结实验要求不因此减免。

本轮不修改冻结SPEC正文、旧报告或任何阈值；这些项转入后续准入决策，而非补发训练授权。

## 5. CI与本机收束

基线`85e7267ef`的[fast lint](https://github.com/liulang5945-netizen/Seed/actions/runs/40853390200)失败在mypy：trace-cache的异构payload推断与三个新模块的Tensor/Any返回共9处。已定位到远端job日志，本机也复现同样9处；建议以局部类型注解/`typing.cast`修正。用户随后明确本轮只修计划，临时源码修改已撤回，实际修复/回归/推送由用户执行。完整主CI与m7-ci在查询时尚未完成，不能据fast-lint或本机检查声称远端全绿。

本次只修改方案/计划与导航；不改训练源码、冻结SPEC正文、参数或原结果，也不执行标定/训练/恢复。冻结原件、训练数据、checkpoint、机器失败报告与独有实现保留。

用户实施CI修复时按下表定位；仅补类型信息，不放宽CI基线或改张量运算。代码变更后的来源身份须重新记录，不能替旧实验追签恢复。

| 文件与基线位置 | 待处理内容 | 验收 |
|---|---|---|
| `taiji/joint_t1_trace_cache.py:362–363` | 异构payload推断为object，两项SHA应保持str类型 | 核心mypy、cache来源/恢复回归 |
| `taiji/joint_state_value_encoder.py:35/40/63` | 三处Tensor返回被推为Any | 类型约束＋encoder梯度/形状/恢复回归 |
| `taiji/joint_state_readout.py:57` | Tensor返回推为Any | 类型约束＋读出控制回归 |
| `taiji/joint_state_consumption.py:348/382/391` | 三处embedding表达式返回推为Any | 类型约束＋选择/来源/消费回归 |

E0实施顺序：先按各终点绑定的实现历史确定源码与制品SHA；在独立历史运行图中，用固定train/合成输入比较主进程与新进程parse/feature/answer；覆盖15臂并保存逐臂回执；明确哪些B真实链、有效mask、optimizer/cursor隔离已经覆盖。缺失源码/不等价先记录工程缺项，不能重训或重读holdout替代。F0b与恢复计时分开，避免用争用资源的样本报价。

## 6. 清理及历史恢复

建议候选仅为已结束的五个测试目录：`.seed_test_tmp/parallel-credit-20261028`、`parallel-credit-check-20261028`、`parallel-credit-complete-20261028`、`parallel-credit-final-20261028`、`parallel-credit-modes-20261028`（后四个同位于.seed_test_tmp）。本轮未删除、未统计释放量，也未产生清理完成回执。用户执行前逐个核对仓内绝对路径、tracked子项、ignore和链接边界，删除后记录实际状态/逻辑字节；不能整根清.seed_test_tmp，因为其中仍有源码overlay。

原有`.zcode/`、`delivery-runs/`、旧UI输出及未跟踪监督报告继续保留，未据“untracked”代删。阶段快照和重复长流水从活动正文蒸馏，完整原文可从`85e7267ef:plans/active/roadmap/03_CURRENT_EXECUTION.md`及同提交的10、E决策、N177恢复；原报告引用保持原位，不再复制整篇到archive。

## 7. 本轮交付边界

按用户最新要求，本轮只提交计划/方案文档。临时源码修正已撤回，未实际清理任何目录，也未执行新训练、模型恢复、F0b或sealed评价。九份计划文档的文件链接/章节锚点及diff静态检查通过；后续修复、回归及清理由用户按上述清单实施。
