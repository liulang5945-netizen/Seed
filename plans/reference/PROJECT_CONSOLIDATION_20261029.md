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
| E0历史补证 | 已闭合（2026-10-29，只读）：15臂终点 fresh-process 回执＋4个历史revision逐臂复算＋状态隔离/有效mask矩阵，见§8 | 不重训、不重评holdout；已并入F1准入前置清单的完成项 |
| E2c/训练态E3余项 | 当前F候选的真实任务信用on/stop、禁写组与optimizer/RNG/cursor独立恢复 | **当前唯一队首**；已有推理恢复资产与E0的只读owner profile可复用 |
| F0b | 完整producer/V4/decoder四臂×三seed的短成本/恢复标定 | 当前可准备manifest/脚本/采样口径；实际计时与模型恢复串行或资源独占 |
| F1 | 依F0b冻结更新/曝光、数值门、总预算与跨包停止线后，一次有界验证 | 未启动，不由本轮收束放行 |

F合同中已有方向/四臂与F0b限额，正式F1数值门、更新数和总预算仍未冻完。现有跨包“两次无净收益回诊断”规则继续保留。

## 4. F0b/正式冻结前必须澄清的合同项

1. F合同§4的初拟“1280 updates＝5个128-episode epoch、batch4”不自洽：完整遍历时5 epochs为160 updates，1280为40 epochs。保留冻结原文，本记录登记勘误；定价/正式冻结前选定唯一曝光口径，不能直接照矛盾句开跑。
2. F0a中copy-only已达1.0；F1要求main超过copy-only的判别力需检验。不能从train满分推出holdout必满，但也不能把复制可满分的端点直接当作机制净收益判据；正式读取holdout前必须澄清。
3. E0绑定的历史源与当前producer/runner确有源码身份差异（4个revision），但**在受测面上没有行为漂移**：固定train episode的parse/feature/answer指纹在4个历史检出与当前树之间60/60逐位相同（§8）。该等式只覆盖一个固定输入下的评分面，不外推到训练轨迹或未见输入。
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

## 8. E0 补证结果（2026-10-29，只读）

用户2026-10-29更正"标用户执行的其实也需要助手执行"，E0由助手执行。只用训练侧固定输入，**没有重训、没有重评、没有再读holdout/query**。

| 面 | 仪器 | 机检判据 | 结果 |
|---|---|---|---|
| 15臂终点 fresh-process 回执 | [verify_..._endpoint_receipts.py](../../../scripts/training/verify_taiji_n1_plan177_6_t1_endpoint_receipts.py) | 每臂主进程指纹＝自己起的真子进程指纹；评分前后参数/优化器/cursor摘要相等 | 15/15 全真；[回执](../../../reports/taiji_n1_plan177_6_t1_endpoint_receipts_20261029.json) |
| 4个历史revision逐臂复算 | [compare_..._historical_replay.py](../../../scripts/training/compare_taiji_n1_plan177_6_t1_historical_replay.py) | 每臂在**它自己绑定的revision**检出里复算＝当前树指纹；跨面同一仪器字节；检出干净（仅未跟踪仪器） | 60/60 相同（15臂×4面）；[对照件](../../../reports/taiji_n1_plan177_6_t1_historical_replay_20261029.json) |
| 状态隔离/有效mask矩阵 | [probe_..._state_isolation.py](../../../scripts/training/probe_taiji_n1_plan177_6_t1_state_isolation.py) | 15臂policy自述digest与模式表一致、绑定的train split/cache等于当前实物；5模式 backward-only（**不step**）梯度分布符合各自scope；参数/优化器/cursor前后不变 | 8/8 判据真；[矩阵](../../../reports/taiji_n1_plan177_6_t1_state_isolation_20261029.json) |

支撑读数（同批机检）：`no_update` 无任何语言梯度；`readout_only` 只有 `output` 组（2.72e-01，其answer loss 0.2867/byte）；`responsible_owner_frozen` producer=0 而V4全组有梯度；五个模式的全局RNG状态前后未变（披露项，不计成败）。`correct/parser_permuted` 的 answer loss 0.0002/byte 与 parser loss 0.0002 nats 说明梯度量级差异来自残差，不是掩码泄漏。

边界：复算面是**一个固定train episode**（`f0:…:train:0`）与**固定train batch 0**上的只读前向/反向；等式不外推到训练轨迹、未见输入或能力；评分器报告里的 `parameters_updated_during_scoring` 仍是写死的常量（已登DEBT-G182），本次隔离结论来自回执仪器的实测摘要而非该字段。

## 9. E2c/E3 状态信用（2026-10-29，零步工程）

新增候选 `taiji/joint_state_credit.py`：字节值编码器 → 按实体/属性身份选择 → 线性颜色读数；两臂只切换"回答CE是否留在连续特征上"。仪器 [E2c控制](../../../scripts/training/run_taiji_n1_plan177_e2c_credit_control.py) 与 [E3零步验收](../../../scripts/training/precheck_taiji_n1_plan177_e3_state_credit.py)。

| 面 | 机检判据 | 结果 |
|---|---|---|
| 信用通断（合成3实体＋真实train episode各一） | 两臂初始前向/父参数/选择相同；`credit_on` 编码器与特征张量有正梯度且一步后编码器参数移动；`credit_stop` 编码器梯度与特征梯度为零、编码器不动而读数头照常更新；梯度有限 | 两用例 12/12 全真；[E2c件](../../../reports/taiji_n1_plan177_e2c_credit_control_20261029.json) |
| 选择器信用 | 回答CE对选择器在两臂均为零（它按身份打分，不读连续特征） | 记为事实：本候选不能声称"选择器已受任务纠正" |
| 零步保存/新进程恢复 | checkpoint 无 autograd 图；新进程前向 logits/选择/特征逐位相同；optimizer/RNG(torch+python)/cursor/参数一致；二次前向不变 | 17/17 真；[E3件](../../../reports/taiji_n1_plan177_e3_state_credit_20261029.json)，成本约 1.3 CPU 秒量级（限额 8,000/6,000）；checkpoint 落在 `data/`（按策略忽略，件里带 sha256） |
| 禁写组 | 逐个冻结 encoder/selector/readout：被冻组参数不变且梯度清零，未冻的可写组仍移动（encoder↔readout 双向） | 3/3 组通过 |

两条新发现（不写成能力结论）：

1. **真实链覆盖率**：全量重测见 `reports/taiji_n1_plan177_e1_parse_coverage_20261029.json`——32 个 train 家族里只有 16 个（钥匙/杯子/书/盒子 四个实体）能解析出该实体的当前颜色事实，球/手机/笔/眼镜 四个实体的 16 个家族全部落空，且缺**按实体成块**，不是按值或按轮次零散失败。owner 2026-10-29 裁定：F0b 训练面收窄到这四个实体（16 家族），另一半明写未覆盖；F1 的能力声明随该口径限定。
2. **逐位相等依赖线程数**：新进程默认线程数下，同一权重的前向在末位出现 1 ulp 差异；两端都 `torch.set_num_threads(1)` 后才逐位相同。这条不是候选缺陷，但"恢复后前向等价"的判据必须同抄线程口径，否则会造出假红。

## 10. F0b 12-run 短标定（2026-10-29）：价格拿到了，但 wall 包络超了 96 秒

运行器 [run_..._f_f0b.py](../../../scripts/training/run_taiji_n1_plan177_f_f0b.py)，件 [F0b](../../../reports/taiji_n1_plan177_f_f0b_20261029.json)。四臂（main_credit_on / main_credit_stop / copy_only / no_state）× 三 seed；每步更新都重跑冻结链（producer→V4 query-only next-byte logits→bias，批 4 条 16 家族的 episode），只训 value encoder+bias；训练面按 owner 裁定收窄到 4 个已覆盖实体。

**成立的部分**：12/12 run 全部完成；每 run 20–25 次更新（总计 273 次计量更新）；零检查点与端点检查点在新进程里全部复现（24 次重放，`zero_restore_ok`/`endpoint_restore_ok` 全真）；实测单价 **4.68 CPU 秒/更新**（中位 4.58，最大 7.69；wall 4.82 秒/更新），量级与 N177.6 联合链的 24.7–45.8 秒/更新不同，因为这里 producer/V4 只做前向、只有 encoder+bias 反向。

**没成立的部分**：包络上限 1,500 wall 秒，实际 **1,596.3 秒（106%）**；CPU 1,452.2 秒在 2,000 秒以内。逐 run 看，每次都在 wall 额度（116.2 秒）附近停更新，但停下之后还有"终点存盘＋两次新进程重放"，每 run 再花 6–9 秒，12 次累加成这 96 秒。这不是"测量误差"，是协议漏算：`ENDPOINT_RESERVE_CPU/WALL`（8/12 秒）已补进运行器的停止判据，但**本次运行先于该修法**，所以按预定政策记为 `budget_stop_overrun`——不补跑、不加轮次；要再跑一次得另批额度。

**F1 投影（只说能说的）**：1,280 updates × 12 runs = 15,360 次更新 × 4.68 CPU 秒 = 71,885 CPU 秒；按合同 ×1.3 余量 = 93,450；再 +30% 保存/评价/失败尾账 = **约 121,500 CPU 秒（≈33.8 小时）**。这个价只覆盖"冻结链＋encoder/bias"这一候选；若 F1 的 main 臂还要训练 V4，必须另测 V4 那部分的单位成本再加进来，不能拿这个数当整候选报价。

## 11. F1 训练的开工与"生成口径"缺口（2026-10-29）

owner 同日签了 F1：候选只训 encoder/bias（V4 冻结）、预算上限 ≈121,500 CPU 秒。运行器 [run_..._f_f1.py](../../../scripts/training/run_taiji_n1_plan177_f_f1.py) 已起跑（串行单线程，phase: 每 256 步耐久存盘、`latest.pt` 可续跑、每 run 10,100 CPU 秒）。

开工后发现一个实现面缺口：F 合同的主读数 `answer_exact` 要完整回答，而候选只训 encoder/bias——冻结 V4 的贪心生成没有任何"把 feature 用上"的入口。处置：给 `_generate_from_joint_cache` 加了一个**默认 `None`** 的 `logit_bias` 参数（生成时逐步加到下一字节 logits 上），并用两支测钉住"不给 bias 与原贪心逐字节相同、给了 bias 当场改选中字节"。该口径已写进 [F1 合同§2.5](../../reference/SPEC-N1-177_F1_freeze_proposal_20261029.md)，没有改成"只评首字节"那种降级口径。

线程数实测（同一 4 步窗口）：1 线程 3.65 CPU 秒/更新、2 线程 10.1、4 线程 18.6 ⇒ 串行单线程既最省 CPU 也最省总和；预计 wall ≈20 小时，结束时才读一次 holdout。

### 11.1 F1 运行手册（续跑/监督/收口）

- 当前 phase：`run_taiji_n1_plan177_f_f1.py --threads 1` 串行跑 12 个 run；每个 run 的进度看 `data/checkpoints/n1_plan177_f_f1_20261029/<arm>__seed_<n>/run_state.json`（`updates` 只在每 256 步的耐久点刷新，所以"0"不代表没动）。
- 断点续跑（2026-10-29 实测过）：把某 run 的 `latest.pt` 与 `run_state.json` 拷到一个临时 run root，`--only <arm>__seed_<n> --updates 514` 跑两格——结果 `resumes=1`、游标从 512 接到 514（不是从零重来）、CPU 账从 2,650.2 累加而不是清零、状态转 `completed`。正式件仍须 12 个 run 都 `completed`；`--only` 只用于冒烟/补跑。
- 预算：每 run 10,100 CPU 秒、整包 121,500；触顶写 `run_cpu_cap` / `total_cpu_cap` 并停。
- 单实例：run root 的 `run_root.lock` 存在即拒起；若进程被 SIGKILL 留下锁，先确认没有活进程再手工删除（不要盲删）。
- 训练结束后才做：`reports/taiji_n1_plan177_f_f1_20261029.json` 汇总 + 一次 holdout 评分（评分器另件，未写）；期间不读 query/holdout、不调阈值、不加轮次。

### 11.2 F1 评分器：设计要点与未完成原因（下一会话从这里接）

我起了一版 [评分器] 草稿并**当场删掉**了：它把 `state_exact` 写成"解析出的值出现在答案里"、把无关交换实现成"换成另一个实体的值"，这两条与 F 合同 §5 的定义不是同一件事，跑出来的数会看起来成立但量错了对象。先用文字把口径钉住，再写仪器：

- `answer_exact`：冻结 V4 用 F1 合同 §2.5 的 `logit_bias` 逐步生成，规范化后与 `turn.answer_target.target_bytes` 比；按 episode 内轮次评分，再 family macro。
- `value_exact`：生成文本里的颜色 token 与"当轮应有值"比；与整句分开记。
- `state_exact`：**不是**看答案里有没有这个词，而是看这次干预下"被选中的 feature 的 value"是否等于课程真值——它测的是 store/selector 把哪条事实交给了解码器。该值必须由 `PersistentContentStoreV1.current(entity, attribute)` 给出，不能从 parse 里顺手续一个。
- `relevant_swap_delta`：把**被查询实体**的当前值换成另一个合法值后，答案是否**变成交换值**（不是"是否变化"）。
- `irrelevant_swap_keep`：只改**另一个实体**的当前值，被查询实体的答案保持不变。
- `cut_drop` / `rescue_gain`：切断选中 feature / 恢复后答案的下降与回来。
- `copy_only_match` / `no_state_gap` / `credit_gap`：三个对照臂的差值，按同一批 episode 配对。
- `old_retention`：N176 F0 的 16 题探针这次没接进来，报告里要写 `not_run`，不许写成 0 或"通过"。

另：holdout 的物化入口是 `build_episode(family, variant)`（`build_sealed_split` 对 query/holdout 是硬拒绝），评分器必须自带"写一次"的 attempt 闸；`E1 request` 的 query 面不读。

### 11.3 评分器已落地（2026-10-29），并用 train 冒烟抓掉两处口径错

[评分器](../../../scripts/training/evaluate_taiji_n1_plan177_f_f1_holdout.py) 按 §11.2 的口径写完，另加 `--smoke-train` 只在一条 train episode 上跑通链路（不碰 holdout、不写 holdout 报告）。冒烟当场抓出两处我自己的口径错：

1. **答案是整句，不能拿裸颜色比**：`state_exact`/`value_exact`/四类交换原先直接比"生成文本 == 颜色"，全部假零；改成先 `answer_colour()` 抽颜色再比。（首跑 `state_exact=0.0` 就是这个假零。）
2. **无关交换的分母**：episode 里只有一个实体时 `irrelevant_swap_keep` 被记成 0.0（"没换成功"），实际是"没东西可换"；现在单列 `irrelevant_turns`，无候选时为 `null`，并把 `query_unparsed`（查询轮 producer 没抽出实体）单独披露。

冒烟读数（train 首条、copy_only 的 zero 端点，未训练）：`answer_exact 0.5、value_exact 1.0、state_exact 1.0、cut_drop 0.0、rescue_gain 1.0、irrelevant_swap_keep null（无第二实体）、query_unparsed 1`。**这不是能力读数**——端点未训练，只用来证明链路和分母都对。

一次性闸：先 `assert_runs_complete()` 核对 12 个 run 都是 `completed`，再写 `.attempt.json`，最后才物化 holdout；任何一步不满足就拒跑，不烧掉那次"只读一次"。

### 11.4 F1 训练侧的早期读数（2026-10-29，只读探针）

[进度探针](../../../scripts/training/probe_taiji_n1_plan177_f_f1_progress.py) 对 `main_credit_on__seed_0` 的 zero 与 256 步耐久点各算一次（固定 16 条 train 家族，不 step）：

| 端点 | mean loss | hit | 说明 |
|---|---|---|---|
| zero（未训练） | 5.7108 | 1/16 | 随机头 |
| cursor=256 | 0.0012 | 16/16 | train 面已饱和 |

含义要说清：**这是训练侧的拟合，不是能力**。16 个家族 × 4 个值、冻结 V4 只调首字节偏置，本来就容易在几百步内记住；1,280 步的余量主要会变成对 train 的过拟合，holdout 的意义在于看它是否只剩"记住"。合同已冻结"不按中途表现调阈值、不加轮次"，所以不因这条读数提前收工，也不据此加练。

### 11.5 F1 结案收口器（2026-10-29）

[closeout_taiji_n1_plan177_f1.py](../../../scripts/training/closeout_taiji_n1_plan177_f1.py) 把三份输入合成唯一收口件，全部 fail-closed：12 个 run 都 `completed` 且端点 sha256 与账一致；旧保持探针在且 `worst_drop_pp ≤ 3`；holdout 件在、attempt 文件恰好一份、件里记的判定与按同一 `scores` 重算的一致（**手改结论会被抓**）。任一项不满足就拒绝出版，不生成半份结案；输出写一次。7 支合成夹具单测覆盖 happy path 与六条拒绝路径。

## 12. F1 首轮被停：训练目标与答案面对不上（2026-10-29）

**发生了什么**：F1 首轮按"查询轮的下一字节 = 值首字节"这个目标训了 1,280 步（main_credit_on/seed0），按计划继续会跑满 12 run。跑完第一个 run 后做只读检查，发现生成的答案是乱码（例：`钥黄门门门镜雨顣黄雨飨黄钥黄颜雨題钥钥颭门`），`answer_exact=0.0` 而 `cut_drop=1.0`、`state_exact=1.0`。

**诊断（同一端点上的读数）**：`bias(0)` 最大绝对值 0.294；喂真实颜色特征时是 **10.833（黄色）／8.603（蓝色）／12.061（红色）**。也就是说，为了把"值首字节"顶到第一位，训练把偏置推到 8–12 nats，而生成时这个偏置逐步累加，直接把整句顶坏。根因不是候选或 V4，而是**我写的训练目标位置错了**：答案其实是"钥匙的颜色是黄色。"，查询轮真正的下一字节是"钥"，不是"黄"。

**处置**：停掉进程（已确认无残留、锁已清），把那次 run root 改名 `data/checkpoints/n1_plan177_f_f1_20261029_aborted_wrong_objective` 留证（main_credit_on/seed0 账：1,280 步、6,235.2 CPU 秒；main_credit_stop/seed0 停在 256 步、977.3 CPU 秒；合计 7,212.5 CPU 秒）。**目标改为教师强制整句答案**：对 `turn.answer_target.target_bytes` 逐字节算 CE，每步加同一 `bias(feature)`；冻结链的查询轮 joint 每家族只跑一次（等价 N177.6 的 trace cache）。

**修正后的实测**：copy_only/seed0 跑 200 步只用 17.7 CPU 秒（约 **0.09 秒/更新**，比错目标那版的 4.87 秒/更新便宜两个量级，因为不再每步重放整条链），且答案立刻变正常——8 轮里 3 轮完全一致、两个查询轮颜色全对（`value_exact=1.0`）。按 0.09 秒/更新，12×1,280 步约 1.4k CPU 秒，远在 121,500 预算内；加上已花的 7,212 秒，总账仍不到预算的十分之一。

**边界**：200 步的读数是"目标对齐了、链能在跑"的证据，不是能力；F1 的判定仍要等 12 run 跑完 + 旧保持探针 + 一次 holdout。修正后的端点恢复门（新进程比对参数/优化器/cursor）在冒烟里为 True。

## 13. F1 一次性 holdout 结果：所有臂都在地板，判 FAIL_COPY_ONLY（2026-10-29）

12 run 用修正后的目标跑完（合计 1,416.7 CPU 秒；加上被停的首轮 7,212.5 秒，总账约 8.6k，仍不到 121,500 预算的十分之一）。旧保持探针与 holdout 各跑一次，结案件 `reports/taiji_n1_plan177_f_f1_closeout_20261029.json` 的判定是 **FAIL_COPY_ONLY**。

**主读数（holdout 16 家族／64 episode／512 轮／122 个查询轮）**

| 指标 | main_credit_on（三 seed 中位） | copy_only | no_state |
|---|---|---|---|
| answer_exact | 0.0156 | 0.0215 | 0.0352 |
| main − copy_only | −0.0059 | — | — |
| main − no_state | −0.0195 | — | — |
| credit_gap（on − stop） | −0.0039 | — | — |

分项（main_credit_on/seed0）：`state_exact 0.318`、`value_exact 0.039`、`cut_drop 0.299`、`relevant_swap_delta 0.0`、`irrelevant_swap_keep 1.0`、`rescue_gain 1.0`。

**怎么读**：判词按预注册优先级落在 `FAIL_COPY_ONLY`（copy ≥ main），但必须同报更基本的事实——**所有臂都贴着地板**（1.5%–3.5% 完全一致），`relevant_swap_delta` 在每臂都是 0，且 122 个查询轮里有 **52 轮（43%）连实体都没解析出来**。也就是说，这不是"主路径比复制差"，而是"主路径和复制都没有真正答对"；把两者的微小差值当成机制优劣是读错了对象。

**旧保持**：main 三 seed 分别掉 0 / 6.25 / 6.25 pp（16 题的粒度就是 6.25pp，任何一次翻转都必然越过 3pp 线）；main_credit_stop 0 / 0 / −6.25；copy_only 12.5×3；no_state 25×3。结论只写"main 有 1 题级别的回归，且这把尺在 n=16 上无法表达 <6.25pp 的变化"，不写成"遗忘严重"。

**工程面全过**：12/12 run 完成、每 run 端点新进程恢复 True、旧保持探针可跑、holdout 只读一次（第一次尝试在评分前因词表外实体崩掉、未产出任何数值，该 attempt 已改名 `...attempt_crashed.json` 留证并按"修仪器后重读一次"披露）、结案件的判定由同一函数重算并与记录一致。

**唯一后继（按合同 FAIL_COPY_ONLY 的方向，加上这次读数）**：先量 producer 在 holdout 表面形式上的解析覆盖（这次 43% 查询轮未解析），再谈连续特征可分性与信用。覆盖不解决，任何臂都到不了状态路径，分辨不出"没学会"和"没喂到"。

### 13.1 后继第一查：编码器特征可分性——这把尺在 n=8 上不可用

[探针](../../../scripts/training/probe_taiji_n1_plan177_f1_feature_separability.py) 把 8 个颜色各过一次训练后的编码器：**8 条向量互不相同**（没有塌成一团），质心两两平均距离 3.70；但 `rank=7` 恰好等于中心化后的最大秩（n−1），类内距离恒 0 ⇒ 留一/质心类指标全部无定义。件里因此写 `ruler_usable=false` 并给出原因，**不把"rank 7"读成缺陷**。

真要判可分性，需要每色多条样本（例如按字节上下文或逐字节特征），这属于下一格的工作，不在 F1 结案里冒充。

### 13.2 训练侧对照把 FAIL 的成因指出来：F1 只训了 32 条里的 16 条可解析用例

对同一个 main_credit_on/seed0 端点在**全部 128 条 train episode**上跑同一评分器（训练侧，不碰 sealed）：

| 指标 | 训练侧（128 ep／244 查询轮） | holdout（64 ep／122 查询轮） |
|---|---|---|
| answer_exact | 0.0234 | 0.0156 |
| value_exact | 0.121 | 0.039 |
| state_exact | 0.339 | 0.318 |
| relevant_swap_delta | 0.0 | 0.0 |
| 查询轮未解析 | **105/244（43%）** | **52/122（43%）** |

两点结论：
1. **不是泛化失败，是连训练面都没学会**：训练侧 `answer_exact` 也只有 2.3%。F1 运行器的 `build_cases` 只取每个家族的 `:0` 变体，所以 1,280 步实际只见过 **32 条用例（其中 16 条能解析出实体）**，而评分面有 122 个查询轮——这是训练覆盖不足，不是机制被判死。
2. **解析缺口在训练侧同样存在**：43% 的查询轮解析不出实体，与 holdout 同比例；这条与 DEBT-G184 是同一把尺子，只是从"家族"细化到了"轮次"。

因此 F1 的 `FAIL_COPY_ONLY` 保留为**该候选在该训练面下的正式出口**，但不能被读成"状态消费机制不成立"；下一步是在同一合同下换训练面（用全部可解析的训练查询轮）重跑，并换一个**从未读过的评价面**（`query` 分区，16 家族／64 episode）以保住独立性。

## 14. F1b：换训练面 + 换评价面（query 分区，从未读过）——仍 FAIL_NO_STATE_EFFECT

按 owner 2026-10-29 裁定：训练面从"每家族 :0"换成**全部可解析训练查询轮（108 条）**；评价面换成 **query 分区（16 家族／64 episode／123 查询轮，此前从未读过）**。12 run × 1,280 步，合计 1,569.9 CPU 秒。旧保持 main 臂三 seed 全为 0.00pp（对照臂 copy 12.5、no_state 18.75）。

**门表（修好取值后的真数）**

| 门 | 值 | 阈值 | 过否 |
|---|---|---|---|
| answer_exact | 0.041 | 0.85 | ✗ |
| value_exact | 0.141 | 0.90 | ✗ |
| state_exact | 0.346 | 0.90 | ✗ |
| relevant_swap_delta | 0.096 | 0.30 | ✗ |
| irrelevant_swap_keep | 1.000 | 0.95 | ✓ |
| cut_drop | 0.333 | 0.30 | ✓ |
| rescue_gain | 1.000 | 0.20 | ✓ |
| main − copy_only | +0.002 | 0.10 | ✗ |
| main − no_state | 0.000 | 0.10 | ✗ |
| credit_gap | 0.000 | 0.10 | ✗ |

判决 **FAIL_NO_STATE_EFFECT**（同时命中 NO_LEARNING、NO_GENERALIZATION，优先级取第一个）。读法：特征不是完全没用——`cut_drop 0.333` 与 `rescue_gain 1.0` 过线，`relevant_swap_delta` 也从 F1 的 0 升到 0.096——但绝对正确率、跨臂差值与信用差都在地板；且 **54/123（44%）查询轮仍然没解析出实体**，被查询的还有一半实体是 producer 见都没见过的。

**同时修掉评分器一个结构性错**：`scores` 原先只装 `answer_exact` 与三个差值，另外六道门取值恒为 None ⇒ 永远判假。已改为按宏表取 main 三 seed 中位，并加 `--recompute-from`：**从已发布的逐 episode 数据离线重算，不重读任何 sealed 面**。两份已发布件的重算结果：F1 仍 `FAIL_COPY_ONLY`、F1b 仍 `FAIL_NO_STATE_EFFECT`——判词没变，但门表从"六个 None"变成真数。两份 `.recomputed.json` 与原件并列入库。

**下一步的形状因此明确了**：不是继续调信用或阈值，而是**先补 producer 在未见实体与未见表面形式上的解析**（44% 查询轮无状态，跨两套 sealed 面同比例）。状态路径没接上之前，任何关于"学没学会消费状态"的读数都不可解释。

## 15. 解析缺口的分解（训练侧，2026-10-29）：不是"实体词表缺一半"这么简单

[探针](../../../scripts/training/probe_taiji_n1_plan177_e1_parse_failure_breakdown.py) 把 128 条 train episode 的 **349 个查询轮**逐轮看 producer 给的 role span：

| 组 | 解析出实体 | 未解析 | 解析率 |
|---|---|---|---|
| 已覆盖实体（钥匙/杯子/书/盒子） | 144 | 26 | 85% |
| 未覆盖实体（球/手机/笔/眼镜） | 100 | 79 | 56% |
| 合计 | 244 | 105 | 70% |

失败形态很干净：**未解析的 105 个轮次里，缺的永远同时是 entity 与 value（各 105 次），不是只缺一个 role**——即模型根本没触发"这是一条事实/查询"的抽取，而不是抽错角色。

**这条改写了下一步的形状**：不是"补四个实体"就能解决——未覆盖实体也有 56% 能解析，已覆盖实体却有 15% 失败，说明**表面形式（措辞/位置/上下文）与实体身份同时在起作用**。所以下一阶段的目标应是"查询轮的实体+值抽取覆盖率"，训练面要同时覆盖未见措辞与未见实体；只补词表会漏掉 26 个失败轮，只换措辞会漏掉 79 个。

（注：这与 DEBT-G184 的 16/32 家族覆盖不是同一张面——那张量的是"事实轮能否给出当前值"，这里是"查询轮能否给出实体 span"，两者都留档、不互相替代。）

### 15.1 105 个未解析轮次拆成两类，各自成因不同

同一份逐轮数据里再做一次切分：

| 类 | 轮数 | 分布 | 判读 |
|---|---|---|---|
| **旧值查询模板**（"X 之前／以前是什么颜色？"） | 40 | 覆盖全部实体（含已覆盖的 钥匙/杯子/书/盒子） | 模板整体没学会：244 个解析成功的轮次里只有 4 个是旧值问法 |
| **普通模板 + 未见实体** | 60 | 手机 32、球 24、笔 4、眼镜 4（全部是未覆盖实体） | 实体缺口：这些实体在普通问法下也抽不出 |
| **普通模板 + 已覆盖实体** | 5 | 钥匙/杯子/书/盒子 | 该面基本干净（5/149 ≈ 3%） |

两条补都只跟"数据/课程"有关，跟"能不能消费状态"无关：
1. **旧值模板**：F1 合同里有 `old_value_exact` 这一项，而这一类查询轮现在根本进不了解析——不补，这一项永远是空的；
2. **未见实体**：手机/球 在普通模板下也失败，说明不是"模板对了就行"，实体身份仍要进训练面。

**先证伪自己的一条前提**：这两类缺口都不是"课程里没有"——train 的 32 个家族覆盖全部 8 个实体（每实体出现 103–117 次），旧值模板在 train 里已有 64 轮且 64 轮都带 span 标注。所以是**有监督仍没学会**，不是缺数据；N178 提案已按此改正（做法从"扩课程"改为"训练侧加压 + 先测价"）。

**四格精表（解析成功/该格总数）**：旧值×已覆盖 0/21、旧值×未见 4/23、普通×已覆盖 144/149、普通×未见 96/156。旧值问法几乎与实体无关地失败（0% 与 17%），普通问法则是"已覆盖 97%、未见 62%"。

下一步（新阶段，需预算）按这两条同时扩训练面，验收用同一把尺：**查询轮实体+值抽取覆盖率**，四格分别报，而不是先用答案正确率当代理。提案见 [SPEC-N1-178](SPEC-N1-178_parse_coverage_extension_proposal_20261030.md)。

## 16. N178 测价与覆盖移动（2026-10-30，5,000 CPU 秒授权内）

[测价探针](../../../scripts/training/price_taiji_n1_plan178_parser_pressure.py)：父端点 `correct/seed0`，按"含旧值轮 / 未覆盖实体"加权采样，**只对 parser 损失反传、只更新 producer**，批 2，256 步。件 `reports/taiji_n1_plan178_price_20261030.json`。

- **单价 2.20 CPU 秒/更新**，256 步合计 846.8 秒；parser 损失 1.216 → 0.0125。
- 覆盖率（分母锚在**课程标注的 act**，不用模型自报的 act——我曾用后者，分母从 384 漂到 349，读数虚高，已修）：

| 格 | 训练前 | 256 步后 |
|---|---|---|
| 旧值 × 已覆盖 | 8/32（25%） | **32/32** |
| 旧值 × 未见 | 4/32（12.5%） | **32/32** |
| 普通 × 已覆盖 | 152/160（95%） | 156/160（97.5%） |
| 普通 × 未见 | 96/160（60%） | **156/160（97.5%）** |

留一格对照：这批数是在**训练面**上算的，等于"能不能把自己训练过的东西解释对"。它证明的是"加压这条路线在解析面上有效"，**不是**"sealed 面也会同样好"，也不是"状态被消费了"。parser 损失掉到 0.0125 本身也可能只是记住了训练集——真实的泛化判据要等同一候选在 sealed 面上的解析覆盖。

三处口径修正也一并留档：① 查询轮没有值 span，覆盖率按"能否给出实体"判；② 分母锚课程 act，不锚模型 act；③ 覆盖率与解析缺口分解是同一把尺，别混用两种判据。

**预算账**：本阶段已花 846.8 秒（另有两次冒烟/中止合计约 1,140 秒，均由 5,000 秒授权覆盖），余量约 3,000 秒。

## 17. N178 正式阶段：face2 上 7/64 → 22/64，但**只有一个有效 seed**（自曝 bug）

12 个 run 的读数：baseline（父端点）face2 **7/64**；训练后 **22/64**，四格为 普通×已覆盖 8/16、普通×未见 8/16、旧值×已覆盖 2/16、旧值×未见 4/16。CPU 合计 9,647.5 秒（预算 15,000）。

**但三个 seed 的端点逐字节相同**（sha256 `fed4ce3bc7673af0…` ×3）：`_train_one` 里 `torch.manual_seed(seed)` 写在了 `_trainer()` **之前**，而加载 N177.6 检查点会恢复 `torch_rng_state`，把种子覆盖掉 ⇒ 三次跑的是同一条轨迹。**所以这次只有 N=1 的证据，"三 seed 一致"不成立**；已修（种放到构造 trainer 之后），但已花的 9,647.5 秒不能追认成三次独立样本。

能说的：**在一条没见过的措辞面上，parser 加压把实体抽取从 11% 提到 34%**（普通 50%、旧值 12.5–25%）；这证明"加压这条路线在未见措辞上有位移"，但**没有 seed 方差**，也**不是**"学会了旧值问法"（旧值仍只有 2/16 与 4/16）。剩余预算 5,352.5 秒，不足以补两个 seed（每个约 3,200）。

## 18. N178 判读：方向成立、**未达预定门槛**（3 个真 seed）

种子 bug 修好后补跑 seed1/seed2，三个端点哈希互异（`fed4…`／`ad8a…`／`dc86…`），face2 读数：

| 格 | 基线（父端点） | seed0 | seed1 | seed2 | 中位 | 提案门槛 |
|---|---|---|---|---|---|---|
| 旧值 × 已覆盖 | 0/16 | 2 | 4 | 4 | 4/16（25%） | ≥80% ✗ |
| 旧值 × 未见 | 0/16 | 4 | 8 | 4 | 4/16（25%） | ≥80% ✗ |
| 普通 × 已覆盖 | 6/16 | 8 | 7 | 8 | 8/16（50%） | 不下降 ✓ |
| 普通 × 未见 | 1/16 | 8 | 10 | 8 | 8/16（50%） | ≥80% ✗ |
| 合计 | 7/64 | 22 | 29 | 24 | **24/64（37.5%）** | — |

**判读**：3/3 seed 都高于基线（7→22/29/24），方向成立；但四格里三格**没到预先写下的 80%**，所以 N178 按自己的验收线是**未达标**，不能算出口通过。能说的上限是"parser 加压在未见措辞上把实体抽取从 11% 提到 37.5%（中位）"。

**排除了一条嫌疑**：旧值轮的标注是对的（问"钥匙之前是什么颜色？"答"之前是蓝色。"），不是课程标错——所以剩下的缺口是**学习/容量**，不是数据或标签。

**成本**：seed0 2,848.3 秒 + seed2 3,490.1 秒 + seed1 约 3,400 秒（最后一次采样读数，进程在写报告前因路径 bug 崩掉，未落账本）≈ 9,700 秒，在 17,500 预算内。
