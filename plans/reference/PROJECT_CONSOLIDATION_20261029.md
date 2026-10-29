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
