# 84 四条件学习三种子三臂正式训练（2026-10-16）

83工程完整，报告SHA `4b157a1eaf2f0ec6cae20eef705a441a408f24a5cc7816dd1c84fa59acb9dd06`，CPU37.3125秒，九零态实际独立恢复、36副本步/144合成QA、67源链守卫，主轨迹0更新/真实课程0查询。同seed三臂完整零态相同、三seed不同；cut writer不变/无Adam。82训练校准六门通过，报告SHA `a0dd6e74c669275fbfd7e629e17a8ff7a6122e232bb6797574f7bfaf0a73160e`；不把单seed当确认。

## 固定主轨迹

seed11/12/13×position/terminal/cut九fresh模型，每模型从对应83完整零态开始，结构62056参数、两次读取/相对地址/问题与上下文及局部关系机制不变。固定24576更新/55296QA/43008训练会话；共221184更新/497664QA/387072会话。三臂父题、43改名流、四个6144周期、四条件等权目标完全相同；非transition保留69原目标，五组实际Adam位移上限0.001/lr0.001不变。cut自fresh第一步阻断writer信用，4参数无Adam且不变，其余37步24576；position/terminal41步24576。不能引入82训练权重/Adam或继承历史。

单线程串行，独立整批CPU≤14400秒（240分钟，含源链、保存、子进程恢复、评分与分片），依据82每模型1301.89秒估算并留余量。每模型zero/final各一次独立恢复，每次父子各两步同臂合成四条件签名，共18恢复/72副本步/288合成QA，主轨迹和副本开销单列。固定顺序seed11三臂→seed12三臂→seed13三臂；完整保留失败种子/控制，不选中点、不补训/重启/替换。

五评分点0/6144/12288/18432/24576，每点四原view576条＋原/改名transition四条件384条，共960/模型/点、4800/模型、43200已知行。逐项保留teacher/free/raw/stop，每个四条件original须复现同点原view对应行，评分前后完整参数/Adam/RNG/历史不变。所有已知训练变体不能充作未见能力。0dev/v3holdout/v4/v5；三原生seed全部六训练门通过、九模型全轨迹完整并冻结前，85不准入。

## 冻结门与证据

各position终点必须过82原四训练门（原/改名≥173/192且每类≥40/48，pair/改名pair各≥87/96）以及原/改名四条件联合各≥44/48。控制如实计分但不拿控制充分性代替position前置。三position合取；任何不足保留失败，85不绑定训练SHA，不消耗v5。即使过训练前置也不宣称信用行为优势/泛化，通过85原14四门才有对应小面证据；全项目出口另核。

每模型全部24576 loss、24576五组measurement、4800评分行以无损gzip分片独占保存，mtime0、压缩/解压双SHA和字节数/精确计数核验。主报告记录完整来源/代码/CPU/18恢复/模型Adam步与全部387072会话计数。独立receipt验证主报告与九分片逐字段一致、精确五点评分与六门；不能只看聚合布尔值。attempt独占；异常保留partial checkpoint及完整已得行/损失分片，不自动恢复或覆盖。

## 预声明产物与顺序

`scripts/training/run_taiji_n1_transition_factorial_paired.py`；`scripts/training/_taiji_n1_transition_factorial_training_receipt.py`；`tests/taiji_native/test_native_transition_factorial_paired_training_contract.py`；`tests/taiji_native/test_native_transition_factorial_training_receipt_contract.py`；`reports/taiji_n1_transition_factorial_paired_train_20261016.json`、`_attempt.json`及`_s{11,12,13}_{position,terminal,cut}.json.gz`（精确族binary属性）；忽略目录`data/checkpoints/n1_transition_factorial_paired_train_20261016/s{seed}_{arm}_{zero,final,partial}.pt`；结果裁决`plans/reference/EXP-N1-84_transition_factorial_paired_training_adjudication_20261016.md`。

先完成84/85准备实现、artifact-free合同/根Ruff/相关计划导航检查并提交，然后独占训练；运行时冻结所有共享代码。85执行器TRAIN_SHA初始None，84全终点后仅在前置全部成立时另提交精确SHA绑定。不能运行旧78绕过新目标准入。所有旧实验/报告/模型不改、不删、不推远端。五计划保持实时，N1消费/保持/C4/同bundle三器官/N系列/完整HEAD CI债仍欠。

## 最新执行状态（2026-10-16）

本批已于Codex客户端更新重启时中断。唯一可验证检查点为seed11/position/zero；attempt不含进度字段，因此中断前实际更新数未知。结果裁决见[EXP-N1-84](EXP-N1-84_transition_factorial_paired_training_adjudication_20261016.md)。84不续接、不覆盖；同一固定实验转由全新执行尝试86采用fresh初始化和独立产物族运行，见[PLAN-N1-86](PLAN-N1-86_detached_factorial_execution_20261016.md)。85保持未准入、训练SHA未绑定、v5未消费。

## 实施前置记录

84/85共30项artifact-free合同通过，两根Ruff通过；84真实来源只读预检77来源守卫与九完整零态路径通过，0更新/查询。逐步QA调度/有限损失与五组实际位移、五阶段六view精确目标/原始复现/汇总逐项一致、无损双SHA、cut状态与全历史恢复均有合同；85 TRAIN_SHA保持None，未准入/未查询。共享代码提交后冻结并开始84，不能在训练运行中改执行器/目标/原门。完整HEAD CI债仍保留。
