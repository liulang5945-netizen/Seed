# N1-74 · 完整fresh历史三臂工程

## 来源和唯一修订

73完整，报告SHA `f32596a25a8dd032f0416eb647280d872fcf9e04b9a5990de226cea8b2131aaf`。70/71测试的是继承52后的最后6144步writer信用，不是从头信用。74仅改初始化/计步归属，不改现有模型结构、截断臂、位移规则或历史门；旧候选保留。

每seed11/12/13重新构造62056参数LocalRelationEvidenceCandidate（默认embedding16/hidden32/key16、双hop），不迁移任何52/70/其他权重、Adam、RNG或会话。用同seed重建position/terminal/cut，三臂完整初始bundle逐位相同；三个seed全部随机参数各自初始化，现有零残差head按原类保持零。Adam lr0.001，初态state为空、会话为空；不加载任何模型checkpoint来源。

## 工程验证

严格新checkpoint schema/初始化marker、原native模型marker、seed/arm/phase、位移policy；本批只能恢复completed0和空Adam/空会话，旧母档/工程档/partial拒绝。九fresh零档各独立新进程恢复。每次主进程保存签名及worker恢复签名均在独立完整副本上做同臂两步合成双问题学习，检查完整下一bundle、loss、五组位移和梯度逐位一致；主轨迹始终0更新/0查询/空历史。

第一步零relation head获得学习，第二步relation GRU信用正且有限；position/terminal writer信用正、两步Adam为2；cut writer梯度0、参数不变且从头无Adam状态，其他37项Adam为2。两步均保持原moment规则及实际位移限relative0.001/floor1。副本不得回写父档、RNG或会话。

九零档×两次签名×两步＝36合成副本更新/72合成QA，单独列账；0正式训练/课程/dev/holdout/v4模型查询。独立CPU≤120秒、一个Torch线程，含前置、保存、九独立恢复及所有签名。预算/错误独占保存报告并停止，不缩seed/臂、不重跑，不改现有源代码或冻档。报告前后核73及其固定报告/代码守卫；活计划仅快照，不禁止正常更新。

## 产物和后继

`scripts/training/audit_taiji_n1_fresh_full_history.py`、`tests/taiji_native/test_native_fresh_full_history_contract.py`、`reports/taiji_n1_fresh_full_history_preflight_20261016.json`，九零档在`data/checkpoints/n1_fresh_full_history_preflight_20261016/`。先合同和静态CI，再提交并工程执行。

通过后唯一后继为另冻单seed固定剂量的train-only充分性校准；从fresh零档正常训练，不追加70/不查询已消费v4或v3留出。剂量/CPU和新确认面在校准协议中另冻，不将工程成功称能力、C4或项目出口成功。正式跨seed/控制配对和任何全新未见面仍需先冻，原71失败不回改。

## 实施前验证

五项artifact-free合同通过：三seed全fresh/同seed三臂逐位配对、三臂两步信用与空parent、position真实独立恢复、严格marker/phase/换seed拒绝，以及剥离moments后训练权重不能伪装fresh。默认Ruff与CI B/SIM忽略B008通过；新执行器不读取任何母档checkpoint，identity仅引用既有报告/源码。提交后执行九档工程。
