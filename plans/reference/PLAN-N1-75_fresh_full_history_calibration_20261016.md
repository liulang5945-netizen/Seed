# N1-75 · 全fresh模型固定train-only充分性校准

## 准入与单一配方

74完整，报告SHA `24c62ecf659555792d8193bf420da160f0cbf5e90ac2b0e3be93ce780d733760`。只用74 seed11 position原零档（SHA `b95c440a1671f0cd6f132353730b43598714b3e5c16737ed610e76052c6f913c`）；其全部参数按原类fresh、Adam/history为空，主轨迹从未更新，不能用工程副本或70/52权重。先严格恢复并逐位核初态，再新protocol保存/独立恢复zero，才做正式训练。

结构62056参数、双hop/相对地址/残差问题对齐/关系tape不变；原Adam lr0.001、QA两问题均值＋0.25观察保真、67/69相同五组relative0.001/floor1.0实际位移约束和原moments。唯一训练臂position，fresh seed11；正式更新固定24576，QA36864，不据中点早停/选档/改剂量。

使用43原有6144步固定流完整重复四遍，`train_item(step % 6144, train, curriculum)`；192原train、原3072已注册改名stream和40切换问题，重复使用相同数据而非生成新名称或扩测面。总训练会话24576个唯一ID，不能重用session导致历史混入。

## 完整读数/预算/恢复

0/6144/12288/18432/24576五次完整576项原题/pair/已知改名/改名pair，合计2880组teacher+free；全部24576 loss与五组实际/提议位移保留。零档/终点两次独立新进程恢复，完整参数/Adam/RNG/历史及两步同臂下一更新签名一致；主进程/worker各两步，两档共8合成副本更新/16合成QA单列，不能并入24576正式更新。74只恢复zero的新工程协议不能被改成训练加载器，75自有versioned训练schema和计步验证。

独立CPU≤1200秒（20分钟），一个Torch线程，含前置、训练、全部读数/保存与独立恢复/守卫。所有41项终点Adam step应为24576、完整Adam状态41项、初始会话空且终点新增24576；签名副本不能回写父模型。来源52/70/71/72/73/74证据链、74零档、全部课程与代码身份执行前后守卫；0dev/v3holdout/v4或其他新未见查询。预算/错误保存完整partial和已得读数，停止，不重跑或追加。

## 原门与后继

沿原四门：原题≥173/192且每类≥40/48、pair≥87/96，已知改名≥173/192且每类≥40/48、改名pair≥87/96。只作从头已知任务校准，不是端到端三seed信用优势或新泛化证据。未通过则定位固定曲线主要瓶颈，不循环加训；通过后另冻正式三seed三臂等预算训练与全新未见确认面，不能复用已消费v4。原71按原门保持失败，绝不据新配方补发成功。

## 产物与全项目边界

执行器`scripts/training/run_taiji_n1_fresh_full_history_calibration.py`，合同`tests/taiji_native/test_native_fresh_full_history_calibration_contract.py`；checkpoint在`data/checkpoints/n1_fresh_full_history_calibration_20261016/`；报告`reports/taiji_n1_fresh_full_history_calibration_20261016.json`采用完整无损紧凑JSON，裁决放plans/reference。先检查保存/恢复合同并提交，再一次执行固定包。

保持旧强候选，不覆盖默认、不推送/删除数据，不将隔离候选表现替代C1/C2原三器官协议或C4、保持与N系列其余出口。目标仍是可持续学习、真实联合能力的高上限模型。

## 执行前合同

三项artifact-free合同通过：实际fresh zero新进程同两步恢复、明确synthetic final布局的全部状态/会话/计步恢复与错误拒绝、四周期取模边界及固定范围。两Ruff通过。partial独立phase只保存全量状态用于审计，严格加载器拒绝自动续训；74代码不改。提交执行器后按冻结配方一次执行。
