# N1-76 · 完整fresh配对协议、v5与证据体积准入（零模型查询）

## 输入与目的

75完整且原四门通过，报告SHA `61dced2fe042f6f0e320b7e354afff227c732f82ec4acd601b18d8417f848e5a`；单seed固定24576更新实际851.109375秒，终点186/96/184/96。74三seed九共同fresh零档工程完整，不能将校准终点迁入正式配对。76不做模型加载/训练/teacher/free，只完成正式包的可核执行身份、全新未见面和资源/数据落点。

## 固定77正式包

seeds11/12/13×position/terminal/cut，共九模型；均从74各自原zero完整恢复，全部模型参数/空Adam/空历史初始指纹与74一致。同seed三臂完整初态相同，三seed全部随机参数独立；原零残差初始化保留。原62056参数、双hop/相对地址/问题残差/关系tape、paired QA＋0.25保真、Adam lr0.001与五组relative0.001/floor1位移不变。

每模型固定24576更新，43原6144流重复四遍，九模型共221184正式更新/331776正式QA；不选seed或中点。0/6144/12288/18432/24576各576已知评分，合计25920。九zero/九final独立恢复，18档主进程/worker各两步同臂签名，额外72副本机械步/144合成QA单列。position/terminal终点41项Adam均24576；cut writer四参数从头不变且无Adam状态，其余37项均24576。每模型新增24576唯一会话。

独立串行单线程总CPU≤9000秒（150分钟），含所有前置/更新/评分/18恢复/签名/压缩完整证据与守卫；估计约7660秒，预算不能复用旧40分钟或校准预算。所有九模型固定次数完成，不因某position失门而省掉控制；任何预算/工程错误保存当前partial和已得全部分片/汇总并整批停止，不自动重启/续训。训练中0dev/v3holdout/v4/v5查询，只核封存字节。

## v5新确认面

独占生成`plans/manifests/n1_fresh_confirmation_v5_20261016.json`：固定seed76、32项、lookup/revision/transition/unknown各8，两条观察、随机来源顺序沿现行任务语义；10字节实体、6字节材料值，全部实体/材料与已注册v3/34/43/v4排除集合分离，四类/自然boundary/cap64计分不改。只离线生成/校验，0模型查询，不查看结果选面。精确字节固定LF保存，来源SHA与生成器身份入receipt；拒绝覆盖/重新抽面。

九position终点全部原四门通过、九模型/18恢复/全部守卫完整且冻结后，另版78才一次查询九模型×32＝288。沿14原门：每position seed≥16/32、每类≥2/8；position总分分别领先terminal/cut各≥12/96。不放宽原门，不追认71；v5使用后保守消费、不能重训后再当未见。

## 完整无损证据布局

75单模型原JSON31.2MB，无损gzip6实测约6.45MB；九模型必须采用逐模型分片，而非删样本/字段。77每模型完整loss/五组位移/2880原始teacher+free行/阶段摘要，写`reports/taiji_n1_fresh_paired_train_20261016_s{seed}_{arm}.json.gz`，固定gzip mtime0。主`reports/taiji_n1_fresh_paired_train_20261016.json`记录每分片路径、压缩文件SHA、解压内容SHA/字节数、完整计数/协议身份、milestones/来源/恢复与门判定；读取须验证两层SHA和精确计数，错/缺片不能完整。

76以合成记录验证分片无损往返、独占写入、压缩/解压内容篡改拒绝、计数不全拒绝；在内存对75真实原报告再做无损预算/体积预检但不修改它。压缩输出只在已声明reports命名族，checkpoint只在`data/checkpoints/n1_fresh_paired_train_20261016/`，不新建根文件夹。完整原始读数仍可恢复，不把gzip当删数据手段。

## 76预算/产物/后继

76独立CPU≤30秒，0模型加载/更新/teacher/free/新未见查询。产物：`scripts/training/build_taiji_n1_fresh_paired_admission.py`、报告分片工具`scripts/training/_taiji_n1_evidence_shard.py`、对应artifact-free合同，`reports/taiji_n1_fresh_paired_admission_20261016.json`，上述唯一v5manifest与plans/reference裁决。先验证/提交，再独占生成；75原报告/所有旧档不改。

通过后唯一下一步77正式执行器实施、全部恢复/分片/信用合同与提交，再一次启动完整固定九模型训练。当前不是正式多seed能力通过，更不是默认采用或项目收官；旧强候选、71失败、保持/C4/同bundle三器官/N系列其他出口及CI历史债保持。
