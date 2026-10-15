# PLAN-N1-63 · 关系tape三seed三臂固定训练

## 前置与共同来源

62工程通过，v4确认面SHA256 `a148dbf6c65812b08b3f25103b014da82b97796c18134724dc2f03dcb6f12529`；全程只校验文件SHA和冻结回执静态计数，不把内容传入模型。源为52固定终点，共享旧55480参数/35项Adam及history；关系6576参数由62迁移器seed11/12/13初始化，同seed三臂完整bundle指纹须与62初始指纹逐位相同。61终点不参与训练起点；不用62合成session作为训练内容。

## 固定执行

按seed11/12/13、arm position/terminal/cut顺序串行九fresh模型，每模型6144新更新，共55296；总独立本机CPU≤2400秒，包含前置、所有train/probe评分、九zero及九final独立恢复与失败partial保存。线程1，不删样本或改上限凑时，不并发跑九模型。每模型0/3072/6144里程碑，原43固定3072改名流、40固定双问题、9216QA序列（九模型82944序列），每步QA均值+0.25观察保真，无55一致性/58margin。

正常position全逐字节信用；terminal原writer/问题条件和新增关系输入均遵守各自source末态重复定义；cut阻断观察writer及观察局部特征/embedding路径信用而保留问题/关系学习，沿已验证同臂loss arm参数，不另造control。三个arm用同初始化、样本、目标、次数、优化器配方，唯一差异为既定读取/信用干预。

四组Adam终点reader24576/local18432/context12288/relation6144，所有41参数moment齐全；每模型历史27648→33792。自身session唯一、不能碰其他模型会话。完整checkpoint包括参数/Adam/RNG/history、seed/arm/protocol，exclusive zero/final/partial不覆盖62或61。zero独立恢复前不得更新；final独立恢复后才认完整。来源报告、checkpoint、所有训练manifest、v4字节SHA及代码身份前后守卫。

所有九模型都完成固定配方，不能在某seed中点挑模型或某个position先失败就改配方。遇预算/工程错误立即保存当前partial并停止整批，不以partial或未跑模型宣称完整；不自动重启。完整跑完后每seed的position必须通过原四门：原题≥173/192且每类≥40/48、pair≥87/96、已知改名≥173/192且每类≥40/48、改名pair≥87/96。控制臂不因未达position门而补训。若任一position四门不足，本批失败且v4保持零查询，先证据修订不循环追加。

## 确认准入与边界

本63不查询v4。仅九模型完整、18次独立恢复及全部position四门通过且源码/档案冻结后，另冻64一次性九模型×32＝288条确认评估。沿14原门：每position seed≥16/32、每类≥2/8，position三个seed平均各领先terminal/cut≥4/32（总差各≥12/96）。固定同一v4面对九模型，评估后即消费，不能重新训练再拿它作未见。

0v3dev/holdout、0v4查询、0其他新面，0远端推送或默认推广。共享母档的新关系初始化稳健性不能冒称端到端独立三seed。旧能力保持/C4/同bundle三器官及N系列其他线与CI历史债继续欠；真实出口不缩减为本课程过门。

## 提交传输字节守卫修订（训练前、0确认查询）

62生成器默认Windows换行导致工作区v4/回执为CRLF，而Git blob仅换行为LF。只读逐字节核对确认：原v4 SHA a148dbf6c65812b08b3f25103b014da82b97796c18134724dc2f03dcb6f12529，对应Git LF SHA ea1cd7df5ff18c90a9799220d771c14ac740fc7642d0761a0050e2d8942684b4；原回执221982613df310877b2fc37cb767f8fbafce864710e8b8086138bfad7856cd47，对应LF 8fe8ae348aa93644b9fb62b2a80ab3f91da9bdd99a08df5f9f110046b737235b。

不重写62报告或重新生成面。63/64读取守卫只接受这两种已冻字节形态，且必须能仅通过全量LF/CRLF变换互相重构；任何第三个hash、混合换行或内容改动均拒绝，不采用宽松JSON语义比较。记录原冻结SHA与本次实际SHA，执行前后实际字节仍须一致。62来源守卫中的这两项按固定双seal验证，其余全部保持原单SHA。所有题目/答案/顺序、训练门、预算及确认准入不变，0确认查询。

### cut优化器计账澄清（正式更新前）

cut干预使writer四参数无梯度，Adam对应step和moment应保持母档18432，不能伪造为24576。其reader组因此为[18432,24576]，其他旧参数及新关系组仍按原规定推进；position/terminal全部reader24576。按参数名逐项验证，并核对cut writer参数与moment不变。此为冻结信用干预的机械结果，6144更新/四门/确认准入不变。
