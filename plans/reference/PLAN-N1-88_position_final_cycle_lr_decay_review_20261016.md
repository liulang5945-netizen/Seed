# PLAN-N1-88 position末周期学习率衰减复验（2026-10-16）

## 依据与目标

run86c工程完整但seed12/position在末点未过六门。PLAN-N1-87在不加载模型的前提下分析已封存逐步loss和更新读数：seed12/position cycle4平均QA loss约为cycle3的18.8倍，增量主要来自transition；observation loss反而下降。五组Adam位移均未越上限，末点评分也显示transition/unknown错误与高loss部分关联。seed11/13 position保持六门，但cycle4 loss和组位移下降。该诊断不能确立因果，却把可检验的首要机制指向特定seed的末周期过度更新。

本轮只改变position臂最后6144次更新的实际Adam学习率：前三周期0.001，末周期0.00025。其余架构、Adam矩估计连续性、逐步样本/命名/QA目标、五组relative update上限、评分视图和六门全部沿run86c原协议。该方案用较小末周期步幅保留已形成能力，同时完成固定更新数；它不选中间checkpoint。

## 固定实验

- fresh seeds 11、12、13，全部position；每个24576更新、55296 QA序列、43008训练会话。
- 所有模型从各自新建零态开始。不得从run86c训练权重继承；可只比较记录的初始bundle签名以确认可比性。
- 五阶段0/6144/12288/18432/24576；每模型4800评分行，视图仍为original/pair/rename/rename_pair及原/改名四条件。 前18432步loss、五组measurement及0/6144/12288/18432评分行必须与run86c同seed position封存轨迹逐项完全相等，证明仅改变最后周期。
- 正式训练总计73728更新、165888 QA序列、129024训练会话、14400评分行、6次独立恢复、24副本签名步、96合成QA。正式训练前的三seed零态保存/独立恢复另记3次恢复、12副本步和48合成QA；全部前置CPU也计入4800秒上限。
- 独立本机CPU上限4800秒，包含zero/final完整保存及独立恢复；串行单线程。训练前先执行三fresh零态checkpoint格式保存/恢复预检，预检不通过则不得训练。
- 正式输出使用独立run88c报告/任务/checkpoint族；隔离执行并校验sources/code/hash，禁止自动重试或覆盖。

## 冻结验收

沿用run86c/85的原六门，不降低、不换测面：position每seed原题≥173/192、四类各≥40/48、pair≥87/96、改名≥173/192且各类≥40/48、rename_pair≥87/96、原与改名四条件joint各≥44/48。三seed必须全部通过才算训练充分。要求完整SHA/原始行复算、零态与终点独立恢复、样本/更新/session/QA准确、sources/code unchanged及CPU accounting完整。

v5、dev、v3 holdout、v4及其他未见集查询一律为0；`TRAIN_SHA=None`、`can_promote=false`。run86c的失败与控制证据保持原状。即使本轮三position通过，也须另行完成准入裁决与确认面授权，不自动绑定85或晋升候选。

## 边界与后续

这是学习率配方变更，使用新的attempt/report/checkpoint族；不修改冻结runner或run86c输入。仅做一次正式三seed处理，不追加或挑选checkpoint。若任何seed六门失败，如实失败，先从全程逐步轨迹检视是否末周期loss/位移仍异常，再提出唯一下一版。CPU资源已由所有者全权授权，不需逐次确认；任何远端推送或删除数据仍视为不可逆操作。

## 开始条件

正式训练前完成独立fresh zero checkpoint保存/恢复，核对参数、完整Adam空态、RNG和历史一致，独立恢复CPU纳入4800秒总限。完成后的hash与preflight回执必须提交后才可启动三seed训练。


## 授权

本方案及其可撤销的范围/实现选择按所有者2026-10-16更新的AGENTS.md持续授权执行。CPU上限仍是本计划冻结的4800秒；远端推送或数据删除不在授权内。


## 实施身份（冻结，2026-10-16）

- run88c控制器SHA256：`801073f6eec312eb8e147a98700e4130ec9f856eda4402fcb18ec2568a38e350`。
- run88c Task Scheduler启动器SHA256：`36bb48ae495de0962feeb1fc3c31de54aac62f47bb260ae6ff43d37ec3862434`。
- 冻结84训练器SHA256：`38018dbe08b9eb4c35d0b5c97196727b34f94efee68d68113e80269b8d3c9491`。
- 相关文件：`scripts/training/run_taiji_n1_transition_factorial_lr_decay_run88.py`、`scripts/training/launch_taiji_n1_transition_factorial_lr_decay_run88c.ps1`。


## 首次checkpoint前置失败与处理

2026-10-16首次checkpoint前置在读取三份run86c position封存分片后，因将control_records的二项返回误解包为两个嵌套结果而停止；发生在run88零态/模型/checkpoint反序列化前。运行更新/optimizer/model query均为0；run88 checkpoint根目录已创建但文件数为0，失败回执`reports/taiji_n1_transition_factorial_lr_decay_run88_preflight_failure.json`保留。已修正解包逻辑，并在原目录为空这一可验证条件下，按所有者授权另开`preflight_attempt_02`，保留第一次attempt与失败回执；不删除或覆盖任何文件。再次核验三seed zero saves、独立恢复及调度边界；成功前不得训练。

此处只新增一次checkpoint前置重试，不增加正式训练尝试数、不改变4800秒上限或任何训练/能力判据。


## 前置执行记录（2026-10-16）

preflight_attempt_01在读取3个run86c position封存分片后，误将(control report, model map)二项返回再次按二项嵌套结构解包，停止于零态模型加载前。没有run88候选模型/checkpoint反序列化、保存、optimizer更新、评分或查询；run88 checkpoint根目录已创建但经核验为空。精确失败记录见`reports/taiji_n1_transition_factorial_lr_decay_run88_preflight_failure.json`，attempt_01和空目录保留。已修复解包逻辑；依所有者最新持续授权，唯一下一步是对空根执行`preflight_attempt_02`，生成新attempt并再次完整核验。成功前不启动正式训练。


## run88前置失败与run88b隔离处理（2026-10-16）

run88的preflight_attempt_02在seed11 zero checkpoint独立签名恢复时失败。诊断确认父进程PyTorch线程数为12，而restore worker的OMP/MKL/OPENBLAS环境固定为1；以12/1配置可复现签名差异，同线程时签名匹配序列化的zero checkpoint。run88 root保留seed11零态checkpoint（SHA256 `c0522ce292b632113d032044d2aded407744bdba2e5609cbd2b0cc85d26126f9`）及两次attempt/失败记录，不会读作run88b训练源或删除。失败证据为`reports/taiji_n1_transition_factorial_lr_decay_run88_preflight_failure_02.json`。

最终训练与checkpoint preflight迁至隔离族run88b；controller在载入任何run88b状态前将父进程设置为单线程，子进程也维持单线程。对run88两次前置与签名诊断未单独计量的CPU预留300秒；总预算仍≤4800秒，run88b正式预算相应扣减。run88b不得复用run88 checkpoint，只用冻结83的fresh零态来源。唯一下一步是提交修复并执行三seed run88b零态保存/独立恢复前置；成功前不训练。


## run88b独立输出族与历史前置保留

前两个零态前置尝试均发生在任何正式更新前：attempt_01在control结果解包失败；attempt_02在多线程父进程/单线程restore worker之间签名不一致，根因为父进程未强制单线程。两份report、attempt和run88下已保存的s11零态checkpoint全部保留。最终候选预检与训练使用新的run88b报告/任务/checkpoint族，父进程和restore worker统一单线程；run88b不得加载或续接run88的零态文件。为前置调试中的未单独计量CPU预留300秒，仍包含在4800秒总限内。当前唯一下一步为run88b三seed零态checkpoint保存/独立恢复前置。


## run88b形式启动失败与run88c隔离族（2026-10-16）

run88b Task Scheduler任务已启动后，worker在来源守卫拼装时因把不存在的可选`preflight_attempt_02`文件无条件纳入哈希列表而退出；OS退出码1，PID14620已消失，正式`attempt`、model/checkpoint加载、评分和optimizer更新均未发生。精确失败记录`reports/taiji_n1_transition_factorial_lr_decay_run88b_startup_failure.json`、launcher/worker/completion回执保留；精确临时任务已注销。run88b三zero预检报告和零态checkpoint亦保留。

控制器现将可选attempt文件按存在性纳入来源守卫，正式运行切换到新隔离族run88c，仅从冻结83 fresh zero source重新保存零态checkpoint；不反序列化、复制或续接run88b模型文件。仅按hash将旧run88b preflight零态文件纳入历史来源清单。父/子进程均单线程，预留既往未计量CPU 300秒并计入4800秒总限。唯一下一步先执行run88c三seed零态保存/独立恢复前置，成功且提交回执后再启动正式训练。
