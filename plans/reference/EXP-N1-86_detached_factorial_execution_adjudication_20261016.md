# 86 隔离执行启动裁决（2026-10-16）

## 状态

**首轮状态：启动失败，未进入训练；不构成训练结果。** Task Scheduler任务Seed-N1-Run86-20261016注册及Start均成功，随后控制器在60秒task_receipts等待后以FileNotFoundError退出。OS任务状态Ready、LastTaskResult=1。证据记录见reports/taiji_n1_transition_factorial_run86_startup_failure.json。

## 根因与证据边界

PowerShell启动器将启动返回记录写入reports/taiji_n1_transition_factorial_run86_returned.json，控制器却等待reports/taiji_n1_transition_factorial_run86_launcher_returned.json。因此worker清单未写出。失败发生在控制器验证Task Scheduler清单阶段，早于读取84冻结attempt、调用runpy加载训练器、调用readiness()核验并加载83来源模型或课程样本。

失败目录内没有worker、attempt、主报告或checkpoint；主模型更新、dev和holdout查询均为零。现存launcher/return/completion/error日志保留，精确失败任务已注销。84中断产物未改，85 TRAIN_SHA=None，v5未查询。

## 后续处置

修正formal return文件名并将后续运行隔离到run86b报告与checkpoint家族。先重做修正版Task Scheduler合成心跳和run86b路径checkpoint保存/独立恢复预检；相关代码、计划及新回执提交后，才启动相同的fresh九模型固定配方。样本、目标、更新数、CPU上限、训练门和未见测面限制均未改变。


## run86c最终批次裁决（2026-10-16）

原文记录的是86首轮启动失败；该失败结论保留。后继隔离批次run86c已自然结束并独立裁决如下。

### 工程与来源核验

- Task Scheduler `Seed-N1-Run86C-20261016` 自然结束，OS `LastTaskResult=0`；PID6428已退出。completion回执`exit_code=0`、`engineering_complete=true`，其报告SHA256 `0a8fc62e9aa00577d67001c70008773e6e5cee54cebad4115956e0360446d785`与主报告原字节SHA完全相等。之后已按PLAN-N1-86注销该精确临时任务；不涉及删除数据。
- 九个seed×arm分片逐一经冻结receipt校验器核对原始43200评分行、目标/预测符号、样本身份、四条件原始复现、24576条loss/measurement轨迹、五阶段及六视图汇总；9/9工程完整。
- 汇总为221184更新、497664 QA序列、387072训练会话、43200已知评分行、18次独立恢复、72副本签名步、288合成QA；包含子进程恢复在内CPU 9692.03125秒，低于14400秒，CPU accounting完整。
- 18个zero/final checkpoint均存在且SHA与封存恢复记录匹配；每模型两次恢复均`equal=true`，每次签名副本步2。83项运行来源守卫逐项重算全部匹配，主报告记录`code_unchanged=true`、`sources_unchanged=true`。
- dev、v3 holdout、v4及v5确认查询均为0；`can_promote=false`。

### 六门结果

seed11/position与seed13/position均通过六门。seed12/position未通过：原题总分176/192（≥173），但unknown仅38/48（<40）；配对96/96；改名181/192且四类均≥40；改名配对96/96；原始和改名四条件联合均24/48（各低于44）。因此三position共同前置失败。控制臂均完整封存，不因position失败而补训、重选控制臂或改门。

| seed | arm | 原题 | 配对 | 改名 | 改名配对 | 联合四条件 原/改名 | position六门 |
|---|---|---:|---:|---:|---:|---:|---|
| 11 | position | 192 | 96 | 191 | 96 | 48 / 48 | 通过 |
| 11 | terminal | 134 | 38 | 123 | 36 | 48 / 48 | 控制 |
| 11 | cut | 192 | 96 | 190 | 96 | 43 / 37 | 控制 |
| 12 | position | 176 | 96 | 181 | 96 | 24 / 24 | **失败** |
| 12 | terminal | 132 | 33 | 122 | 26 | 48 / 48 | 控制 |
| 12 | cut | 192 | 96 | 191 | 96 | 48 / 48 | 控制 |
| 13 | position | 192 | 96 | 192 | 96 | 48 / 48 | 通过 |
| 13 | terminal | 116 | 32 | 108 | 25 | 48 / 48 | 控制 |
| 13 | cut | 180 | 89 | 179 | 88 | 24 / 24 | 控制 |

因此PLAN-N1-85条件准入拒绝：`TRAIN_SHA=None`，85 attempt不存在，v5未消费。不能以控制臂、某些position seed或中间评分点替代失败前置。

### 已记录轨迹的边界性观察

seed12/position在18432步原题/配对/改名/改名配对均为192/96/192/96，原始与改名四条件联合均48/48；24576步变为176/96/181/96和24/24。每步loss日志显示QA均值从cycle3的0.000250升至cycle4的0.004688，而observation均值从0.000248降到0.000029；QA波峰出现在20992–22016及24064–24576。末点错误落在原题transition 6项与unknown 10项。它们是已记录量之间的关联，不能单独证明是何种Adam位移造成回退，也不能定位两评分点之间的首次错误步。

唯一建议的下一步是先审议[PLAN-N1-87](PLAN-N1-87_run86c_offline_trajectory_audit_review_20261016.md)：在60秒CPU内只分析九个已封存分片，不加载模型/checkpoint、不做新评分或留出查询、0更新；重点按计划对齐末周期loss、五组实际位移及失败项。该新增诊断待所有者批准；它不授权新训练配方或改变任何门。


## PLAN-N1-87首次离线诊断执行状态（2026-10-16）

所有者已批准PLAN-N1-87。首次执行在Python顶层导入阶段因入口未配置仓库根`sys.path`而失败，未读取报告/分片/训练样本，未加载模型或checkpoint，未执行评估/查询/更新；CPU时间未由该失败预检单独计量，正式分析报告不存在。错误与零副作用回执见`reports/taiji_n1_transition_factorial_run86c_offline_trajectory_audit_attempt.json`。已修复导入入口。因原批准方案规定仅一次且不重试，未擅自再次运行；已根据所有者对可撤销范围调整的明确授权，按[PLAN-N1-87R](PLAN-N1-87R_run86c_offline_audit_retry_review_20261016.md)在原60秒上限及全部只读边界下额外重跑一次。


## PLAN-N1-87 / 87R离线轨迹审计结项（2026-10-16）

审计一次成功，CPU3.015625秒，9/9分片、216个512步窗口；模型/checkpoint加载、模型评分/查询、训练更新和任何未见集查询均为0。报告SHA256 `c5b21641de584d6e871ad4747ce8e181205e790a886c7b58bbb1305c9e46b2a0`。详细统计见[机器报告](../../reports/taiji_n1_transition_factorial_run86c_offline_trajectory_audit.json)。

重点读数：seed12/position cycle4 QA loss均值0.004687574，cycle3为0.000249594；transition QA loss0.000626488→0.013793352，observation loss0.000248035→0.000029176。cycle4五组更新的actual-to-cap平均比为7.3%–10.7%，cycle3为1.3%–2.3%；未发现越限。末点原题16个错项、改名11个错项以transition/unknown为主；错误项平均loss较高，但存在高loss仍答对的条目。此为关联性证据，不证明因果。seed11/13 position cycle4误差不增，且update ratio回落，显示种子轨迹差异。

唯一下一步已按最新授权自动进入[PLAN-N1-88](PLAN-N1-88_position_final_cycle_lr_decay_review_20261016.md)：fresh三seed position，仅末6144步学习率降至0.00025，以检验晚期回退；保持原六门、样本和零留出测面。训练前先完成独立检查点保存/恢复验证。即使训练门通过，也不会自动消费v5或晋升候选。
