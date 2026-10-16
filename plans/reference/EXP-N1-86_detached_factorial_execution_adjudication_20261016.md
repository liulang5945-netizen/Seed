# 86 隔离执行启动裁决（2026-10-16）

## 状态

**首轮状态：启动失败，未进入训练；不构成训练结果。** Task Scheduler任务Seed-N1-Run86-20261016注册及Start均成功，随后控制器在60秒task_receipts等待后以FileNotFoundError退出。OS任务状态Ready、LastTaskResult=1。证据记录见reports/taiji_n1_transition_factorial_run86_startup_failure.json。

## 根因与证据边界

PowerShell启动器将启动返回记录写入reports/taiji_n1_transition_factorial_run86_returned.json，控制器却等待reports/taiji_n1_transition_factorial_run86_launcher_returned.json。因此worker清单未写出。失败发生在控制器验证Task Scheduler清单阶段，早于读取84冻结attempt、调用runpy加载训练器、调用readiness()核验并加载83来源模型或课程样本。

失败目录内没有worker、attempt、主报告或checkpoint；主模型更新、dev和holdout查询均为零。现存launcher/return/completion/error日志保留，精确失败任务已注销。84中断产物未改，85 TRAIN_SHA=None，v5未查询。

## 后续处置

修正formal return文件名并将后续运行隔离到run86b报告与checkpoint家族。先重做修正版Task Scheduler合成心跳和run86b路径checkpoint保存/独立恢复预检；相关代码、计划及新回执提交后，才启动相同的fresh九模型固定配方。样本、目标、更新数、CPU上限、训练门和未见测面限制均未改变。
