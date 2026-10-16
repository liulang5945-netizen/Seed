# run86b 正式任务启动失败裁决（2026-10-16）

## 裁决

run86b 在训练材料就绪之前失败，LastTaskResult=1，退出码1、worker CPU 2.25秒。没有生成run86b attempt、主报告或分片；候选未载入、训练材料未进入`readiness()`、0更新、0课程/dev/holdout查询。Task Scheduler任务已注销，启动清单、返回回执、worker、completion和stderr均保留。控制器在失败前创建的`data/checkpoints/run86b`目录目前为空，保留原状。

## 根因证据

stderr栈落在冻结runner的`audit()`首个独占批次检查（`run_taiji_n1_transition_factorial_paired.py:463-469`），错误为`FileExistsError: exclusive fixed nine-model batch; no restart`。该守卫先于`readiness()`、训练材料读取、样本构造、attempt写入和训练循环。

复现确认`runpy.run_path()`返回映射与`module["audit"].__globals__`不是同一个对象。控制器把run86b的`REPORT`/`ATTEMPT`、外部checkpoint路径、readiness包装和print包装写进返回映射，冻结runner函数仍读原始globals：它看见84既有attempt `reports/taiji_n1_transition_factorial_paired_train_20261016_attempt.json` 和84检查点目录 `data/checkpoints/n1_transition_factorial_paired_train_20261016`，所以按拒绝重启合同停止。完整机器裁决在`reports/taiji_n1_transition_factorial_run86b_startup_failure.json`。

同时发现控制器的外部checkpoint路由尾段仍硬编码为`run86`。它不是本次守卫报错的直接原因（函数globals未被覆盖使该闭包未生效），但若只修globals，会把模型检查点导向错误目录，因此必须同批修为新的唯一族`run86c`。

## 后续冻结动作

保留run86b全部失败证据与空检查点目录，不覆盖或删除。修正控制器以更新`audit.__globals__`实际使用的命名空间，并把report、attempt、checkpoint和source guards逐项验证为run86c；新增针对runpy函数globals绑定的回归测试。用全新run86c文件族完成静态/单测、生命周期与正式回执交接预检、fresh零态保存/独立恢复预检，提交修订与证据后才启动run86c。训练样本、九fresh模型、每模型24576更新、总221184更新、CPU≤14400秒、原六position门和零dev/v3 holdout/v4/v5边界完全不变。
