# PLAN-N1-162G：真实训练步的 CPU 线程吞吐基准

**状态：已依用户授予的执行计划批准权批准并冻结**<br>
**日期：2026-10-20**<br>
**触发：** 用户询问提高 CPU 占用是否会加快训练；N1-162B runner 将 PyTorch、OpenMP、MKL、OpenBLAS 限制为单线程。<br>
**相关结果：** [N1-162B训练结案](PLAN-N1-162B_execution_closeout_20261020.md)、[N1-162F1来源与逐条复核](PLAN-N1-162F1_execution_closeout_20261020.md)

## 问题与设计

在同一N1-162B seed11/surface_control final checkpoint、同一批冻结训练槽位、同一训练步骤上比较1、2、4个CPU线程的吞吐。每档做两次独立内存恢复；固定执行顺序为 `1,2,4,4,2,1`，每次先运行32步预热，再计时256步，共6次×288步=1,728次内存内更新。每次均从同一checkpoint重新恢复，不把上一次试跑的参数或optimizer状态带入下一次。

训练槽位取N1-162B已用的seed11/surface_control schedule第0–287步，使用原上下文构造及`n132._training_step`。源码/schedule构建和checkpoint恢复时间不计入每步吞吐，但计入总CPU预算；从预热后开始的256步计时覆盖逐步上下文构造及原训练步骤。另报告有效CPU核数（进程CPU秒/墙钟秒）、每更新CPU时间、各重复的更新/秒及相对单线程的加速比。

只有当某个多线程档两次更新/秒都至少比单线程两次结果的中位数高10%，且六次均从相同fingerprint恢复、源checkpoint哈希不变，才记为“稳定吞吐收益”；否则结论为无明确收益。该阈值只用于解释本次资源基准，不是模型能力门。

## 前置、预算与边界

- 使用封存checkpoint `data/checkpoints/n1_plan162b_question_surface_20261020/s11_surface_control_final.pt`，预期SHA-256 `6976af645eff7213aa4d210d0919451c37e430bf3821b3c1e0c1360f17acd913`。每次计时前独立恢复并核对optimizer/history/fingerprint；每轮后及结尾复核原checkpoint SHA未变。N1-162B的zero/final完整bundle已保存并独立恢复6/6，本计划不写任何checkpoint。
- 固定训练schedule manifest文件SHA-256 `225b5d6e55df9cec3b62a977e2a34173a9dc39b9c54d32d5258c2451e9e99fd1`，其9216行schedule摘要SHA-256 `bf6cc77a9f569b3d9dca151d455bfbe6ccf875715d17f2a123d94dfc2c88c84c`；N1-162B result SHA-256 `19dfaeb2086abaa503ee5150dba3bbfc9756d0892e8b6b54222fbfc4ecba2c31`。来源、runner或checkpoint身份不符即停止。
- CPU累计上限900秒；不计训练验收分、不生成或查询题目，不访问dev/holdout/confirmation；不写回任何模型状态、不改正式训练线程配置、数据、目标、判据或晋升状态，`can_promote=false`。
- 这是资源性能读数，不是学习质量证据。任何线程数的数值差异不用于能力比较；只有稳定的墙钟吞吐提升才支持后续单独调整训练执行线程数，仍须保留可复现性与正式checkpoint恢复检查。

## 产物

- runner：`scripts/training/run_taiji_n1_plan162g_training_thread_scaling_benchmark.py`
- 契约测试：`tests/taiji_native/test_n1_plan162g_thread_scaling_contract.py`
- 报告：`reports/taiji_n1_plan162g_training_thread_scaling_20261020.json`
