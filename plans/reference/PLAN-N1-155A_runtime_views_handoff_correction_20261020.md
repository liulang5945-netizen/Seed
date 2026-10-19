# PLAN-N1-155A：N1-155 schedule 上下文传递更正

**状态：已批准，执行中**  
**日期：2026-10-20**  
**上位冻结方案：** [PLAN-N1-155](PLAN-N1-155_short_context_unknown_rebalance_and_fresh_diagnostic_20261020.md)，SHA-256 `6664a7c1e51778f67917a310832dbfa1cbd66033efc3db50fc03c0dc13a9cd82`

## 触发与结论

N1-155 首次执行在第一个训练 slot 应用上下文时停止：schedule builder 保留了 slot 与长度表，却没有把 N1-150 schedule 中的 `views` 一并传给训练器，因此触发 `KeyError: 'views'`。故障发生在第一次参数更新、评分或输出查询之前。

失败回执 SHA-256 为 `86a1b99508a1986818600c70ea4e9202c26ad8f596324bd551d5a9caa038ce07`；attempt SHA-256 为 `ace1aa1c2ad75445fb04533a0f87ed86524ca6c66114f9f522704d8e4317e4da`。回执记录训练更新 `0`、train-only 评分 `0`、dev/holdout/confirmation/fresh 查询 `0`，进度文件为空，累计 CPU 为 `130.921875` 秒。执行器仅恢复了 seed11 candidate 的 zero bundle，并写出 `additional_updates=0` 的局部 checkpoint（SHA-256 `21b032076d1ad0e93800062f222085f210c232204c0469e19abde27215a50550`）。六个原 zero bundle、readiness 与 preflight 均保留。

## 批准的更正

依据用户授予的执行计划批准权，批准一次隔离的 N1-155A 技术更正执行。修复仅把冻结 N1-150 schedule 的上下文 `views` 透传到 N1-155 运行 schedule；不改变训练输入、长度分配、模型、优化器、目标、更新量、评分分母、原训练门、seed128 面、查询上限或晋级状态。

N1-155A 使用独立的 readiness、schedule、preflight、checkpoint、attempt、progress、result 和 seed128 face 路径，保留 N1-155 首次失败的所有文件。新 readiness 与源审计将绑定本更正、失败回执、原 readiness/preflight、原 zero bundles、零更新局部 checkpoint，以及修复后的代码和测试。N1-155A checkpoint 仍记录上位 PLAN-N1-155 SHA，并额外记录本更正文件 SHA。

训练仍为六 endpoint、每个 9,216 更新，共 55,296 更新；若六个原门全通过，才生成并查询 seed128 fresh face，最多 2,880 条。train-only 评分仍为 7,488 条。0 dev、旧 holdout 和 confirmation 查询，`can_promote=false`。

CPU 总上限沿用 PLAN-N1-155 的 18,000 秒，并计入首次失败的累计 130.921875 秒；N1-155A 剩余阶段合计最多 17,869.078125 秒。readiness 和六个 zero full bundle 独立恢复仍须先通过，然后才开始训练。

## 新的 write-once 输出

- checkpoint：`data/checkpoints/n1_plan155a_v28_unknown8_rebalance_20261020/`
- schedule：`plans/manifests/n1_plan155a_unknown8_rebalance_schedule_20261020.json`
- 其余审计、readiness、preflight、attempt、progress、result：`reports/taiji_n1_plan155a_unknown8_rebalance_*_20261020.*`
- seed128 face：`plans/manifests/n1_plan155a_v29_short_unknown_length_seed128_20261020.json`

本更正只允许这一次隔离执行。任何 N1-155A write-once 阶段在启动后再次失败，都封存已写出的回执，不复用路径、不自动重试；后续执行须先有新的故障归因和更正裁定。
