# M5 R2-H3.5-A v3 matched 训练预注册

> 状态：训练前已冻结，尚未授权执行正式训练或读取 final 能力结果。
>
> 可机读合同：`plans/reference/contracts/r2_h3_5a_matched_run_v1.json`
>
> 数据：`tests/fixtures/r2_h3_5a_response_plan_v3.jsonl`，digest `0bc5b5540d4b622f907942f42d7b809e825932e50c6e6c505a3cd091edabd691`

## 冻结范围

- 12 train、8 dev、4 final，family 与完整 response 跨 split 隔离。
- train/dev/final 都覆盖 answer、say_unknown、clarify、refuse；同时包含共享起点不同内容、history 和 context permutation。
- 三个种子：20260916、20260917、20260918；每臂 10 epoch，每 epoch 最多 12 个 train episode，CPU，300k 总预算。
- legacy control 使用 response-phase；treatment 使用32维 response-plan；主要因果对照是同一 treatment checkpoint 的 plan bridge 消融，以排除额外参数解释。
- 开发运行必须带 `--defer-final`。脚本只生成 train/dev 输出，paired 诊断也只读取 dev；final 在三种子 dev 报告和消融冻结后至多读取一次。

## 执行门

每个 seed/arm 必须使用独立 checkpoint、report 和 preflight 目录。任何一臂出现保存恢复失败、parent 漂移、超预算、运行时 oracle、只改善 train、只改善 boundary 或消融无效，立即停止全部后续 seed，不读取 final。

进入 final 前必须同时具备：六次磁盘 preflight 全通过、六次预算通过、三个 seed 的 dev plan margin 与 continuation 改善方向一致、parameter-matched plan 消融撤销 treatment 收益。final 只用于确认冻结结论，不调参、不换题、不追加 epoch。

该包即使通过也只允许进入 H3.5-B 机制复评，不自动形成 S2、L2、Mini 或默认入口授权。
