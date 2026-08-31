# R5C-S52：artifact consumption policy 与 verified 默认边界 Gate

## 目标

把 S51 的 `require_verified_measurements` 布尔开关收敛为显式、可审计的 artifact consumption policy，避免调用方散落硬编码 strict/legacy 语义。策略必须明确新运行时、受控迁移和 legacy 回放分别允许什么，同时不把外部 artifact 自动注册为认知内容。

## 推荐方向

- 新建或受控成长路径默认 `verified-only`：新 measured artifact 必须带可独立验证的 measurement sidecar。
- 历史 checkpoint/replay 只通过显式 `legacy-compatible` policy 运行，并把 policy、原因和 artifact status 绑定到调用审计；不隐式升级、不伪造 facts。
- policy 本身内容寻址、可 checkpoint、可回滚；不把策略判断散落成多个布尔分支。
- multi-candidate preflight、runtime bridge、replay、admission、rollback 和 retention 继续复用既有原子 contract。

## Gate

真实 native/CPU canary 必须证明：

1. verified-only policy 接受 verified bundle，拒绝 legacy/missing/tampered sidecar，且失败不改变 runtime；
2. legacy-compatible policy 只允许明确的旧证据回放，并在 projection/audit 中保留原因；
3. policy save/load/replay/rollback 稳定，策略变更不能绕过 artifact、candidate、batch 或 parent-checkpoint 校验；
4. 多 candidate policy resolution 保持 all-or-nothing，任何不满足策略的 candidate 不会提前消费 sibling。

## 验证入口

- 定向测试：`tests/taiji_native/test_artifact_consumption_policy.py`
- CPU canary：`scripts/training/eval_taiji_artifact_consumption_policy.py`
- 报告：`reports/taiji_w7_r5c_s52_artifact_consumption_policy_20260831.json`

本 slice 是 S51 之后的策略收敛节点；在 policy 默认边界明确前，不改变现有默认 legacy 兼容，不处理 CI、CUDA、前端、Windows shell、自动删除、无限扩张、开放域收益或通用智能声明。
