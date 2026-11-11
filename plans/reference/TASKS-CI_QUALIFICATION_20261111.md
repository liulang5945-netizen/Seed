# TASKS：CI 5 条定性 + 修复（2026-11-11）

依赖 spec：`plans/reference/SPEC-CI_QUALIFICATION_20261111.md`
前置：用户批准 §6 的三个决策。

## 块 A：跨进程摘要（C1-C4）

- [x] A1 改 `tests/taiji_native/test_conditional_language_candidate_v3.py`：父进程取线程数，子进程 `torch.set_num_threads`
- [x] A2 改 `tests/taiji_native/test_native_content_core.py`：同上
- [x] A3 改 `tests/taiji_native/test_ordinary_baseline.py`：同上
- [x] A4 改 `scripts/training/run_native_content_core_r5.py`：`_fresh_probe` 传 `--threads`，子进程钉定
- [x] A5 验证：上述 4 文件单跑全绿；再以「r022 → core → ordinary → v3 → r5」敌意顺序合跑全绿（预期 `37 passed`，当前为 `1 failed, 36 passed`）
- [x] A6 简报：条数、涉及文件、残余

## 块 B：r022 可诊断性（C5）

- [x] B1 `test_living_core_credit_r022.py:859` 改为带 `gate` 载荷的断言
- [x] B2 本机重跑确认仍绿（本机 gate 通过，断言消息不影响通过态）
- [x] B3 简报：判定为环境决定、失败字段待下一次 CI 定位，未改阈值

## 块 C：两条关系契约容差（C6-C7）

- [x] C1 `test_native_local_relation_contract.py:72` 改为 `8 ulp × 量级` 自缩放容差 + spread 打印
- [x] C2 `test_native_relation_confirmation_contract.py:72` 同上（`transfer.torch`）
- [x] C3 验证：两文件单跑全绿
- [x] C4 简报：判定为判据本身写错（逐位相等跨 BLAS 不成立）

## 块 D：deselect 清单（只出清单，不强行恢复）

- [x] D1 从 `.github/workflows/ci.yml` 提取 79 条/腿的 `--deselect` 目标（51 个文件），标注理由（缺件/慢/环境）
- [x] D2 分「已知环境不可跑」与「被顺手排除」；后者且零风险者恢复
- [x] D3 简报：条数、恢复数、剩余数

## 块 E：收束与提交

- [x] E1 台账新增 `D-20261111-xx`：三条 checkpoint 判据修正、R5 runner 缺陷、r022 待定位、两条容差修正
- [x] E2 `FINDING-CROSS_PROCESS_EVIDENCE_20261110.md` 追加三条新实例
- [x] E3 更新 `plans/active/roadmap/03_CURRENT_EXECUTION.md` 当前动作
- [x] E4 清理本轮临时产物，确认无残留机制
- [ ] E5 提交并推送（只提交本轮相关文件，不带 worktree 既有脏文件）

## 明确不做

- 不跑正式训练、不读 confirm、不动无关冻结读数
- 不放宽任何既有阈值；r022 在拿到失败字段前不 rebaseline
- 620 份报告豁免、278 条 advisory 不作硬目标
