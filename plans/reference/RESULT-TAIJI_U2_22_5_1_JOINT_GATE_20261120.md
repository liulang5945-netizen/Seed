# 结果卡：22.5.1 联合门 fit 面（`joint_gate_fit_face_pass`，含 verify-only 修订披露）

> 文档状态：[结果卡] — 2026-11-20。绑定合同：[`taiji_u2_22_5_1_joint_gate_20261120.json`](../manifests/taiji_u2_22_5_1_joint_gate_20261120.json)（`status=frozen`、`training_authority=owner:standing-authorization:2026-11-20#recommendation-default`、`result.closed=true`）。回执：[run](../../reports/taiji_u2_22_5_1_joint_gate_run_20261120.json)／[verify（含修订）](../../reports/taiji_u2_22_5_1_joint_gate_verify_20261120.json)／[smoke](../../reports/taiji_u2_22_5_1_joint_gate_smoke_20261120.json)／[定价](../../reports/taiji_u2_22_5_1_joint_gate_price_20261120.json)。设计卡：[DESIGN-TAIJI_U2_22_5_1_JOINT_GATE_20261120](DESIGN-TAIJI_U2_22_5_1_JOINT_GATE_20261120.md)。

## 1. 范围、授权与身份

- **范围**：MAINLINE §2 `22.5.1` 联合门在 **fit 面**的 B 纠正＋多记录＋相对父代 D＋可失败 A 保持＋因果对照＋恢复。**不是**正式 check② 裁决（22.5 check 曝光保持 2/2 用尽，`0 check / 0 final` 读取）。
- **授权**：owner 常设授权「按推荐项先推进、后续回溯修订」→ 合同记录 `owner:standing-authorization:2026-11-20#recommendation-default`；S1 由 owner 豁免（生成链未重跑风险已登记，D-20261119-12）。
- **父身份**：单一主候选＝已接受 v1 A 父 seed `20261112` bundle（`pixel-salience`、`d=32`、2,000 更新）。合同绑定文件 sha256＋strict load＋记录 fit 复现；run preflight `fit 31/32` 与记录一致，父参数全程逐位不变。
- **3 seed**：同一父 bundle × 3 个训练子种子（`20261110/20261111/20261112`）；父不重训/不换/不池化。

## 2. 协议与臂（读前冻结）

反馈训练 256 步、batch 2 world、已见 2 模板；评测 fit 32 world 的 24 个非未知 B 项（已见 2 模板／未见第 3 模板两面）。臂：`main`（器官训练）、`frozen_update`（同调度 `lr=0`，零参数更新）、`permuted_feedback`（闭环轮换监督目标）、因果切断 `record_organ`/`fast`/`record_injection`、`overwrite_probe`（+2.0 无保护扰动）。判据与出口顺序见设计卡 §2–§3；配对 bootstrap 10,000 次的单侧 95% 界。

## 3. 读数（三 seed 完全一致；run `1359.22 ≤ 帽 1400 CPU`，wall `1370 s`）

| 指标（分母 24/24） | 读数 | 门 |
|---|---|---|
| B 全应答轮成功 / 纠正率 | **24/24 = 1.0 / 1.0** | ≥0.80 ✓ |
| 反馈学习净益 vs `frozen_update` | **1.0**，95% 下界 **1.0**，3 seed 同向 | ≥0.05、下界>0 ✓ |
| A 保持（child / parent） | **23/24 / 23/24**，单侧 95% 上界 **0.0** | ≤0.02 ✓ |
| D 相对父代（child / 父代同条件） | **22/24 / 9/24**，相对下降上界 **−0.375** | ≤0.02 ✓ |
| 多记录 / 版本覆盖 | 21/24 / 24/24 | 报告项 |
| 因果切断 B | `record_organ` **0/24**、`fast` 24/24、`record_injection` 24/24 | 必要项 ≤2 ✓ |
| 置换反馈 B | **0/24** | ≤2 ✓ |
| 可失败保持探针 | 扰动后 A 下降 **1.0** | ≥0.02 ✓ |
| 恢复（独立进程重载） | **3/3 `recover_exact`**（同条件 A/B 逐字节） | exact ✓ |
| 父代逐位不变 | 3/3 | ✓ |

**判词：`joint_gate_fit_face_pass`**（按冻结出口顺序，verify 修订后重算；原 run 判词同）。

## 4. verify-only 修订（必须披露）

首次 run 的 A 行把「A 真值」误写成「B 修订目标」，导致 A 保持判据空转（读数 0/24 对 0/24）。按 E5.1「verify-only amendment」先例，新增 `--verify` 模式：用已保存的三颗子体 artifact 重测 A 面（**0 优化器步、0 check 读取**），以**不变判据摘要**重算判词；B/D/对照/恢复读数不改。修订后 A child `23/24`、parent `23/24`、单侧上界 `0.0`。修订成本 `54.42 CPU`；合同 `result.amendment` 永久留痕。

## 5. 签署边界（严格）

- **签**：fit 面联合门读数与判词；B 纠正/多记录/反馈净益/相对-D/A 保持/因果/恢复在该面上的成立。
- **不签**：正式 check② 裁决（曝光已尽）；**B 父资格正式化**（spec 要求 check② 整包裁决，未做）；架构优越性门（`architecture_gate: not_run`，等算力配平前置未满足）；22.6 解锁；final；生成链可复现性（S1 豁免的残留风险）。
- **口径歧义披露（建议 owner 回溯时裁定）**：本卡按 D-20261119-11 冻结口径以「父代同条件行为」为 D 参考（child `22/24` vs parent `9/24`，通过）。若改用更严的「父代 A 真值」参考（parent A `23/24` vs child D 22/24），drop `1/24 = 0.0417`、单侧上界约 `0.083 > 0.02` ⇒ 会判 `d_backoff_breach`。同一份读数在两种口径下判词相反，已如实并列，不埋。

## 6. 成本与后继

- **成本**：S2 设计/定价/smoke `≈ 22.12 + 定价 12.6 CPU`；S3 授权 `0`；S4 run **`1359.22 CPU（帽 1400，余量 40.78）／1370 s wall`**；verify 修订 `54.42`。总计 `≈1,448 CPU`，对先前口头估价（包价 1,102.71／帽 1,378.38）超包价但未超帽；阶段账继续按「超支永久记事故」处理，单列本卡实耗。
- **唯一后继（S5 已在本卡完成同步）**：owner 回溯裁定三选一——(a) 认可 fit 面证据并按具名 `fit_face_only` 豁免 check②，解锁 22.6 并把残留风险登记；(b) 另批一次 check② 曝光做正式整包裁决（需新曝光账）；(c) 先补更严 D 口径的重测（`verify` 模式可扩展，估 <80 CPU）再裁。执行者建议：先做 (c) 的零训练双口径复核与 (a) 的具名豁免并行呈报，不自动解锁 22.6。
