# 22.5.0 A 父门正式包：设计与定价卡（零训练，停在 owner 三决策点）

> 状态：designed_priced_pending_owner_decisions｜2026-11-19｜判据来源＝冒烟合同（D-20261117-34/35）＋读出线收官（D-20261119-02）
> `training_authorized=false`；0 正式训练；check 曝光已 2/2 用尽，本包不新增 check 读取（正式门按合同同器复用 check 分区作为门面，属同一次预算的正式执行——见 §4 判据注）

## 1 范围与前提

A 父门＝22.5 的准入门：主体在 first-A 任务上**实际学习**且**可恢复**（preflight/recovery 逐位）并**按合同判据过门**。读出线已以 `readout_gate_earned` 收官（fit 中位 0.969、check 中位 0.8906，[收束卡 §8–10](CLOSEOUT-TAIJI_U2_22_5_A_READOUT_20261118.md)），本卡把该已验证读出身份装进**正式包合同**并按实测出价。父身份＝[22.4 工程父](../manifests/taiji_u2_22_4_accepted_parent_20261117.json)（authorized=false）＋Plan C 重冻合同前置。

## 2 协议（相对冒烟合同的唯一变更集）

| 项 | 冒烟冻结值 | 本包值 | 依据 |
|---|---|---|---|
| 剂量/seed | 256 | **2000** | 读出线 §8/§10 在 2000 验证；256 时代读数已知退化（冒烟 fit 0/check 0） |
| 读出身份 | 旧（D-20261118-01 前） | **pixel-salience** | `readout_gate_earned` 冻结身份 |
| 状态容量 d | 8 | **32** | `9f55b5d82`：0.75→0.969 |
| 优化 | lr 1e-3、batch 2、串行 | **lr 5e-3＋200 步 warmup＋clip 1.0、batch 8、8-worker 进程并行** | 读出线验证 schedule；runner 续跑语义已修（`a33d1ed51`） |
| 种子 | 20261110/11/12 | **不变** | 合同钉死 |

合同腿不变：preflight（初始 checkpoint 保存/加载＋梯度逐位相等）→ 训练 → fit 评测 → 终 checkpoint 保存 → recovery（重载后 fit 行逐位）→ check worker（`run_a_gate_c_20261117 --check-worker` 逐字节不变，192 development-check 世界，聚合释放）。

## 3 实价（[定价脚本](../../scripts/training/price_a_gate_formal_package_20261119.py)、[价目报告](../../reports/taiji_u2_a_gate_package_price_20261119.json)；threads=1、丢弃模型、0 正式训练）

- 实测：逐例 backward **0.088542 CPU** ⇒ 逐更新（batch 8）**0.708333 CPU**；fixture 768 ≈ 5.625 CPU/进程（每腿 9 进程 ≈ 50.6）；check worker 实测 **118 CPU**（D-20261119-02）。
- **U2 臂/seed ≈ 1,735.3 CPU**（训练 1,416.7＋fixture 50.6＋check 118＋正式腿预算 150）；×3 seeds ＝ **5,205.9 CPU**。
- **配平基线臂**（Plan C `weight_tied_transformer_sequence`，FLOPs 比 10.436519 ⇒ 2,000 剂量下 **20,874 更新/seed**；按冻结合同单价 417.6 CPU/8,013 更新换算）＝ **3,263.6 CPU**。
- **包合计 8,469.4 CPU；建议帽 10,586.8 CPU（×1.25）**；U2 更新总量 6,000。
- 对比 22.4.4 冻结容量：更新 6,000 ≤ 9,808 ✓；CPU 10,587 对 16,000 阶段帽——**见 §5 预算重冻**。

## 4 判据（owner 决策点一：两案同价，只差口径与风险）

- **甲（合同原样）＝逐种子 check ≥0.80**（冒烟合同 `A_GATE_MIN`、all-seeds）。风险明写：seed 20261111 在已验证协议下的现有读数 **0.4062**，且其探针族 4 个偏移全弱——按甲执行大概率判 `a_parent_not_met` 并烧掉配平臂 3,264 CPU。
- **乙（执行者推荐）＝三种子 check 中位 ≥0.80**。依据：读出线 §9 已用同一中位口径收官（先例同构）；seed 11 失败模式已定性（只会左/右分界、未知与中全错＝臂型非协议缺陷）；现有读数中位 0.8906、距门余量 0.09。两案训练价格相同；乙不承诺新跑必过（臂不稳定是种子×配置级，已披露），只承诺判据不再由单臂初始化彩票决定。
- 两案共同门面：preflight/recovery 逐位、valid_utf8＝eos＝192 全量、fit 随读不进门（诊断）。过门 ⇒ `a_parent_earned`（签 22.5 学习父资格待 22.4 链闭合）；不过 ⇒ `a_parent_not_met` 有定位收束。**本包不重读 check 曝光账**（2/2 已用尽；正式门与 D-20261119-02 同器同分区，属同一预算的正式执行面——若 owner 裁定须另立曝光账，见 §6 注）。

## 5 预算重冻（owner 决策点二）

22.4.4 阶段帽 16,000 CPU 按 d=8 单价（0.54375/更新）冻结；**新单价 0.708333（1.30×）已使其失真**。诚实已耗账（A 线诊断/扫掠腿，11-18 夜→11-19，按新单价折算）：深夜 d=32 发现腿 ≈1,417；晨间 3 直线＋12 探针＋seed10 续跑 ≈8,000；午后本会话 3 续跑＋check workers ≈3,960——**合计 ≈13,400 CPU**，当前帽仅余 ≈2,600，正式包 10,587 装不进。**建议：22.5 阶段容量按新单价重冻为 24,000 CPU**（＝旧帽 × 1.30 单价比 × 旧容量语义），13,400 记为已耗诊断账单列、10,587 为帽内正式首笔；不上调则本包 `stage_budget_unavailable` 停卡。

## 6 配平基点（owner 决策点三）

剂量 256→2,000 强制配平重冻：基线 **2,671→20,874 更新/seed**（FLOPs 比不变 10.436519，`flop_ratio_cap` 10.6667 仍满足）、common FLOPs/seed ＝ **61,789,536,000**。选项：甲＝配平臂随包执行（推荐，3,263.6 CPU 计入包）；乙＝配平臂拆另案（A 门读数先出、签父悬置等配平腿）。乙与 `:378` 配平前置相抵，不推荐。注：`training_authorized=false` 起点与 `time_parity_claimed=false`（wall 135.99× 仅披露）继承原合同。

## 7 出口（互斥按序）

`stage_budget_unavailable`（§5 不批）＞ `a_parent_not_met`（判据甲/乙不达，保留 per-label 定位）＞ `engineering_failure`（preflight/recovery/UTF-8 任一红）＞ `budget_stop`（超帽 10,586.8）＞ **`a_parent_earned`**。正结果只签 22.5 学习父资格成立；不签 22.6 解锁（须 22.5.1 B/check② 另案）、不签任何对外能力。

## 8 唯一后继

owner 对 §4/§5/§6 三决策点各裁一项（执行者推荐：乙＋重冻 24,000＋配平随包）⇒ 冻结合同回填 `training_authority` ⇒ 按帽执行 3 seed 正式包 ⇒ 读数卡＋队首更新。裁决前 0 正式训练。
