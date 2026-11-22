# S1-RT 合同修订草案：以 RT 机制重立 S1 判据（待 owner 审阅；零训练、零 check）

> [修订草案] — 2026-11-22（D-20261122-270）。依据 owner 裁定 A（以 RT 机制重立 S1 并回灌 MAINLINE）。**本文不运行任何训练、不签 S1；零 check/final。**

## 1. 修订关系

- 旧 `§3A.1 S1`（同主体联合生命流，四轮 `s1_partial`）按 D-240 收束为历史证据，不再追加探针；
- 本草案以 **RT 机制**（任务子群路由＋参数级保护＋`capsule_v2` 继承）重立 S1；通过后 `23-S..28-S` 才解锁；
- 旧 S1 的判据原件与四轮读数保留，不做改写。

## 2. 主体与范围（合成受控面）

- 单一可继承 `SelectivePersistentGraph`：72 单元（12 群×6）、`per_sample_state=True`、学习式写门；
- 任务顺序：`P-A` 中文双向 8 类 → `P-B` 四类纠错（4+conflict）→ `P-D` 遮挡物理 4 类；
- 每任务 **24 单元固定路由**（`task_routes`，可由 capsule 继承）；训练采用**参数级保护**：每阶段只更新本任务单位行/列与 `read_gate` 对应行，其余梯度置零；保护阶段 `weight_decay=0`；
- 读出头按任务独立（`PhaseHead`）；读出头随 `capsule_v2` 之外的 head state dict 保存（本轮显式披露）。

## 3. 判据（读前冻结；G1–G8）

| 门 | 指标 | 判据 |
|---|---|---|
| G1 共存 | 每 seed 终值 | CN ≥0.95；P-B ≥0.85；P-D ≥0.80 |
| G2 纠错 | P-B conflict | ≥0.95 |
| G3 消融 | P-D 无探针输入下降 | ≥0.20 |
| G4 旧技能 | CN/P-B 相对峰值损伤 | ≤0.02 |
| G5 恢复 | capsule 前向差；恢复后三任务差 | ≤1e-6 / ≤1e-6 |
| G6 对照 | 共享臂；冻结臂（历史同协议读数） | 共享臂 CN 或 P-B 终值 <0.65 或损伤 >0.05；冻结臂 P-B/P-D <0.80 |
| G7 预算 | 3 seeds、单卡 ≤800 CPU、零 check/final | 硬帽 |
| G8 纪律 | `criteria_sha256` 读前冻结；run conditions 记录；不调阈 | 必须 |

出口：G1–G8 全过 ⇒ `s1_rt_verified`（解锁 23-S 待 owner）；任一不过 ⇒ `s1_rt_partial`（按预注册缺口返工，不加探针）；超帽 ⇒ `budget_stop`；工程失败 ⇒ `engineering_failure`。

## 4. 与已有证据的映射（如实）

- D-257（`s1rtf_verified`，3 seed，629.62 CPU）已按下表读数达 G1–G5：CN 1.0；P-B 0.984–1.0；P-D 0.852–0.914；conflict 0.992–1.0；消融 0.367–0.508；旧损伤 0–0.008；胶囊/恢复 0.0；
- G6 对照来自 D-260（共享臂 CN/P-B 终值 0.17/0.26）、D-240（冻结臂 P-B 0.36/P-D 0.46，同协议变体）；
- **但**：S1-RT 合同本草案晚于 D-257，若直接采信 D-257 属「先读后冻」。合规做法：以本草案冻结 `criteria_sha256` 后，**按新冻结合同重跑一次 3 seed（≈630 CPU、零 check）** 作为正式 S1 证据；D-257 保留为预验证。

## 5. 授权与流程

1. 本草案交 owner 审阅（范围/门/预算）；
2. owner 批准后：冻结 `criteria_sha256` → 单卡 `training_authority` → 3 seed 正式跑 → 出回执 → 判 `s1_rt_verified` / `s1_rt_partial`；
3. 通过后回灌 MAINLINE：S1 标记 `s1_rt_verified`，23-S 解锁（仍需新卡授权）。

## 6. 明确不含

- **不引入蒸馏**（owner 决定）；生长/协同不在本卡（见附录 B 参考）；
- 不签模型成型/check/final/C/D/J；不测开放中文/真机/多模态。

## 附录 B：B 路线非蒸馏参考（未来卡用，不属 S1）

1. **生长**：PNN/Net2Net 式**函数保持初始化＋旧列冻结＋横向连接**（无蒸馏；用读出混合旧/新 + lesion 门验证贡献）；
2. **协同**：DeepSeekMoE 式**共享专家＋细粒度 routed experts＋负载均衡**（共享通识、专长分开，避免全部共享的折中）；
3. **容量分配**：HAT 式**学习 mask＋累积掩码＋稀疏正则**（替代手工固定分区）；
4. **评测**：3 seeds、ACC/BWT/FWT、mean±std；单 seed 只作机制归因。

## 7. 声明
零训练、零 check/final；草案待 owner 审阅；不签任何能力结论。
