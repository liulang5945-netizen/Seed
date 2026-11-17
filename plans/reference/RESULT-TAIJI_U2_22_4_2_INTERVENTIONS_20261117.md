# 22.4.2.5 零训练固定干预矩阵结果（3 seed 全臂通过）

> 卡号：`22.4.2.5`｜日期：2026-11-17｜判定：**`intervention_matrix_valid`**｜性质：**零训练 fixture/评分器核验**，不是模型能力/父结果。
> 依据：[计划 §22.4.2.5](PLAN-TAIJI-CLOSED_LOOP_DEVELOPMENT_20261110.md)（动作因果性、经历必要性、跨模态依赖；固定干预矩阵与原始轨迹；信息不足进未知/求证集合）、[22.4.2.4 结果卡](RESULT-TAIJI_U2_22_4_2_CDJ_CERT_20261117.md)、[世界规格](SPEC-TAIJI-U2_WORLD_22_4_2_20261112.md)。
> 纪律：**0 模型前向、0 optimizer step、0 项目数据读取、0 封存/check/final 读取**；实耗 **63.7812 CPU／9.672 wall**（帽 120／240）。
> 预注册：判据与出口在读数前固定，`criteria_digest=6d3af92c1f61a6f49fc3eb95f70e9f95b4842b5ebd039815d746ae81f18fc51b`；报告落盘 [reports/taiji_u2_intervention_matrix_22_4_2_20261117.json](../../reports/taiji_u2_intervention_matrix_22_4_2_20261117.json)。

## 1 固定矩阵（3 seed × 8 臂 × 2 case）

| 臂 | 对照构造 | 判据 | 读数（seed 124／125／126） |
|---|---|---|---|
| `action_causality` | 同初态（x=-0.55）、同首帧；18 步右推 vs 左推 | 首帧逐位相等、动作不同、末帧/当前答案不同 | 右推**中间**、左推**左边**；3/3 过 |
| `history_necessity` | 不同初态（-0.55 vs -0.25）与不同动作长度；**末帧逐位相同** | 末帧相等、首帧不同、previous 答案不同；只看当前视图不可辨 | previous **左边 vs 中间**；当前视图仅见"未知/未知"⇒**不可辨**；3/3 过 |
| `appearance_dynamics` | 同初态、**同动作**；mass/drag 1.0/1.0 vs 2.0/2.0 | 首帧相等、动作相等、末帧/当前答案不同 | 轻**中间**、重**左边**；3/3 过 |
| `occlusion` | 同轨迹；末帧可见 vs 遮挡该实体 | 可见 answer、遮挡 unknown；末帧不同 | 可见**中间**、遮挡**未知**；3/3 过 |
| `audio_missing` | 真实接触声 vs 缺失；RGB 可见 | 有声答实体、缺失 unknown | **绿色／青色／红色圆片** vs **未知**；3/3 过 |
| `audio_conflict` | 真实接触声 vs 异频合成声 | 有声答实体、冲突 verify | 实体 vs **求证**；3/3 过 |
| `proprio_missing` | 本体有效 vs valid 全 0 | 一致 vs unknown | **一致 vs 未知**；3/3 过 |
| `proprio_conflict` | 本体强度 0.2 vs 篡改为 0.8 | 一致 vs verify | **一致 vs 不一致**；3/3 过 |

**全局不变量**：按 `(question, legal_digest)` 分组，不存在"同一问题、同一合法输入、不同答案/状态"的 case；3/3 seed 通过。

## 2 原始轨迹与可重建性

- 每个 case 记录 `trajectory_id`、seed、初态、mass/drag、48/30 tick 的**紧凑动作序列**（`[op, target_x, target_y, force_x, force_y, intensity, duration]`）、首/末帧 SHA-256、峰值音频 SHA-256、本体 SHA-256、首/末位置词与 world version。
- 原始 RGB/audio/proprio **不直接入库为张量**；由 seed＋动作序列可重放生成，报告 550,647 字节。
- 动力学臂使用**固定 target=-0.4** 的开环动作，保证 light/heavy 两 case 的动作序列逐项相同；差异只来自 mass/drag。
- 经历必要性臂的末帧逐位相等是实测量（`last_frames_equal=True`、`first_frames_equal=False`），不是共享张量伪造。

## 3 范围与限制

- **能签**：22.4.2.5 固定干预矩阵的 8 个轴在 3 seed 上均可被评分器区分；缺失/冲突臂分别进入原合同 `未知` 与本卡本地 `求证` 集合；动作因果、经历必要、外观/动力学、遮挡、音频/本体缺失与冲突均有固定配对与原始轨迹。
- **音频/本体臂的合同边界**：冻结任务合同只有 `current/previous/unknown` 位置时间意图，没有音频身份或本体一致性请求。本卡为这两类臂使用**本地 `CrossModalCase` 评分规则**（音频主频/本体强度与记录一致性），**未修改冻结合同**；若 22.4.2.6 要把它们接入模型，须先扩展合同并重新冻结。它们目前只签"矩阵与评分器可区分"，不签模型会听/会本体感。
- **不签**：任何父/能力/check②/22.6；22.4 七项交付仍缺第 2（真实任务族 recipe 及分区 digest）与第 7（`accepted_parent`）；22.4.2.6／22.4.3／22.4.4 未做。干预矩阵通过不等于 22.4 准入。

## 4 唯一后继

**`22.4.2.6` 生成器入口**：合拢世界/执行器/模型原始观测的 shape、schema 与资源样例，交付生成器入口及微型恢复回执；记录生成/编码/前缀重建/保存的真实成本。之后 22.4.3 分区 → 22.4.4 完整定价 → 22.4 `accepted_parent`。在 22.4 准入前：0 训练新增、0 能力、0 父；check② 不读、22.6／22.7 不解锁。
