# M4.V2 C 阶段 formal v2 预注册（G1 操作化修正 + fresh sealed v5）

> 冻结日期：2026-09-10。前置：[C 阶段 formal v1](M4V2_C_STAGE_FORMAL_PREREGISTRATION_20260910.md) §7（G3/G4 通过、G1 失败——映射缺口）。本文冻结 G1 的操作化修正与全新 sealed v5 的评分合同；其余判据与臂构成与 v1 完全相同。冻结后实现 runner 并运行。

## 1. 唯一变更：G1 操作化

**v1 G1**：C 与 FS 的 **validation** combined delta 在 ≥ 2/3 课程上 < 0。

**v2 G1（冻结）**：C 与 FS 的 **sealed** combined delta 在 ≥ 2/3 课程上 < 0（实际改善的证据在主测量上验证）。

**修正理由（v1 §7.1 已记录）**：FS 的 replay 巩固收益是类选择性的——在覆盖 D/R 弱类的 sealed 上强负（−0.131），但在不覆盖弱类的 3-episode validation 上高方差略正。validation 不是 replay 类选择性机制的良好学习代理。学习门应问「训练是否在主测量上产生了改善」，而非「是否在小样本 validation 代理上改善」。

**不混淆 G1 与 G3 的边界**：G1 是「臂学到东西了吗」（delta < 0，绝对证据）；G3 是「FS 比 C 好吗」（FS < C，比较证据）。两者量尺不同，G1 过不意味着 G3 过。

## 2. 其余判据与臂构成（全部与 v1 相同，零变更）

- G2 灾难界、G3 弱类主判据、G4 整体非劣、G5 描述性——结构、阈值、epsilon 派生方式不变；
- 四臂（F / C 300 步 / FS 300 wake+100 replay / XL 2× 次要-bar）不变；
- v4 artifacts 原样复用（digest 对 v4 build report 校验）。

## 3. Sealed v5（全新 materialize）

- `materialize` task_seed 未用过（≠0/19/47/73/101）；3 episodes 首 file 覆盖 D/R/A（与 v4 同构但 fresh 内容）；不交性对全部前序 sealed（v1/v2/v3/v4）与 train/validation 强制。

## 4. 停止线与边界

与 v1 相同；sealed v5 读取后禁止任何判据/阈值调整；`can_promote=false` 固定。

## 5. 结果映射（修正后覆盖全分支）

- G1∧G2∧G3∧G4 全过 → FS 成为 K 相位默认学习机制候选；
- G3 失败 → continuation 收束为默认、B 阶段闭合；
- G4 失败 → replay 设计回炉；
- G1 失败（sealed delta ≥ 0）→ FS 未在主测量上产生改善，continuation 收束为默认；
- G2 失败 → 灾难性劣化，回学习设计归因。

## 6. 产物

1. `materialize_taiji_m4v2_c_stage_sealed_test_v5.py` + sealed v5 artifact；
2. `eval_taiji_m4v2_c_stage_formal_v2.py`（复用 v1 runner 机械，G1 改 sealed delta；先 py_compile/ruff）；
3. pre-sealed 产物 + `reports/taiji_m4v2_c_stage_formal_v2_20260910.json`；
4. 路线图执行记录 + 独立提交。

## 7. 执行记录（2026-09-10，formal v2 已运行，四门全过）

1. sealed v5 materialize：task_seed=151（未用集 {0,19,47,73,101}），3 episodes 首 file 覆盖 D/R/A；对 v1/v2/v3/v4 sealed 与 train/validation 不交校验通过；artifact sha256 `c7680ad5…34b5d`、internal digest `d620bf74…c8b64`。
2. 两阶段纪律：pre-sealed 产物先落盘（`reports/taiji_m4v2_c_stage_formal_v2_presealed_20260910.json`）——9 行 validation 评分 + epsilon 派生（epsilon_cat=0.01、epsilon_ni=0.003846）在 sealed 读取前冻结。
3. 结果（sealed v5，统计单元 course n=3，`sealed_read_count=1`）：
   - **G1（v2 sealed 口径）3/3 通过**：C −0.12879/−0.12875/−0.12876、FS −0.13070/−0.13043/−0.13078，全课程 <0；
   - **G2 通过**：全部课程 delta 远离 +epsilon_cat=0.01 灾难界；
   - **G3 弱类主判据 3/3**：FS −0.19392/−0.19352/−0.19412 vs C −0.19140/−0.19134/−0.19133，每课程好 ~0.0022（D 类 −0.2244 vs −0.2237、R 类 −0.1632 vs −0.1595 量级一致）；
   - **G4 整体非劣 3/3 且 FS 实际更优**：好 ~0.0018，亦优于 XL（−0.12868）。
4. `formal_passed=true`、`can_promote=false` 不变（无晋级、不接默认运行时）。按 §5 冻结映射：**FS（fast/slow+replay）成为 K 相位默认学习机制候选**。v1 的映射缺口（G1 败 + G3/G4 过）被本次操作化修正消解——replay 的类选择性收益在主测量上成立，validation 代理的高方差不再是学习门瓶颈。
