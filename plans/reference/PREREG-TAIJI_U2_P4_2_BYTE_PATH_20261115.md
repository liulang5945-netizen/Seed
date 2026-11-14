# 22.5.P4.2 预注册：列通路接进字节路径（2026-11-15）

> 状态：判据与实价已冻结，**未获授权、0 训练、0 optimizer step**。契约 [`taiji_u2_p4_2_byte_path_20261115.json`](../manifests/taiji_u2_p4_2_byte_path_20261115.json)（`criteria_sha256=633a8618…`，帽 **3960 CPU／7920 wall**），实价 [`taiji_u2_p4_2_price_20261115.json`](../../reports/taiji_u2_p4_2_price_20261115.json)，梯度普查 [`taiji_u2_p4_2_census_20261115.json`](../../reports/taiji_u2_p4_2_census_20261115.json)。

## 1 一句话

本卡只问一件事：**主体自己的字节头，能不能把 P4.1 教会编码器的列颜色路由说成位置词。** 服务输出是主体自己的 `output()` 257 符号分布，不是旁路读出；成功是一次完整应答轮（精确答案字节 + EOS + 合法 UTF-8），分母是全部 fit 轮。

## 2 为什么它不是"直接跑 22.5.1 父门"

22.5.1 的 B 父门有五项，本卡只有其中一部分有对象：

| 22.5.1 项 | 本卡状态 | 原因 |
|---|---|---|
| 全应答轮成功率 ≥0.80 | **可测** | fit 32 轮，全部计入 |
| 纠正率 ≥0.80 | **可测（条件）** | 首轮错误 ≥4 才算分母；否则 `not_identifiable_by_design` |
| 真实反馈学习增益 ≥0.05 | **可测（条件）** | 同学科冻结更新对照，同反馈暴露、0 更新 |
| A 仍有效旧能力下降 ≤0.02 | **`not_applicable_no_A_parent`** | 22.5.0 未过，没有 A 套件 |
| world 配对 95% 下界 >0 且 3seed 同向 | **`not_in_scope_single_seed`** | 单 seed，无 world 配对 |
| check② 整包裁决 | **`not_read`** | 无 check 套件、不消费 check 额度 |

因此本卡最多产出**父候选**（development fit 层），**不能写 `accepted learning parent`**。这与 [22.6 准入卡](REVIEW-TAIJI_22_6_ENTRY_20261115.md) G1–G3 的判定一致：先有字节路径的读数，再谈父。

## 3 前置：P4.1 的权重已经在工作区丢失

`data/experiments/taiji_u2/p4_1/20261114T2215Z/` **为空**，`data/` 未入库，报告里也没有权重摘要。因此第 0 步不是"加载 B 臂"，而是**确定性重建**：

- 用 HEAD 上冻结的 P4.1 runner（`fe722cb4…`，与 P4.1 manifest 一致）重建 B 臂：同 seed、同数据/顺序、同 7376 步；
- 只在三个剂量的 **10 个签名字段**逐项与已发布报告完全相等时才接受（`fit_correct`、`train_correct`、两种颜色准确率、列命中、两种弃权、两种探针）；任一不等 → `p4_1_reconstruction_mismatch`，**不进入第 1 阶段**；
- 重建成本实测 ≈ **118 CPU**；产物落在 P4.2 自己的 run 根，不写 P4.1 目录。

## 4 构造：两阶段，服务输出只有一个

**观测重分组（零新参数）**：同一 packet、同一像素值，rgb 从 16 个 patch（48 值）改为 16 个**列**（48 值）＋列坐标（x 取列中心、y=0）；audio/proprio 原样。这是 `column-observation-v1`，主体 `state_dict` 不变。

**阶段 1：让主体自己的读把答案露出来**。编码器冻结在重建的 B 臂；其余全部参数（22,350）加一个 **100 参数的 4 向训练期脚手架**（读 `[h,o,fast]`，即 `joint_readout` 的输入，向量已按 branch 平均）训练。脚手架**只在训练期存在**，评估时不存在。

**阶段 2：让字节头把读数说出来**。全部参数冻结；只训练 `joint_readout`、`byte_head`、`copy_gate`（**8 张量／3,919 标量**）。训练数据是冻结主体**自己传给 `emission` 的实参缓存**（cursor/alpha/beta/ids/valid/mask）加 `joint_readout` 输入与上一字节；训练开始前用 3 个位置做**回放一致性检查**：缓存重算的 log-prob 必须等于主体自己 `output()` 的 log-prob（容差 1e-6），不等 → `engineering_failure`。

**为什么拆两段（实价依据）**：端到端 7 字节全图训练实测 **0.625 CPU/步**，7376 步 ≈ 4,600 CPU；"读"训练 **0.281/步**、缓存后的字节头训练 **0.016/步**。两段把同一件事拆成两条可支付的信用路径，服务输出仍是字节头。

## 5 判据（冻结原文）

| 项 | 定义 | 门 |
|---|---|---|
| 全应答轮成功率 | fit 32 轮全部计入；精确答案字节＋EOS＋合法 UTF-8；greedy，65 字节上限；失败/多余字节不删不筛 | **≥0.80** |
| 纠正率 | 首轮错误 ≥4 时，对每个错误释放正确答案并做 **8 次 Adam（lr 0.001）**，即时重问；分母＝首轮错误 | **≥0.80** |
| 反馈学习增益 | 学到副本第二遍 fit 准确率 − 同主体冻结更新对照第二遍 fit 准确率；对照收同样反馈、**0 参数/状态写入** | **≥0.05** |
| 恢复 | 保存 `final.pt`；本卡先做 in-process 重载逐字节一致；守卫再做**独立进程**重载 | 逐字节相同 |
| A 保持 | 无 A 父 | `not_applicable_no_A_parent` |
| 3seed／95% 下界 | 单 seed | `not_in_scope_single_seed` |
| check② | 不读 | `not_read` |
| accepted parent | 不签 | `false` |

阶段 1 脚手架门只用于**早停**：train 与 fit 都 ≥0.95 → 收敛停止；前两个剂量 fit 都 ≤0.35 → `scaffold_cannot_read_answer`（第 2 阶段仍跑一次，给出可比读数）。

## 6 剂量、早停与出口

- 阶段 1 剂量 **1844／3688／7376**（与 P4.1 同剂量，便于比较）；阶段 2 剂量 **461／922／1844**。
- 预算帽 **2650 CPU／5300 wall**；超帽 `budget_stop`，保留全部 checkpoint 并报告最后剂量。
- 非有限梯度/参数、缓存回放不一致、source pin 不符 → `engineering_failure`。
- 出口：`byte_path_carries_answer`（主门＋纠正/增益（若可测）＋恢复）、`byte_path_carries_answer_gates_incomplete`、`byte_path_learning_below_gate`（末剂量比首剂量提升 ≥0.10）、`byte_path_blocks`。

## 7 实价（0 训练，2026-11-15）

| 量 | 空闲机读数 | 加载机读数 |
|---|---:|---:|
| 单次 prepare | 0.015625 | 0.03125 |
| 阶段 1 步（batch 8） | 0.28125 | **0.421875** |
| 缓存位置（含 prepare） | 0.005441 | 0.007952 |
| 阶段 2 步（8 位置） | 0.015625 | 0.015625 |
| 完整单例步（纠正用） | 0.046875 | 0.0625 |
| fit 轮评估 | 0.9375 | 1.40625 |
| P4.1 重建步 | 0.015625 | 0.03125 |
| P4.1 探针 | 0.265625 | 0.375 |

两次测量相差约 1.5 倍且**整体同比例**，判断为机器负载而非代码差异。帽取**慢机读数**作上界：最坏情形 3,959 CPU ⇒ 帽 **3,960 CPU／7,920 wall**；快机上会远低于帽。实价测量本身 7.97 CPU／8.17 wall（慢机）。

## 8 产物与守卫

产物：本卡 runner（`scripts/training/run_p4_2_22_5.py`）、守卫（`scripts/training/verify_p4_2_20261115.py`）、census／price／reconstruction／run／guard 五份回执、本预注册、结果卡。

守卫 21 项（见 manifest `guards`），含 4 个红方向：改 `fit_correct`、从分母删一个错误、对照出现更新、声称父。所有 checkpoint 记 sha256；P4.1 权重丢失的教训以"重建＋签名"处理，不以"文件可能还在"为前提。

## 9 不主张

- 不主张 `accepted learning parent`（缺 A 保持、三 seed、check②）；
- 不主张任何能力、旧能力保持、架构优越性；
- 通过只说明：**在 development fit 上，主体的字节头能把列颜色路由说成位置词，并能在反馈后纠正、保存恢复后逐字节一致**。

## 10 唯一的下一步

**请求 owner 授权执行本卡**（`authorization.asked` 已写入 manifest）。未获授权不训练；授权后按 `run` 模式一次跑完，先重建、后两阶段、再纠正/恢复，最后守卫复核。
