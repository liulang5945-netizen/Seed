# 包 21.3 表示可分性诊断 · 预注册冻结规格（2026-11-07）

> 状态 [预注册已冻结·待 owner 裁定执行]。owner 2026-11-07 批准**准入** 21.3，本件只冻结判据，**不跑读数、不做任何前向、不训练**。判据的唯一机读源是 [21.3 预注册 manifest](../manifests/taiji_package21_3_prereg_20261107.json) 的 `criteria` 对象，`criteria_sha256 = 02b727f958dbad8437a73c2914bf321a81263f5c85dc08183b3bd56f11d8c512`（对 canonical JSON 的 SHA-256，自洽可重算）；本件**不复述另一套可抄的门值**，只写口径理由、已知先验、执行契约与出口语义。判据先落盘、读任何数之后不得改。
>
> 上位：[包 21 计划 §4](PLAN-PACKAGE21_READOUT_DECOMPOSITION_20261107.md)、[详细开发计划 P1](PLAN-DEVELOPMENT_DETAIL_20261107.md)、[ROADMAP-NEXT §2b](ROADMAP-NEXT_20261104.md)、账本 I22.2 / D16-10 / D16-12。准入证据：[21.1 报告](../../reports/taiji_package21_1_readout_decomposition_20261107.json)（判 U 4/4）＋[manifest](../manifests/taiji_package21_1_readout_decomposition_20261107.json)＋[P0 结项卡](../../reports/taiji_package21_1_p0_closeout_20261107.json)。

## 1. 准入与边界

- **准入成立**：21.1 冻结判据的 U 分支（≥3/4 格 `template_forced` 内容位 <0.40 且相对 `natural` 增益 <0.05）实测 4/4 格成立，`verdict=upstream_representation_is_first_wall`；P0 已把证据入库并写回三处队首。
- **本包回答的唯一问题**：即使句框正确、内容仍错，是「F 表示不可分／无权威」还是「写入时机」还是「发射读出」？三轴不相互代签（账本 I22.2）。
- **硬边界**：0 训练 / 0 优化器步 / 0 Adam / 0 新参数 / 0 写 checkpoint / 0 新图 / 0 定价 / 0 confirm；confirm 索引 240–319 继续封存、0 读取；不重开包 20、不追加补丁链。
- **不是 20.2s 的重跑**（四点实质差异）：① 输入改为 20.4c_r2 的 **4 个末步训练后 checkpoint**（20.2s 的父包是 `output/taiji_package19_compare/28031-B_stop_lock-step300.pt`，未训参考图）；② 特征改为**问句条件化键**（20.3a 已冻结的 key 定义）而非 20.2s 的 F 展开 8192 维直读；③ 增加逐询问键可分性（余弦分布／top1–top2 边距／秩）与逐轮写 vs release 写的 F 向量数；④ 对照口径按 §3 统一并增加窗口身份对照。若实现上退化为 20.2s 的原管线原特征，判 `engineering_stop`，因为那是一个已知 `inconclusive` 的重复实验。

## 2. 已知先验（不因本包改写）

| 来源 | 读数 | 对本包的含义 |
|---|---|---|
| 20.2s（[报告](../../reports/taiji_package20_2s_f_information_20261105.json)，`inconclusive`） | 主探针留一决策位 **0.6818**；标签打乱 5 seeds **mean 0.5091 / max 0.7727**；按 mean 的 margin **0.1727**；`window_identity_matches_f=true`；每个 world **distinct_f_vectors=2** | 22 行不是 22 个独立样本；窗口身份捷径已能解释主探针 |
| 20.3a（[报告](../../reports/taiji_package20_3a_addressed_store_20261105.json)，`option2_failed`） | 冻结口径 `strongest_control = max(permutation per-seed max, random same-norm, no-F state)`；top1–top2 边距中位 **0.00317**、22/22 <0.05；杠杆 **8.7e-6**；信息闸 margin **−0.1818** | 正式冻结的「最强对照」是 **max**，不是 mean |
| 20.3b（[报告](../../reports/taiji_package20_3b_f_authority_20261105.json)，`authority_weak`） | oracle F 替换仍 **0/22、0 字节变化** | F 权威是次级缺陷，不据此宣称一般定理 |

**先验推论（写进结论模板）**：若 4 个 checkpoint 上 `distinct_f_vectors` 仍 ≤2，或窗口身份对照仍等于主探针，则「表示轴关闭」是当前图／数据上的**结构性读数**，不是训练剂量不足；不得据此加剂量、换阈值或加第三臂（D16-10 / D16-12）。任何判负必须写成范围限定，不得升格为「表示不可能有权威」。

## 3. 必须在读数前钉死的口径（本件核心）

### 3.1 「最强对照」的统计量冲突 —— 采用 20.3a 已冻结口径

同一批数据、同样的门值，两份历史报告给出**相反结论**：20.2s 用标签打乱的 **mean**（margin 0.1727 → `inconclusive`），20.3a 冻结的是 **per-seed max**（margin 0.6818−0.7727 <0 → 判负）。P1 与包 21 §4 只写「最强对照」，未指定统计量。

**裁定**：取 20.3a 已冻结的 `max(permutation per-seed max, …)` 作为**主口径**。理由三条：① 它是仓库里唯一写成冻结条款的口径；② 「最强」的字面即 max；③ 标签打乱的某个 seed 已超过主探针时，用均值把它平均掉会制造假的 `inconclusive`，掩盖「闭式读出并不比乱序标签更强」。20.2s 报告的 `inconclusive` **不改写**，只作为历史敏感性锚登记；本包不复用其 mean 口径。

### 3.2 双口径敏感性（防事后挑口径）

报告必须同时给出主口径（max）与 20.2s 遗留口径（mean）的 margin 与 verdict。**只有主口径的 verdict 生效**；附报不得用于改判，也不得在看到读数后切换口径。这一条本身可被 `criteria_sha256` 与报告字段机器核验。

### 3.3 对照集合：取三份文档的并集（四条，缺一不可）

| 对照 | 来源 | 冻结统计量 |
|---|---|---|
| `window_identity` | P1、包 21 §4、20.2s `window_features` | 二值窗口 one-hot 上的同一闭式探针 |
| `label_permutation_5_seeds` | 三份文档 + 20.2s/20.3a | 5 seeds（`20261105…20261109`）的 **per-seed max**；mean 仅作敏感性 |
| `random_projection_same_dim_same_norm` | 20.2s、20.3a | 同一管线、同维同范数随机向量 |
| `no_f_other_persistent_state` | P1、20.2s、20.3a | 20.2s `other_features(row)` 管线 |

`strongest_control = max(四条冻结统计量)`。这满足 P1「三条对照缺一不可」与包 21 §4「两条最强对照」，并覆盖 20.2s／20.3a 历史口径以便直接对比；**门值一个不改**，只是把对照集合与统计量补全、收紧。

### 3.4 有效自由度闸门（沿用 20.3a 结构闸）

20.3a 已冻结 `distinct_f_vectors_global_strictly_greater_than: 2`。本包沿用：`distinct_f_vectors < 3` 时，任何「通过」结论无效，自动落 `inconclusive`；配对 bootstrap 必须以**不同 F 向量**为重采样单位（不是 22 行），并报次数与区间。这是把 P1「可能出现的问题」第 3 条落成可执行守卫，不新增门值。

### 3.5 4 个 checkpoint 的最严判定（防挑最好）

4 个 20.4c_r2 末步 checkpoint（2 seed × 2 arm）**各自独立**出一套主探针／四对照／margin／verdict／有效自由度；主结论取 **margin 最小者**，并附报 4 份原值。若 4 份结论不一致（有的过、有的不过），判 `inconclusive`，不得挑表现最好的 checkpoint 报通过。

### 3.6 与 21.4 / P3 的转段冲突（登记，不自动转训练）

包 21 §4 与 P1 替换方案表写「判负／不可判 → 转 21.4 或 P3 单项诊断」，但 **21.4 与 P3 的准入条件都不包含「表示判负」**（21.4 准入＝21.1 判 I，或 21.3 证明表示有权威但未落图）。因此本包：判负与不可判只登记具名 `verdict`、对照原值与有效自由度，**不自动进入任何训练包**；后续去向由 owner 另立合同裁定。此不一致在本件登记，不擅改上位计划的冻结文字。

## 4. 输入与执行契约

- **图与 checkpoint**：复用 `scripts/training/probe_taiji_package20_4c_r2_formal_comparison.py` 的 `build_graph(candidate, gate, arm)`（已入库 sha `112c76ae…`），`addressed_store_enabled=False`；4 个 checkpoint 路径与 sha256 见 21.1 报告 `cells[*].checkpoint_sha256`，执行前逐一重算，不符即 `engineering_stop`。
- **特征卡先行**：问句条件化键的具体构造、岭回归 λ 网格、留一折划分、解码函数、seed 列表与归一化方式，必须在**读任何读数之前**落盘为 `feature_card`（独立 JSON，含 `feature_card_sha256` 写进报告）。未先落盘、或读数后改动 ⇒ `engineering_stop`。判据允许复用 20.2s `probe()` 管线与 20.3a 的 key 定义，但组合方式必须在特征卡里冻结。
- **双列 runner sha**：`runner_sha256`（产生读数）与 `submitted_runner_sha256`（提交字节）分列；runner 未写之前两列均为 `null`，**不得预填**。执行合同要求两者与入库字节一致。
- **产物**：`reports/taiji_package21_3_representation_separability_20261107.json`（write-then-rename）、`plans/manifests/taiji_package21_3_representation_separability_20261107.json`、`…_progress.jsonl`（逐行 append＋fsync）、逐例读数（22 面 × 4 checkpoint）、四条对照原值、双口径敏感性、配对 bootstrap 区间与有效自由度数、归因结论三轴分账。
- **成功语义**：`representation_has_authority` 只产出「把闭式读出变成图上可学读出头」的**定价输入**，不是定价通过、不是训练准入、不是能力成立。

## 5. 出口（五类，具名 `verdict`）

全部按 manifest `criteria.decision_rule` 判定，本件不复述门值：`representation_has_authority` / `representation_axis_closed_for_this_graph` / `inconclusive` / `engineering_stop` / `budget_stop`。CPU 到 1,800（90%）即停并如实报，不缩数据后重跑；wall 上限按包 21 家族 CPU:wall=1:1.5 的既有比例补冻为 3,000（21.1 为 4,800/7,200），由本预注册登记，不改变 CPU 门。

## 6. 执行前硬门（全为真才准开跑）

1. 本预注册 manifest 已入库，且 runner 内 `CRITERIA_SHA256` 常量与 manifest 的 `criteria_sha256` 逐字一致；判据自 2026-11-07 冻结后未改。
2. `feature_card` 已落盘并登记 sha（§4）。
3. 4 个 checkpoint sha 与 21.1 报告一致；图身份与 `candidate_sha_v2` 未漂移。
4. 双列 runner sha 齐备且 runner 已入库；0 训练 / 0 confirm 声明落盘。
5. owner 已按本件单独裁定**执行**（本轮只冻结预注册，**准入 ≠ 执行**）。

## 7. 判据、出口与验收（索引）

冻结判据、门值与控制统计量**只以** [21.3 预注册 manifest](../manifests/taiji_package21_3_prereg_20261107.json) 的 `criteria` 为准（附 `criteria_sha256`）；出口语义见本件 §5，执行前硬门见 §6，上位判据见[包 21 计划 §4](PLAN-PACKAGE21_READOUT_DECOMPOSITION_20261107.md) 与[详细开发计划 P1](PLAN-DEVELOPMENT_DETAIL_20261107.md)。**本节只作索引，不改上述任何判据文字、门值或预算**；改动须另出预注册版本并保留原块审计（D16-38）。

## 8. 停止线

本件冻结后停在 owner 裁定节点：**是否按本预注册合同执行 21.3 读数（0 训练 / CPU ≤2,000）**。21.3 已准入、预注册已冻结，但**尚未执行、尚未完成**；21.2 与 21.4 保持冻结，U1 不进队列。范围口径：准入批准 ≠ 执行批准 ≠ 里程碑关闭。
