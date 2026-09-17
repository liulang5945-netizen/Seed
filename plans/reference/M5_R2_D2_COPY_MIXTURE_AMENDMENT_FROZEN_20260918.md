# R2-D2 修订：H-A2 copy 混合证据读出（最小 H-A1 learnability 失败后的图修订，冻结版）

2026-09-18。状态：**已冻结**（本文提交即冻结）。本修订是 [R2-D2 预注册](M5_R2_D2_PER_POSITION_EVIDENCE_PREREGISTRATION_FROZEN_20260918.md) §6 预承诺路径的一次图修订，沿用 H3.8 v1→v2 先例：learnability 阶段定位到具体机制缺陷后，在同假设族内修订图，不扩训练预算、不读 final、不移动 dev 判据。

## §1 最小 H-A1 的结项（learnability 阶段，不进入 matched dev）

1. checkpoint 前置检查通过（零步保存/新进程 logits 逐位一致/续训一致/v3 恢复）。
2. 冻结 lr=0.05 probe：loss epoch 5 到 2.09 后震荡（4.75/5.79/6.29/5.00/3.52），与 H3.8 learnability 报告记录的「workspace 图 0.05 不稳定」同型；§6 禁止在门上调 lr/加 epoch。
3. 稳定 lr=0.01 三臂诊断（[报告](../../reports/r2_d2_lr_stability_diagnostic_20260918.json)）：A/C1/C0 均自 epoch 1 平台（CE 0.60–0.68、byte acc 75–80%、train M1≈0.17）；C0/C1 塌缩为边际常量（「绿」），A 仅在个别形状有形状条件命中。
4. 寻址诊断（[报告](../../reports/r2_d2_addressing_diagnostic_20260918.json)）：训练后 softmax 权重熵=0、max=1，**全部落在题干位置**（fact 28/29、negation 7/19；材料从句始于 byte 30+），材料区质量数值为 0（~1e-26）。
5. **判定**：图接线与梯度正确（实现门全绿、梯度有限非零、读取可被 lesion 关闭），但「纯生成解码 + 软内容寻址 + 仅答案 CE」在冻结预算下没有动力把寻址特化到材料——读出塌缩为题干条件常量。这是**机制设计缺陷在训练集上的直接观察**，不是 dev 能力结论；最小 H-A1 不进入 matched dev，假设本身不在此判死刑（信息路径尚未获得公平的提取通道）。

## §2 修订假设 H-A2（冻结陈述）

> 在逐位置证据条目上为回答 byte 增加一个**仅在可见 prefix 位置归一化的 copy 分布**并与原生 vocab 分布混合（`p=π_gen·p_vocab+(1−π_gen)·p_copy`，π 由模型逐 byte 产生），将使材料选择获得直接 CE 梯度：copy-supported 形状（fact/negation/same_opening_fact，train 144/174）在训练集可拟合（M1≥0.90），且在 dev 反事实对上答案随单一事实正确翻转（M4 fact_flip≥2/6），超过等机制 broadcast 对照与固定策略天花板；非 copy 可解形状（unknown=abstain、combination=关系结论）必须仍由生成分支承担，其成败是 B/C 路由证据，不计入 copy 的成功。

依据：[VISION §15.3](VISION_FUTURE_TECHNOLOGY.md)「先不启用 copy；单纯寻址不改善时，copy 作为独立变量」——最小 H-A1 正是「单纯寻址」失败，copy 是该节预先分阶段的下一变量，不是临时加损失。

## §3 graph v4 精确规格（增量；v2/v3 图与 checkpoint 读路径全部保留）

1. 配置新增 `copy_mixture: bool = False`（默认关 ⇒ v3 行为逐位不变）；仅允许在 workspace 臂且 `evidence_source ∈ {per_position, broadcast_final}` 开启（v2 4 槽与 prefix byte 位置不对齐，不允许组合）。checkpoint 版本 3→**4**，载荷含 `copy_mixture`；v2/v3 旧 payload 继续可读，缺省即关。
2. `WorkspaceState` 增 `entry_bytes: tuple[int,...] | None`：v3 两源的 L 个条目与 prefix L byte 一一对应，携带该位置 byte；v2 槽为 None。
3. 每生成/teacher-forced 步 j（renderer 状态 r_j、寻址权重 w_j=softmax(K q_j/√sw)）：
   - `p_vocab = softmax([r_j; read_j]·decoder + bias)`（既有 logits，257 维）；
   - `p_copy[b] = Σ_i w_j[i]·1[entry_bytes[i]=b]`，b∈0..255，仅在可见 prefix 位置上归一化（不对答案标签、不用答案派生位置；材料/题干不预切分，位置选择由模型自己学）；
   - `π_j = sigmoid(r_j·W_gate + b_gate)`；`p = π_j·p_vocab + (1−π_j)·pad_257(p_copy)`；boundary 256 只能由 p_vocab 供给（p_copy[256]=0）；
   - 训练目标 = 真实下一 byte（含 boundary）对 `p` 的 NLL；梯度经 p_copy→w→K/证据投影回 prefix 扫描。
4. 新增参数仅 `copy_gate_weight (rw,)=64` 与 `copy_gate_bias (1,)=1`（初值：权重小尺度 1/√rw、bias=0 ⇒ π≈0.5，不预置偏向）；**A2 参数 82,658**，C1-copy 同清单同计数。
5. 贪心生成走同一混合：每步取混合概率 argmax；推理不更新参数。
6. lesion 语义扩展：`zero_read=True` ⇒ read 置零**且** (1−π)·p_copy 项移除（p=π·p_vocab，π 由 renderer 给）；misbind（V 行移位）保留，另新增 **copy-key 错位**评测 lesion：复制权重对**错行位置**寻址（K 与 entry_bytes 整体错位 L//2），预期直接打掉 copy-supported 答案；两 lesion 确定性、仅评测。

## §4 臂（同 D1 digest、3 seeds、lr 修正为 0.01、30 epochs、从零构建）

| 臂 | 图 | 参数 | 角色 |
|---|---|---:|---|
| **A2** | per_position + copy | 82,658 | treatment |
| **C1-copy** | broadcast_final + copy（L 行全是 h0 投影；p_copy 退化为 prefix byte 频率先验） | 82,658 | 主对照：等机制等参数，无逐位置信息 |
| **A0** | per_position 无 copy（=最小 H-A1，同 lr） | 82,593 | copy 消融锚点；最小 H-A1 诊断已给 seed917 轨迹 |

无上下文模型基线、M1–M5 口径、表面天花板（M1 0.1735/M3 0.0581/M4 0.0）沿用原合同 §4，三臂同做。

**lr 冻结更正（0.05→0.01）**：原合同 §3 的 0.05 写作「trainer 现默认值」，与 H3.8 learnability 已冻结结论（workspace 图 0.05 不稳定、0.01 两臂稳定并收敛）相冲突；本次以该**先于本修订存在的**冻结证据为依据更正，0.05 失败运行原样保留在报告中，不删除不覆写。

## §5 门（修订；看 dev 前固定）

1. **实现门（零训练）**：v3 七门保留；新增 copy 门——p_copy 对可见 byte 求和为 1、不含 256；NLL 对 copy_gate/证据投影的数值差分；zero/copy-misbind lesion 接线与确定性；teacher-forced 与逐步混合前向一致；copy 关闭时 v3 逐位不变；v4/v3/v2 checkpoint 三版本恢复矩阵。
2. **probe（1 运行：A2，seed 20260917，30 epochs，lr 0.01，wall 20 min）**：checkpoint 前置检查同前；通过线 = copy-supported 形状（fact+negation+same_opening_fact，144 题）train M1 **≥0.90** 且 loss 有限下降；整体 M1/M3/M4 与非 copy 形状（unknown/combo）仅描述。
3. **matched dev（3 seeds×3 臂，30 epochs，wall 30 min/运行）**：G1 M3≥0.26 且 A2−C1 δ≥0.20；G2 M4≥0.50 且 δ≥0.25，**fact_flip≥2/6**（combo_flip 单列，作为 C 路由证据）；G3 copy-misbind 使 M4 绝对降 ≥0.50；G4 无上下文 ΔM3≥0.30 且无上下文 M1≤0.50；G5 seed 一致（≥2/3、最差 seed 非负）；G6 boundary/恢复。全部满足才支持 H-A2。

## §6 停止线与路由

1. probe 不过（含 copy-supported M1<0.90）：这是同假设族第二次 learnability 失败 ⇒ **不再自主修订图**，冻结全部诊断、升级人工复审与用户裁决（候选：renderer/表达容量、B 绑定、课程更新方式；带证据讨论，不自动选）。
2. matched 任一 G 门不过：A2 未获支持，按形态路由——copy 形状过而 combo 全垮→C；copy 选择随模板不随事实翻转→绑定/查询（B）；lesion 不掉→捷径否决。
3. 不读 final；不扩 epoch/数据/宽度；不把 copy 成功外推到 L2 或真实语料。

## §7 产物

`reports/r2_d2_copy_implementation_gates_20260918.json`、`reports/r2_d2_copy_learnability_probe_20260918.json`、`reports/r2_d2_matched_dev_20260918.json`（format 升级含 copy 字段与三臂）；checkpoint 仍写隔离目录、digest 入报告；`growth_admitted=false`、`can_promote=false`。

## §8 不得主张

copy 是 Taiji-owned 的可见材料概率通路（无外部模型、无答案位置泄漏、不复制材料中不存在的结论）。即便全门过，也只支持「D1 合成 dev 上逐位置证据 + copy 混合形成经反事实验证的内容提取」；combo/unknown 的生成能力、真实语料问答、L2 均不随之成立。
