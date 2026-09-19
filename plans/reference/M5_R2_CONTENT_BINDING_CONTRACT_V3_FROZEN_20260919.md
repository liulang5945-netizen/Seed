# M5 / R2 内容绑定实施合同 v3（容量扩展，冻结版）

2026-09-19。状态：**已冻结**。冻结出处：v2 判停后用户在路线裁决中选择「更大容量 v3」（问答记录：选择即批准对应代价，实现门＋标定约 1 小时内；全程不碰 sealed 确认集）。依据：[v1/v2 判停备忘录](M5_R2_CONTENT_BINDING_CALIBRATION_STOP_20260919.md)（§6 v2 追记）。以 [v2 FROZEN 合同](M5_R2_CONTENT_BINDING_CONTRACT_V2_FROZEN_20260919.md) 为基线，**只改一个主设计变量：问题→内容通路的容量**，其余逐字继承。

## §E1 主设计变量：问题侧容量扩展

**假设 H-C**：v2 诊断显示 object_swap/relation/negation 的对比 margin 在 train 上从未满足（hinge≈softplus(γ)），而材料内可区分的对比（fact_flip/missing）已满足——瓶颈是"问题→对象绑定"的表示容量：单一 64 维 GRU 终态不足以从问题中读出"问的是哪个对象"并条件化答案选择。把问题侧通路加宽（问题编码 64→128、关系/内容 MLP 96→128）后，object_swap 的 train margin 应变得可满足（hinge 下降），进而标定复制/绑定读数改善。若容量加大后 **train margin 仍不可满足**，表示轴在该设计族内判否（干净负结果）；若 train margin 满足而标定仍不达门槛，则瓶颈转为泛化（数据/配方轴）。

**冻结数值**：`question_hidden_width` 64→128；`relation_hidden` 96→128；其余全部不变（char/prefix 48、evidence 48、6 槽×48、3 轮绑定、renderer 64、数据、目标函数含 pair-contrastive γ=1.0/λ=1.0、两配方、调度、微批、选择规则、停止条款、seed）。

## §E2 预注册的主检验与观察量（标定前冻结）

1. **主检验（train-only）**：每个标定评估点（500/1000/1500/2000）附带逐类 train pair-hinge 诊断（每类 32 组，仅 train 数据）：object_swap 的 hinge 相对 v2 终值（≈1.29/1.34）的下降是容量假设的直接证据；hinge→0 为 margin 可满足。
2. **选择规则不变**：复制≥0.90 且未知≥0.90 且值有限 → 三类成对宏最大；两臂皆无合格候选 → v3 判停结案（同 v1/v2 纪律，不调参再试）。
3. 正式 6 次与确认批按 v1 FROZEN §A/§B 原文执行（合格候选存在时）。

## §E3 执行顺序

1. 实现门（零训练）：v3 配置的参数形状/计数核验（问题矩阵 ×128）；v2 配置（64/96）与旧 checkpoint 继续可加载（向后兼容）；margin 诊断函数在 B/v2 checkpoint 上复现 v2 终值（仪器连续性）；v3 checkpoint 往返。
2. 标定 4 次（产物 `reports/r2_content_binding_v3/<arm>/<seed>/<config>/`）→ 按选择规则判读 + 主检验读数。
3. 结果三分支：合格候选 → 正式/确认批；train margin 满足而标定不过 → 泛化轴结案；train margin 仍不满足 → 表示轴判否结案。任一分支都更新 03 并留判停/结案备忘。

## §E4 边界

不碰 sealed；不调 γ/λ/宽度再试；不自动晋级；容量假设判否不证明"绑定不可学"，只判否该设计族在该尺度族内的该轴。
