# P3.9 A/B 双图比较预注册（2026-11-14）

> 22.5.P3.9.1–4。判据、门值、控制统计量与出口**只以** [机读 manifest](../manifests/taiji_u2_p3_9_prereg_20261114.json) 的 `criteria` 为准（`criteria_sha256` 可重算）；本件只作索引与边界说明，不复制第二套数字。守卫见 [verify_taiji_p3_9_prereg_20261114.py](../../scripts/training/verify_taiji_p3_9_prereg_20261114.py)。

## 0. 准入链

[执行包 22 §P3.9](PLAN-TAIJI-CLOSED_LOOP_DEVELOPMENT_20261110.md) 要求「先按两图定价，再冻结」，原预算草案未经冻结不得执行或自动翻倍；[路径备案 D-20261114-01](DECISION-TAIJI_NEURAL_CORE_PATH_20261114.md) §2/§4 定义 A 为兼容嵌入、B 为单位原生核心，并禁止把未做的行为比较补写成事实；[生命周期规格](SPEC-TAIJI_U2_NEURAL_UNIT_LIFECYCLE_20261113.md) §3/§8 给出同步递推与恢复合同。

[P3.8.4 全局收束卡](CLOSEOUT-TAIJI_U2_P3_8_4_GLOBAL_20261114.md) 已按负结果＋预算停止有限出口结案，`native_parent_qualified=false`。本预注册**不转移 P3.8 的任何额度**，两图均为新的研究身份，不冒称同一主体的连续成长。

## 1. 定价（准入依据，非结果）

[实价报告](../../reports/taiji_u2_p3_9_1_price_20261114.json) 已在冻结前落盘：两路线 × 两种精度各测初始化／前向反向／Adam 步／单次评估，`verdict=price_measured_admission_possible`。保守取各操作最大值估计配对更新与评估单价，按 manifest 的预算帽算出可支付剂量。

**必须随定价一起引用的限制**：本机 `process_time` 分辨率约 0.0156 s，多个单次操作读数为 0.0 是**低于一个计时刻度**，不是零成本；所有估计只使用保守最大值，行为窗另按真实总 CPU 分账。定价用了每图 3 次 Adam 调用作单价测量，0 项目任务数据读取、0 checkpoint 写、0 产生父。

## 2. 四张卡的交付与有限出口

| 卡 | 交付 | 出口 |
|---|---|---|
| P3.9.1 两路线正式图 | [共享算子与 A/B factory](../../taiji/u2_neural_units.py)、[契约测试](../../tests/taiji_native/test_u2_neutral_units_graph.py)、[图验收报告](../../reports/taiji_u2_p3_9_1_graph_20261114.json) | 全部检查通过签 `reference_graphs_verified`；缺项签 `reference_graph_incomplete`，不得进入 P3.9.2 |
| P3.9.2 更新、成长与恢复 | 两图通用完整 Capsule（参数／状态／单位边 registry／moment／RNG／双时钟）、独立进程恢复、两代出生与休眠／唤醒／回滚 | `restore_and_lifecycle_verified` 或 `engineering_failure` |
| P3.9.3 核心组织有界比较 | 一次预注册行为窗；同父 spec／同 seed／同样本暴露／同更新数；配对 bootstrap、控制臂、CPU 分账 | 按 manifest `decision_rule` 具名结案；两臂无动态范围签 `budget_cannot_express`，**不是路线被否决，也不是加剂量理由** |
| P3.9.4 全局结案与交接 | 出生／信用／行为／恢复／成本五本账、能力账与六轴卡登记 | `p3_9_closed_limited_exit`；收束 ≠ 里程碑关闭 ≠ 范围排除批准 |

## 3. 口径边界（不得代签）

- **A 兼容范围**：声明精确映射的是「c=0／b=0／E=∅ 端点的单位递推子图」与旧 R/P 形式的逐值等价，按 manifest 的双精度容差验收；不覆盖 U2 全主体 `native` 全图，也不签 U2 旧权重继承。
- **B 直接单位作用**：读出必须走门控后的单位活动、不经 H；H 仅作 scratch 且切断后读数逐位相同。「B 不是旧 H 后新增头」由结构断言与 H／单位活动干预共同钉住，不以参数量或生物名称代签。
- **逐参数信用**：每个声明可学习参数都要有梯度，首刻结构性零梯度**如实列出**，禁止用总范数非零代替负责单位信用。
- **行为窗**：固定合成合法输入序列，标签来自合法输入本身；不是自然中文任务、不是项目任务数据、不读取封存面、0 能力声称。无已验旧能力，**保持不签**；初始化／迁移／蒸馏费用没有可比较的已训练 A，本窗只比较同初始化的核心组织，不把差异独立归因于单位思想。
- **统计**：读结果前冻结 Δ 下限、seed、样本暴露、CPU 配平带与 bootstrap 轮数；结果按配对重采样报区间，不挑表现最好的 checkpoint。
- **一次工程包＋一次行为窗后必须全局收束**，不靠反复加剂量延长单位图设计。

## 4. 判据、出口与验收（索引）

冻结件为上述 manifest（含 `criteria_sha256`）；`execution_runner_sha256` 与 `execution_report_sha256` **恒保持 `null`、不预填、不回写**（冻结件只追加），实际入库字节由收束卡与独立审计按文件 sha 记录。守卫重算 sha、核对 SPEC 无第二套门值、核对三处队首逐字一致，并做红绿双向演练；回执 `reports/taiji_u2_p3_9_prereg_guard_20261114.json`。

父身份、预算帽、封存面权利与纪律计数以 manifest 为准；本卡 0 check／final／confirm 读取、0 能力声称、0 新接受父。
