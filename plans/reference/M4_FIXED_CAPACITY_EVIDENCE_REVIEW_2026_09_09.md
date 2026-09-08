# M4 固定容量连续成长证据链复盘（R0～R12）

> 归档日期：2026-09-09；设计复审：2026-09-09。范围：M4.R0～M4.R12 十三轮实验的原始结果及复审后的有效解释边界。本文不含执行顺序；执行状态以 [03_CURRENT_EXECUTION.md](../active/roadmap/03_CURRENT_EXECUTION.md) 为唯一来源，替代设计见 [Taiji 继承式成长架构 v2](../active/architecture/TAIJI_CONTINUAL_DEVELOPMENT_V2.md)。所有数值均来自 `reports/` 下的版本化 JSON 报告，原始文件未改动。

> **复审更正：** R0～R12 可靠地关闭了若干 F1 byte-prediction 候选，但没有证明 Taiji 的结构成长无效，也没有穷尽持续学习规则。原归档中的“更新规则轴已穷尽”“分布切换主导遗忘”“cycle3 是固有难度”超出了实验设计可支持的范围；下文 §4、§8 已按代码、report schema 和对照臂重新限定。决策链图保留当时的推进逻辑，不再作为当前架构结论。

## 1. 主线问题

Taiji 的继承式成长要求：在同一继承 checkpoint 上连续吸收新记录（新任务周期），且旧能力不退化。M4 主线检验的单一问题是——**固定容量（不扩参数）的前提下，连续多周期 continuation 的失败到底由什么主导**：更新规则、巩固机制、数据源，还是课程分布本身。

实验载体始终是同一个三 seed 谱系（seed11 `taiji-m2s-*`、seed29/47 `taiji-m2u-*` identity-generation child）、同一 record-disjoint C→C2→C3 三周期课程（partition seeds 43/44/45）、同一 read-only scorer、同一预注册 retention Gate（C″ gain 全 seed 为正 且 C cycle2/cycle3 退化 0/3）。

## 2. 决策链概览

```
R0  固定容量三周期 cascade ──技术链闭合，能力不晋级（C″ -0.011，cycle2/3 退化 3/3）
 ├─ R1 巩固规则 consolidation=0.5 ──远期收益无，cycle2/3 仍 3/3 退化 → 撤回
 │   └─ R2 增量容量槽（零初始化）──旧能力不退化但新收益 2/3 负 → 不支持扩容
 ├─ R3 owner credit 归因（readout/context/joint）──joint 有信号但 pilot 2/3 正
 │   └─ R4 参数差分审计──owner 写入幅度大、课程分布无断裂 → 转尺度变量
 │       └─ R5 update_scale=0.5 ──C″ +0.116 3/3 正，cycle3 首次成为唯一失败点
 │           └─ R6 周期归因──cycle3 失败非 owner outlier → 大预算 formal
 │               └─ R7 formal（65536/16384）──cycle3 退化 2/3 → 否决
 │                   └─ R8 失败归因──非 owner 幅度 → R9 课程审计──非课程 outlier
 │                       └─ R10 consolidation × scale 0.5 单规则 audit ──cycle3 3/3 恶化
 │                           └─ R11 只读归因──保守项机制证实但“参数稳定性换不来边界泛化”
 └─ R12 数据源替换（UltraData）──A 指标比 simple_zh continuation 臂差 +1.62 BPB、语料内 cycle3 +0.249（均 3/3）
     → abrupt replacement 风险很高，数据密度假设被分布变化混淆，替换源否决
```

## 3. 各轮结论表

| 轮次 | 问题 | 关键数值 | 判定 | 报告 |
|---|---|---|---|---|
| R0 | 固定容量 readout cascade 是否可重复成长 | C″ gain `-0.01078`（1/3 正）；cycle2/3 退化 `3/3` | 技术成功、能力失败 | `taiji_m4r0_cascade3_aggregate_20260908.json` |
| R1 | 巩固规则（`consolidation_strength=0.5`）能否修复保持 | C″ `+0.0565` 3/3 正；cycle2 `+0.0132` 3/3、cycle3 `+0.0051` 3/3 退化 | 否决 | `taiji_m4r1_consolidation_aggregate_20260908.json` |
| R2 | 零读出槽 + 固定等权混合能否解除退化 | A/B 旧 C 指标未退化但 C″ `-0.0028/-0.0197`；`capacity_pressure_supported=false` | 此容量实现无收益；不能否定表征扩宽 | `taiji_m4r2_capacity_diagnosis_canary_seed11_20260908.json` |
| R3 | 退化来自 readout、context 还是 joint 更新 | joint smoke `+0.0509` 3/3 正；pilot `+0.0171` 2/3 正且 cycle2 2/3 | owner 归因链可信，无 formalization | `taiji_m4r3_update_interference_pilot_aggregate_20260908.json` |
| R4 | 课程分布与更新幅度哪个是优先变量 | 相邻 phase JS≈0.01；readout relative L2≈0.23 | 优先变量是 update scale | `taiji_m4r4_course_shift_audit_20260908.json` |
| R5 | 半速更新（scale 0.5）能否修复 | pilot C″ `+0.1157` 3/3 正；cycle2 1/3 退化 | 改善但不过严格 Gate | `taiji_m4r5_update_scale_pilot_aggregate_20260908.json` |
| R6 | cycle 级退化定位 | scale 0.5 仅 seed47 cycle2 `+0.0037`；非 owner outlier | 允许一次 formal | `taiji_m4r6_cycle_retention_attribution_20260908.json` |
| R7 | 大预算 formal | C″ `+0.1528` 3/3 正；cycle3 退化 2/3（seed11/47） | 否决 | `taiji_m4r7_formal_aggregate_20260908.json` |
| R8 | cycle3 失败=更新幅度？ | 失败组 relative L2 ≤ 通过组 1.10× | 非 owner magnitude outlier | `taiji_m4r8_cycle3_failure_attribution_20260908.json` |
| R9 | C3 的 byte 边际分布是否异常 | C3/C2 最大 boundary JS 1.19×，无共享 byte-level 异常 | 未见 byte-marginal outlier；未排除序列/语义差异 | `taiji_m4r9_foundation_course_audit_20260908.json` |
| R10 | readout-only 路径上的 consolidation × scale 0.5 | candidate C cycle3 2/3 退化且比 baseline 变差 3/3 | 当前静态 preservation 规则否决 | `taiji_m4r10_rule_audit_formal_aggregate_20260908.json` |
| R11 | R10 preservation 为何加重 C cycle3 | candidate 位移更小 3/3（0.143~0.151 vs 0.162~0.173）但 delta 变差 3/3 | 在该 readout-only 路径上，限制位移未改善 Gate | `taiji_m4r11_cycle3_interaction_attribution_20260909.json` |
| R12 | abrupt UltraData replacement 能否改善 | 语料内 C cycle3 `+0.2489` 3/3；A 指标比 simple_zh 臂差 `+1.62` | 替换源否决；数据密度假设被分布变化混淆 | `taiji_m4r12_data_source_aggregate_20260909.json` |

注：R10 报告名为 `taiji_m4r10_rule_audit_formal_aggregate_20260908.json`（同目录含三 seed formal 与 smoke）。

## 4. 复审后的因果结论与边界

**C1（技术底座闭合，保留）**：owner 隔离、原子 checkpoint、fresh-process 恢复、record-disjoint 数据链、read-only scoring 在全部十三轮均通过各自技术 Gate。报告可以用于分析其实际训练路径；这不等于不同轮次拥有相同 owner 图或相同能力 Gate。

**C2（窄规则结论）**：已测试的全局 update scale 与“在当前输入上逼近 protected logits”的固定 preservation 规则不满足原 Gate；后者在 R10 的 readout-only 路径上确实减少参数位移但没有改善 C cycle3。实验没有实现 fast/slow synaptic state、importance、真实经历 replay、learned router 或成熟结构扩宽，因此不能称“更新规则轴已穷尽”。

**C3（abrupt replacement 结论）**：R12 的 UltraData 臂相对 simple_zh continuation 臂，C cycle3 差约 `+0.254 BPB`，A 绝对 BPB 的臂间差为 `+1.62`。报告没有记录同一臂训练前的 A parent baseline，因此 `+1.62` 不能直接命名为“相对父代遗忘”；它证明的是突然全量换语料在当前 kernel 上风险远高于同源 continuation。数据密度、文体、领域和序列结构同时变化，不能由该实验分离。

**C4（Gate/课程结论）**：R7 的 scale 0.5 在 C3 新能力上 `+0.1528 BPB`（3/3），C cycle3 平均 delta 为 `−0.0050 BPB`，但 seed11/47 分别有 `+0.0021/+0.0095`，因 exact-zero all-seed Gate 被否决。该 Gate 的否决判定必须保留，但在没有 course-order 复现、非劣界和完整 A/B/C 累计矩阵时，不能外推为“第三周期是固有难度”。R9 只测 byte marginal，也不能排除更高阶序列差异。

**C5（成长实现错位）**：R2 是新旧 readout 概率固定 50/50 的实验 artifact；R7 更新 protected predictive context+readout，R10/R12 则因 Workbench active boundary 冻结 context、只更新 active readout；`AdaptiveNeuronNetwork` 也未被 M4 课程的普通 forward/credit 路径消费。因此 R0～R12 没有测试“进入主路径、零影响出生、学习路由、成熟后准入”的结构成长，不能据此关闭 CR-4/A8。

## 5. 开放假设（已登记、未预注册）

1. **渐进混合课程**：UltraData 与 simple_zh 按比例过渡以分开数据密度与 abrupt shift；转换语料与 preflight 资产保留，但须先使用 v2 scorecard，不能沿用 R12 Gate 直接长跑。
2. **继承式结构增长**：旧权重迁移为 slow state，新结构以 zero-gated residual shadow 出生并学习路由；R2 的固定等权读出混合不是这个假设的反证。
3. **快适应—慢巩固**：fast delta、eligibility、importance、真实 episodic replay 到 slow weights 的机制尚未测试；当前 logits preservation 只作为否决对照。

## 6. 成本事实（CPU，Windows，12 线程）

- 三周期 formal（1 MiB×3）：训练约 3.6k~5.2k 秒/seed，评分约 0.8k~2.0k 秒，峰值工作集约 370 MB，checkpoint 约 33.3 MB
- R7 规格三周期（64 KiB×3）：训练约 10~12 分钟/seed（两臂约 25 分钟）
- 吞吐约 620~860 body bytes/s；这是当前 CPU 运行成本基线，不外推到大模型训练

## 7. 资产清单

- 脚本：`scripts/training/eval_taiji_m4r1..r12*.py`、`aggregate_taiji_m4r1/r7/r10/r12*.py`、`scripts/data_prep/convert_ultradata_sft_nothink.py`
- 共享 harness：`eval_taiji_m2r1_phase_c_canary.py`（cascade 臂支持 `consolidation_strength` + 向后兼容 `predictive_update_scale`）
- 数据：`data/ultradata/`（T0 本地化，gitignored）、转换语料 `data/ultradata/derived/`
- 报告：`reports/taiji_m4r*`（版本化 JSON，SHA-256 互链）
- 撤回/冻结候选：consolidation（R1/R10）、增量槽（R2）、gated temporal（M2.R2）、UltraData 替换（R12）——全部保留为可回滚 artifact，默认关闭

## 8. 对下一步的修订含义

M4 v1 的正确答案是：**当前 F1 continuation harness 能可靠保存和测量，但它的学习对象、扩容方式、owner 一致性、课程和 Gate 都不足以代表 Taiji 的继承式成长。** 因此：

1. 不再继续 R13 式标量调参，也不直接在 R12 后启动混合语料长跑；先修量尺语义与累计能力矩阵。
2. R0～R12 的技术 Gate、数据 lineage、checkpoint 和 scorer 资产复用；exact-zero Gate 作为历史严格观测保留，但由预校准非劣界、最坏域上限和多 course-order 共同决定 v2 晋级。
3. M4 v2 先做 checkpoint-compatible fast/slow 状态，再做真实 replay，最后才允许 zero-gated structural shadow；结构必须进入主 forward/credit 路径。
4. 当前唯一执行入口见 [03_CURRENT_EXECUTION.md](../active/roadmap/03_CURRENT_EXECUTION.md)，完整替代设计见 [Taiji 继承式成长架构 v2](../active/architecture/TAIJI_CONTINUAL_DEVELOPMENT_V2.md)。
