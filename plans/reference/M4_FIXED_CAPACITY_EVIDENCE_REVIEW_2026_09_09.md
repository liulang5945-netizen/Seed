# M4 固定容量连续成长证据链复盘（R0～R12）

> 归档日期：2026-09-09。范围：M4.R0～M4.R12 十三轮实验的全部能力结论、因果判定与开放假设。本文是证据解释与决策依据，不含执行顺序；执行状态以 [03_CURRENT_EXECUTION.md](../active/roadmap/03_CURRENT_EXECUTION.md) §4 为唯一来源。所有数值均来自 `reports/` 下的版本化 JSON 报告，原始文件未改动。

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
 └─ R12 数据源替换（UltraData）──A 跨语料遗忘 +1.62 BPB、语料内 cycle3 +0.249（均 3/3）
     → 分布切换是数量级最大的遗忘源，替换源否决
```

## 3. 各轮结论表

| 轮次 | 问题 | 关键数值 | 判定 | 报告 |
|---|---|---|---|---|
| R0 | 固定容量 readout cascade 是否可重复成长 | C″ gain `-0.01078`（1/3 正）；cycle2/3 退化 `3/3` | 技术成功、能力失败 | `taiji_m4r0_cascade3_aggregate_20260908.json` |
| R1 | 巩固规则（`consolidation_strength=0.5`）能否修复保持 | C″ `+0.0565` 3/3 正；cycle2 `+0.0132` 3/3、cycle3 `+0.0051` 3/3 退化 | 否决 | `taiji_m4r1_consolidation_aggregate_20260908.json` |
| R2 | 旧能力退化能否被增量容量解除 | A/B 旧能力不退化但 C″ `-0.0028/-0.0197`；`capacity_pressure_supported=false` | 不支持扩容 | `taiji_m4r2_capacity_diagnosis_canary_seed11_20260908.json` |
| R3 | 退化来自 readout、context 还是 joint 更新 | joint smoke `+0.0509` 3/3 正；pilot `+0.0171` 2/3 正且 cycle2 2/3 | owner 归因链可信，无 formalization | `taiji_m4r3_update_interference_pilot_aggregate_20260908.json` |
| R4 | 课程分布与更新幅度哪个是优先变量 | 相邻 phase JS≈0.01；readout relative L2≈0.23 | 优先变量是 update scale | `taiji_m4r4_course_shift_audit_20260908.json` |
| R5 | 半速更新（scale 0.5）能否修复 | pilot C″ `+0.1157` 3/3 正；cycle2 1/3 退化 | 改善但不过严格 Gate | `taiji_m4r5_update_scale_pilot_aggregate_20260908.json` |
| R6 | cycle 级退化定位 | scale 0.5 仅 seed47 cycle2 `+0.0037`；非 owner outlier | 允许一次 formal | `taiji_m4r6_cycle_retention_attribution_20260908.json` |
| R7 | 大预算 formal | C″ `+0.1528` 3/3 正；cycle3 退化 2/3（seed11/47） | 否决 | `taiji_m4r7_formal_aggregate_20260908.json` |
| R8 | cycle3 失败=更新幅度？ | 失败组 relative L2 ≤ 通过组 1.10× | 非 owner magnitude outlier | `taiji_m4r8_cycle3_failure_attribution_20260908.json` |
| R9 | C3 课程边界异常？ | C3/C2 最大 boundary JS 1.19×，无共享异常分布 | 非课程 outlier | `taiji_m4r9_foundation_course_audit_20260908.json` |
| R10 | consolidation × scale 0.5 单规则 audit | candidate cycle3 3/3 恶化（seed11 由过转不过） | 否决，固定规则撤回 | `taiji_m4r12_data_source_aggregate_20260909.json` 同目录系列 |
| R11 | consolidation 为何加重 cycle3 | candidate cycle3 位移更小 3/3（0.143~0.151 vs 0.162~0.173）但 delta 变差 3/3 | **参数稳定性换不来边界泛化**；巩固方向冻结 | `taiji_m4r11_cycle3_interaction_attribution_20260909.json` |
| R12 | 数据密度（换语料）能否改善 | 语料内 cycle3 `+0.2489` 3/3；A 跨语料遗忘 `+1.62` 3/3 | 替换源否决；**分布切换主导** | `taiji_m4r12_data_source_aggregate_20260909.json` |

注：R10 报告名为 `taiji_m4r10_rule_audit_formal_aggregate_20260908.json`（同目录含三 seed formal 与 smoke）。

## 4. 已确认的因果结论

**C1（技术底座闭合）**：owner 隔离、原子 checkpoint、fresh-process 恢复、record-disjoint 数据链、read-only scoring 在全部十三轮均通过 Gate。M4 的失败全部是能力失败，没有任何一轮是测量或保存失败。

**C2（更新规则轴已穷尽）**：巩固强度（R1/R10）、容量槽（R2）、owner 组合（R3）、更新尺度（R5/R7）逐一否决。R11 给出机制级解释：保守项把 active readout 拉向 protected（位移更小、距离更近，3/3 可观测），但 cycle3 边界处 holdout 需要**参数移动去适应分布变化**——限制移动只会更差。门控保守项候选即使完美实现，上限也只是 baseline（仍 1/3 退化），因此该方向整体冻结，候选闸门 2 关闭。

**C3（分布主导，数量级证据）**：R12 的单变量语料切换显示——语料内 cycle3 退化 `+0.249 BPB`（simple_zh 参考臂 `−0.005`）、A 跨语料灾难遗忘 `+1.62 BPB`，比 R5～R10 全部更新规则效应（±0.05 内）大 **5～50 倍**。遗忘的主导变量是训练分布与旧知识的分布差异，不是任何已实现的更新/巩固机制。

**C4（课程结构属性）**：R9 证明 C3 phase 本身无分布异常、R8/R11 证明 cycle3 失败对更新规则不敏感、R12 证明分布切换使遗忘放大——三条证据合起来：**在同源连续课程上，第三周期的旧能力保持是当前架构+课程的固有难度属性**，不是可以通过单一规则修复的缺陷。

## 5. 开放假设（已登记、未预注册）

1. **混合语料课程**：UltraData 与 simple_zh 按比例混合以降低分布冲击——R12 的 +1.62 BPB 跨语料遗忘只否定“替换”，未检验“渐进混合”。转换语料与 preflight Gate 已保留（`data/ultradata/derived/ultradata_sft_nothink_simplezh.jsonl`，7/7）。
2. **继承式增长（容量扩宽）**：R2 否决的是“零初始化 readout 槽修复保持”，不是“有证据的容量压力下扩宽表征”；若未来 cycle3 退化被证明伴随真实容量压力（而非分布适应），该线按 §8 日程重开。
3. **架构外挂（attention/递归跨时间信用）**：候选闸门 1 保持关闭——前提 (a) 巩固方向完成（已满足）但 (b) 归因指向时间信用（当前归因指向分布适应，不满足）。

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

## 8. 对下一步的含义

M4 主线问题已有答案：**固定容量下的连续成长瓶颈是分布适应，不是规则缺失**。这意味着：

1. 任何“先修规则再成长”的路线已无剩余单变量；继续在固定容量 + 同构课程上做规则实验的期望收益为零。
2. 若要继续能力成长研究，剩余可检验假设集中在：混合/课程式数据引入（开放假设 1）、有容量压力证据的继承式增长（开放假设 2）、或转向 M5 的知识内化与真实任务（能力验证场景不同，不再以三周期 retention Gate 为唯一量尺）。
3. 三 seed cascade + 预注册 retention Gate 的测量资产完全可复用，是后续任何路线的公共量尺。
