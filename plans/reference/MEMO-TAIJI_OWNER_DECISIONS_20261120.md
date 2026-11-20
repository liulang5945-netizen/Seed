# 待裁事项汇总（2026-11-20）：四项决策与执行者建议

> 文档状态：[决策输入/只读汇总] — 2026-11-20。**零训练、零 check/final。** 依据：包 22 收口账与各卡回执；九轴当前值见 §0。

## §0 九轴当前值（同一父 seed `20261112` 谱系）

| 轴 | 当前值 |
|---|---|
| `engineering` | verified |
| `task` | verified（`fit_face_only`；正式 check② 未结） |
| `learning` | verified（路由技能）＋**任务通路级新能力有界 verified（D-78/79）**；内容积累 uncovered |
| `retention` | verified（严格 D 口径 breach 已披露） |
| `autonomous_choice` | **negative**（22.6.2 `policy_gain_absent`） |
| `mature_growth` | 结构/机制 verified＋**通路级新增/回收有界 verified（D-79/81）**；收益 `growth_benefit_not_established_in_package_22`（**单元级**，不得与通路级混写） |
| `recursive_development` | **uncovered**（两次相依纠正多 seed 不成立） |
| `physical_transfer` | **verified（有界：22.4.2b 推/位置桶动作片段）**；碰撞/遮挡与真机 uncovered（D-57/58/59） |
| `lifetime` | **verified（有界：≥3 会话跨进程，参数逐位继承＋身份不变＋准确率不降反升，D-63）**／100 会话、24h 级 uncovered |

附加证据：`uncertainty_expression` verified（有界：joint24 ECE 0.028→0.021，D-62）。

## §1 决策点 A：正式资格路径（A 案 vs B 案）

- **A 案**：`fit_face_only` 豁免，0 CPU；当前工作路径已按此记录（D-20261119-21）。
- **B 案**：1 次 check 曝光，192 seeds，包价 430、帽 540 CPU；需先冻结 D 口径（严格 D 会 breach，已披露）。
- **建议**：若目标要求「正式资格」，选 B 并先冻 D 口径（同条件参考）；若接受 fit 面证据，维持 A。**执行者默认 A 不动，等 owner 裁定。**

## §2 决策点 B：22.8 与 23–30 准入

- 22.8 需 22.7 接受父＋`>=3` session×32 轮 J 轨迹；23–30 为长期扩展与交付确认。
- 当前九轴：`autonomous_choice` negative、`recursive_development` uncovered、长跨度 lifetime uncovered、任务轴仅 fit_face_only ⇒ 不满足整包接受条件。
- **建议**：不自动解锁；若 owner 要推进，先立 22.8 单卡（`>=3` session 连续存活），或先解决 autonomous_choice/recursive 再谈包接受。

## §3 决策点 C：真机/多动力学族（27 阶段）是否投入

- 22.4.2b 已签**有界**仿真动作片段（N=3 推、位置桶、单主体）。
- 真机需 27 阶段（三动力学族：推阻/碰撞/遮挡声响；模拟/回放/真机分签），涉及外部依赖与预算。
- **建议**：先把仿真多动力学族（碰撞/遮挡声响）在现有 world 内扩测（零外部依赖，帽 150 CPU），再决定真机；真机需 owner 明确授权。

## §4 决策点 D：阶段账处置

- 阶段帽 24,000；已知位置 ≈35,250（D-20261119-15）＋22.7b 超支 ≈1,800（D-20261119-43）。
- **建议**：维持「超支永久记事故、不改帽」，后续每卡预算单独定价、超出即预算停止；若要重冻阶段帽，需 owner 明确指令并留档。

## §5 结论

四项均待 owner；执行者默认路径：A 案维持、不解锁 22.8/23–30、先做仿真多动力学族扩展（零外部依赖）、阶段账按事故记录。**任何一项若 owner 未回溯反对，下一轮按常设授权继续推进相应零依赖部分。**

## 附：正式 check② 决策准备（D-20261119-97）

[A/B 对照材料](MEMO-TAIJI_CHECK2_DECISION_20261120.md) 已备：推荐 **A 案维持**（0 暴露、`fit_face_only`），B 案需先完成六项前置（D 参考冻结、B 父资格、候选/评估脚本冻结预注册、预算、失败出口）。8 通路新证据不改变任务轴；未见任务仍 `uncovered`。
