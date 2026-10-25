# CI 浮点判侧证据包（族 A 第五例）——供 owner 裁定

状态：**owner 已于 2026-10-25 裁定方案乙并实施**（admission 阈值判定改
`math.isclose` 相对容差，`taiji/interaction_group_online.py` 的
`ADMISSION_FLOAT_TOLERANCE = 1e-12`；族 A 全部 8 条 deselect 同日摘除，
红绿语义钉在 `test_interaction_group_online` 的
`test_online_admission_thresholds_absorb_cross_platform_float_noise`）。
本文件保留为裁定留痕与实施依据。
背景：2026-10-24/25 CI 修绿期间，8 条测试因跨 runner 浮点判侧翻转入 deselect
（ci.yml 批次三「族 A 第五例」），按批次一⑤先例「不修核心库＝不动研究语义」搁置。

## 一、翻转记录（同一提交、不同 runner、结论相反）

### relation 两测（Ubuntu 腿）

| run（同一提交 ad12c2e02） | 腿 | 结果 |
|---|---|---|
| 40024125106 attempt 1 | Ubuntu | **红**×2（`assert False`） |
| 40024125106 attempt 4 | Ubuntu | 绿×2 |
| 40075515421 attempt 2 | Ubuntu | **红**×2（`assert False`） |

- `test_native_local_relation_contract.py::test_relation_credit_cut_terminal_sources_and_reuse`
- `test_native_relation_confirmation_contract.py::test_each_seed_terminal_and_cut_observation_branch`

同提交的 Windows 腿（1h22m／1h24m／1h34m／1h50m 四轮）与本机（torch 2.13.0+cpu，
7/7 通过、3.88s）稳定绿。三次 Ubuntu 两红一绿＝公开 runner 池硬件轮换下约五五开。

### structural-gain 门族 5 条（Windows 腿）

| run（同一提交 a3ce9beb8） | 腿 | 结果 |
|---|---|---|
| 40087142025 | Windows | **红**×5 |
| 40096384119 | Windows | 绿×5 |

- `test_cross_domain_structural_gain.py::test_cross_domain_structural_gain_gate`
- `test_interaction_group_multifamily.py::test_interaction_group_multifamily_leave_one_out_gate`
- `test_online_interaction_structural_bridge.py::test_online_interaction_structural_bridge_gate`
- `test_structural_workspace_net_gain.py::test_structural_workspace_net_gain_gate`
- `test_terminal_three_domain_governance.py::test_terminal_three_domain_governance_gate`

报错签名：五条中的四条同报 `online structural bridge requires two applied feedbacks`，
multifamily 报 `selector did not choose a group for held-out complement`。
本机 5/5 绿（146s）。两轮绿后第三轮翻红＝runner 池内 CPU/BLAS 差异。

## 二、代码位点与机制

`taiji/interaction_group_online.py:311` 与 `:550`：`minimum_interaction` 默认 0.0
配**严格 `<`** 比较。跨平台浮点（不同 CPU 的 SUMM 顺序／BLAS 内核）在阈值附近
算出任何微小负值（哪怕 1e-18）即被判到另一侧——批次一⑤已登记此机制（族 A 第四例
`test_continuous_structural_growth` 同根）。

历史判例一：H19c（2026-09-30）放弃 3.10 腿，直接原因即
`test_interaction_group_multifamily` 在 3.10 与 3.12 因「浮点/容器迭代边界」判到
不同侧且本机无法复现。
历史判例二：⑤（㊵-469）当时裁定「不修核心库：那会动研究语义（接纳阈值本身）」。

## 三、待取证据（裁定后实施阶段的第一步）

现有 CI 失败只出断言消息、无浮点读数。需在 runner 上插桩复测取到：
1. `minimum_interaction` 判侧处的实际浮点差值与符号（预期量级 1e-16～1e-18）；
2. 翻转发生在 `< 0` 边界还是 `< threshold` 边界（决定容差写法）；
3. multifamily 的容器迭代顺序敏感性是否独立于浮点（3.10 判例提示可能双因）。

**2026-10-25 插桩读数到位（run 40185913504，Ubuntu）**：
`reason=feedback_resource_cost_exceeds_admission_budget, realized_interaction=0.6000000000000001,
uncertainty=1.165359456941537, resource_cost=3.0`——与静态分析吻合（interaction 门 0.6 vs 0.1
余量巨大、uncertainty 1.165 < 1.25 均安全），拒绝来自 **resource_cost 门：第二个 episode
(d,e) 的资源消耗 3.0 精确超过 admission 预算 2.0**。而 40171918472 的 Ubuntu 绿过同一测——
机制＝上游模型数值的平台差异改变 `select` 结果 → 不同候选 → episode 资源消耗不同（有时
≤2.0 有时 =3.0），量级型发散而非渲染噪声，isclose 容差不适用。**归 owner 二次裁定**：
（i）该 eval 的 `maximum_resource_cost` 2.0 提到覆盖 (d,e) 家族固有消耗（参数级修正，
不改结构增益判据）；或（ii）`_extend_online_feedbacks` 的两轮 select 确定化（钉住与
第一轮一致的候选路径）。裁定前该测保持 deselect。

**2026-10-25 二次裁定落地**：owner 裁 (i)——该 eval 扩展相 admission 资源预算
2.0→4.0（eval_taiji_continuous_structural_growth.py 的 evaluate()，判据本身
不变，本机绿 47.65s），deselect 条目摘除。
另：本轮 b0_n2 又接两次登记（N1-167/168 收口脚本，registry 94/pending 12/live 106）——
并行会话结案波仍在落盘，按 CI_CONTRACT_RULES §2.1 逐波收口。
**2026-10-25 静态分析更新（continuous_structural_growth 专属）**：方案乙
（isclose 1e-12）实施后该测在 run 40166946185 Ubuntu 仍翻一次、在
499c3892f 又绿——但静态分析**基本排除阈值门**：该 eval 的
`realized_interaction = derived_score − _single_score(first) − _single_score(second)`，
其中 derived_score 有 1e-12 护栏精确等于 1.0、`_single_score` 是纯常量函数
（0.2），realized ≈ 0.6 对 minimum 0.1 余量巨大，三道阈值门都在远离边界的
安全区。真因指向**candidate 生命周期语义**：第二个 second-cycle feedback
`(d,e)` 与第一个已准入的 `(a,e)` 共享成员 e，e 被观察后 `(d,e)` 候选的命运
取决于 learner 状态，而上游 torch 数值的平台差异会改变 select 的候选组成。
此层属研究语义（candidate 生命周期），归 owner 裁定；诊断插桩
（`eval_taiji_continuous_structural_growth.py:152`，断言携带
admission.reason 与三阈值读数）已在其树上，下次翻红直接读出确切 reason。
另注意：其「Linux 稳定红」旧认知已被推翻——isclose 前后各红绿交替一次，
实为间歇性（runner 池硬件轮换放大的上游数值差异）。

## 四、补丁草案（裁定同意后才实施）

- 方案甲（最小）：两处严格 `<` 改带绝对容差比较（如 `< -1e-12`），
  并在该处注释「跨平台容差，owner 2026-10-XX 裁准」；同时更新批次一⑤的
  deselect 重启条件并摘除族 A 全部 8 条。
- 方案乙：admission 判定改 `math.isclose`（相对容差）——语义更平滑但改动面大。
- 方案丙：不修库，把 8 条维持 deselect 至下次语义版本一并处理（现状）。

## 五、附带事项：预注册措辞棘轮的 224 份隐形文档（G74②）

批次三重冻后 `plans/reference/PLAN-*.md` 共 329 份＝105 ok／224 invisible／
0 ambiguous。G74② 的逐份措辞修复归人判；agent 只提供批次建议：
- 修法模板（已在该棘轮测试断言中）：给判据段加一个含「判据/出口/验收」的二级标题；
- 建议批次：①N1 训练链（PLAN-N1-0xx～1xx，约 120 份，模板化程度最高）；
  ②N1 run/N3/N5 面（约 80 份）；③C2/N02/N3 裁定书（约 24 份，需逐份语义核）。
  每修一份即从 FROZEN_INVISIBLE 白名单删除对应名（棘轮只许单向收紧）。
