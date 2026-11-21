# 结果卡：联合生长烟测（growth_joint_partial；133.77 CPU；1 seed；零 check）

> [结果卡] — 2026-11-21（D-20261121-260）。上游 [生长绑定](RESULT-TAIJI_U2_GROWTH_BINDING_SMOKE_20261121.md)。

## 1. 读数（cap 回执 reports/taiji_u2_growth_joint_smoke_20261121.json）
baseline P-D 0.8359；combined 0.8828；old_only 0.8828；new_only 0.3594；lesion_new **0.0**；old_skill_damage **0.3438**。

## 2. 判读
1. 联合训练把 P-D 提到 0.8828，但**完全来自旧单位**（old_only 同值）；新单位仍被忽略（lesion 0.0）；
2. **旧技能损伤 0.344 异常**：CN/P-B 单位未训练却被破坏，最可能是 `grow_units_neutral_` 后 **read_gate 布局（2N→2(N+12)）重映射不完整**（与早期 head a-offset 问题同类），使旧单位读出漂移；
3. 因此本卡只记录为 **`growth_joint_partial`**，旧损伤读数视为**测量/实现缺陷**，不当作生长代价证据；新单位仍未被功能绑定。

## 3. 后继
先修 `grow_units_neutral_` 后 read_gate 的旧行迁移（单测：生长前后旧任务读数差 ≤1e-6），再重跑联合生长。


## 5. 更正（D-20261121-261，零训练）
旧「损伤 0.3438」是**评估布局 bug**：生长后 `a` 与旧头输入拼接错位（旧头应取 `z[:old_n]`+`a[:old_n]`）。修正后重跑：**old_skill_damage = 0.0156**（≤0.02），其余不变：combined 0.8828 = old_only 0.8828、new_only 0.3594、lesion_new 0.0。结论不变：联合生长仍**没有**让新单位产生功能贡献（lesion 0），但旧技能保持成立。
