# PLAN-N2-03 判读 · "代价来自收束还是巩固的权重更新"——2026-10-08 跑完并入库

> **判读对象**＝[PLAN-N2-03 预注册](PLAN-N2-03_reset_attribution_prereg_20261008.md) §2 的 J-N6-归因（三档互斥）＋§3 的四条守卫。
> **一句话结论**：**判 `cost_from_weight_update`** —— Arm R（只收束、参数族零差）在 run-2 跌破的那 7 列上
> **一列都没复现**（`n_reproduced = 0 / 7`，七列 `attrib_R` 全为 `−0.0`，即臂档读数与**巩固前逐列相等**）
> ⇒ 这一格代价不归"落盘前的 `reset_dynamics` 收束"，归**巩固写进权重的变化**。
> 判读仪＝新件 `scripts/training/count_taiji_n2_03_attribution.py`，读数件＝`reports/taiji_n2_03_attribution_20261008.json`（rc=0）。

## 1. 七列逐格（before＝母档面／run-2＝治疗后面／armR＝收束臂面）

| 面:列 | 巩固前 | run-2 巩固后 | Arm R | run-2 差 | Arm R 差 | `attrib_R` | 复现 |
|---|---|---|---|---|---|---|---|
| `cap0:E` | 1 | 0 | **1** | −1 | 0 | −0.0 | 否 |
| `replay_strict_hits:1.0` | 6 | 5 | **6** | −1 | 0 | −0.0 | 否 |
| `replay_strict_hits:2.0` | 6 | 5 | **6** | −1 | 0 | −0.0 | 否 |
| `replay_well_formed:0.0` | 24 | 18 | **24** | −6 | 0 | −0.0 | 否 |
| `replay_well_formed:0.5` | 24 | 17 | **24** | −7 | 0 | −0.0 | 否 |
| `replay_well_formed:1.0` | 24 | 15 | **24** | −9 | 0 | −0.0 | 否 |
| `replay_well_formed:2.0` | 25 | 14 | **25** | −11 | 0 | −0.0 | 否 |

`columns_considered = 7`、`columns_excluded = []`（没有一列因分母为 0 或缺读数被剔除）。
判级线：`attrib_R ≥ 0.5` 的列数 ≥5 判"收束致损"、2..4 判中间带、≤1 判"权重致损"——实测 **0** ⇒ 落第三档。

三档都曾被测：契约测 `tests/taiji_native/test_n2_03_attribution_contract.py`（15 passed）用夹具把
`cost_from_wake_reset`（7 列全复现）与 `partially_resolved`（3 列）各走通过一次，**这把尺不是只会输出一个答案的尺**。

## 2. 四条守卫的实测（任一不成立则整件不判，实测全绿）

| 守卫 | 冻结口径 | 实测 |
|---|---|---|
| **G-N6-1** 该臂真的没学 | 臂档与源档按**参数族**零差（§3ter 更正后的口径） | `g_n6_1_parameter_family_unchanged=true`；`family_diff.parameter_family` 的 `changed/dropped/added` 全 **0**（`g_n6_1b_parameter_entries_stable=true`）；参数摘要 `1605` 条目前后同 |
| **G-N6-2** 装载中性 | 臂档可被产品装载器读回，读回后参数族仍等于母档 | `g_n6_2_arm_loadable_and_params_identical=true`（臂档 `output/n2b_resetarm/checkpoint.pt`，12,331,669 B） |
| **G-N6-3** 面同源 | 三张 CAP-0 面同一题集、复述三面同 `items/item_offset/first_item`、主列 `items_sha256` 与基线同一份 | `eval_set` 三面同为 `plans\manifests\cap0_eval_set_v2.json`；复述三面同为 `items=24, offset=0, first_item=V001`；主列两侧 `items_sha256` 同为 `8b974e9b62d8e4ca` |
| **G-N6-4** 不新增仪器 | 读数走 run-2 那三台现成仪器，取数函数从 run-1/run-2 判读器**导入** | 判读器只 `from scripts.training.count_taiji_n2_faces import _cap0_tally, _main_column_entries, _replay_arms`，零重算生成链 |

臂自述：`wake_episode_id = wake-after-sleep`，`tick 273 → 0`，情节/活动族被清掉的叶子与变化数如实登记在
`reports/taiji_n2b_resetarm_20261008.json`（这一臂的**处理本身**，不是缺陷）。

## 3. 主列（`never_lf_eaters_counted`）只作旁证，不入归因分数

| 面 | 主列 | 题集 sha | 档 sha |
|---|---|---|---|
| 巩固前（`taiji_n2_stop24_before_20261008.json`） | 56 | `8b974e9b…` | `d6169a35…` |
| run-2 巩固后 | **47** | 同 | `504f7c34…` |
| Arm R 收束臂 | **57** | 同 | `8f6ee3fe…` |

run-2 在主列上的**改善**（56→47，方向＝越小越好）在收束臂上同样没复现（56→57）。
⇒ 与 §1 同向：这一列的变化也不来自收束。**注意**：57 对 56 是**变差 1 枚**，本件不判它是不是噪声
（CAP-0 已两次实证确定性的只有那台；停摆主列没做过空档复算），只报数。

## 4. 顺量到的一条仪器级事实（比归因本身更通用）

同一份权重、情节/工作记忆被整块清空（掉 1,612 枚叶子、68 处活动态变化、`tick 273→0`）之后，
CAP-0 的 E 维、复述两档命中、复述四档成句**七列读数逐列回到巩固前的值**。
⇒ 这七列读数**不消费情节/工作记忆那两族状态**；它们是被权重（＋固定输入题集）决定的函数。
这条对本线的用处：以后任何"掉了一格"的读数，不能再用"是不是刚收束把态清了"来解释——**已被量掉**。

## 5. 去向（按预注册 §5 的分支，不引入新判据）

1. **判"主要来自权重更新" ⇒ 保持集设计照 §8.3 那一支走**（候选版本不覆盖 parent、`L_keep` 与调参集分离），
   这才是**第三件改权重实验**的输入；本件不自行改产品码，也不启动任何改权重跑（回 owner 批文）。
2. **DEBT-G47 修法甲/乙之争的"代价"论据被削**：预注册 §5 原本写"若判收束致损 ⇒ 修法乙重新上桌"。
   判的是反方向 ⇒ **修法甲之下读到的代价并不含"收束清态"那一份**，所以没有理由为了保住情节态而回到乙。
   （这只是**取消一条反对甲的论据**，不等于对甲的重新裁定——G47 已按甲结清，本件不重开。）
3. **09 §4 分流第四行的这一格已出读数**：代价住在"巩固写入范围"这一层，下一步的候选是"哪些权重被写坏了"，
   那是**逐族消融**类的实验（需要改权重跑 ⇒ owner）。

## 6. 本件不证明什么

* 不证明"权重更新里具体是哪一族造成代价"——Arm R 只把两族变量分开，没有指认族内哪一支；
* 不证明 E 维那 1→0 的**机制**（100 项里 1 项，粒度问题仍挂在 DEBT-G51 那一族）；
* 不证明"回退到母档就一切复原"（回退面在 run-2 已单独测过，本件的臂档不是回退档）；
* 不给任何"能力恢复了多少"的承诺，也不动 `sleep_pass` 默认位、不放大剂量。

## 7. 复算（零改权重，全部只读）

```bash
python scripts/training/run_taiji_n2b_reset_arm.py           # Arm R 驱动（实测 2.9 s，写 output/n2b_resetarm/）
# 三支面与 run-2 同名同参，只换 --checkpoint（四条都在 2026-10-08 实跑，逐条 rc=0）
python scripts/training/eval_taiji_cap0_baseline.py --checkpoint output/n2b_resetarm/checkpoint.pt --report reports/taiji_n2b_resetarm_cap0_20261008.json
python scripts/training/measure_taiji_a30_repetition_penalty.py --checkpoint output/n2b_resetarm/checkpoint.pt --limit 24 --out-report reports/taiji_n2b_resetarm_replay24_20261008.json
python scripts/training/probe_taiji_a30_stop_failure.py --checkpoint output/n2b_resetarm/checkpoint.pt --limit 24 --out-report reports/taiji_n2b_resetarm_stop24_20261008.json
python scripts/training/count_taiji_a30_eater_p_boundary_floor.py --report reports/taiji_n2_stop24_before_20261008.json --report reports/taiji_n2b_resetarm_stop24_20261008.json --expect-items-sha 8b974e9b62d8e4ca --out-report reports/taiji_n2b_resetarm_stop24_main_20261008.json
python scripts/training/count_taiji_n2_03_attribution.py     # 判读器（缺件 rc=2）
python -m pytest tests/taiji_native/test_n2_03_attribution_contract.py -q   # 15 passed
```

面件的行很宽，这里为可读性写成单行；实跑用的是同一串参数。臂档那三张面的自述里
`checkpoint` 字段逐条为 `output/n2b_resetarm/checkpoint.pt`（已核，不是母档、不是候选档）。

## 8. 不变项

M6 收官、M8 挂起；`DEFAULT_CHECKPOINT` 与厂档不动（Arm R 只写 `output/`）；甲臂起跑、
`git push origin main`、第三件改权重实验三格仍等 owner；N3 乙 τ 那格空着；N4/N5 后置；台账行序以行首标号为准（㊵-419⑥）。
