# N 主线完成度对照表（2026-10-10 ㊵-662 立，只登记现状、不改判据、不起跑）

> 用途：把 [09_NEXT_MAINLINE_PLAN](../active/roadmap/09_NEXT_MAINLINE_PLAN.md) §2 六项的**出口交付条款**逐条对到
> 现有证据上，并明确每条缺口**谁能动**（我这侧／owner 终端／owner 批文）。
> 本表不宣布任何一项完成；它的作用是防止把"仪器更完善"读成"计划更接近完成"
> （分层口径见 [08 ㊵-659](../active/roadmap/08_UPSTREAM_SYNC_PLAYBOOK.md) 与 `explain-ordering-and-separate-instrumentation-from-capability`）。
> 分层定义：**L1 产品码**＝代码里那格存在；**L2 入口可达**＝默认或显式旗标能走到；**L3 判读可信**＝读数有面、有守卫、有兼容锚；
> **L4 能力读数**＝分数/行为变了；**L5 计划条款结清**＝出口交付的原文要求全部有证据。

## §1 六项对照（一行＝一条出口交付要求）

| N 项 | 出口交付原文里的要求 | 现有证据（件/测/读数） | 目前能声称到哪一层 | 缺什么 | 谁能动 |
|---|---|---|---|---|---|
| N1 | S5 判读报告入库 | `PLAN-N1-00_S5_ADJUDICATION_20261007.md`（J1＝209＞118 不成立，按冻结出口路由） | **L5 该条已结** | — | — |
| N1 | S1 预注册＋判据先冻 | `PLAN-N1-01_s1_readout_prereg_20261007.md`；阶段0 已跑并判读（`PLAN-N1-01_ADJUDICATION_20261008.md`：J-S1a 不成立、第三出口"读取可修但不充分"） | **L3**（判据面可红） | 阶段1（再激活重放＋训练）未跑；§7.5 否决门四条未取；§18.4 生产者→消费者映射未登记 | **owner 批文**（训练机时＋产品码面） |
| N1 | S1 后三器官同读验收（附第 0 点对照） | 无 | **未做** | 全部 | 依赖上一格 |
| N2 | 一次真实长跑＋离线巩固完整读数 | 两次通电均跑完并判读（`PLAN-N2-01_ADJUDICATION`／`PLAN-N2-02_ADJUDICATION`）；回退与持久化侧 DEBT-G47 在真产品链结清 | **L3＋L4 负结果** | 计划要的"加速点①成立"未成立（J-N2b' `Δquality ≤ 0`）；按 §4 第四行分流去查巩固写入范围与保持集 | **owner 批文**（PLAN-N2-04 甲层要产品码） |
| N2 | 巩固产物形成候选版本、不覆盖 parent | `reports/taiji_n2_postcheck_20261008.json`（回退方向实测完好、摘要逐位同） | **L2＋L3** | — | — |
| N3 | 容量决定（甲／乙／并行）＋对应读数 | 乙：τ 已按 PLAN-N3-09 现冻（β 0.64／circuit 0.66，四张 `freezable`、守卫臂 `not_freezable_at_this_grid`）＝[PLAN-N5-07 判读](PLAN-N5-07_ADJUDICATION_20261010.md) §2 | **L3 已到位／L4 只到触发侧** | 甲的两臂（250k×2）未跑 ⇒ "容量是否封顶"这句没有读数；乙的**生长**读数（真晋升）未取 | **owner 终端**（两臂已批，时序＝第三档与整族之后，owner #22③） |
| N4 | S3 轨迹态＋S4 模式分离的实施与 §3.3 项验收 | 前置链已铺：产品默认挂载（PLAN-N4-04 实施格）、档位自述传递、扰动强度判据、寻址尺升版（J-N4f-2 不放行＝**否证我自己的角度对**）、写路径普查（`reports/taiji_n4_write_path_census_20261009.json`，默认位 `write_happens_at_product_default=false`） | **L1＋L3（仪器与读数面），能力侧 L4 未取** | S3／S4 本体未实施；对象重命名／属性交换／干扰对象／槽占满四类验收未跑 | **owner 批文**（产品码）＋依赖 N1 的 S1 通路 |
| N5 | ①训练器挂 bridge+trigger | `--n5-shadow`／`--n5-shadow-gate` 在 argparse 面上 12/12 旗标现读在场；信封自述 `metadata.command_surface.argv` 可读 | **L2 已证** | — | — |
| N5 | ②真实长跑走完整四层循环 | 双臂 60,000 tick：`shadow_forward_hits 59,762`、`shadow_branch_hits 119,524`（`reports/taiji_n5_07_shadow_gate_full_20261010.json` 本表落盘前现读）、配对四元组五项相等且 `tau` 两臂同 0.64（同件 `pairing.j_n5b_3=valid`） | **L2＋L3**；**归因未定** | 台账 ㊵-656 那句"逐张量独立比出 `DIFFERING 20 of 220`"**既无封存件也无留下脚本**（`git grep DIFFERING -- reports/` 现读 0 命中）⇒ 此刻不可复算，已登记 **DEBT-G92**；20/220 的两个解释（影子写入 vs 浮点非确定性）还没分开 | 我这侧＝把逐张量比做成仪器并封存（G92）；分开解释那一半＝**owner 终端**第三档（owner #22②裁"现在跑"） |
| N5 | ③世界学习器首次真实调用 | `WorldDynamicsLearner` 公开 `propose`／`snapshot`／`restore`，普查器读到 `contract_complete`（`reports/taiji_n5_world_model_contract_census_20261009.json`；[08 ㊵-651](../active/roadmap/08_UPSTREAM_SYNC_PLAYBOOK.md)） | **L1＋L3** | 合同齐≠被调用：接进语料训练链那一格未动（DEBT-G78 要产品码批文） | **owner 批文** |
| N5 | ④生长前后任务矩阵＋`J-N5b-6` 合取 | `J-N5d-1` 有读数且为负（两臂 `cost_persists`）；`J-N5d-2` 分辨率陈述（`ruler_usable=false`）；`j_n5b_5` 现在由冻结取数式出值＝`unverified_noise_floor_missing`（`reports/taiji_n5_08_jn5b5_first_read_20261010.json`） | **L3 齐／L5 差一格** | 四元合取的第 5 项要 `noise_floor` ⇒ 仍等第三档；出值后 `J-N5b-6` 才允许离开 `not_adjudicable` | **owner 终端**（同一支档） |
| N6 | 工程债随实施项清 | DEBT 表至 **G92**；本轮结清 G87（夹具现场缺失）、G89 转半结；新登记 G90（`--line` 阈值字面量）、G91（五枚门只读封存件不读活跑）、G92（一条已发表的独立读数无件无脚本、不可复算） | **L3 持续改善** | G81 活队列 43 枚；G85 修法②③；G74② 已结、G76 待双臂读数；PLAN-B-03 拆分未完成（owner 签字件，只登记不代改） | 我这侧可自办：G85②、G91 读数面仪器化、G92 仪器化 |

## §2 一句话现状（不许被读成"接近完成"）

- **能力层（L4）本主线今日的净新增＝零**：N5 触发侧打通发生在 ㊵-656，保持侧今日是**负结果照常出版**，收益侧是**分辨率陈述**；
  N1／N2／N3 甲／N4 的能力读数全部尚未取得。
- **判读可信层（L3）今日净新增三格**：`j_n5b_5` 由占位符变成有取数式＋正反例＋兼容锚的仪器格；
  p2-11 门的歧义支从"没被走到"变成"可证被走到"；查出一族"读封存件冒充现行绿"的假安心并登记为债。
- **队首唯一性**：以 [03_CURRENT_EXECUTION](../active/roadmap/03_CURRENT_EXECUTION.md) 那一条 `## 当前唯一下一步：` 为准；本表不是队首，不据本表起跑任何东西。

## §3 完成判据（这张表什么时候可以删）

六项的 L5 列全部为空、且 N5 的 `J-N5b-6` 离开 `not_adjudicable_*`、N3 甲有容量读数、N1 三器官验收过 §7.5 四条否决门 ⇒ 本表作废并在 08 台账点名；
在此之前任何"N 主线已完成／接近完成"的说法都不该出现。作废时本表**不删**，加日期标注"已被取代"并点名取代它的那一笔。
