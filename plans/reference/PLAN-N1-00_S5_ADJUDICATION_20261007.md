# PLAN-N1-00 · S5 便宜证伪判读（2026-10-07 判读入库；判据未调）

> **效力**：[预注册](PLAN-N1-00_s5_endpoint_falsification_prereg_20261007.md)（2026-10-07 冻结；owner 同日弹窗裁"批准开跑"）的唯一判据 J1 按冻结文本判读，两个出口都是真知识。**判定＝J1 不成立 ⇒ 病在更深（基底）**，按预注册 §3 路由转 [09 §4](../active/roadmap/09_NEXT_MAINLINE_PLAN.md) 分流表第一行；S1（PLAN-N1-01）预注册顺延重审。判据未调、未重跑凑线；本轮预算 1 配置×1 面，仪器自检红 0 次、修复 0 次。

## 1. 判据判读（唯一判据，读数单位＝行，分母冻结 288）

- **J1（never-LF 拖写行数 ≤118）＝ 209 ⇒ 不成立**（基线 236；降幅 27 行＝11.4%，远未达腰斩线）。
- 判读仪器＝在库 [count_taiji_a30_eater_p_boundary_floor.py](../../../scripts/training/count_taiji_a30_eater_p_boundary_floor.py)（fail-closed）：status ok、自计数与仪器声明双一致、`base_sha256_unchanged`＝True、无缺列行。

## 2. 守卫判读（G1–G3 全绿；缓解数字在守卫生效前提下取得）

- **G1 面冻结** ✓：判读件 `items_sha256`＝`0541a3f4568a9c5b`、`max_length`＝256、α＝1.0、checkpoint/circuit sha 与基线件逐位同、`base_sha256_unchanged`＝True。
- **G2 α=0 锚点** ✓：S5 代码＋旗标开、α=0 臂（`taiji_a30_stop_failure_self_v42_alpha0_endpoint_guard_96_20261007.json`）与历史 v38_alpha0 件 [compare_taiji_a30_report_identity](../../../scripts/training/compare_taiji_a30_report_identity.py) `--subtree per_item` ⇒ **identical=true、behavior_diff 0、schema_diff 0**——改动未从 α=0 路径泄漏。
- **默认路径逐位不变**（v42 format 注核）✓：无旗标重跑 v37 命令（`..._v42_noflag_identity_96_20261007.json`）对 v37 件 per_item ⇒ **identical=true、0 差异**——harness 与生产路径无损。
- **G3 非末字节分支** ✓：单测层 5/5 绿（`tests/taiji_native/test_copy_circuit_endpoint_yield.py`）＋既有电路合同 49 测全绿；面层由上两条覆盖。

## 3. 读数对照（第 0 点对照如实附）

| 读数 | v37 基线 | v42 S5 | Δ |
|---|---|---|---|
| 生成行分母 | 288 | 288 | 冻结 |
| eaters（拖写代） | 247 | 219 | −28 |
| **never-LF 拖写行（J1）** | **236** | **209** | **−27（11.4%，判据线 118，不过）** |
| 发 LF 的 eaters | 11 | 10 | −1 |
| stoppers（提前停止） | 41 | 69 | ＋28（占比 14.2%→24.0%） |
| never-LF 群 p_boundary_max（中位/最大） | 0.000158 / 0.00724 | 0.000161 / 0.00724 | 量级不动 |

## 4. 归因输入（如实登记，不构成本轮新判据）

- **终点提议真实开过火且有解**：28 条拖写代提前停止（27 条自 never-LF 群转化＋1 条自发 LF 群）——"末字节 ⇒ 让位＋边界符提议"这条机制本身能把走完轨迹的代放走，方向与 H-S5 一致。
- **但被困群几乎走不到终点**：209 条仍 never-LF 的行里，逐行配对后仅 **5 行**的 `p_boundary_max` 被终点分支动过——即 copy 轨迹在这群代上**极少真正到达所选事件的末字节**，终点检测器无火可点；该群边界概率量级（max 0.00724）逐位不动。
- **归因指向（接 09 §4 第一行）**：病不在"缺终点信号"单因子——更深处是**轨迹走不到末字节**（池化分布在轨迹中段就失焦/漂移，或发射根本没沿事件字节序走）。按 §8.6 纪律先列完整计算图、stop-gradient 与离散操作边界，再定最小归因实验；§16.1 纪律：本轮预算已用尽且无新可辨识假设的**判据级**重跑，归因实验须另立预注册。

## 5. 资产与状态

- 判读件：`reports/taiji_a30_stop_failure_self_v42_s5_endpoint_circuitseedA_96_20261007.json`（format v42）＋守卫臂两件（α=0 臂、无旗标身份臂，同日入库）。
- 实施：`taiji/copy_circuit.py` 终点分支（`endpoint_yield_override`，默认关）＋探针 v42 旗标 `--copy-evidence-endpoint-yield`＋守卫单测 5 件。
- **N1 状态**：S5 出口＝B（病在更深）；P1"接口病"假设**降级为部分成立**（终点信号能放走走完轨迹的代，但不是 never-LF 陷阱的主因）；PLAN-N1-01（S1 再激活读取预注册）**顺延重审**——重开前置＝基底归因实验（轨迹为什么走不到末字节）另立预注册并出读数。09 §2 N1 步骤与 N 系列排序不变。
- 台账：08 ㊵-477；M6 收官、M7-CI 绿、M8 挂起不变。
