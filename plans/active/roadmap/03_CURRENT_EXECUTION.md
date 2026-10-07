# 当前执行（N 主线，2026-10-06 起）

> **本文是 N 主线的活动文档**。M 主线（M0–M8）已于 2026-10-06 收束：其执行日志（含全部停靠块与历史读数）封存于 [archive/history/m_series_execution_20261006/03_CURRENT_EXECUTION_M.md](../../archive/history/m_series_execution_20261006/03_CURRENT_EXECUTION_M.md)——**历史读数与"下一步"指示不得引用为现行**；引用历史请带批次出处。

## 现行状态（2026-10-06 收束时点）

- **M6 provider 与产品收口**：收官（[批准书 20261006](../../reference/M6_SCOPE_EXCLUSION_APPROVAL_20261006.md)，02:145 声明入档；台账 ㊵-455）。
- **M7 CI 与发布**：限定开线，**CI 已绿**（`m7-ci` workflow：build＋doc-sync 43 叶＋hygiene 18 叶，run 37453182065；台账 ㊵-456/457）。发布面 owner 裁暂缓；剩余＝manifest 对齐、测试面接 CI（前置 ㊵-145 未裁）、ubuntu-26 复验（2026-10-19 后）。
- **M8 CUDA 与规模化**：摘除主线（重启＝真设备＋能力就绪＋owner 单独批准）。
- **A 支线／R2 线**：已收官（SPEC-A-26 结案归档；R2 v1 标定判停结案）。

## 活动规划与导航

- **下一轮主线（N 系列）**：[09_NEXT_MAINLINE_PLAN.md](09_NEXT_MAINLINE_PLAN.md)——架构与模型（2026-10-07 三版重订，最终版为**内容级构造**：VISION 的回答计算链（§7.1）/否决信号（§7.5）/错误分账（§7.2）/训练配方（§19.3）/世界模型四接口（§6）/惊讶度处理（§8.3）/失败路径分流（§13）等具体内容直接转化为各 N 项"实施设计"块；§1.2 接回 VISION 能力演进路线（§17.3 五级演进＋§19.10 实施包地图＋§19.5 联合行为——N 系列是第一段"修复＋加速"，收官接回实施包地图）；§4 失败路径分流表；§5 全局不变量（§1））。北极星＝经验→能力转化速率，S7 四组基线为第 0 点。排序不变：N1 P1 接口修复（S5 便宜证伪最先）→ N2 巩固通电 → N3 容量决定 → N4 读取形态补全 → N5 自进化唤醒（N3 裁乙即启动，一体两裁）→ N6 工程债随实施项清。
- **项目收束**：[M0_M7_PROJECT_CONSOLIDATION_20261006](../../reference/M0_M7_PROJECT_CONSOLIDATION_20261006.md)——逐阶段终态、跨阶段遗留登记（10 项带重启条件）、继承资产。
- **连续台账**：[08_UPSTREAM_SYNC_PLAYBOOK.md](08_UPSTREAM_SYNC_PLAYBOOK.md)（㊵-455＝M6 收官条；㊵-456 起＝M7/N 系列时代；更早条目为 M 时代历史）。

## N 系列执行日志（自下而上追加，每项开工前预注册判据）

【2026-10-07 · S5 判读入库：J1 不成立 ⇒ 病在更深（基底）】三支面同日跑完（每支约 13 分钟，全部 rc=0）：①无旗标身份臂对 v37 件 per_item **identical=true/0 差异**（默认路径逐位不变）；②α=0＋S5 旗标守卫臂对 v38_alpha0 件 **identical=true/0 差异**（G2 过）；③判读面 `taiji_a30_stop_failure_self_v42_s5_endpoint_circuitseedA_96_20261007.json`（G1 面冻结全对上）。**J1＝never-LF 拖写行 209 ＞ 118 ⇒ 不成立**（基线 236，降幅 27 行＝11.4%）；辅读数：stoppers 41→69（终点提议真实放走 28 条拖写代，其中 27 条自 never-LF 群转化）、never-LF 群 p_boundary_max 量级逐位不动（max 0.00724）、209 条被困行里仅 5 行被终点分支动过 ⇒ **copy 轨迹在被困群上极少真正到达事件末字节，终点检测器无火可点**。按预注册 §3 路由：P1"接口病"降级为部分成立，转 09 §4 分流第一行（§8.6 完整计算图归因：轨迹为什么走不到末字节）；PLAN-N1-01（S1 预注册）顺延重审，重开前置＝基底归因实验另立预注册。判据未调、未重跑凑线；预算用尽（1 配置×1 面、修复 0 次）。判读报告＝[PLAN-N1-00_S5_ADJUDICATION_20261007](../../reference/PLAN-N1-00_S5_ADJUDICATION_20261007.md)。

【2026-10-07 · owner 弹窗四裁＋S5 开跑】弹窗按 09 §3.2 时间序出全四项，owner 裁：①**N1/S5 预注册批准开跑**（PLAN-N1-00 解冻生效）→ S5 进入实施与复跑（实施落点 `taiji/copy_circuit.py` 终点分支＋探针 v42 旗标 `--copy-evidence-endpoint-yield`，零训练、开关默认关，守卫单测 5/5 绿＋既有电路合同 49 测全绿）；②**C6 六决策点按推荐组合批**（①a workbench_artifact＋②b no_prose＋③现在做＋④sleep_pass project 段＋⑤消费者同批＋⑥面板行加）——N2 材料线解冻，实施按 N2 推进步骤另起；③**N3 路线＝甲乙并行**——scaling 探针与 R4 通电双线启动，各自预注册（乙压强阈先在线面冻结）；④**主线外挂账均不处理**（1.9GB／发布面／icon.ico 继续挂账）。裁定的完整状态登记见 [09 §3.2](09_NEXT_MAINLINE_PLAN.md)。

【2026-10-07 · N1/S5 预注册落盘】[PLAN-N1-00_s5_endpoint_falsification_prereg_20261007](../../reference/PLAN-N1-00_s5_endpoint_falsification_prereg_20261007.md) 冻结：假设 H-S5（never-LF 拖写陷阱＝复制证据通路缺终点信号，`_successor_bonus` 进度链在事件末字节设计性消失）；基线＝v37 peakrun 在库件复算（288 行分母、eaters 247、never-LF 236＝95.5%、never-LF 群 p_boundary_max 中位 0.00016/最大 0.0072；checkpoint/circuit sha 实测在盘逐位对上）；唯一判据 J1＝never-LF 拖写行数 ≤118（基线 236 腰斩），守卫 G1 面冻结＋G2 α=0 逐位恒等＋G3 非末字节分支逐位不变；预算 1 配置×1 面、修复 1 次仅限仪器机械修复。成立 ⇒ P1 坐实接 PLAN-N1-01；不成立 ⇒ 病在基底转 09 §4 分流第一行。零训练、开关默认关。~~状态＝待 owner 点头~~ ⇒ **同日已批，开跑**。
