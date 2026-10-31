# R026：09.4 持续可塑性与结构出生——同父主体六臂有界比较合同

> 文档状态：[合同] — 预注册09.4的比较范围、对照臂、主终点、判据与预算。**本页在运行前冻结**；跑后不改判据。调度只见[MAINLINE](../MAINLINE.md)。

日期2026-10-31。对象：[R021参考图](SPEC-TAIJI-CORE_REFERENCE_GRAPH_R021_20261031.md)＋[R023](SPEC-TAIJI-CORE_GROWTH_R023_20261031.md)门控出生＋[R024](SPEC-TAIJI-CORE_RESTORE_R024_20261031.md)声明更新表达式的合成夹具。`living_core.py`字节不变。实现落`taiji/native_development.py`。

## 1. 同父主体与声明更新（合法反馈/重放改负责参数）

六臂全部从**同一父主体**（同config/seed/初态/fit journal）出发。声明反馈更新＝教师强制重放分支上的平均NLL，Adam(lr=1e-3, betas=(0.9,0.999), eps=1e-8)，每臂**8步**（每步重算重放梯度）；参数全体参与（与R024同一声明路径），新生R行的入边梯度即"反馈改负责参数"的直接读数。除声明8步外无任何更新。

## 2. 六臂（同父、同更新日程）

| 臂 | 出生规则 | 单元数 | 门表 |
|---|---|---|---|
| coverage_birth | R023全链：信号条件→max-min分配→fit窗SVD方向 | 分配结果 | 混合门表（父端点trace） |
| gradmax_schedule | 固定日程单事件：全局信用top-R_max方向，无分配/无标定 | R_max=8 | 常数1 |
| conditional_svd | 逐支路top-r_c，r_c=min(rank_num,4)，**不做max-min与预算** | 逐支路合计 | 混合门表 |
| random_gate | 随机门信用（R023电池同种子）的top方向 | 与coverage合计相同 | 随机门值表 |
| replay_only | 无出生 | 0 | — |
| capacity_random | 种子随机正交方向（QR），容量匹配 | 与coverage合计相同 | 常数1 |

## 3. 预注册主终点与判据（跑前冻结）

- **主终点**：每臂`ΔNLL_holdout = NLL_holdout(更新后) − NLL_holdout(父)`，holdout=声明探针生涯（prompt=`b"probe-holdout-"*6`、target=`b"probe target ab"*4`，与fit journal不同字节、同store），教师强制NLL自规范初态评估。
- **保持**：保持探针＝**第三条声明生涯**（retention journal：prompt=`b"retention-context-"*6`、target=`b"keep target ab"*4`，与fit/holdout字节不同、同store），每臂`drift_retention = |NLL_retention(更新后) − NLL_retention(父)|`；容差0.05，超限记arm_retention_breach。fit journal是更新数据本身，其NLL下降是学习不是保持。
- **预注册问题**（"覆盖/信用改善是否转为后继任务净收益"）的裁决规则：`coverage_birth`的ΔNLL_holdout严格小于`min(replay_only, capacity_random)`且差>1e-3，**且**其drift_fit≤0.05 → `positive`；否则若coverage_birth不劣于两对照（差≤1e-3内）→ `indeterminable`；否则 `negative`。三值之一如实落账，**不追加seed/步数/臂**。
- 伴随量（每臂）：出生时kappa、完整X/Delta/Z的数值秩（tau沿用R021 §4）、fit窗跨窗信用余弦、更新后fit信用谱变化（学会后秩降如实记录）、CPU/wall全成本。X＝fit窗设计矩阵[S,n,p]（R025 collect口），Delta＝逐微步∂L/∂pre，Z＝逐支路门控堆叠设计[S,K·p]。
- 负结果与不可判按原出口结案；失败不自动添组件/加剂量/加臂。

## 4. 预算、出口与交付

- 预算：全比较≤900 CPU秒/20墙钟分钟；超限`budget_stop`（已完成的臂保留）。
- 出口：`passed_r026_preregistered_comparison`（＝预注册问题已有positive/negative/indeterminable之一且全部臂无工程失败）／`failed_arm_execution`／`failed_retention_all`／`budget_stop`／`engineering_failure`。
- **边界**：本比较在合成夹具、声明更新表达式与未训练主体上进行；positive不签任务能力、不签真实世界收益、不自动准入训练（09.5–09.8按序）；negative同样不证伪v1.2候选于真实条件。
- 交付：本合同、`taiji/native_development.py`、`tests/taiji_native/test_plasticity_arms_r026.py`、`scripts/training/verify_taiji_plasticity_arms_r026.py`、`reports/taiji_core_r026_plasticity_arms_20261031.json`（逐臂单元数/门表/主终点/保持/kappa/秩/成本＋预注册裁决）。
