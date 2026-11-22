# 主线盘点与公开研究对照（2026-11-22；零训练、零 check）

> [盘点/研究] — 2026-11-22（D-20261122-269）。回答「主线到哪、卡在哪、社区有什么可借鉴」。**不做训练、不改判据；零 check/final。**

## A. 主线位置（合同 vs 实证）

| 层面 | 状态 | 证据 |
|---|---|---|
| 合同队列 | `S0 → S1 → 23-S..28-S → 29 → 30 → §15`；S1 仍为队首 | MAINLINE §3A |
| S1 旧协议 | `s1_partial`（11/16→13/16→12/16→9/16 四轮） | D-231/232/233/235 |
| 23-S.0 窄合同 | `23s0_partial`（9/13、7/13）；0c 5/13；joint 不成立 | D-238/239/240 |
| RT 新机制线 | **s1rtf_verified / j_rt_verified / capsule_route_inheritance_verified**：3 任务共存＋纠错＋消融＋旧损伤≤0.008＋胶囊/恢复差 0.0 | D-249/251/253/257 |
| 结构生长 | 负结论收束（5 张一致：lesion ≤0.0625、新任务 gain −0.117） | D-237/258–265 |
| 协同 | `shared_coexist_no_synergy`（CN −0.039、P-B +0.023） | D-266/268 |
| 未测 | 开放中文/对话/生成、真机/多模态行动、跨会话迁移 | — |

**结论**：实证已换代（RT 机制解决了 S1 原失败点），但**未回灌 MAINLINE、未取得 owner 单卡 training_authority、未签 S1**；主线处于「S1 合同内、协议已实质换代」的过渡点。

## B. 问题清单

| 编号 | 问题 | 现状 | 关键读数 |
|---|---|---|---|
| P1 | 任务条件化容量分配/保护 | 已解（有界）：固定子群＋梯度掩码＋wd=0；capsule routes | S1-RTF 3 seed 全过 |
| P2 | 生长功能绑定 | 未解：中性/联合/残差/新任务 lesion<0.20 | 最深 0.0625，gain −0.117 |
| P3 | 协同（联合>单轴） | 未解：共享=折中、交替=干扰 | CN −0.039 / P-B −0.0625 |
| P4 | 跨任务/跨会话正迁移 | 未测（capsule 只证恢复，不证迁移） | — |
| P5 | 评测规范 | 单 seed 多、指标混用、缺 ACC/BWT/FWT | 过程披露 |
| P6 | 工程债：保护/路由在脚本不在内核 | fixed partition 为实验层 | 待 API 化 |

## C. 公开研究地图（可借鉴，附来源）

1. **持续学习总框架**：Wang et al., *A Comprehensive Survey of Continual Learning*（arXiv:2302.00487）——稳定性-可塑性、三类方法、标准指标；我们属参数隔离类。
2. **参数隔离（P1 源头）**：
   - HAT（Serrà 2018, arXiv:1801.01423）：**学习式硬注意力 mask＋梯度屏蔽**；我们固定分区+grad mask 是其简化版，可升级为累积 mask+稀疏正则自动分配。
   - PackNet / SupSup：剪枝/超掩码式容量分配（先释放容量再冻结）。
   - Progressive Networks（Rusu 2016）：**每任务新列＋横向连接、旧列冻结**——我们生长线缺的正是 lateral 连接与读出绑定。
3. **MoE（P3 协同）**：
   - Switch Transformer（arXiv:2101.03961）：top-1 路由＋负载均衡辅助损失。
   - DeepSeekMoE（arXiv:2401.06066）：**细粒度专家＋共享专家**——我们「共享 24 单元」缺 shared/routed 分工，导致折中。
   - MoE-PEFT 路由坍塌（检索到 SIMPA 2026）：单专家吸走 82.5% token，修在「输入视图」；对应我们的路由漂移/新单位被忽略。
4. **生长（P2）**：
   - Net2Net（arXiv:1511.05641）：**函数保持式加宽/加深**（复制/拆分），避免随机初始化；我们 grow 是零读出列，需 Net2Net 初始化＋蒸馏。
   - *The Neuron Addition Problem: A Survey on Growing Neural Networks*（HAL hal-05842579）：masked growth α=0 与 transient mask ramp；指出需 ramp/信用分配新单元才会被使用。
5. **合并/组合（P4）**：task arithmetic/model merging（Ilharco 2022、TIES 2023）、LoRA experts（Hu 2021；MoLE）——用于跨会话整合；需先标准化评测。
6. **记忆/状态**：Mamba/RWKV 与「固定状态 vs attention 记忆几何」讨论——与我们 per-sample 持久状态 vs 外部历史读出的取舍同题。

## D. 可落地改造（建议，未执行）

1. P1 升级：HAT 式学习 mask（累积+稀疏正则+容量上限）或 PackNet 式先剪枝；固定分区保留为下界基线。
2. P2 生长：PNN/Net2Net 路线——新列/新单元函数保持初始化＋旧列冻结＋横向连接＋蒸馏；读出显式混合旧/新，lesion 门不变。
3. P3 协同：DeepSeekMoE 式「共享专家＋细粒度 routed experts＋负载均衡」；记录路由分布防坍塌。
4. P5 评测：3 seeds、ACC/BWT/FWT、mean±std；单 seed 只作机制归因，不写能力结论。
5. P6 工程：保护/路由 API 化进核心（capsule 已有 routes），加「生长前后旧任务差 ≤1e-6」单测。

## E. 下一步（等 owner 裁定，非训练）
1. 首选：以 RT 机制重立 S1 判据并回灌 MAINLINE §3A（需 owner 单卡授权）；
2. 生长：按 PNN/Net2Net 新卡（1 seed 烟测 ≤300 CPU）；
3. 协同：按 DeepSeekMoE 共享+细粒度新卡；
4. 先标准化评测再谈能力签署。

## F. 声明
零训练、零 check/final；外部资料仅为映射，不代表已验证；不签模型成型/final/C/D/J。
