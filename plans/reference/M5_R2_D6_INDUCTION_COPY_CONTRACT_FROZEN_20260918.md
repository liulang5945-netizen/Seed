# M5 R2-D6：induction 持续 copy 假设包（发射匹配偏置，冻结版）

2026-09-18。状态：**已冻结**（用户在 [D5 结项评审](M5_R2_D5_MATCHED_DEV_REVIEW_MEMO_20260918.md) 裁决「B 槽位包/绑定方向」后提交即冻结）。前置：D5 双判否收敛——D4 驻留（v7）失败根因结构化为「bonus 条件于模型自身上一步寻址命中（self-confirming），首步 miss 即链条断裂，且无覆盖压力时不学习」；而字节域绑定的强先验是**实际发射字节与历史字节的字面匹配**（emitted 匹配在 teacher-forcing 下由真值确认，在自由生成下由自身输出确认，二者一致时形成 induction 链）。D5 同时确认：覆盖（v2 train）使多字节持续 copy 在 train 完全可学（72/72）但摧毁 v6 的 dev 泛化（0.178→0.008）——位置模式记忆压过了序列规律。

## §1 假设 H-B（冻结）

> 在字节级 copy 寻址中引入 **induction 偏置**：回答第 t 步，对每个可见 entry 行 j（j≥1），若 `entry_bytes[j-1] == 上一步发射字节 s_{t-1}`，则**行 j**（匹配位置的后继）的寻址得分获得可训练加性 bonus `beta`（所有头共享、softmax 前）。该偏置把「复制值词首字节后链条断裂」变成「沿字面匹配的行序列持续复制」：值词 {b1,b2,…} 在材料中是连续行，第一步经既有问题寻址锁定 b1 行（D4/D5：首字节正确率 ~50%），此后每步由「前一字节等于刚发射字节」的匹配自动指向 b2、b3…直到值末。配合 D5 已证的覆盖数据（train 含多字节值），模型应学到正的 beta 并在 v2 train 上把双字值完整复制；若机制为真，dev 上双字节值 full_rate 应从 ≈0 显著上升且覆盖数据的有害性（位置模式记忆）被先验替代。
>
> **规格修订（冻结前，2026-09-18）**：首稿把 bonus 目标写成匹配行自身（复读 s_{t-1} 而非其后继），实现评审时纠正为**匹配行的下一行**——induction 的语义是读取延续符号。合同随此修订冻结。

**与「对象×值槽位」的关系（诚实声明）**：本包实现的是 B 绑定族在字节域的最小充分形式（行序列 induction 链 = 值段绑定的逐字节读法），不是显式 key-value 槽位表。若 H-B 判否，显式槽位表（对象池→值段写读）是下一候选假设，本包不混做。

## §2 graph v8 精确规格（冻结）

1. 在 v7 代码族上新增配置标志 `copy_induction: bool = False` 与参数 `copy_induce_bias`（标量，初值 **0.0**，+1 参数：96,034→96,035；canonical 清单**追加表尾**——D5 §4 教训制度化）。版本 7→**8**，旧版全可读。
2. 注入点（唯一）：`_head_weights` 计算 scores 后、softmax 前——`scores[:, J] += beta`，其中 `J = { j≥1 : entry_bytes[j-1] == s_{t-1} }`（匹配位置的**后继行**；与 §1 修订一致），`s_{t-1}` 为该步的输入符号 previous_symbol（teacher-forced=真实前一字节；自由生成=上一发射字节；episode 首步 previous=boundary，boundary 不在 entry 字节空间 → 自然无 bonus）。
3. 与 v7 驻留正交共存：两标志独立；本包实验臂只开 induction 关 v7 驻留（v7 bias 冻结在 0 参数路径=关闭）。内容读出与 copy 分布共享同一 weights（与 v6/v7 一致）。
4. 单测可注入：`_head_weights(..., induce_from=int)` 允许显式指定匹配字节（默认 None=从该步 previous_symbol 取）。
5. 目标沿用 D4：λ=1.0 copy 值监督 + 答案 CE；数据臂：v2 train（主臂）与 v1 train（回归/γ 读数臂）；训练微批 8、固定顺序、30 epochs、lr 0.01；seed 20260917（probe）/ 三 seeds（matched）。

## §3 实现门（零训练，含 D5 教训条款）

1. 标志关闭或 beta=0：与 v6 **逐位一致**（logits/混合/生成/损失）。
2. **历史 checkpoint 重放**：λ=1、induction-off 路径重放 D4 冻结 checkpoint 记录 loss（0.008076…，实现门直接断言，不许留到 matched 才抓）。
3. 匹配集结构：构造含重复字节的前缀，断言 `J` = 全部字面匹配行；boundary 输入 → J=∅；beta 设大值时下一步各头 argmax ∈ J（命中行自身，若其后有合法 continuation 行则按既有寻址）。
4. beta 梯度：数值差分；induction-off 参数清单不含新名；v8/v7/v6/…/v2 恢复矩阵 + fresh preflight。
5. 冻结态回归：v7 flag-off、v6、v5 在 canonical 追加表尾后 init 不变（与既有报告重放一致）。

## §4 probe（train-only）

- **B2**（v2 train + v8 induction + λ=1）：D5-A01 同集全门（多字节 72/72、单字节回归、copy 概率、错位、稳定）**+ 新描述读数：beta 终值与方向**。判据附加：beta 终值 >0 登记为「机制使用」证据（不设门）；beta≤0 且 D5 门过 = 模型不需要 induction 也能拟合 train（信息保留）。
- **B1**（v1 train + v8 induction + λ=1）：D4 全回归门 + beta 读数（v1 无多字节压力，beta 符号预示 induction 在单字节值上的干扰/收益方向）。
- probe 门失败 → 该臂不进 matched，按 §5 路由。

## §5 matched dev 与失败路由（预承诺）

- 主臂 B2×3 seeds；对照 A00=D4 T2 历史（v1，漂移复核后引用）+ D5-A01 历史（v2 无 induction，漂移复核后引用）。G1–G6 沿用，δ 基准=A00；**关键判据**：
  - K1 dev 双字节值 full_rate ≥0.30（D4/D5 全 ≈0 的精确目标）；
  - K2 dev M4_flip ≥0.50（绑定读数）；
  - K3 B2 的 M3 > A01（v2 历史 0.008）×10（覆盖有害性被 induction 修复）。
- 失败路由：
  | 结果 | 路由 |
  |---|---|
  | G1–G6+K1 过 | H-B 支持 → 进入包评审（是否集成/扩槽位表） |
  | K1 过、K2/G2 败 | 持续 copy 修复但绑定精度不足 → 显式槽位表候选（H-B2'）升级裁决 |
  | K1 不过、K3 过（induction 救回 v2 但仍截断） | 字面匹配先验不足以产生组合 → **字节图族判否**，阶段评审（token 化/收束） |
  | K1、K3 全不过 | 同上，且覆盖交互证伪 → 阶段评审 |
  | probe 失败（train 崩） | induction 与既有寻址冲突 → 回实现门排查，不放预算 |
- beta 符号与大小全程单列（机制使用读数）；combo 线 C/B 信号继续记录。

## §6 边界与预算

不读 final；仪器不动；`growth_admitted=false`、`can_promote=false`。probe 2 运行 + matched 3 新臂运行 + 2 漂移复核 + 3 seeds B2 ≈ 9 次训练（<20 min CPU）；checkpoint 隔离 reports/r2_d6_checkpoints/。本包关闭 induction 变体线（beta 形状/匹配窗口调整=变体，不追加）。
