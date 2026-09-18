# R2-D2 修订二：H-A3 问题条件化回答起点（graph v5，O1，冻结版）

2026-09-18。状态：**已冻结**（用户裁决 O1 后提交即冻结）。
依据：[首字节升级备忘](M5_R2_D2_FIRST_BYTE_ESCALATION_MEMO_20260918.md) §4、[H-A2 copy 修订](M5_R2_D2_COPY_MIXTURE_AMENDMENT_FROZEN_20260918.md) §6。本修订是合同允许的最后一次同假设族自主图修订：**v5 probe 再失败 ⇒ 无第三次，A 族路线交回用户全面评审**。

## §1 修订点（仅一个变量）

renderer 起点从 prefix 无关学习常量改为**问题干条件化**：

```
h_question = prefix 因果扫描在「材料标记之前一个 byte」的状态
r_0 = tanh(h_question · answer_start_weight + answer_start_bias)
```

- 材料标记（`背景：/线索：/已知：`）在 byte 流中最早出现处切分；h_question 不含标记与材料（RNN 因果性保证材料无法经 r_0 旁路）。无标记的合成输入回退为 h0（仅测试用）。
- 新增参数 `answer_start_weight (rw,rw)=4,096`、`answer_start_bias (rw,)=64`，共 **4,160**；v5 臂总参数 **86,818**。
- 配置 `question_conditioned_start: bool=False`（默认关 ⇒ 图 v4 逐位不变）；仅允许 evidence 臂（per_position/broadcast）开启；v2 final_state_slots 清单与计数不变。
- checkpoint 版本 4→**5**，载荷含该标志；v2/v3/v4 payload 全部继续可读；缺省即关。
- copy 混合、证据来源、目标、lesion 与 D1 评测口径全部不变。

## §2 臂与门（其余沿用 H-A2 修订，不重述）

| 臂 | 图 | 参数 |
|---|---|---:|
| **A5** | per_position + copy + question start | 86,818 |
| **C5** | broadcast_final + copy + question start（等机制无逐位置信息） | 86,818 |
| A4 锚点 | per_position + copy + 常量 start（=H-A2 已跑臂，留档） | 82,658 |

**新增实现门（零训练）：**
1. 同材料不同问题干 ⇒ r_0 不同；同问题干不同材料 ⇒ r_0 逐位相同（r_0 不含材料）。
2. 同题干下材料被替换时，首步分布的变化**只能**来自读出：zero_read（含去 copy）后两条 prefix 的首步混合分布逐位一致。
3. h_question 切分位置对三 split 标记正确（合成用例）；无标记回退 h0。
4. 清单 86,818、梯度到达 answer_start_weight/bias、v5/v4/v3/v2 恢复矩阵、copy-off 仍逐位等于 v3。

**probe（1 运行：A5，seed 20260917，30 epochs，lr 0.01，wall 20 min，preflight 先行）：** 门不变——copy-supported（fact/negation/sof，144 题）train M1 **≥0.90**，loss 有限下降；M3/M4 与 uncopyable 形状描述。**对本修订的核心观察点：negation 首字节「不」必须进入贪心输出（negation M1>0）。**

**matched dev（probe 过后才启动）：** 3 seeds×{A5,C5}（A4 仅在需要时作 copy-off 锚点补跑），G1–G6 阈值与 H-A2 修订完全相同（M3≥0.26/δ≥0.20；M4≥0.50/δ≥0.25、fact_flip≥2/6；copy-misbind ΔM4≥0.50；上下文 ΔM3≥0.30；seed；boundary/恢复）。

## §3 预承诺

- v5 probe 不过（含 negation 仍为 0 或 M1<0.90）⇒ **停止 A 族全部自主图修订**，冻结诊断，由用户对「表达容量 / 课程更新方式 / B 绑定 / 规模」做全面评审；不得改门、加权数据、加 epoch。
- 不读 final；不主张 L2；训练仅用 D1 train split；所有产物 `growth_admitted=false`、`can_promote=false`。
