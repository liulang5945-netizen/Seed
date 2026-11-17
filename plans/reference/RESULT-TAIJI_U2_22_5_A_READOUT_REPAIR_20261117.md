# 22.5.0.A fit-only 读出修复：EOS 偏置已修，首 token 选择仍恒定

> 卡号：`22.5.0.A`｜日期：2026-11-17｜状态：**`readout_partial_eos_fixed_first_token_constant`**｜性质：**fit-only 读出实验**，0 check、0 final、无正式 A 门。
> 报告：[v1](../../reports/taiji_u2_22_5_a_readout_repair_20261117.json)、[v2 首 token 加权](../../reports/taiji_u2_22_5_a_readout_repair_weighted_20261117.json)。
> 授权：owner 2026-11-17 指示「按建议执行 fit-only 读出修复，检查 EOS 偏置和 goal/emission 路径」。

## 1 EOS 偏置：已定位并修复

- 冒烟模型的 teacher-forcing 首字节：目标字节概率 **-1.5844**，EOS **-1.4088**，argmax=EOS ⇒ 推理立即空串＋EOS。
- 在**首字节屏蔽 EOS** 后（训练与解码同一约束），输出不再是空串，开始发射内容；EOS 偏置是"空输出"的直接原因。
- 结论：**EOS 偏置真实存在，且可通过首字节 EOS mask 修复**；不是 UTF-8/EOS 缺失/恢复问题。

## 2 goal/emission 路径：可发射，但首 token 不选类

- teacher-forcing 下，目标内容字节在各位置的概率约 **-1.5 到 -2.0**，说明 goal/request/emission 路径能把内容送到字节头。
- 但自由解码的首 token 在**所有** train/fit 样本上都是 **"左边"**；后续字节随后正常发射并 EOS。
- 两组只改首 token 监督的 fit-only 实验：

| 实验 | 首 token 权重 | 更新 | train | fit | 输出 |
|---|---:|---:|---:|---:|---|
| v1 | 1.0 | 512 | **16/64 = 0.25** | **8/32 = 0.25** | 全 "左边" |
| v2 | 7.0 | 512 | **16/64 = 0.25** | **8/32 = 0.25** | 全 "左边" |

- 两组都正好 25%（4 类机会线）；v2 的 loss 更高（首 token 权重放大），但选择不变。
- 结论：**首 token 类别没有从观测/位置状态中被选出**；不是 EOS 偏置单独造成，也不是简单加权能修。与 U/V 线"值侧可读、选择/查询不足"的定位一致。

## 3 成本与纪律

- v1：**2,784.78 CPU**；v2：**2,886.59 CPU**；均只跑 train/fit，**0 check、0 final、0 正式 A 门**。
- check 曝光保持 **1/2 used、1/2 remaining**；未因这次修复增加。
- 没有改任何门值、剂量、check 曝光或 final seal。

## 4 归因与边界

- 已排除：EOS 缺失、UTF-8、恢复、空输出解码。
- 已确认：发射路径能把内容字节送出去；首 token 类别选择是瓶颈。
- 不能写成"U2 架构证否"；这是**读出/选择机制**缺口，和 U/V 的值侧/选择缺口同族。
- 需要的是**首 token 选择头/查询适配器**：从观测的位置/值状态读出 4 类（左/中/右/未知）并选择首 token，而不是继续调 loss 权重。

## 5 唯一后继

设计 **首 token 选择头**（新 U2 读出身份/版本 + 重冻），先在 fit 上验证选择准确率 >0.80，再考虑用剩余 1/2 check 跑正式 3-seed A 门。**在 fit 选择头有信号前**：0 check、0 final、0 A 父、0 B；22.5.1／22.6／22.7 锁定。
