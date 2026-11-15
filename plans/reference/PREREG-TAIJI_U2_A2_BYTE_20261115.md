# A.2 预注册：把冻结的四类读写成自然中文 6 字节

> 卡号：`22.5.A.2`｜日期：2026-11-15｜状态：**预注册与定价已冻结，训练未授权**。
> 契约：[`taiji_u2_a2_byte_20261115.json`](../manifests/taiji_u2_a2_byte_20261115.json)（`criteria_sha256=9b12d593…`）。
> 前置：[A.1b 结果卡](RESULT-TAIJI_U2_A1B_COLOUR_20261115.md)、[A.1b-cal 结果卡](RESULT-TAIJI_U2_A1B_CAL_20261115.md)。
> 零更新 smoke：[报告](../../reports/taiji_u2_a2_byte_smoke_20261115.json)；定价：[报告](../../reports/taiji_u2_a2_byte_price_20261115.json)。

## 1 这张卡问什么

A.1b-cal 用**零训练先验校正**把读从 fit 25/32 抬到 **32/32**（train 61/64，交叉半最低 0.844，后验）。
读已经能给出“左边／中间／右边／未知”，但 P4.2 只测过“读没有答案时字节头会不会说”，从未公平测过“读有答案时字节头能不能说”。

A.2 只问一件事：**冻结这个读，把它的四类答案写进主体自己的字节发射器官，主体能不能把答案写成自然中文 6 字节？**

这是**表达正对照**，不是内容条件化绑定测试：答案类别由冻结的读提供，码里不含场景像素。通过只说明“字节通路能写出已知答案”，不说明“主体能从场景里读出答案”。

## 2 冻结的读

- 主体：P4.1 B 编码器（`reconstruct-B-007376.pt`，`f6705088…`），22,742 标量／194 张量，**全部冻结**。
- 颜色头：A.1b 的 263 参数逐列 7 类头（`a1b-00200.pt`，`3ebac8b3…`），**冻结**。
- 读法（非学习）：`argmax(logits − log(train64 逐列类别先验))` → 取被判为请求颜色的列 → 列均值 x → 左/中/右/未知。
- 先验：只用 train64 列标签，`[0.0234375, 0.0302734375, 0.015625, 0.0146484375, 0.0146484375, 0.021484375, 0.8798828125]`；运行器重算并逐值核对（`atol=1e-6`）。
- 读的读数在 A.2 中**只记录、不设门**：fit 32/32、train 61/64。

## 3 read-code-v1：读怎么进入字节通路

- 四类答案按固定顺序 `("左边","中间","右边","未知")` 编码为 **24 维定长码**：`zeros(24); code[class] = 1.0`；两个分支收到同一份码。
- 注入点：`joint_readout`（主体自己的读出头，`3*d = 24 → 8`）。训练、评测、反馈三条路径都用同一个前向钩子把 `joint_readout(x)` 替换为 `joint_readout(code)`。
- **没有新参数编码读**；码是常量，不参与训练。
- 码宽由 `3 * d` 校验（当前 d=8）。

## 4 只训什么

`joint_readout`、`byte_head`、`copy_gate` 共 **8 张量／3,919 标量**；主体其余参数（含编码器、binding、native 递推、解码器）全部冻结，梯度不得到达。
声明的 8 个张量：

```
byte_head.bias, byte_head.weight, copy_gate.bias, copy_gate.weight,
joint_readout.first.bias, joint_readout.first.weight,
joint_readout.last.bias, joint_readout.last.weight
```

### 4.1 读数解读注意（开跑前的结构事实）

本任务的 448 个缓存位置**全部没有可用记录**：`valid` 形状 `(0,0)`、`valid_records_total=0`、`alpha[:, -1]=0`（最后一路路由质量为 1、`q=1`）。在这组条件下：

- 复制项不存在，`log_gen = log(gamma + (1-gamma)·q) + log_generator = log_generator`，**gamma 被消掉**；
- `copy_gate` 的梯度**恰好为 0**（零步 census 逐张量记录见运行报告），它会停在初始值；
- 因此 A.2 实际检验的是 **`joint_readout` + `byte_head` 的生成通路**能否写出答案，`copy_gate` 只是声明在可训集合里、结构上不起作用。

不得把 gamma、1−gamma 或“复制门被训练”写成机制证据；报告必须带零步梯度 census 与 `copy_gate_inert=true`。

## 5 数据与剂量

- 数据：`first_a/20261112T074836Z/data.bin`（digest `40813b4c…`），train 64、fit 32；**check／final／confirm 不读**。
- 训练 600 更新，batch 8，Adam lr 0.01（betas 0.9/0.999，eps 1e-8，wd 0）；剂量点 200／400／600。
- 上下文缓存：与 P4.2 同一模式——冻结主体自己传给 `emission` 的实参（cursor/alpha/beta/ids/valid/mask，外加 goal、previous、x），共 448 个位置；训练前必须通过 3 个位置的**回放一致性**检查（`body_log_probability` 与缓存重算差 ≤1e-6）。
- 落盘：`initial.pt`、`engineering-step.pt`（1 次工程更新，跑在副本上）、每个剂量点 `a2-00N.pt`、`final.pt`、`feedback-learned.pt`；先存包再做评估。

## 6 门（数值在读数前冻结）

| 门 | 定义 |
|---|---|
| 主门 | 全 32 fit 轮**精确应答**（答案字节 + EOS + 合法 UTF-8，贪心，65 字节帽）≥ **0.80**；失败/多字节/非法轮计入分母，不得筛除 |
| 纠正 | 首轮错例逐个释放正确答案、按反馈 lr 做 **8** 次 Adam，再整体重问一次；纠正率（重问后变对的错例／首轮错例）≥ **0.80** |
| 反馈增益 | 学到副本第二遍 fit 正确率 − 对照副本第二遍正确率 ≥ **0.05** |
| 稳定 | 首轮正确、反馈后变错的题数 ≤ `max(1, 10% × 首轮正确数)` |
| 对照 | 同一训练后主体、同一重问，**0 参数更新、0 状态写入**；前后状态摘要逐位相同 |
| 恢复 | 保存训练后 bundle，独立进程重载，重问 32 轮；输出**逐字节一致** |

不可识别规则（沿用 P4.2）：首轮错例 < 4 时，纠正率与反馈增益记为 `not_identifiable_by_design`，不记通过也不记失败。

## 7 出口

- `byte_expression_confirmed`：主门 + 纠正 + 增益 + 稳定 + 恢复全部满足。
- `byte_expression_confirmed_feedback_unmeasured`：主门 + 稳定 + 恢复满足，错例 < 4 使反馈两项不可识别。
- `byte_expression_partial`：主门满足，但纠正／增益／稳定／恢复有任一项不满足。
- `byte_expression_below_gate`：主门不满足。
- `budget_stop`：到达 CPU／wall 帽，保留全部 checkpoint 与最后剂量读数。
- `engineering_failure`：缓存回放不一致、非有限梯度、身份 pin 失配、checkpoint 丢失等。

不设“再加剂量／换读出”的自动续跑；任一门不过即按出口结案并回报。

## 8 预算

- 设计帽 **600 CPU／1,200 wall**。
- 实测定价（5 次内存工程更新 + 1 次 fit 与 train 评估）：曲线 **20.794 CPU**，采用帽 **22.873 CPU／45.746 wall**（曲线 ×1.1，远低于设计帽）；定价本身 7.141 CPU、0 checkpoint、0 正式训练更新。
- 采用帽只许下调，不得上调；越帽按 `budget_stop` 结案。

## 9 能主张与不能主张

**能（若通过）**：在冻结的 P4.1 B 主体与 A.1b-cal 读下，只训 8 张量／3,919 标量，字节发射器官能把四类答案写成精确的自然中文 6 字节；缓存回放、独立进程恢复、反馈纠正与 0 更新对照均按预注册计数。

**不能**：不能写“A 父成立”或 `accepted_parent=true`（单 seed、check② 未读、A 保持不适用）；不能写“主体从场景读出答案”（码是读的答案类别，不是像素）；不能把 fit 32/32 写成独立确认（它是 A.1b-cal 的后验读数，A.2 的新证据是字节读数）；不能把 0.80 写成能力上限或跨任务泛化。

## 10 授权状态

实现、零更新 smoke 与定价已完成；**正式训练未授权**。契约 `status=price_frozen_authorization_pending`、`training_authority=null`，运行器的 `run` 模式在冻结前拒绝开跑。按当前指令，不申请放行、不写账本、不跑训练。
