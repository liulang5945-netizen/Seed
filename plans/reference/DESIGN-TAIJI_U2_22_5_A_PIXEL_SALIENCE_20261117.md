# 22.5.0.A 读出身份 `pixel-salience`：设计卡（**不训练**，待定价与重冻）

> 卡号：`22.5.0.A`｜日期：2026-11-17｜状态：**`designed_unpriced`**｜性质：**设计＋接线合同**，0 训练、0 重冻、0 check、0 final。
> 依据：[读出上限复核卡](RESULT-TAIJI_U2_22_5_A_READOUT_CEILING_20261117.md) §7–§8（像素算子 2/3 种子 0.93；patch 级 0.45–0.64；oracle 重建 1.000）。

## 1 要装的是什么

`pixel-salience` 读出算子（≈400 参数的量级，原型实测）：

```
s_i  = <q(request), k(pixel_i)> / sqrt(d)        # 乘性匹配
w    = softmax([s_0..s_255, s_null])
f    = [μx, μy, mass, max_w, Σwx, Σwy]          # 线性坐标矩
read = MLP(6 → 32 → d)
```

- `q` 由 **request 字节自编码**（字符 embedding 256→text、位置参数、mean/peak 池化、`text→d`）；**不喂实体 one-hot**——one-hot 已被实测证否为无增益（D-20261117-48），且它是声明先验，不是主体自己的读。
- `k` 由像素的 3 个通道线性映射；像素坐标**不加新观测字段**：由观测自带的 16 个 patch 中心坐标（`coordinates["rgb"][:,:2]`）加已知的 4×4 子网格偏移 `(±3/32, ±1/32)` 展开，得到 256 个精确 x/y。子网格偏移是渲染器的固定几何，不是数据。
- 读出向量 `read` 与 `joint_readout([h,o,fast])` **相加**注入，注入前经 **零初始化** 的 `d→d` 投影 ⇒ 出生严格中性（见 §3）。

## 2 接线合同（必须动被钉文件，先列清楚）

| 位置 | 改动 | 是否改参数 |
|---|---|---|
| `taiji/u2_pixel_read.py`（新文件） | 算子本体 + `install_pixel_readout(model)` | 新增 6 个张量（byte/text/query/pixel/null/out） |
| `taiji/u2_subject.py` `__init__` | 按 `config.readout == "pixel-salience"` 调 `install_pixel_readout` | 新配置项 `readout: str = "patch"` |
| `taiji/u2_subject.py` `correct()` | 把**已 detach** 的 `observations["rgb"]` 与 `coordinates["rgb"]` 存进 `result` | 不改计算图、不改输出 |
| `taiji/u2_subject.py` `branch_context()` | `context = joint_readout(...) + out(pixel_read(state))` | 由 `readout` 开关控制 |

**出生中性**：`out` 的权重与偏置零初始化 ⇒ 任何继承 v4/v6/v7 权重的候选在新开关**关闭**时逐位复现父代；开关**打开**时首个前向的 `delta = 0`，logits 与父代逐位相同（§3 要求实测，不是声明）。

**身份**：`readout=pixel-salience` 作为**新版本身份**，不复用 `taiji-u2-subject-v4/v6/v7`；旧身份与其 bundle 全部不变。

## 3 出生中性的红/绿演练（接线的验收条件）

1. 绿：`readout="patch"`（开关关）的候选与父代 bundle 逐位相同（`value_digest` 相等）。
2. 绿：`readout="pixel-salience"` 但 `out` 零初始化时，`output()` 的 logits 与开关关逐位相同（Δ 严格 0.0）。
3. 红：把 `out` 的权重扰动 1e-3 后，Δ 必须非零——否则零初始化测试是装饰。
4. 绿：一次 `task_loss.backward()` 后，`out`／query／pixel 各有非零梯度（否则该通路不可学）。

## 4 判据（先声明，跑之前冻结）

- **算力**：以原型实测的"每例前向+反向 CPU"定预算；**不得沿用 256 更新**。原型在 768 世界上需要 ≈4.6M 例更新才到位，A 门现预算 512 例更新，差约 4 个数量级；若按 batch 8、每例 0.4 CPU 计，4.6M 例≈1.8M CPU，**明显超出可行性** ⇒ 因此本卡必须同时给出**降规模的方案**（见 §5），否则这条线在预算上不可执行。
- **fit**：3 体 fit ≥0.80，才算读出身份可用；未过不读 check。
- **保持**：A 任务本身不写记录，保持门按构造不可失败 ⇒ **不得引用为证据**（沿用既有纪律）。
- **种子**：3 种子，报中位与逐种子；任一种子若未拟合 train（<0.99），该臂记 `not_convergent`，不出判词。

## 5 必须先解决的两件事（本卡不解决，交给定价脚本）

1. **预算不可行**：4.6M 例更新是"专职头在 768 世界上收敛"的量，按 U2 单次学习 0.453 CPU 会到百万 CPU 量级。可行方向只有两条，且都要在定价里量出来：
   - **蒸馏**：先用已收敛的原型头在 train 上产出软标签/显著度图，U2 只学"复制该显著度读"（把 4.6M 例更新降成几千次监督更新，但引入"教师在体外"的口径问题，须在合同里写明这是**教师蒸馏**而非主体自读）；
   - **缩小世界数并把 fit 改成同分布**：768→192 世界、fit 同为 1–2 体，用"同分布 held-out"换更新数，代价是 fit 与 check 的口径要重写。
   两条都不许悄悄做：要先出定价卡与 owner 裁决。
2. **重冻范围**：`grep` 计到 **45 个 manifest** 引用 `taiji/u2_subject.py`。**池分解（`source_hashes`／`source_guards`／`exclusion_sources.*`／`construction.*`）必须在冻结时用四池扫描重算**——本次试扫用了 `k == "taiji/u2_subject.py"` 的形状、返回 0 命中，说明这些 manifest 用的是别的键形，**不得沿用该次扫描的结论**（记为一次未成形的扫描，避免以后误引）。

## 6 不签

不签 A 父、不签能力、不签 `pixel_read_feasible`（原型判词仍是 `not_convergent`）、不签 22.6／22.7、不解锁 22.5.1、不动 check（仍 1/2 used、1/2 remaining）、不读 final。

## 7 定价（已跑，`priced_no_training`）

[定价报告](../../reports/taiji_u2_22_5_a_pixel_salience_price_20261117.json)｜脚本 [price_a_pixel_salience_20261117.py](../../scripts/training/price_a_pixel_salience_20261117.py)｜**0 源文件改动**（实例级 `add_module` ＋绑定方法替换 `branch_context`）。

| 检查 | 读数 | 判定 |
|---|---|---|
| 出生中性（零初始化） | 首步 log 概率 **Δ = 0.0**（支撑集差异 0 项） | 绿 |
| 红方向（`out.weight` +1e-3） | **Δ = 7.224e-4** | 红，测试能响 |
| 新张量梯度（扰动后一次 backward） | 11 个张量**全部非零** | 绿 |
| 继承参数 | 逐位未变 | 绿 |

**成本**：每例前向+反向 **0.961 CPU（开）对 0.992（关）**，增量落在噪声内 ⇒ **算子本身几乎免费，墙不在算力速率**。

**墙在更新数**：按原型所需 ≈4.6M 例更新折算 **≈4.42M CPU**；A 门现配额 512 例更新 ≈ **492 CPU**，**差 ≈8,984 倍**。所以"把算子接进去再训 256 更新"注定只会得到"算子对了但预算不够"的负结果——这正是本卡要求先定价的原因。

**结论**：接线在技术上已无障碍（中性、可学、免费），唯一未决的是**预算口径**：要么走教师蒸馏（把 4.6M 例更新换成几千次监督更新，但合同须写明"教师在体外"），要么缩小世界数并重写 fit 口径。两条都要 owner 裁决。

## 8 四池重冻扫描（已跑，只读）

[扫描报告](../../reports/taiji_u2_22_5_a_refreeze_scan_20261117.json)｜脚本 [scan_refreeze_scope_20261117.py](../../scripts/training/scan_refreeze_scope_20261117.py)｜0 文件改动、0 训练。

**先说结论：本仓的 manifest 里没有 `source_hashes`／`source_guards` 这两个池**（此前记为"四池"的词汇来自另一代合同）。实际形状是：

| 池（顶层键） | 命中 manifest | 说明 |
|---|---:|---|
| `source_pins` | **35**（**55** 条摘要项） | 真正的钉：模块路径为键、sha256 为值 |
| 文本提及（`coverage/*/planned_source` 等） | 6 | 不是钉，不会因改文件失效 |
| 其他嵌套位置 | 11 | 未归类，冻结前需逐个看 |

- **55 条摘要项全部与当前文件不匹配**（历史 blob 口径）。⇒ 只有走"历史字节"校验的通道能容忍；按**当前值**校验的通道（`first_a_support.load_learner` 就是这种）会在文件一改就抛 `learner_source_identity_mismatch`。
- ⇒ **接线的真实成本不是"改一个文件加一个开关"，而是：1 个文件 + 35 个 manifest + 所有已存 learner bundle 的加载路径**（bundle 里的 `source_pins` 由 `_source_pins()` 生成，同样钉这个文件）。
- 因此接线必须与"bundle 兼容策略"同时定：要么新身份下**不加载旧 bundle**（旧 bundle 只作归档、不再 resume），要么给加载器加一条"新身份声明"通道。这条要先定，否则接线当晚就会把 22.5.0 之前所有可恢复运行变成不可恢复。

**本卡仍不授权任何训练。** 下一步是定 bundle 兼容策略 ＋ 预算口径（蒸馏／缩规模）二选一，两件都要 owner 拍。

## 9 学习曲线：0.80 在 **1e6 例更新** 处跨过（4.6M 高估 4.6 倍）

[曲线报告](../../reports/taiji_u2_22_5_a_gate_learning_curve_20261117.json)｜脚本 [probe_a_gate_learning_curve_20261117.py](../../scripts/training/probe_a_gate_learning_curve_20261117.py)｜0 U2 训练、1,649.5 CPU。

| 例更新 | 中位 train | 中位 fit3 |
|---:|---:|---:|
| 1e5 | 0.522 | 0.375 |
| 3e5 | 0.939 | 0.776 |
| **1e6** | 0.986 | **0.9271** |
| 3e6 | 0.987 | 0.9323 |

- 跨越点在 **1e6**，3e6 只多 0.005 ⇒ **4.6M 是把"6000 步收敛"直接折算造成的 4.6 倍高估**。
- ⇒ 读出规模定为 **≈1e6 例更新**（≈1300 全批步，3 种子）。按每例 0.961 CPU 计约 **96 万 CPU**，A 门现配额 512 例（≈492 CPU）仍差约 2,000 倍，但这是**实测断点**，不再是粗估。
- **预算已由 owner 解除限制**（2026-11-17）；**「蒸馏」已撤回**——它与"主体自己读"冲突，不是预算问题。**「缩规模」不再需要**，768 世界 + 3 体 fit/check 的口径保持不变。

**本卡仍不授权训练**：下一步是出生式接线（`birth_units` 记账，不是 `add_module` 外挂）＋ 55 条 `source_pins` 的兼容处理，接线通过中性/梯度演练后才开训。
