# M5 CAP-0 整模型清点结果：默认入口服务未训练基底，训练态无法加载

日期：2026-09-15。状态：**已执行（只读清点段）**；基线段**未执行**（须先冻结评价集）。
依据：[07 整模型能力评价与最小用户验收版本](../active/roadmap/07_MINI_MODEL_DELIVERY.md) §3.A / §4.1 / §5.1 / §7。
产物：[清点脚本](../../scripts/training/eval_taiji_cap0_inventory.py) /
[报告](../../reports/taiji_cap0_inventory_20260915.json) /
[契约测试](../../tests/taiji_native/test_cap0_inventory_contract.py)。

## §1 一句话结论

**能加载、能出字、零报错——但它不是训练过的模型。**
默认入口 `SeedRuntime.load()` 加载的 `checkpoints/seed_corpus.pt` 处于 **`tick=2`**（API 运行时保存的刚初始化基底），
对话输出是**同一个固定模板**（7/7 轮只有 1 个签名，仅把提问回显进去）；
而**真正训练过的检查点无法被这条链路加载**（`seed_beta.pt` 16,000,000 ticks 报
`ValueError: enabled identity organ checkpoint payload is missing`）。
⇒ **接线缺陷与训练态搁浅均已确认**；整模型对话能力**仍属"未测"**，不是"失败"——
因为没有任何训练态能被服务出来。

## §2 检查点清点（只读读取信封元数据）

| 文件 | 大小 | `metadata.tick` | `trainer` | 保存时间 | 默认 | 可加载 |
|---|---|---|---|---|---|---|
| **`seed_corpus.pt`** | 43.2 MB | **2** | `api_seed_runtime` | 2026-09-14T19:36Z | **是** | **是**（但未训练） |
| `seed_beta.pt` | 4.1 MB | **16,000,000** | `train_seed_corpus` | 2026-08-24T02:23Z | 否 | **否** |
| `resumed_seed_corpus.pt` | 4.1 MB | **4,800,200** | `resume_checkpoint` | 2026-08-23T12:57Z | 否 | 否（同族） |
| `seed_corpus_prev_20260823.pt` | 4.1 MB | — | — | — | 否 | 否（旧格式） |

结构差异：三个旧文件**不含当前格式的 `taiji` 组件块**（只有 `config`/`format`/`metadata`/`substrate`），
只有 `seed_corpus.pt` 带 `taiji.components`。
`identity_organ_enabled` 在 `taiji/config.py:222` **默认为 `True`**，
而 `taiji/model.py:2726-2732` 对"活模型有身份器官、检查点无该载荷"**fail-closed 拒绝**（不静默补默认）。
该严格性本身是正确设计，但**没有任何迁移路径**，于是旧格式的训练态被永久搁浅。

## §3 三段事实（必须分开读，不得互相代替）

| # | 事实 | 证据 | 性质 |
|---|---|---|---|
| 1 | **接线缺陷** | 默认 tick=2 vs 训练态 tick=16,000,000；`wiring_defect=true` | **已确认**，可修 |
| 2 | **训练态搁浅** | `seed_beta.pt` 加载抛 `ValueError: enabled identity organ checkpoint payload is missing`；旧文件无 `taiji` 组件块 | **已确认**，需迁移/重导出决策 |
| 3 | **整模型对话能力** | 诊断探针 7 轮、0 报错、固定模板 | **未测**（无训练态可服务） |

## §4 原始输出证据（逐字，不评分）

默认入口、`learn=False`（07 §4.1 要求测试时不训练），7 轮全部回答、0 报错、共 1015 字节：

| 轮 | 提问 | 原始输出 |
|---|---|---|
| identity | 请用一句话说明你是谁、由什么运行。 | `我已收到你的问题："请用一句话说明你是谁、由什么运行。"。当前原生语言表层正在形成稳定表达。` |
| greeting | 你好。 | `我已收到你的问题："你好。"。当前原生语言表层正在形成稳定表达。` |
| knowledge_short | 水的沸点在标准大气压下的多少摄氏度？ | `我已收到你的问题："水的沸点在标准大气压下是多少摄氏度？"。当前原生语言表层正在形成稳定表达。` |
| context_a / context_b | （记名称 / 问名称） | 同上模板，**未保留也未复述上下文** |
| unknown_boundary | 请举一个你确实无法回答的问题…… | 同上模板 |
| material_qa | 材料：……B2 的结果是什么？ | 同上模板 |

**模板签名检测**：把提问替换为 `<PROMPT>` 后，7 轮收敛到**唯一 1 个签名** ⇒ `templated=true`。
按 07 §3.B「**固定话术不计内容能力**」，这不是"对话弱"，而是"没有内容能力"。

## §5 维度结论（未测记未测，不记 0、不记通过）

| 维度 | 状态 |
|---|---|
| A 模型真实性 | **已测**：加载链、模式（`native-readable`）、缺检查点拒绝（`FileNotFoundError`，确定性）、tick 来源 |
| B 基本对话 / C 基本问答 / D 上下文 / E 指令推理 | **未测**（评价集尚未冻结） |
| F 项目代表能力 | **部分**：B1 表示门 `representation_discriminative`、B2 选择门**负结果**——都是选择器面结果，**不等于**整模型对话能力 |
| G 不确定性与安全 | **未测** |
| H 性能与稳定性 | **未测**（尚无标定上限） |
| I 持续适应 | **未测**（首次 L3 交付非必达） |

**外部 provider 未使用**（`external_provider_used=false`）；全程新进程、未训练、未写检查点。

## §6 与 07 §7 缺口归属表对应

07 §7 第一行：「**无统一加载/推理链** → M2 基础合同＋M5 集成」，并明令
"**仓库有模块就算模型存在**"不是可接受的理由。本轮实测把该缺口从"担心"变成"证据"：
加载链存在且可用，但**它服务的对象不是训练态**。

## §7 界限：本轮**不**能主张什么

- **不能**说"Taiji 不会对话"：没有训练态被服务过，那是**未测**，不是能力结论。
- **不能**说"训练无效"：`seed_beta.pt` 的 16M ticks 从未被成功加载与评测，其能力未知。
- **不能**把诊断探针当作 B～H 基线：07 §4.1 要求评价集**在看到候选成绩前**冻结，
  本探针是**清点用**，已在报告 `probe_kind` 字段显式标注，**不评分、不设阈值**。
- **不能**把 B1/B2 的选择器分数折算成整模型能力（07 §7 第三行）。
- **不能**在本轮顺手改加载策略：那涉及"fail-closed 拒绝 vs 迁移"的安全语义，属决策。

## §8 下一步（**需决策**）

按顺序两件：

1. **默认检查点 / 加载策略处置**（缺陷修复，非研究问题）。三选一，各有取舍：
   - **(a) 迁移**：为旧格式补一条显式迁移路径（缺身份器官载荷时按既定语义补齐），
     代价是**放宽当前 fail-closed**，必须写明"哪些缺失可补、哪些仍拒绝"；
   - **(b) 重导出**：把训练态在**当前格式**下重新保存（不放松加载器），代价是需要一次可信的重建/续存流程；
   - **(c) 明示降级**：保持加载器严格，改为在产品入口**显式声明"当前无可用训练态"**，
     并把语言训练排期——最诚实但最短时间内仍不可对话。
2. **冻结 CAP 评价集**（07 §4.1：B/C/D/E 各 20 项、G ≥ 20、A/H 确定性、F 用既有能力合同），
   冻结后才跑首次**计分** B～H 基线。

**本轮不启动训练、不改加载器、不改 binder、不做产品采用。**

## §9 附：搁浅缺陷的精确定位（决策所需证据）

报告已把下列事实落为字段（`model_reality.default_model_format` / `most_trained_model_format` /
`stranded_is_legacy_format` / `stranded_defect_diagnosis`）：

| 事实 | 值 |
|---|---|
| 默认 `seed_corpus.pt` 的**模型格式**（读 `substrate.format`） | `taiji-native-v10`（当前格式） |
| `seed_beta.pt` / `resumed_seed_corpus.pt` / `…prev_20260823.pt` | **`taiji-native-v8`**（legacy） |
| `taiji/model.py` 是否支持 v8 | **是**：`LEGACY_CHECKPOINT_FORMATS = {v8, v9}`，且 `is_legacy_checkpoint` 在 **9 处**被使用（2611/2623/2637/2652/2669/2682/2708/2790/2856） |

**注意一个易错点**：模型格式在 **`substrate`** 键下；同级的 `taiji` 键是 Seed 适配层，带**另一条版本轴**
（`seed_corpus.pt` 的 `taiji.format = taiji-native-v1`，而 `substrate.format = taiji-native-v10`）。
读错键会把格式报错——本报告第一版就踩了这个坑并已修正。

**缺陷形态（与同文件既有模式对照）**：

```python
# 2611 行：已有的正确 legacy 模式（并附 M2-2h 迁移说明）
if predictive_context_payload is None and not is_legacy_checkpoint:
    raise ValueError("v10 checkpoint is missing its predictive context")

# 2726-2732 行：身份器官分支漏了同一个守卫
else:
    if not isinstance(identity_payload, Mapping):
        raise ValueError("enabled identity organ checkpoint payload is missing")
```

⇒ 该分支对**任何**缺载荷的检查点都拒绝，**包括身份器官尚不存在的 v8 文件**。
修法即沿用文件内既有模式（`and not is_legacy_checkpoint` + 一条迁移说明），
**属于"拒绝 vs 迁移"的安全语义决策，不是代码难题**——这正是本轮停在此处的原因。

**同时说明**：曾尝试"把信封 config 的 `identity_organ_enabled` 改成 False 再读入"，
被第二道守卫拒绝（`checkpoint configuration does not match architecture`，`taiji/model.py:2604-2606`：
`TaijiConfig.from_dict(checkpoint["config"]) != self.config`）⇒ **重导出不是改配置就能做到的**，
需要显式的迁移实现或重建流程。这把原先三选项收敛为：**(a) 加 legacy 守卫（小改、有先例）** 为首选。
