# M5 P5.1d 预注册：语义 encoder 接入产品内化边界（注入合同工程化）

- 状态：**FROZEN**（冻结于 2026-09-12；冻结前未训练、未接新数据源、未读取 sealed）
- 前置：P5.1c 已闭合（`8f71bde7`，disc 0.731412 ≥ 0.3 九门全过，outcome=`contrastive_content_discrimination_supported`）
- 依据：`plans/active/roadmap/03_CURRENT_EXECUTION.md` P5.1c 执行完毕条目所声明的唯一下一步
- 性质：工程预注册。P5.1c 在 probe 层验证了内容寻址语义内化；P5.1d 把该验证过的机制正式接入产品 `ArtifactInternalizationTrainer`，消除 probe 式运行时替换工程债，并以真实 governed 语料替代脚本内合成 fixture 完成规模化验收。

## 1. 摸底结论（只读 recon，冻结依据）

### 1.1 现状工程债（probe 层已验证、产品层未接入）
- `SemanticArtifactKnowledgeEncoder`（FORMAT=`taiji-semantic-artifact-knowledge-v1`，VERSION=1）现定义于 probe 脚本 `scripts/training/probe_taiji_p5_1b_semantic_paraphrase_transfer.py` line 192-231，384 维直通（`feature_dim = embedder.dimension`），`encode(artifact)` = MiniLM 嵌入渲染文本（`unit_kind` + `content` 全键值，剔除身份键 `scope_id`），checkpoint 仅存锚定元数据（`embedder_model_id` / `embedder_revision` / `embedder_config_digest` / `feature_dim`），模型权重不入 checkpoint。
- 注入方式为运行时替换：`trainer.encoder = SemanticArtifactKnowledgeEncoder(embedder=embedder)`（P5.1b line 244、P5.1c line 123），绕过 `__init__` 类型假设。
- `from_checkpoint` 恢复需 stub 绕行（P5.1b `_semantic_arm` line 247-255：deepcopy payload → 换入结构 encoder checkpoint → 重算 digest → 恢复 → 再手动换回语义 encoder）——类型不安全、往返不忠实，属明确工程债。
- probe 版 `SemanticArtifactKnowledgeEncoder.from_checkpoint` 忽略 payload 锚定字段直接 `cls()`——锚定漂移不设防，晋升时必须修复。

### 1.2 产品 trainer 注入面（`taiji/artifact_internalization.py`，776 行）
- 两处硬编码：`__init__` line 315（`self.encoder = ArtifactKnowledgeEncoder(feature_dim)`）与 `from_checkpoint` line 733（`encoder = ArtifactKnowledgeEncoder.from_checkpoint(payload["encoder"])`）。
- encoder 调用点仅两处：`_example` line 353（semantic 特征，`feature = self.encoder.encode(artifact)`，feature digest 进 example_id）与 `_procedural_records` line 422（procedural cue）。
- 查询读出 `semantic_value_from_feature` line 679-695 直接消费裸 feature tensor，不经 encoder——encoder 注入不影响读出合同。
- `dataset_digest` line 523-533 绑定 `self.encoder.checkpoint()`；trainer checkpoint line 697-720 含 `payload["encoder"]` + `checkpoint_digest`（排除该键自摘要）——encoder 身份已内容寻址入两条摘要链，注入新 encoder 自动继承，零改动。
- 维度一致性：`from_checkpoint` 漂移校验要求 semantic.feature_dim / procedural.cue_dim / affordance.input_dim 均等于 encoder.feature_dim；三器官构造共享同一 feature_dim。
- admission 原子性：`consolidate` 在三器官 trial 副本上训练（line 629-646），semantic.passed ∧ procedural(holdout>lesion ∧ retention≥0.5) ∧ affordance(native<frozen) 全过才整体提交（revision+1、child_digest），否则 `rolled_back=true` 不落盘。

### 1.3 消费侧与维度合同
- `InternalizedFeatureLearner`（`taiji/internalization_learner.py`）对特征来源零假设：`_features` 仅校验 `grounding.numel() == feature_dim` 与 finite；checkpoint 格式 `taiji-internalization-learner-v1` 含 feature_dim/weights/bias/lineage。→ 384 维直通无消费侧障碍。
- 对比训练机制已内建：`pairwise_margin` + `consolidate(ranking_pairs=...)`，P5.1c 冻结 margin=0.5，在 `FEATURE_DIM=384` 直通上实测 disc 0.731。→ **维度合同：384 直通，不投影**（投影有损且引入额外冻结组件，上限更低；P5.1c 判别力已在 384 维全信息上验证）。
- `DocumentEmbedder`（`taiji/document_embedding.py`）：MiniLM-L12-v2 本地缓存，CPU/eval/no_grad/mean-pooling/L2 归一化确定性嵌入，384 维硬校验，`to_payload()` 供锚定（model_id/revision/config_digest）。

### 1.4 真实 governed 语料管道
- 既有 governed 链：`DeclarativeSourceRegistry`（register/transition/checkpoint，`seed_platform/source_registry.py`）→ `SkillArtifactAdapter.project` → `EvolutionCorpusArtifact` + 事件（`seed_platform/evolution_adapters.py`）。
- P5.1c gate 语料为脚本内合成 fixture（A_WORKFLOWS/B_WORKFLOWS 手写 workflow，每族十文本）→ P5.1d gate 语料改由 governed 管道构建。

## 2. 注入合同（冻结）

### 2.1 选型裁定
| 方案 | 裁定 | 理由 |
| --- | --- | --- |
| A. 构造参数注入 + from_checkpoint 格式分派 | **采纳** | 单一 encoder 概念、复用现有漂移校验与摘要绑定不变量、保留未来可插任意 encoder 的扩展上限 |
| B. 子类覆写 trainer | 否决 | 双继承链污染、checkpoint 往返仍需分派、上限低 |
| C. 维持 probe 式运行时替换 | 否决 | 类型不安全、from_checkpoint 需 stub 绕行，即本次要消除的工程债 |

### 2.2 冻结条款
1. **encoder 晋升**：`SemanticArtifactKnowledgeEncoder` 从 probe 脚本晋升至 `taiji/artifact_internalization.py`（与 `ArtifactKnowledgeEncoder` 并列，复用 `EvolutionCorpusArtifact` 类型与文本渲染语义）；probe 脚本删除本地定义改 import 产品模块（机制收敛，禁残留副本）。
2. **晋升修复**：晋升版 `from_checkpoint` 校验 payload `embedder_model_id`/`embedder_revision`/`embedder_config_digest`/`feature_dim` 与本地缓存 `DocumentEmbedder` 解析值逐项一致，任一不符 raise（锚定漂移拒绝）；FORMAT/VERSION 不变（`taiji-semantic-artifact-knowledge-v1` / 1）。
3. **构造注入**：`ArtifactInternalizationTrainer.__init__` 新增 `encoder: Any | None = None`——None 保持现行为（`ArtifactKnowledgeEncoder(feature_dim)`，向后兼容零回归）；传入对象须暴露 `feature_dim`/`encode(artifact)`/`checkpoint()` 协议，且构造时即校验 `encoder.feature_dim == feature_dim`（fail-fast，漂移校验前置）。
4. **checkpoint 分派**：`from_checkpoint` 按 `payload["encoder"]["format"]` 分派：`taiji-artifact-knowledge-v1` → 结构 encoder；`taiji-semantic-artifact-knowledge-v1` → 语义 encoder；未知 format → `ValueError`。废除 stub 绕行模式。
5. **对比更新合同**：新增 trainer 构造参数 `semantic_pairwise_margin: float = 0.0`（默认 = 现行为），传入 semantic organ；`consolidate` 新增可选 `ranking_pairs` 参数透传 `semantic.consolidate`（配对构造策略属调用方/门 runner，trainer 保持通用）。
6. **维度合同**：384 直通，无投影；三器官统一 384；`procedural_hidden_dim` / `affordance_feature_dim` 维持独立超参不变。
7. **锚定继承零改动**：`dataset_digest` 与 trainer checkpoint 对 encoder 的绑定逻辑不改，语义 encoder 锚定元数据自动入两条摘要链。

## 3. 真实 governed 语料规模化（冻结）

1. P5.1d gate 语料由 governed 管道构建：`SkillArtifactAdapter` 投影真实注册源为 `EvolutionCorpusArtifact` + 事件，partition（train/holdout/retention）在构建期冻结；禁止脚本内手写 workflow fixture 作为 gate 语料。
2. 划分纪律沿用：train/holdout/retention 按 artifact 身份不相交（`consolidate` 既有校验兜底）；holdout/retention 仅评估不训练。
3. 规模实测值入报告（不冻结绝对数，冻结下限：train artifacts ≥ P5.1c 等价规模即 ≥ 20；实测值 §7 对账为准）。
4. 查询文本沿用 P5.1c 改写查询纪律（与语料表层不相交的 paraphrase 对，A/B 两族），由 gate runner 内冻结字面量。

## 4. 九门式验收判据（冻结）

| # | 门 | 冻结 margin |
| --- | --- | --- |
| 1 | 静态四项：py_compile / ruff / mypy / black --no-cache 全过（改动文件全集） | — |
| 2 | 默认回归：不传 encoder 且 margin=0 时，trainer 默认构造结构 encoder，consolidate 行为与现行实现一致（同 fixture digest 等价） | — |
| 3 | 构造 fail-fast：`encoder.feature_dim != feature_dim` 时 `__init__` raise | — |
| 4 | checkpoint 往返：语义 encoder 注入 → consolidate → checkpoint → `from_checkpoint` 自动分派恢复，同 artifact encode 特征 digest 恒等、semantic score 恒等，全程无 stub | — |
| 5 | 锚定漂移拒绝：篡改 payload["encoder"]["embedder_revision"] 任一字符 → `from_checkpoint` raise | — |
| 6 | 格式分派：未知 encoder format → raise；旧 format `taiji-artifact-knowledge-v1` 仍恢复结构 encoder | — |
| 7 | 对比辨别力（核心门）：真实 governed 语料 + pairwise 对比训练后，A-para mean − B-para mean ≥ 0.3；placebo 对照臂反向辨别 ≥ 0.3 | 0.3 |
| 8 | 三器官 admission：semantic.passed ∧ procedural(holdout>lesion ∧ retention≥0.5) ∧ affordance(native<frozen) 在 384 维下成立 | retention ≥ 0.5 |
| 9 | 确定性 + 预算：同语料同 seed 两次 consolidate 的 dataset_digest 与 child digest 一致；gate wall ≤ 1200 s（MiniLM 嵌入开销较 P5.1c 放宽） | wall ≤ 1200 s |

- 判定文案：九门全过 → `semantic_encoder_injection_supported`；任一失败 → 如实报告 `rejected`，禁止反向篡改报告凑数。
- 探针双路径互证：gate 主臂（产品 trainer 内化路径）与 P5.1c 冻结探针值（disc 0.731412）交叉对照入报告。

## 5. 产物顺序（§6 前置）

1. `taiji/artifact_internalization.py`：晋升 `SemanticArtifactKnowledgeEncoder`（含锚定校验修复）+ `__init__` encoder 参数 + `semantic_pairwise_margin` + `from_checkpoint` 格式分派 + `consolidate` ranking_pairs 透传。
2. `taiji/__init__.py`：导出 `SemanticArtifactKnowledgeEncoder`。
3. probe 脚本收敛：`probe_taiji_p5_1b_semantic_paraphrase_transfer.py` 删本地定义改 import（机制收敛，防双副本漂移）。
4. gate runner：`scripts/training/eval_taiji_p5_1d_semantic_encoder_injection_gate.py`（真实 governed 语料 + 九门）。
5. 报告：`reports/taiji_p5_1d_semantic_encoder_injection_20260912.json`。
6. 静态四项 + 本文件 §7 对账 + 路线图同步 + 独立提交。

## 6. 纪律

- 冻结后执行期禁止：改 margin、改语料划分、改判据、读取 sealed、绕过 admission 原子性。
- 报告以 gate 实测值为准；预注册与报告冲突时以实测对账（§7），不回改判据。
- 机制收敛：probe 重复定义必须删除，禁止残留旧副本干扰后续调用。
