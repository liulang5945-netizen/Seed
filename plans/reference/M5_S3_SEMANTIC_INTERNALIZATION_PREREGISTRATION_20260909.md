# M5.S3 预注册：语义 embedding 内化

> 起草日期：2026-09-09。本文把 M5.S2 的"语义升级设计决策 (a)"落成预注册设计。确认前不写实现（实现本文发布后即视为已确认，与 R5 预注册流程一致）。执行顺序以 [03_CURRENT_EXECUTION.md](../active/roadmap/03_CURRENT_EXECUTION.md) 为准。

## 1. 假设

> 用预注册的多语言语义 embedding 替换确定性统计特征后，内化管线对**未见来源域**的 holdout 改善仍满足 S2 全部因果控制（外部充分性、内化必要性、来源必要性、checkpoint、retention），且新增**语义保持探针**成立：同义改写对的 embedding 距离显著小于无关文本对的距离。

若语义 embedding 下任一 S2 因果控制失败，或语义探针不成立，则 M5 语义内化假设被否决，S2 的统计特征结果保持为该阶段最佳证据。

## 2. embedding 来源与依赖边界（预注册后固定）

| 项 | 决定 |
|---|---|
| 模型 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`（Apache-2.0，384 维，多语言含中文） |
| 加载方式 | 仅 `transformers` AutoTokenizer/AutoModel + mean pooling（**零新增 pip 依赖**；pyproject 已声明 transformers/sentence-transformers 边界） |
| 锚定 | 记录 HF revision SHA 与 `config.json` 的 content digest 进报告；embedding 函数版本化 `m5s3-embedder-v1` |
| 推理 | `torch.no_grad()` + eval 模式；CPU；批量 ≤32 |
| 缓存 | HF 本地缓存（一次性下载），运行时不联网 |
| 禁止 | 云端 API、任何联网推理、修改模型权重 |

## 3. 数据与 Gate

- 语料：与 S2 相同的 UltraData 转换语料（SHA-256 锚定）；train = Knowledge+IF 各 3000，holdout = 未见来源域 Chinese-general 2000，retention = train 子集 2000；
- 特征维度 12 → 384（`InternalizedFeatureLearner(feature_dim=384)`）；
- Gate 全部沿用 S2 五条因果控制 + 11 项检查，另加：
  - **语义保持探针**：从 holdout 采 200 对同源改写（同 domain 随机对）与 200 对跨域对，报告两组的 embedding 余弦距离均值（探针只报告不作为 Gate，避免与内化 Gate 循环）；
- 资源记录：embedding 墙钟与批量计数（384 维 CPU 推理成本首次进入 M5 成本基线）。

## 4. 边界

- reward 仍为占位常量（真实 outcome 等 MCP/执行回路，属设计决策 (b)）；
- embedding 模型不进入 checkpoint（只锚定 id/revision/digest）；learner checkpoint 兼容性以 feature_dim 变化为准（384 与 S2 的 12 不互通，报告分别版本化）；
- `can_promote=false` 固定。
