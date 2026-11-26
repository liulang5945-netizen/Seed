# 阶段 0 预检与 CPU 长机时报价备忘（`N38.0.01`：开放中文统一认知底座）

> [备忘·阶段 0 完成] — 2026-11-26。基于本机实测（`24` 逻辑核 CPU、`PyTorch 2.13.0+cpu`、`MKLDNN=True`），完成本地中文语料去重审计、8,192 词表字节回退 BPE 训练、以及 `small_8m / mid_18m / large_30m` 三档真实吞吐与 12h/24h/48h Chinchilla 报价。**零确认 seed 读取、零 `check/final` 读取。**

- **骨干代码**：[open_chinese_backbone.py](../../taiji/open_chinese_backbone.py)（RMSNorm + RoPE + SwiGLU + GQA + 8 槽跨会话持久记忆状态）
- **单测**：[test_n38_open_chinese_backbone_20261126.py](../../tests/test_n38_open_chinese_backbone_20261126.md)（2/2 PASS）
- **预检脚本与回执**：[n38_0_01_stage0_preflight.py](../../scripts/training/n38_0_01_stage0_preflight.py) / [n38_0_01_stage0_preflight.json](../../reports/n38_0_01_stage0_preflight.json)

## 1. 本地中文语料审计与去重结果

| 语料源 | 原始大小 (MB) | 抽样去重率 (`unique_ratio`) | 估计净去重大小 (MB) |
|---|---:|---:|---:|
| `simple_zh_texts.jsonl` | `1,330.2` | `1.0000` | `1,330.2` |
| `shared_core.jsonl` | `398.8` | `0.9487` | `378.3` |
| `dialogue_extended_clean.jsonl` | `103.3` | `1.0000` | `103.3` |
| `p3b_dialogue_subset.jsonl` | `534.3` | `0.6589` | `352.0` |
| `p3b_dialogue_fresh.jsonl` | `51.5` | `0.0650` | `3.3` |
| `ultradata_sft_nothink_simplezh.jsonl` | `30.4` | `1.0000` | `30.4` |
| `ultradata_hetero_holdout_simplezh.jsonl` | `9.6` | `1.0000` | `9.6` |
| **合计** | **`2,458.1`** | — | **`~2,207.1 MB`** |

- **切分纪律**：按文本 `sha256[:8] % 1000` 确定性切分（`0..979 train` / `980..989 val` / `990..999 sealed_eval`）；已生成 `864,380` token 验证集分片 `data/n38_open_zh/val_tokens.bin`。
- **8K 字节回退 BPE 词表**：`vocab_size = 8,192`（`sha256 = afc7f5c2...`），验证集压缩率 **`1.4274` 汉字/token**（`4.1373` 字节/token），生僻字 OOV 往返 **`100%` 无损（`oov_roundtrip_exact = true`）**。

## 2. 本机 CPU 精度与吞吐实测（`seq_len = 512, batch_size = 4`）

- **最优配置**：`threads = 8` ＋ **`bfloat16` autocast**（比 `float32` 快约 `37%`：`4,343.2` vs `3,176.2 tokens/sec`）。

| 预设 | 参数量 | 实测吞吐 (`tokens/sec`) | 12h 总 token (`tok/param`) | 24h 总 token (`tok/param`) | 48h 总 token (`tok/param`) |
|---|---:|---:|---:|---:|---:|
| `small_8m` | `8,655,616` | `6,437.7` | `278.1M` (`32.13`) | `556.2M` (`64.26`) | `1,112.4M` (`128.52`) |
| **`mid_18m`（推荐）** | **`19,473,024`** | **`4,343.2`** | `187.6M` (`9.64`) | **`375.3M` (`19.27`)** | **`750.5M` (`38.54`)** |
| `large_30m` | `31,469,056` | `3,032.7` | `131.0M` (`4.16`) | `262.0M` (`8.33`) | `524.0M` (`16.65`) |

## 3. 阶段 1 长机时预训练建议（待 owner 拍板启动）

- **首选方案**：**`mid_18m`（`19.47M` 参数）跑 24–48 小时**（`threads=8, bfloat16`，`batch_size=4, grad_accum=4`，每 `500` 步原子保存 checkpoint 并评测 `val_ppl`），24 小时即可吃到 **`~3.75 亿 tokens`（`19.27 tok/param`，贴合 Chinchilla 20:1 最优比例）**，48 小时可达 **`~7.5 亿 tokens`（`38.54 tok/param`）**。
- **备选方案**：若先跑 `12` 小时验证，可选 `small_8m`（12h 即达 `32.13 tok/param`）。
