"""B-3 C-4：从 `Cortex._generate_p7` 抽出的**单步解码纯算法**（生成编排仍留在 Cortex 壳内）。

抽离范围 = 迁移前 `cortex.py` 2477-2531：
    重复惩罚 → no-repeat-ngram 封禁 → EOS bias / 熵停止 → top-k 采样 → 追加 token

语义契约（抽离时逐条钉住，任一条漂移 ⇒ 端到端黄金必红）：
1. `logits` **原地修改**（重复惩罚 / 封禁置 -inf / EOS bias）。调用点每次迭代传入的都是
   `xxx / temperature` 产生的新张量 ⇒ 原地写不污染上游。
2. 三个容器**原地变更**：`generated_token_ids`(set, add)、`generated_token_list`(list, append)、
   `generated_ids_ordered`(list, append) ⇒ 参数按引用传入，不复制。
3. 返回 `None` == **熵停止**：原实现在此 `break`，发生在**任何追加之前** ⇒ 迁移后同样
   先返回、不追加。调用方见 `None` 即 `break`。
4. **RNG 消耗顺序必须与迁移前逐位一致**（`torch.multinomial` 的调用次数与位置不变）
   ⇒ 端到端黄金（`reports/cortex_generation_golden_20260926.json`）才成立。
5. 温度已在调用点除过 ⇒ 本函数**不再除温**（2026-08-23 审计 M5 的口径，勿回退）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def decode_step(
    logits,
    generated_token_ids,
    generated_token_list,
    generated_ids_ordered,
    repetition_penalty,
    no_repeat_ngram_size,
    eos_id,
    top_k,
):
    """单步解码：原地修改 logits 与三个历史容器，返回采到的 token；`None` = 熵停止。

    参数按迁移前的局部变量一一对应，无 `self` 依赖。
    """

    # Repetition penalty: penalize tokens that have been generated
    if generated_token_ids and repetition_penalty > 1.0:
        for tid in generated_token_ids:
            if logits[0, tid] > 0:
                logits[0, tid] /= repetition_penalty
            else:
                logits[0, tid] *= repetition_penalty

    # No-repeat-ngram: ban tokens that would complete an existing n-gram
    if no_repeat_ngram_size > 0 and len(generated_token_list) >= no_repeat_ngram_size - 1:
        ngram_prefix = tuple(generated_token_list[-(no_repeat_ngram_size - 1) :])
        # 查找已生成文本中所有匹配前缀的 n-gram 的下一个 token
        banned_ids = set()
        for i in range(len(generated_token_list) - no_repeat_ngram_size + 1):
            if (
                tuple(generated_token_list[i : i + no_repeat_ngram_size - 1])
                == ngram_prefix
            ):
                banned_ids.add(generated_token_list[i + no_repeat_ngram_size - 1])
        # 将 banned tokens 的 logit 设为 -inf
        for tid in banned_ids:
            logits[0, tid] = float("-inf")

    # P7-修复（2026-08-04）：EOS logit 增强 + 熵停止 + 跑偏截断
    # 训练数据（alpaca clean）无 EOS 标记，模型从未学会输出 </s>，
    # 生成永不自然停止 → 一直生成到 max_tokens 导致长序列崩坏。
    # 1) 每步给 eos_id 加温和 bias，鼓励在自然结束点终止；
    # 2) 连续 3+ 个非中文字符 token（英文/符号/数字碎片）视为跑偏 → 截断停止。
    if eos_id is not None:
        logits[0, eos_id] += 0.5  # 温和 EOS bias（top-k 后可能仍在候选）
    else:
        # 无 EOS：softmax 熵 > 阈值时视为跑偏，提前停止
        # 修复（2026-08-23 审计 M5）：logits 在上方各分支已除以
        # temperature（2704-2757 行），此处不再二次除温——旧行为
        # logits/temperature² 会系统性压低熵，使 8.0 阈值几乎不触发。
        # 注意：阈值语义恢复为单次除温口径，若停止时机变化需重新标定。
        probs_ent = F.softmax(logits, dim=-1)
        ent = -(probs_ent * probs_ent.clamp_min(1e-9).log()).sum(-1)
        if ent[0].item() > 8.0 and len(generated_ids_ordered) >= 8:
            return None

    # Top-k sampling in domain vocab
    if top_k > 0:
        actual_k = min(top_k, logits.shape[-1])
        top_k_vals, top_k_indices = torch.topk(logits, actual_k)
        probs = F.softmax(top_k_vals, dim=-1)
        sampled_idx_in_topk = torch.multinomial(probs, 1)
        next_token = top_k_indices[0, sampled_idx_in_topk[0]].item()
    else:
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()

    generated_token_ids.add(next_token)
    generated_token_list.append(next_token)
    generated_ids_ordered.append(next_token)
    return next_token
