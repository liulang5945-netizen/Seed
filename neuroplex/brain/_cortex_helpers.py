"""
Cortex 纯算法辅助函数（从 neuroplex/brain/cortex.py 抽离，P2 拆分第一步）

设计约束（冻结基线）：本模块只放**无 self 状态、纯函数**的算法，
确保从 Cortex 方法体迁移到此处时逐位等价、不改任何推理数学。
后续 Cortex 大拆分（路由集群 / 域推断集群 / 生成集群）以本文件为样板，
每迁一簇都要配同等数值/行为等价测试。
"""

import re

import torch

__all__ = ["detect_modality", "fuse_leader_quality", "is_degenerate_text"]


def is_degenerate_text(text: str) -> bool:
    """R9（REMEDIATION_PLAN 2026-08-14）：退化输出检测（与 verify 脚本同规则）。

    三类已知退化（欠训练 dialogue neuron 实测）：
    - 编号列表塌缩：`1.\\n` 或字面量 `1.<0x0A>`（裸 prompt 死循环残留）
    - 重复标点/字符：同字符连续 >=4
    - 纯数字长串：剔除数字/空白/运算符/标点后仍 >=2 个非汉字非字母字符
      （>=2 排除 "1 + " 这类短算术 stub——那是截断不是退化循环）
    """
    if not text:
        return False
    if re.search(r"\d+\.\s*\n", text) or "1.<0x0A>" in text:
        return True
    if re.search(r"([。！？，,.！、]{2})\1{1,}", text) or re.search(
        r"(.)\1{3,}", text.replace("……", "。。")
    ):
        return True
    stripped = re.sub(r"[0-9\s,，。.!！?？;；:：、+\-*/=×÷\"\'“”（）()<>]", "", text)
    return bool(stripped) and len(stripped) >= 2 and not re.search(r"[\u4e00-\u9fff\w]", stripped)


def detect_modality(input_data: str | torch.Tensor | dict) -> str:
    """P8: 检测输入数据的模态（从 `cortex.py` 抽离，纯函数、逐位等价）。

    路由顺序：
    1. 显式 dict {"modality": "image", "data": ...} → 直接取
    2. torch.Tensor → 根据维度推断（3D float → image/audio 连续特征）
    3. str → "text"

    Args:
        input_data: str / torch.Tensor / dict

    Returns:
        modality name ("text"/"image"/"audio"/"video")
    """
    if isinstance(input_data, dict):
        return input_data.get("modality", "text")
    if isinstance(input_data, torch.Tensor):
        # [B, L, raw_dim] float → 连续特征（图像/音频）
        if input_data.dim() == 3 and input_data.dtype != torch.long:
            # 默认归为 image，具体模态由调用方通过 dict 指定
            return "image"
        # [B, L] long → token id（文本或离散化的多模态）
        return "text"
    return "text"


def _minmax(values: dict) -> dict:
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-9:
        return {k: 0.5 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def fuse_leader_quality(resonance_scores: dict, nll_quality: dict, alpha: float = 0.5) -> dict:
    """连续 leader 融合分数（共振分与 -NLL 域内 min-max 归一化后加权）。

    Args:
        resonance_scores: {nid: round1_scores}（t=0 场共振分，越大越强）
        nll_quality: {nid: -NLL}（生成质量，越大越好）
        alpha: 共振分权重（0.5 = 等权；质量权重 = 1-alpha）

    Returns:
        {nid: fused_score}（仅含两信号都有的 neuron；空 → 调用方回退）
    """
    common = [k for k in nll_quality if k in resonance_scores]
    if not common:
        return {}
    r_norm = _minmax({k: resonance_scores[k] for k in common})
    q_norm = _minmax({k: nll_quality[k] for k in common})
    return {k: alpha * r_norm[k] + (1 - alpha) * q_norm[k] for k in common}


def infer_domain(neuron_domains: set[str], text: str) -> str:
    """P7: 从文本内容启发式推断域。

    检测顺序：code > math > zh > en > general。
    code/math 检测必须在 CJK 之前，防止英文数学/代码被误判为 en。
    仅在对应域 neuron 已加载时返回该域。

    Returns:
        domain name ("zh"/"en"/"code"/"math"/"general")
    """
    if not neuron_domains:
        return "general"

    # 从 neuron key 提取纯域前缀（支持同域多神经元：zh_aug0_dialogue → zh）
    def _has_domain(prefix: str) -> bool:
        return any(k == prefix or k.startswith(prefix + "_") for k in neuron_domains)

    def _first_domain() -> str:
        """返回第一个 neuron 的纯域前缀（fallback）。"""
        first_key = next(iter(neuron_domains))
        return first_key.split("_")[0]

    # 1. 代码检测：强信号关键字（1 个即判定）+ 结构特征
    strong_code = [
        "def ",
        "class ",
        "function ",
        "async ",
        "const ",
        "let ",
        "var ",
        "SELECT ",
        "CREATE TABLE",
        "docker ",
        "git ",
        "npm ",
        "kubectl ",
        "pip ",
        "package ",
        "#include",
        "require(",
    ]
    # import/from 需要排除自然语言用法（"from 0 to 1", "from the"）
    code_keywords = [
        "return ",
        "if __name__",
        "print(",
        "lambda ",
        "try:",
        "except ",
        "raise ",
    ]
    code_patterns = ["{", "};", "=>", "self.", "std::", "def __init__", "class "]
    if _has_domain("code"):
        strong_score = sum(1 for kw in strong_code if kw in text)
        weak_score = sum(1 for kw in code_keywords if kw in text)
        pattern_score = sum(1 for p in code_patterns if p in text)
        # import/from: 排除 "from 0", "from the", "from a" 等自然语言用法
        import_score = 0
        for m in ["import ", "from "]:
            idx = text.find(m)
            if idx >= 0:
                after = text[idx + len(m) :].lstrip()
                # 如果后面是自然语言词（数字、冠词等），不是代码
                if not (after[:1].isdigit() or after.startswith(("the ", "a ", "an "))):
                    import_score += 1
        if "```" in text:
            pattern_score += 5
        # 强信号 1 个即判定，弱信号需 2 个
        if strong_score >= 1 or import_score >= 1 or weak_score + pattern_score >= 2:
            return "code"

    # 2. 数学检测：多路信号融合（符号 + 关键词 + 公式特征 + 上下标）
    if _has_domain("math"):
        # 2a. 数学符号（Unicode 数学字符 + 基础运算符）
        math_symbols = set("=+-*/^∑∫∏√∞∂∇∈⊂∪∩∀∃≤≥≠≈±→←↑↓⇒⇐")
        math_sym_count = sum(1 for c in text if c in math_symbols)

        # 2b. 数学关键词（英文数学术语，大小写不敏感）
        math_keywords = [
            "derivative",
            "integral",
            "theorem",
            "proof",
            "equation",
            "sin",
            "cos",
            "tan",
            "log",
            "ln",
            "limit",
            "matrix",
            "vector",
            "tensor",
            "eigen",
            "calculus",
            "algebra",
            "geometry",
            "probability",
            "distribution",
            "gradient",
            "fourier",
            "laplace",
            "taylor",
            "riemann",
            "convergence",
            "divergence",
            "differential",
            "polynomial",
            "hypothesis",
            "variable",
            "coefficient",
            "parameter",
            "pythagorean",
            "fibonacci",
            "factorial",
            "logarithm",
            "bayes",
            "gaussian",
            "stochastic",
            "determinant",
            "chain rule",
            "product rule",
            "quotient rule",
        ]
        math_kw_count = sum(1 for kw in math_keywords if kw.lower() in text.lower())

        # 2c. 公式特征：上下标数字、希腊字母、函数调用模式
        superscript = sum(1 for c in text if "\u00b2" <= c <= "\u00b9")  # ²³⁴...
        subscript = sum(1 for c in text if "\u2080" <= c <= "\u2089")  # ₀₁₂...
        greek = sum(1 for c in text if "\u0391" <= c <= "\u03c9")  # Α-ω
        # 函数调用模式 f(x), g(x), h(x)
        fn_call = (
            1
            if text.count("(") >= 1
            and text.count(")") >= 1
            and any(
                p in text
                for p in [
                    "f(",
                    "g(",
                    "h(",
                    "f(x)",
                    "g(x)",
                    "h(x)",
                    "sin(",
                    "cos(",
                    "tan(",
                    "log(",
                    "ln(",
                ]
            )
            else 0
        )

        # 2d. 数字表达式密度（纯数字 + 运算符占比高）
        stripped = text.replace(" ", "").replace("\n", "")
        digit_ops = sum(1 for c in stripped if c.isdigit() or c in "+-*/=()^.,")
        digit_ratio = digit_ops / max(len(stripped), 1)

        # 综合判定（满足任一条件）
        math_total = (
            math_sym_count + math_kw_count * 2 + superscript + subscript + greek + fn_call
        )
        if math_total >= 1 or digit_ratio > 0.4:
            return "math"

    # 3. 中文检测（CJK 统一汉字区块）
    cjk_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    if cjk_count > len(text.replace(" ", "")) * 0.3:
        return "zh" if _has_domain("zh") else _first_domain()

    # 4. 默认：en 或 general
    if _has_domain("en"):
        return "en"
    if _has_domain("general"):
        return "general"
    return _first_domain()
