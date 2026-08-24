"""Anti-corruption regression: verify-style scripts must separate train-fit
from held-out metrics.

背景（metrics_conventions.md）：``verify`` 类脚本历史上把
``learn_bytes(data)`` 之后直接在 ``data`` 上 ``score_bytes`` 的准确率当作通过
条件，即 train == test，无法暴露评估泄漏。修复后口径必须拆分：
- ``train_fit_accuracy``：在训练字节序列上打分，仅信息性，不卡线。
- ``heldout_accuracy``：在未参与训练的确定性字节流上打分，作为通过依据。

本测试以 ``verify_taiji_native_v7.run_benchmark`` 为样本，确认产出的 report
dict 同时包含这两个字段，且二者指代不同口径（held-out 不得与训练拟合混同）。
使用极小 epochs，纯 CPU 运行 < 1s，不触发完整训练。
"""

from __future__ import annotations

from scripts.training import verify_taiji_native_v7 as verify


def test_verify_report_separates_heldout_and_train_fit() -> None:
    report = verify.run_benchmark(epochs=1)

    metrics = report["metrics"]
    assert "train_fit_accuracy" in metrics, "report 必须包含 train_fit_accuracy 字段"
    assert "heldout_accuracy" in metrics, "report 必须包含 heldout_accuracy 字段"

    train_fit = metrics["train_fit_accuracy"]
    heldout = metrics["heldout_accuracy"]

    # 两个字段必须是独立数值，禁止把 train==test 折叠为单一口径。
    assert train_fit != heldout, (
        "train_fit_accuracy 与 heldout_accuracy 必须指向不同口径，" "不可折叠为同源恒等断言"
    )

    # held-out 流是确定性、从未训练过的字节序列，必须区别于训练数据。
    heldout_stream = metrics.get("heldout_stream")
    assert heldout_stream is not None, "report 必须记录 heldout 使用的字节流"
    assert (
        heldout_stream != "abcdabcdabcdabcd"
    ), "heldout 流必须与训练数据 ('abcd' 周期4) 区分，禁止复用训练序列"
    assert heldout_stream == verify.HELDOUT_STREAM.decode("ascii")
