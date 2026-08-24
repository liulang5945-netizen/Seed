"""性能基准测试（F13）——关键路径的延迟 / 吞吐基线，防止性能回归。

设计原则：
- 使用 ``time.perf_counter`` 轻量计时，不引入额外依赖。
- 阈值刻意放宽（覆盖 CI 慢机器），只捕捉"数量级"级别的回归，
  不对微秒级波动敏感；若需更精细的基准请配合 ``pytest-benchmark``。
- 以 ``@pytest.mark.benchmark`` 标记，可用 ``-m benchmark`` 单独运行。

运行：
    python -m pytest tests/test_performance_baseline.py -v
    python -m pytest -m benchmark -v
"""

from __future__ import annotations

import time

import pytest

from taiji import Taiji, TaijiConfig

# 统一的轻量配置：足够小以便在 CI 中快速运行，又足以覆盖真实计算路径。
_BENCH_CONFIG = TaijiConfig(
    region_sizes=(32, 24),
    synapse_fan_in=8,
    motor_fan_in=24,
    memory_units=16,
    memory_fan_in=4,
    memory_readout_fan_in=4,
    memory_meta_dim=8,
    memory_iterations=2,
    memory_time_dim=4,
    memory_episode_dim=4,
    seed=1337,
)
_BENCH_DATA = b"abcdabcdabcdabcd" * 4  # 64 bytes 循环序列


@pytest.fixture(scope="module")
def trained_model() -> Taiji:
    """模块级复用一个已训练模型，避免每个用例重复训练。"""
    model = Taiji(_BENCH_CONFIG)
    model.learn_bytes(_BENCH_DATA, epochs=50)
    return model


@pytest.mark.benchmark
def test_model_construction_latency() -> None:
    """模型构建应在合理时间内完成（< 2s，CI 宽松上限）。"""
    start = time.perf_counter()
    Taiji(_BENCH_CONFIG)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"模型构建耗时 {elapsed:.3f}s 超过阈值 2.0s"


@pytest.mark.benchmark
def test_learn_bytes_throughput(trained_model: Taiji) -> None:
    """learn_bytes 吞吐基线：记录 bytes/s，确保不为退化值。"""
    data = _BENCH_DATA
    start = time.perf_counter()
    trained_model.learn_bytes(data, epochs=20)
    elapsed = time.perf_counter() - start

    assert elapsed > 0, "计时异常"
    bytes_per_second = (len(data) * 20) / elapsed
    # 宽松下限：CPU 上该规模模型每轮 64B 应远高于 100 B/s
    assert bytes_per_second > 100, f"训练吞吐过低: {bytes_per_second:.0f} B/s"


@pytest.mark.benchmark
def test_score_bytes_latency(trained_model: Taiji) -> None:
    """score_bytes 单次推理延迟（< 500ms，CI 宽松上限）。"""
    start = time.perf_counter()
    iterations = 20
    for _ in range(iterations):
        trained_model.score_bytes(_BENCH_DATA)
    elapsed = (time.perf_counter() - start) / iterations

    assert elapsed < 0.5, f"score_bytes 平均延迟 {elapsed * 1000:.1f}ms 超过 500ms"


@pytest.mark.benchmark
def test_generate_latency(trained_model: Taiji) -> None:
    """generate 生成 16 字节的延迟（< 1s，CI 宽松上限）。"""
    start = time.perf_counter()
    trained_model.generate(b"a", 16)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"generate 16 字节耗时 {elapsed:.3f}s 超过 1.0s"


@pytest.mark.benchmark
def test_checkpoint_roundtrip_latency(trained_model: Taiji) -> None:
    """checkpoint 导出 + 恢复的往返延迟（< 2s）。"""
    start = time.perf_counter()
    checkpoint = trained_model.checkpoint()
    Taiji.from_checkpoint(checkpoint)
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0, f"checkpoint 往返耗时 {elapsed:.3f}s 超过 2.0s"
