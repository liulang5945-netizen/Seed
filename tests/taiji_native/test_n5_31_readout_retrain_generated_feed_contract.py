"""`PLAN-N5-20` 的挂接契约册（J-N5r-1／2／3、G-N5r-1／2 的机检形）。

真长跑不在本册里跑——本册只钉四件事：①默认关闭；②落账那一步**不许切本臂 readout**且**关掉所有 learn 位**；
③喂入确实走产品入口而不是第二条写链；④一次真实的小喂入能把 `generated` 那一行从 0 抬起来。
第④用的是 tiny 配置（不是长跑读数）；长跑那一档的现读记在台账 ㊵-740：400 符号、每 50 喂一次
⇒ `feeds=8`、`generated_write_count=8`、行权重 **0.276523322**、奖励和 0.0、`write_surface_ok` 仍为真。
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import math
import sys
from pathlib import Path

import pytest
import torch

from taiji import Taiji, TaijiConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT_ROOT / "scripts" / "training" / "train_taiji_r2_readout_retrain.py"


def _runner():
    spec = importlib.util.spec_from_file_location("n5_31_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _StubSubstrate:
    """记录调用序列的假 substrate：用来单独钉"落账那一步传了什么参数"。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def record_generated_prediction(self, symbol: int) -> None:
        self.calls.append(("record_generated_prediction", (symbol,), {}))

    def observe(self, symbol: int, **kwargs) -> None:
        self.calls.append(("observe", (symbol,), kwargs))

    class _Field:
        PROVENANCE_KINDS = ("experienced", "imagined", "replayed", "external", "generated")
        generated_write_count = 3

        class _Readout:
            import torch as _torch

            edge_weight = _torch.zeros(5, 4)

        provenance_readout = _Readout()

    memory = _Field()


def _tiny_model() -> Taiji:
    config = TaijiConfig(
        region_sizes=(16, 12),
        synapse_fan_in=4,
        motor_fan_in=8,
        memory_units=24,
        memory_fan_in=4,
        memory_meta_dim=8,
        memory_readout_fan_in=8,
        memory_iterations=2,
        seed=5,
        identity_organ_enabled=False,
    )
    model = Taiji(config)
    model.reset_dynamics(episode_id="n5-31")
    return model


def _episodic_write_calls(source: str) -> int:
    """G-N5r-1 的真尺：数「方法名是 `write` 且接收者点分链尾是 `memory`」的调用。

    DEBT-G121 ③ 的修法就在这句上：**越界样本与真码走同一个函数**。原来那条「负对照」是对
    自己刚拼出来的字符串再断言一次子串存在，永远为真、零判别力。
    """

    count = 0
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "write":
            continue
        receiver = ast.unparse(node.func.value)
        if receiver == "memory" or receiver.endswith(".memory"):
            count += 1
    return count


def _tiny_seed_envelope(tmp_path: Path) -> Path:
    """现造一枚极小 Seed 信封当基座：本册要读的是 `run_arm` 的**产出 dict**，不是它的源码。"""

    from seed import Seed
    from seed.config import SeedConfig

    config = SeedConfig(
        taiji=TaijiConfig(
            region_sizes=(16, 12),
            synapse_fan_in=4,
            motor_fan_in=8,
            memory_units=24,
            memory_fan_in=4,
            memory_meta_dim=8,
            memory_readout_fan_in=8,
            memory_iterations=2,
            seed=5,
            identity_organ_enabled=False,
        )
    )
    path = tmp_path / "base.pt"
    torch.save(Seed(config).checkpoint(), path)
    return path


def _tiny_corpus(tmp_path: Path, rows: int = 40) -> Path:
    #: 尺寸是被量出来的：语料必须够跑预算，否则跑器会响亮拒绝
    #: （`corpus exhausted ... the budget does not fit the stream`）。2026-10-13 起行数由
    #: 调用处按「本次实测速率」换算——ETA 只留 3 位小数，预算太小会让那条断言变成
    #: 「机器一快就必然为假」（见定价那一支的说明）。
    row = json.dumps({"text": "abracadabra cadabra abracadabra cadabra"}, ensure_ascii=False)
    path = tmp_path / "corpus.jsonl"
    path.write_text((row + "\n") * rows, encoding="utf-8", newline="\n")
    return path


def test_the_flag_defaults_to_off_and_the_runner_keeps_one_write_path() -> None:
    """J-N5r-1 的形状面＋G-N5r-1：默认关死（读**运行时签名**），跑器里不许第二条写链。

    DEBT-G121 ③ 的两处升级：①默认值不再靠 `source.count("default=0")` 这种文本匹配——
    那枚字符串在别处出现也算数；改读 `inspect.signature(run_arm)` 的参数的默认值。
    ②「负对照」必须走**同一把尺**：合成源码喂进 `_episodic_write_calls` 要数出 1，
    而一枚同名但接收者不是情节场的落盘调用要数出 0（尺过宽会误伤，那是另一族教训）。
    """

    runner = _runner()
    source = RUNNER.read_text(encoding="utf-8")
    assert "--generated-feed-every" in source
    signature = inspect.signature(runner.run_arm)
    assert signature.parameters["generated_feed_every"].default == 0, signature
    assert _episodic_write_calls(source) == 0, "跑器里出现了挂在情节场上的第二条写链"
    assert source.count("record_generated_prediction(") == 1

    fabricated = "def f(model, ctx):\n    return model.memory.write(ctx)\n"
    assert _episodic_write_calls(fabricated) == 1, "这把尺数不出越界样本＝它根本没有动态范围"
    lookalike = "def g(handle, text):\n    handle.write(text)\n    return path.write_text(text)\n"
    assert _episodic_write_calls(lookalike) == 0, "把落盘 write 也计进来＝尺过宽，会误伤"


def test_the_flush_keeps_the_arms_readout_and_turns_every_learn_off() -> None:
    """落账那一步：readout 照抄本臂、所有 learn* 关死（㊵-740 两次撞墙的形状）。"""

    runner = _runner()
    stub = _StubSubstrate()
    arm_kwargs = {
        "learn": True,
        "readout": "predictive",
        "learn_motor": True,
        "learn_fabric": True,
        "learn_predictive_readout": True,
    }
    runner.feed_generated_answer(stub, 66, arm_kwargs)

    names = [call[0] for call in stub.calls]
    assert names == ["record_generated_prediction", "observe"], names
    flush = stub.calls[1][2]
    assert flush["readout"] == "predictive", flush
    learned = {key: value for key, value in flush.items() if key.startswith("learn")}
    assert learned and set(learned.values()) == {False}, learned


def test_a_real_feed_grows_the_generated_row() -> None:
    """J-N5r-2 的小配置版：写计数与那一行的权重必须同时离开 0。"""

    runner = _runner()
    model = _tiny_model()
    row = int(model.memory.PROVENANCE_KINDS.index("generated"))
    before = float(model.memory.provenance_readout.edge_weight[row].abs().max())
    assert before == 0.0

    prior = None
    for symbol in (65, 66, 67, 68):
        step = model.observe(symbol, learn=False, learn_motor=False)
        prior = getattr(step, "prior_prediction", None)
    assert prior is not None

    fed = runner.feed_generated_answer(
        model, int(prior), {"learn": False, "readout": "action", "learn_motor": False}
    )
    after = float(model.memory.provenance_readout.edge_weight[row].abs().max())
    assert model.memory.generated_write_count == 1
    assert fed["write_count"] == 1.0
    assert after > 0.0, (before, after)


def test_the_entry_refuses_to_overwrite_an_in_flight_experience() -> None:
    """无动作入口不许把已在路上的经验挤掉——静默覆盖会让"哪次喂入长了权重"不可归因。"""

    model = _tiny_model()
    model.observe(65, learn=False, learn_motor=False)
    model.act((65,), sample=False)
    with pytest.raises(RuntimeError, match="quiet pending state"):
        model.record_generated_prediction(66)

    model2 = _tiny_model()
    model2.observe(65, learn=False, learn_motor=False)
    model2.record_generated_prediction(66)
    with pytest.raises(RuntimeError, match="quiet pending state"):
        model2.record_generated_prediction(67)


def _run_tiny_arm(
    tmp_path: Path, *, generated_feed_every: int, symbols: int = 600, corpus_rows: int = 40
) -> dict:
    """小信封＋小语料真跑一臂，返回 `run_arm` 的产出 dict（摘要助手全部复用跑器自己的函数）。"""

    tmp_path.mkdir(parents=True, exist_ok=True)
    runner = _runner()
    base = _tiny_seed_envelope(tmp_path)
    corpus = _tiny_corpus(tmp_path, corpus_rows)
    return runner.run_arm(
        arm="A",
        symbols=symbols,
        checkpoint_every=300,
        progress_every=300,
        arm_dir=tmp_path / "arm",
        base_checkpoint=base,
        base_sha256=runner.sha256_of(base),
        corpus_paths=[corpus],
        corpus_digest=runner.corpus_fingerprint([corpus]),
        skip_symbols=0,
        device="cpu",
        fresh=True,
        generated_feed_every=generated_feed_every,
    )


def test_pricing_keys_are_published_by_the_runner(tmp_path: Path) -> None:
    """J-N5r-3 从「读源码」升成「读产出」（DEBT-G121 ②）：小信封＋小语料直调 `run_arm`。

    原来这支只对跑器源码做子串匹配（五枚键名 in source），既不证明这些键真被产出，也不证明
    值是被测出来的——`generated_reward_sum` 那枚恒零字面量就是反例（字面量也能过）。
    现在断的是**产出 dict**：键在场，且速率／ETA／喂入次数／写计数都是现跑出来的正数。
    """

    #: 预算不能是随手挑的常数：ETA 只保留 3 位小数，预算太小会让 `eta_hours > 0.0` 变成
    #: 「机器一快就必然为假」的断言（2026-10-13 这台机器 388.40 符号/秒时就这样红了：
    #: 整族 `1 failed, 3275 passed`，隔离复跑同值 ⇒ 不是争用）。先探一次本次实测速率，
    #: 再把预算定成约 6 秒的符号数 ⇒ eta ≈ 0.0017 h 可表示，那条关系式才真的能被否证。
    probe = _run_tiny_arm(tmp_path / "probe", generated_feed_every=4)
    measured_rate = float(probe["generated_feed"]["measured_symbols_per_second"])
    budget = max(600, math.ceil(measured_rate * 6.0))
    report = _run_tiny_arm(
        tmp_path / "main",
        generated_feed_every=4,
        symbols=budget,
        corpus_rows=budget // 30 + 20,
    )
    block = report["generated_feed"]
    for key in (
        "every_symbols",
        "feeds",
        "generated_write_count",
        "generated_row_weight_max_abs",
        "generated_reward_sum",
        "measured_symbols_per_second",
        "eta_hours_at_requested_budget",
    ):
        assert key in block, key
    assert block["measured_symbols_per_second"] > 0, block
    assert block["feeds"] > 0, block
    assert block["generated_write_count"] > 0, block
    #: J-N5r-3 真正要钉的就是这条关系：ETA 只能由本次实测速率推，不接受引用来的数。
    #: 预算来自上面的 probe ⇒ 无论这台机器这次跑到 225 还是 388 符号/秒，"必须为正"与
    #: "由实测推出"两条都仍能为假（把跑器里的 `symbols / session_symbols` 换成任何引用值都会红）。
    session_symbols = int(report["session"]["symbols"])
    assert block["eta_hours_at_requested_budget"] > 0.0, block
    expected_eta = round(
        session_symbols / block["measured_symbols_per_second"] / 3600.0,
        3,
    )
    assert block["eta_hours_at_requested_budget"] == pytest.approx(expected_eta, abs=1e-9), (
        block,
        session_symbols,
    )
    # 这一枚**仍是跑器按构造写出的 0.0**：把它改成从存储面按 provenance 回读的那一半要动跑器，
    # 而跑器现在被批次 C 占着（两臂必须同源，㊵-754）。所以这里只钉"产出里有这枚键、值为 0.0"，
    # 不许把它读成"零奖励已被实测证实"。
    assert block["generated_reward_sum"] == 0.0, block

    #: 自述必须**自己承认**这枚键的身份：它是构造性的 0.0，不是测量。
    #: 真测量需要 `taiji/memory.py` 里的 generated 奖励累加器＝第四处产品码，仍等 owner；
    #: 而"零奖励"这件事实的真正保证住在入口签名里，所以下面那枚键才是可假的读数。
    assert (
        block["generated_reward_sum_basis"] == "literal_by_construction; not a measurement"
    ), block
    assert block["entry_has_reward_parameter"] is False, block


def test_the_feed_is_off_by_default_on_the_produced_face(tmp_path: Path) -> None:
    """J-N5r-1 的产出面（原来只有源码面）：旗标取 0 时写计数与那一行权重必须都停在 0。"""

    block = _run_tiny_arm(tmp_path, generated_feed_every=0)["generated_feed"]
    assert block["every_symbols"] == 0, block
    assert block["feeds"] == 0, block
    assert block["generated_write_count"] == 0, block
    assert block["generated_row_weight_max_abs"] == 0.0, block


def test_the_runners_help_screen_survives_a_gbk_console() -> None:
    """DEBT-G126 修法②：跑器的 `--help` 在这台机器上实测**改前 rc=1**（`⇒` 在 GBK 不可编码）。

    必须起真进程——capsys 恒绿看不见这一支；且按 DEBT-G85 的教训显式给 `encoding` 并断言
    `stdout is not None`，否则"rc=0 而 stdout 是 None"会被读成"旗标不在帮助里"。
    """

    import os
    import subprocess

    env = dict(os.environ, PYTHONIOENCODING="gbk")
    env.pop("PYTHONUTF8", None)
    proc = subprocess.run(
        [sys.executable, str(RUNNER), "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, proc.stderr[-400:]
    assert proc.stdout is not None
    assert "usage" in proc.stdout.lower()
    assert "--generated-feed-every" in proc.stdout


def test_the_reward_fact_flips_if_the_entry_grows_a_reward_parameter(tmp_path: Path) -> None:
    """`entry_has_reward_parameter` 必须能为真——否则它又是一枚恒假键。

    不动产品码的证法：拿一枚**带** reward 参数的假入口喂给同一个函数，它必须报真。
    """

    runner = _runner()

    class _Entry:
        def record_generated_prediction(self, symbol: int, *, reward: float = 0.0) -> None:
            pass

    assert runner._entry_has_reward_parameter(_Entry()) is True
    assert runner._entry_has_reward_parameter(_tiny_model()) is False
    assert (
        runner._entry_has_reward_parameter(object()) is True
    ), "入口不在场要保守报真，不许静默为假"


def test_the_runner_writes_lf_only_artifacts(tmp_path: Path) -> None:
    """DEBT-G115 的钉子（跑器那一枚实锤）：`run_report.json` 与 `progress.jsonl` 必须是纯 LF。

    实测形状：治疗臂那份 3,328 字节的 `run_report.json` 里有 92 枚 CR——autocrlf 会把入库那份
    归一成 LF，于是"仓库里看着没问题、原件与入库件按字节比就假红"。缺 `newline` 参数的
    `write_text` 在 Windows 上走 locale 行尾，这就是成因。这里用 bytes([13]) 数 CR，避免在测里
    写反斜杠字面量（这一层的传输会把两个反斜杠收成一个，㊵-748 撞过一次）。
    """

    report = _run_tiny_arm(tmp_path, generated_feed_every=4)
    arm_dir = Path(report["report_path"]).parent
    for name in ("run_report.json", "progress.jsonl"):
        raw = (arm_dir / name).read_bytes()
        assert raw, name
        assert raw.count(bytes([13])) == 0, (name, raw.count(bytes([13])))
    checkpoint = Path(report["checkpoint_path"])
    assert checkpoint.is_file()
