"""机器负载读数器（`read_taiji_machine_load.py`，DEBT-G94 修法①）的契约测。

这台仪器的全部用处是"墙钟报价旁边那枚同轮负载数"取得对，所以本册钉的是取法而不是现场值：

* **自证在场**：本器自己是一枚 python 进程，连自己都数不到 ⇒ `python_process_count` 就是假零；
* **排序按累计 CPU 秒**：交付给 owner 的是"哪几枚最可疑"，顺序错了就点名错人；
* **窗口参数必须能为假**：`--sample-seconds 0` 要**响亮拒绝**（rc=2），不许把"取不到"降级成 0；
* **`--top` 是真的在切**：给 1 就必须只出版 1 枚（否则那枚清单是恒在的装饰）；
* `--out` 与 stdout 出版**同一枚读数**（按解析后的对象比），且落盘件里不许有 CR——win32 上子进程
  stdout 自带 `\\r\\n` 翻牌，按字节比会把这条正确行为测成红（本轮第一次就犯在按字节比）。

**本册一律不钉本机当下的枚数或占用率**——那是现场读数，owner 杀掉两枚空转探针就会变；
把它钉进测里就是把一次读数冒充成契约（本仓纪律：状态列会比证据先过期）。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "read_taiji_machine_load.py"


def _run(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        check=False,
        cwd=str(REPO),
    )


def _payload(proc: subprocess.CompletedProcess[bytes]) -> dict[str, Any]:
    assert proc.returncode == 0, (proc.returncode, proc.stderr[-400:])
    #: 调用面必须显式给编码，否则 `-X utf8` 下子进程的 GBK 字节会让 stdout 变成 None（DEBT-G85）。
    text = proc.stdout.decode("utf-8", errors="replace")
    assert text, proc.stderr[-400:]
    return json.loads(text)


def test_the_reader_counts_itself_so_zero_is_never_silent(tmp_path: Path) -> None:
    out = tmp_path / "load.json"
    payload = _payload(_run(["--out", str(out)]))
    assert payload["status"] == "ok", payload
    assert payload["format"] == "taiji-machine-load-v1", payload
    #: 这一枚是整册的判别式：取法坏时它会为 false，取法好时它不可能被"没人抢核"影响。
    assert payload["reader_counted_itself"] is True, payload
    assert payload["python_process_count"] >= 1, payload
    assert payload["python_processes_burning_cpu"] <= payload["python_process_count"], payload
    assert payload["logical_cpus"] >= 2, payload
    assert 0.0 <= payload["system_cpu_percent"] <= 100.0 * payload["logical_cpus"], payload
    assert payload["disclosure"] in {"ok", "single_process_face"}, payload


def test_the_hot_list_is_ordered_by_cumulative_cpu_seconds(tmp_path: Path) -> None:
    payload = _payload(_run(["--out", str(tmp_path / "load.json"), "--top", "5"]))
    rows = payload["top_python_by_cpu_seconds"]
    assert rows, payload
    seconds = [one["cpu_seconds"] for one in rows]
    assert seconds == sorted(seconds, reverse=True), seconds
    #: 每一枚都要能指认身份——只有秒数没有 PID／启动时刻的清单不能交给 owner 去处置。
    for one in rows:
        assert one["pid"] > 0 and one["created"] and "cmdline_head" in one, one


def test_the_top_knob_actually_truncates(tmp_path: Path) -> None:
    """`--top 1` 只出版一枚：证明那枚清单是被参数切的，不是一段恒在的装饰。"""

    payload = _payload(_run(["--out", str(tmp_path / "load.json"), "--top", "1"]))
    assert len(payload["top_python_by_cpu_seconds"]) == 1, payload["top_python_by_cpu_seconds"]


def test_a_zero_sampling_window_refuses_loudly_instead_of_reporting_zero() -> None:
    proc = _run(["--sample-seconds", "0", "--out", "output/should_not_exist.json"])
    assert proc.returncode == 2, (proc.returncode, proc.stdout[-400:], proc.stderr[-400:])
    payload = json.loads(proc.stdout.decode("utf-8", errors="replace"))
    assert payload["status"] == "load_read_failed", payload
    assert "positive" in payload["error"], payload
    assert not (REPO / "output" / "should_not_exist.json").exists()


def test_the_written_file_and_the_screen_are_the_same_reading(tmp_path: Path) -> None:
    out = tmp_path / "load.json"
    proc = _run(["--out", str(out)])
    file_bytes = out.read_bytes()
    #: **比的是解析后的两枚读数，不是两段字节**：win32 上子进程的 stdout 会把 `\\n` 翻成 `\\r\\n`，
    #: 而落盘按仓内规矩钉死 LF——按字节比会把一条正确行为测成红（本轮第一次就这么红的）。
    #: 件里不许有 CR 这一条单独钉，它才是本仓真正要的不变量。
    assert b"\r" not in file_bytes, "落盘件必须是 LF"
    assert json.loads(file_bytes.decode("utf-8")) == json.loads(
        proc.stdout.decode("utf-8")
    ), "两枚取数必须同源"
