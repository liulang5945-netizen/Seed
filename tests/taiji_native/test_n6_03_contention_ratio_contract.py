"""竞争核对照比例尺（`measure_taiji_contention_ratio.py`，DEBT-G94 前置）的契约测。

这台仪器出版的是**两档之间的比值**，所以本册钉的是"两档是否真的各跑过、竞争核是否真的活过"，
而不是比值的大小——比值是机器现场状态，钉死它就是把一次读数冒充成契约。

* 两档各 `repeats` 次 ⇒ `quiet_seconds`／`plus_one_seconds` 长度必须等于 `repeats`；
* **竞争核必须被证实起过且探针跑完时还活着**：`competitor_pids` 长度等于 `repeats`，
  而仪器在竞争核提前退出时是 `raise`（本册第 4 支用一支"自己会退出"的探针把它逼出来）；
* `repeats < 2` 与探针本身失败都必须**响亮拒绝**（rc=2）且不写读数件；
* 仪器不在仓库里留临时探针产物：读数件旁边不许出现 `probe-*.json`。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "measure_taiji_contention_ratio.py"

PROBE_BODY = """
import argparse, json, time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--out-report", type=Path, required=True)
p.add_argument("--spin", type=float, default=0.05)
a = p.parse_args()
end = time.time() + a.spin
total = 0
while time.time() < end:
    total += sum(range(200))
a.out_report.parent.mkdir(parents=True, exist_ok=True)
a.out_report.write_text(json.dumps({"status": "ok", "work": total}), encoding="utf-8", newline="\\n")
"""

FAILING_PROBE_BODY = """
import sys

sys.stderr.write("probe exploded\\n")
raise SystemExit(1)
"""


def _write_probe(root: Path, name: str, body: str) -> Path:
    path = root / name
    path.write_text(body, encoding="utf-8", newline="\n")
    return path


def _run(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, check=False, cwd=str(REPO)
    )


def _payload(proc: subprocess.CompletedProcess[bytes]) -> dict[str, Any]:
    text = proc.stdout.decode("utf-8", errors="replace")
    assert text, proc.stderr[-400:]
    return json.loads(text)


def test_both_faces_run_and_the_competitor_is_proven_alive(tmp_path: Path) -> None:
    probe = _write_probe(tmp_path, "probe.py", PROBE_BODY)
    out = tmp_path / "ratio.json"
    proc = _run(["--probe", str(probe), "--repeats", "2", "--out-report", str(out)])
    assert proc.returncode == 0, (proc.returncode, proc.stdout[-400:], proc.stderr[-400:])
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["format"] == "taiji-contention-ratio-v1", payload
    assert payload["status"] == "ok", payload
    assert len(payload["quiet_seconds"]) == 2, payload
    assert len(payload["plus_one_seconds"]) == 2, payload
    assert len(payload["competitor_pids"]) == 2, payload
    assert payload["median_quiet_seconds"] > 0.0, payload
    assert payload["ratio_plus_over_quiet"] > 0.0, payload
    #: 每档每趟探针前后各一次负载读数 ⇒ 每档 2×repeats 条，且带的是仪器原话而不是本器造的字段。
    for face in ("quiet", "plus_one"):
        rows = payload["load_face_by_run"][face]
        assert len(rows) == 4, (face, len(rows))
        for row in rows:
            assert {"python_process_count", "python_processes_burning_cpu"} <= set(row), row
    assert "not a calibration curve" in payload["reading_limit"], payload["reading_limit"]
    #: 临时探针产物只活在仪器自己的 scratch 目录里，不许落到读数件旁边。
    assert list(out.parent.glob("probe-*.json")) == [], list(out.parent.glob("probe-*.json"))
    assert b"\r" not in out.read_bytes(), "读数件必须是 LF"


def test_repeats_below_two_refuses_and_writes_no_artifact(tmp_path: Path) -> None:
    probe = _write_probe(tmp_path, "probe.py", PROBE_BODY)
    out = tmp_path / "refused.json"
    proc = _run(["--probe", str(probe), "--repeats", "1", "--out-report", str(out)])
    assert proc.returncode == 2, (proc.returncode, proc.stdout[-400:])
    payload = _payload(proc)
    assert payload["status"] == "measurement_failed", payload
    assert "repeats must be >= 2" in payload["error"], payload
    assert not out.exists(), "拒绝路径不许留下读数件"


def test_a_failing_probe_refuses_instead_of_publishing_a_ratio(tmp_path: Path) -> None:
    """反向那一支：探针自己不落产物时，仪器不能拿"零秒"算出一个看起来很合理的比值。"""

    probe = _write_probe(tmp_path, "bad.py", FAILING_PROBE_BODY)
    out = tmp_path / "nope.json"
    proc = _run(["--probe", str(probe), "--repeats", "2", "--out-report", str(out)])
    assert proc.returncode == 2, (proc.returncode, proc.stdout[-400:])
    payload = _payload(proc)
    assert payload["status"] == "measurement_failed", payload
    assert "probe rc=1" in payload["error"], payload
    assert not out.exists(), "拒绝路径不许留下读数件"
