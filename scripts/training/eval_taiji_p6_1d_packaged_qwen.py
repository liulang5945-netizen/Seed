"""Run the real Qwen semantic provider through two packaged backend cycles."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_FORMAT = "taiji-w7-p6-1d-packaged-qwen-lifecycle-v1"


def _request(
    base_url: str,
    path: str,
    payload: dict[str, object] | None = None,
    *,
    timeout_seconds: float = 5.0,
) -> dict[str, object]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_for_health(base_url: str, *, timeout_seconds: float) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        try:
            return _request(base_url, "/api/health", timeout_seconds=5.0)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
            time.sleep(0.5)
    raise RuntimeError(f"packaged backend health timeout: {last_error}")


def _run_cycle(
    package: Path,
    *,
    port: int,
    model_dir: Path,
    model_digest: str,
    data_root: Path,
    log_dir: Path,
    prompt: str,
    timeout_seconds: float,
    cycle: int,
) -> dict[str, object]:
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.update(
        {
            "TAIJI_SEMANTIC_PROVIDER_MODEL_DIR": str(model_dir),
            "TAIJI_SEMANTIC_PROVIDER_MODEL_DIGEST": model_digest,
            "SEED_DATA_ROOT": str(data_root),
        }
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"p6-1d-packaged-cycle-{cycle}.log"
    with log_path.open("wb") as log_file:
        process = subprocess.Popen(
            [str(package), "127.0.0.1", str(port)],
            cwd=str(package.parent),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    try:
        health = _wait_for_health(base_url, timeout_seconds=timeout_seconds)
        activation = _request(
            base_url,
            "/api/runtime/activate",
            {},
            timeout_seconds=timeout_seconds,
        )
        runtime = activation.get("runtime", {})
        provider_status = runtime.get("semantic_provider", {})
        admission = _request(
            base_url,
            "/api/chat/workbench/interpret",
            {"prompt": prompt, "history": [], "constraints": ["只读"]},
            timeout_seconds=timeout_seconds,
        )
        evidence = admission.get("provider_evidence", {})
        execution = admission.get("execution", {})
        return {
            "cycle": cycle,
            "port": port,
            "pid": process.pid,
            "log": str(log_path),
            "health": {
                "reachable": True,
                "state": health.get("status", health.get("state", "")),
            },
            "activation": {
                "status": activation.get("status"),
                "checkpoint_id": activation.get("checkpoint_id"),
                "semantic_provider": provider_status,
            },
            "admission": {
                "format": admission.get("format"),
                "interpretation_status": admission.get("interpretation", {}).get("status"),
                "semantic_step_count": len(admission.get("decomposition", {}).get("steps", [])),
                "provider_id": evidence.get("provider_id"),
                "evidence_digest": evidence.get("evidence_digest"),
                "execution": execution,
            },
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def _probe_invalid_digest(
    package: Path,
    *,
    port: int,
    model_dir: Path,
    data_root: Path,
    log_dir: Path,
    timeout_seconds: float,
) -> dict[str, object]:
    """Ensure a frozen client rejects an unallowlisted model before loading it."""

    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.update(
        {
            "TAIJI_SEMANTIC_PROVIDER_MODEL_DIR": str(model_dir),
            "TAIJI_SEMANTIC_PROVIDER_MODEL_DIGEST": "0" * 64,
            "SEED_DATA_ROOT": str(data_root),
        }
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "p6-1d-packaged-invalid-digest.log"
    with log_path.open("wb") as log_file:
        process = subprocess.Popen(
            [str(package), "127.0.0.1", str(port)],
            cwd=str(package.parent),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    try:
        _wait_for_health(base_url, timeout_seconds=timeout_seconds)
        try:
            _request(
                base_url,
                "/api/runtime/activate",
                {},
                timeout_seconds=timeout_seconds,
            )
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return {
                "status": "rejected",
                "http_status": exc.code,
                "detail": detail,
                "log": str(log_path),
            }
        return {
            "status": "accepted_unexpectedly",
            "http_status": 200,
            "detail": "invalid digest was accepted",
            "log": str(log_path),
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def evaluate(
    package: Path,
    *,
    model_dir: Path,
    model_digest: str,
    data_root: Path,
    log_dir: Path,
    port: int,
    prompt: str,
    timeout_seconds: float,
) -> dict[str, object]:
    if sys.platform != "win32":
        raise RuntimeError("P6-1d packaged Qwen evaluator requires the Windows SeedBackend.exe")
    if not package.is_file():
        raise FileNotFoundError(f"packaged backend not found: {package}")
    cycles = [
        _run_cycle(
            package,
            port=port,
            model_dir=model_dir,
            model_digest=model_digest,
            data_root=data_root,
            log_dir=log_dir,
            prompt=prompt,
            timeout_seconds=timeout_seconds,
            cycle=1,
        ),
        _run_cycle(
            package,
            port=port + 1,
            model_dir=model_dir,
            model_digest=model_digest,
            data_root=data_root,
            log_dir=log_dir,
            prompt=prompt,
            timeout_seconds=timeout_seconds,
            cycle=2,
        ),
    ]
    invalid_digest_probe = _probe_invalid_digest(
        package,
        port=port + 2,
        model_dir=model_dir,
        data_root=data_root,
        log_dir=log_dir,
        timeout_seconds=timeout_seconds,
    )
    checks = {
        "packaged_backend_exists": package.is_file(),
        "both_cycles_reached_health": all(item["health"]["reachable"] for item in cycles),
        "both_activations_succeeded": all(
            item["activation"]["status"] == "ok" for item in cycles
        ),
        "provider_attached_after_each_restart": all(
            item["activation"]["semantic_provider"]["state"] == "attached"
            for item in cycles
        ),
        "same_provider_rebound": (
            cycles[0]["activation"]["semantic_provider"]["provider_id"]
            == cycles[1]["activation"]["semantic_provider"]["provider_id"]
        ),
        "real_semantic_admission_each_cycle": all(
            item["admission"]["format"] == "taiji-semantic-provider-admission-v1"
            and item["admission"]["interpretation_status"] in {"resolved", "candidate"}
            and item["admission"]["provider_id"]
            and item["admission"]["evidence_digest"]
            for item in cycles
        ),
        "no_execution_authority_at_interpret": all(
            item["admission"]["execution"]["action_intent"] is None
            and item["admission"]["execution"]["tool_call"] is None
            and item["admission"]["execution"]["side_effects"] is False
            for item in cycles
        ),
        "invalid_digest_fails_closed": (
            invalid_digest_probe["status"] == "rejected"
            and invalid_digest_probe["http_status"] == 500
            and "allowlisted" in invalid_digest_probe["detail"]
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "package": {"path": str(package), "backend": package.name},
        "model": {"model_dir": str(model_dir), "model_digest": model_digest},
        "request": {"prompt": prompt, "constraints": ["只读"]},
        "cycles": cycles,
        "invalid_digest_probe": invalid_digest_probe,
        "checks": checks,
        "gate": {
            "passed": all(checks.values()),
            "criterion": (
                "the frozen SeedBackend must explicitly rebind the content-addressed Qwen artifact "
                "after restart and admit real semantic evidence without execution authority"
            ),
        },
        "boundary": (
            "This proves two local packaged backend restart/rebind cycles. It does not prove model "
            "quality, CUDA, installer UI, multi-version real rotation, or open-domain intelligence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-digest", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "logs")
    parser.add_argument("--port", type=int, default=18100)
    parser.add_argument("--prompt", default="读取 README.md 并确认当前内容")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_p6_1d_packaged_qwen_20260831.json",
    )
    args = parser.parse_args()
    report = evaluate(
        args.package,
        model_dir=args.model,
        model_digest=args.model_digest,
        data_root=args.data_root,
        log_dir=args.log_dir,
        port=args.port,
        prompt=args.prompt,
        timeout_seconds=args.timeout_seconds,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["gate"]["passed"] else 1)


if __name__ == "__main__":
    main()
