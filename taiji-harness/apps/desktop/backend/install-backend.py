"""Offline backend installer for the Taiji desktop bundle (D2 P1-②).

Runs inside the shipped app with the primary-runtime interpreter.  Idempotent:
if the target venv already satisfies the manifest's pinned wheels the installer
exits successfully without touching anything.  Fully offline: pip is pointed at
the shipped wheelhouse with --no-index, and every wheel is verified against the
manifest's sha256 before installation begins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

RECEIPT_NAME = "backend-install-receipt.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_wheelhouse(wheelhouse: Path, manifest: dict) -> list[Path]:
    wheels: list[Path] = []
    for wheel in manifest["wheels"]:
        path = wheelhouse / wheel["filename"]
        if not path.is_file():
            raise SystemExit(f"backend install: missing wheel {wheel['filename']}")
        if _sha256(path) != wheel["sha256"]:
            raise SystemExit(f"backend install: wheel {wheel['filename']} fails sha256 verification")
        wheels.append(path)
    return wheels


def _venv_satisfies(venv_python: Path, manifest: dict) -> bool:
    """True when the venv already reports every pinned package at its exact version.

    Asked of the venv's own import metadata (not `pip freeze`): a locally built
    seed wheel pins as ``name @ file://…`` and an editable install as ``-e``,
    and freeze's path spellings are separator-unstable on Windows.
    """
    expected = {wheel["name"]: wheel["version"] for wheel in manifest["wheels"]}
    script = """import json, sys
from importlib.metadata import PackageNotFoundError, version
expected = json.loads(sys.argv[1])
def _missing(name, want):
    try:
        return version(name) != want
    except PackageNotFoundError:
        return True
missing = [name for name, want in expected.items() if _missing(name, want)]
if missing:
    print(json.dumps(missing))
raise SystemExit(1 if missing else 0)
"""
    result = subprocess.run(
        [str(venv_python), "-I", "-B", "-c", script, json.dumps(expected)],
        capture_output=True, text=True, check=False,
    )
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", required=True, help="shipped wheelhouse directory")
    parser.add_argument("--manifest", required=True, help="backend-manifest.json path")
    parser.add_argument("--python", required=True, help="primary-runtime interpreter used to create the venv")
    parser.add_argument("--target", required=True, help="target venv directory")
    parser.add_argument("--receipt", default="", help="where to write the install receipt")
    args = parser.parse_args()

    started = time.time()
    wheelhouse = Path(args.wheelhouse)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if manifest.get("format") != "taiji-backend-wheelhouse-v1":
        raise SystemExit(f"backend install: unsupported wheelhouse format {manifest.get('format')!r}")

    target = Path(args.target)
    venv_python = target / "Scripts" / "python.exe"
    if not venv_python.is_file():
        venv_python = target / "bin" / "python"
    if not venv_python.is_file():
        print("backend install: creating venv", flush=True)
        subprocess.run([args.python, "-m", "venv", str(target)], check=True)
        venv_python = target / "Scripts" / "python.exe"
        if not venv_python.is_file():
            venv_python = target / "bin" / "python"

    if _venv_satisfies(venv_python, manifest):
        print("backend install: target already satisfies the manifest; nothing to do", flush=True)
        return 0

    wheels = _verify_wheelhouse(wheelhouse, manifest)
    requirements = wheelhouse / "requirements-lock.txt"
    requirements.write_text(
        "".join(
            f"{wheel['name']}=={wheel['version']} --hash=sha256:{wheel['sha256']}\n"
            for wheel in manifest["wheels"]
        ),
        encoding="utf-8",
    )
    print(f"backend install: installing {len(wheels)} wheels offline", flush=True)
    subprocess.run(
        [str(venv_python), "-m", "pip", "install",
         "--no-index", "--find-links", str(wheelhouse),
         "--no-deps", "--require-hashes",
         "-r", str(requirements)],
        check=True,
    )
    receipt = {
        "format": "taiji-backend-install-receipt-v1",
        "installed_at": time.time(),
        "duration_s": round(time.time() - started, 1),
        "wheels": len(wheels),
        "snapshot": manifest.get("source", {}).get("digest", ""),
        "venv": str(target),
    }
    receipt_path = Path(args.receipt) if args.receipt else target / RECEIPT_NAME
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("backend install: done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
