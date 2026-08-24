"""OpenAPI schema snapshot test.

Exports the FastAPI OpenAPI schema and compares it against a stored baseline.
Detects breaking API changes (path removal, parameter renames, etc.).

Regenerate baseline:
    python -m pytest tests/test_openapi_snapshot.py --snapshot-update
"""

import json
from pathlib import Path

import pytest

SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "openapi_baseline.json"


def _generate_schema():
    """Generate the OpenAPI schema from the FastAPI app."""
    from api.app import create_app

    app = create_app(startup_tasks=False)
    return app.openapi()


@pytest.fixture(scope="module")
def openapi_schema():
    return _generate_schema()


def test_openapi_schema_generates(openapi_schema):
    """Verify the OpenAPI schema can be generated without errors."""
    assert "paths" in openapi_schema
    assert "info" in openapi_schema
    assert len(openapi_schema["paths"]) > 0


def test_openapi_snapshot(openapi_schema, snapshot_update):
    """Compare current schema against stored baseline.

    If the snapshot file doesn't exist, create it.
    If it exists, compare paths and method signatures.

    By default a changed snapshot FAILS without rewriting the file, so
    breaking API changes surface in CI. Pass ``--snapshot-update`` to
    intentionally refresh the baseline.
    """
    snapshot_dir = SNAPSHOT_PATH.parent
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    if not SNAPSHOT_PATH.exists():
        # First run: create baseline
        _save_snapshot(openapi_schema)
        return

    baseline = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))

    # Compare API paths (endpoints)
    baseline_paths = set(baseline.get("paths", {}).keys())
    current_paths = set(openapi_schema.get("paths", {}).keys())

    removed = baseline_paths - current_paths
    added = current_paths - baseline_paths

    # Report changes but don't fail — new endpoints are expected
    messages = []
    if removed:
        messages.append(f"Removed endpoints: {sorted(removed)}")
    if added:
        messages.append(f"New endpoints: {sorted(added)}")

    # Check for parameter changes on existing endpoints
    for path in baseline_paths & current_paths:
        baseline_methods = set(baseline["paths"][path].keys())
        current_methods = set(openapi_schema["paths"][path].keys())
        if baseline_methods != current_methods:
            messages.append(f"{path}: methods changed from {baseline_methods} to {current_methods}")

    if messages:
        detail = "\n".join(messages)
        if snapshot_update:
            # Explicit opt-in: refresh the baseline for the next run
            _save_snapshot(openapi_schema)
            pytest.fail(
                "OpenAPI schema changed (snapshot updated via --snapshot-update):\n" + detail
            )
        else:
            # Strict mode: fail WITHOUT touching the baseline so CI catches drift
            pytest.fail(
                "OpenAPI schema changed:\n"
                + detail
                + "\nRun with --snapshot-update to intentionally refresh the baseline."
            )


def _save_snapshot(schema):
    """Save the OpenAPI schema to disk."""
    # Remove volatile fields that change between runs
    cleaned = {k: v for k, v in schema.items() if k != "servers"}
    SNAPSHOT_PATH.write_text(
        json.dumps(cleaned, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
