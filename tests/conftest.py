"""Shared pytest fixtures for the Seed test suite.

Minimal by design:
- Make logs observable when a test fails (root logger to WARNING).
- Optionally reset the global ``neuroplex`` app-state singleton between
  sessions so one test's side effects don't leak into another. This only
  touches the singleton in conftest; it never mutates library/source code.
- Provide an opt-in ``reset_state`` fixture for tests that need
  function-level state isolation (apply via ``@pytest.mark.reset_state``).
"""

from __future__ import annotations

import logging
from dataclasses import fields

import pytest


def pytest_addoption(parser):
    """Register custom CLI options.

    --snapshot-update: allow snapshot-based tests (e.g. OpenAPI schema) to
    rewrite their baseline files. Without this flag, a changed snapshot fails
    the test *without* updating the file, so breaking changes surface in CI.
    """
    parser.addoption(
        "--snapshot-update",
        action="store_true",
        default=False,
        help="Update stored snapshots instead of failing on drift.",
    )


@pytest.fixture(scope="session")
def snapshot_update(request):
    """True when the user passed --snapshot-update."""
    return request.config.getoption("--snapshot-update")


@pytest.fixture(autouse=True, scope="session")
def _observe_logging() -> None:
    """Ensure WARNING/ERROR logs are emitted so failures are diagnosable."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(name)s %(levelname)s %(message)s",
    )
    yield


@pytest.fixture(autouse=True, scope="session")
def _redirect_default_save_target(tmp_path_factory) -> None:
    """Keep ``SeedRuntime.save()`` from overwriting the model the product serves (DEBT-I7).

    Only the *save* target moves. ``DEFAULT_CHECKPOINT`` -- what the app loads -- is left pointing
    at the real file, because several tests assert against the shipped base and reads are not the
    hazard. The invariant itself is asserted by
    ``tests/taiji_native/test_default_checkpoint_isolation_contract.py``.
    """

    try:
        from api import seed_runtime as _runtime
    except Exception:  # pragma: no cover - optional dependency, same style as the reset fixture
        yield
        return
    original = _runtime.DEFAULT_SAVE_TARGET
    _runtime.DEFAULT_SAVE_TARGET = tmp_path_factory.mktemp("default-save-target") / "seed_corpus.pt"
    yield
    _runtime.DEFAULT_SAVE_TARGET = original


def _reset_app_state_in_place(module) -> None:
    """Preserve imported singleton references while replacing per-test values."""
    fresh = module.AppState()
    for descriptor in fields(fresh):
        setattr(module.app_state, descriptor.name, getattr(fresh, descriptor.name))


@pytest.fixture(autouse=True, scope="session")
def _reset_global_app_state() -> None:
    """Reinitialize the platform app-state singleton for a clean slate.

    Reset fields in place so modules that imported app_state directly observe
    the reset too; replacing the module attribute would leave stale aliases.
    """
    yield
    try:
        import seed_platform.app_state as _app_state_mod
    except Exception:  # pragma: no cover - optional platform dependencies
        return
    try:
        _reset_app_state_in_place(_app_state_mod)
    except Exception:  # pragma: no cover - never break teardown
        return


@pytest.fixture(autouse=True)
def _apply_reset_state_marker(request):
    """The marker opts into the fixture; registering a marker alone does not."""
    if request.node.get_closest_marker("reset_state") is not None:
        request.getfixturevalue("reset_state")


@pytest.fixture(autouse=False)
def reset_state():
    """Function-level state reset for tests with side effects.

    Opt-in: apply ``@pytest.mark.reset_state`` or request this fixture
    explicitly to get a fresh ``app_state`` before each test function.
    This is finer-grained than the session-level reset above.
    """
    try:
        import seed_platform.app_state as _mod

        _reset_app_state_in_place(_mod)
    except Exception:
        pass
    yield
    try:
        import seed_platform.app_state as _mod

        _reset_app_state_in_place(_mod)
    except Exception:
        pass
