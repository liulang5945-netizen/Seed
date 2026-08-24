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
def _reset_global_app_state() -> None:
    """Reinitialize the neuroplex app-state singleton for a clean slate.

    The singleton is created at import time and persists across tests. We
    replace it with a fresh instance at session teardown only if the module
    imported cleanly, so importing this fixture never fails a collection.
    """
    yield
    try:
        import neuroplex.core.app_state as _app_state_mod
    except Exception:  # pragma: no cover - neuroplex optional in some envs
        return
    try:
        _app_state_mod.app_state = _app_state_mod.AppState()
    except Exception:  # pragma: no cover - never break teardown
        return


@pytest.fixture(autouse=False)
def reset_state():
    """Function-level state reset for tests with side effects.

    Opt-in: apply ``@pytest.mark.reset_state`` or request this fixture
    explicitly to get a fresh ``app_state`` before each test function.
    This is finer-grained than the session-level reset above.
    """
    try:
        import neuroplex.core.app_state as _mod

        _mod.app_state = _mod.AppState()
    except Exception:
        pass
    yield
    try:
        import neuroplex.core.app_state as _mod

        _mod.app_state = _mod.AppState()
    except Exception:
        pass
