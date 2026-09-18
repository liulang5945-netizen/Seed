"""DEBT-I7 guard: no test run may rewrite the model the product serves.

``checkpoints/seed_corpus.pt`` is what the app loads at start-up, and ``*.pt`` is not in git.
Measured twice this round: a full-suite run re-saved that file from ``trainer = api_seed_runtime``
and moved its tick ``36 -> 2`` (the 09-18 re-sample also shows a different byte count from the
09-15 sample at the same tick=2), i.e. the writer is **not** idempotent and whatever it overwrote is
unrecoverable.  The root cause was structural: one constant served both "what the product loads"
and "where an unqualified ``save()`` writes", so a test could not call ``save()`` safely at all.

Asserted here, in both directions:

* the save fallback resolves **outside** the repository's ``checkpoints/`` directory;
* an explicit path still wins, and so does the loaded-from path -- a redirect that swallowed either
  would break the product's own resume behaviour;
* the **load** default still points at the real file, i.e. only writes were redirected;
* a real ``save()`` on a runtime built without a path leaves the product file byte-identical.

Note the lookup style: every assertion reads ``seed_runtime.<ATTRIBUTE>`` rather than importing the
name, because ``from api.seed_runtime import DEFAULT_SAVE_TARGET`` would bind the pre-fixture value
at collection time and make this file blind to the very thing it checks.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PRODUCT_DEFAULT = REPO / "checkpoints" / "seed_corpus.pt"
CHECKPOINTS_DIR = REPO / "checkpoints"


@pytest.fixture(scope="module")
def seed_runtime() -> object:
    from api import seed_runtime as module

    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_an_unqualified_save_does_not_target_the_product_file(seed_runtime) -> None:
    target = seed_runtime.resolve_save_target()
    assert target != seed_runtime.DEFAULT_CHECKPOINT
    assert CHECKPOINTS_DIR not in target.parents
    assert not str(target).startswith(str(CHECKPOINTS_DIR))


def test_explicit_and_source_paths_still_beat_the_redirected_default(
    tmp_path, seed_runtime
) -> None:
    explicit = tmp_path / "explicit.pt"
    loaded_from = tmp_path / "loaded_from.pt"
    assert seed_runtime.resolve_save_target(explicit, loaded_from) == explicit
    assert seed_runtime.resolve_save_target(None, loaded_from) == loaded_from
    assert seed_runtime.resolve_save_target(None, None) == seed_runtime.DEFAULT_SAVE_TARGET


def test_the_default_load_source_was_not_redirected(seed_runtime) -> None:
    """只重定向写靶。若连读侧也搬走，"默认入口服务哪个模型"这条判断就没有对象了。"""

    assert seed_runtime.DEFAULT_CHECKPOINT == PRODUCT_DEFAULT
    assert PRODUCT_DEFAULT.is_file()
    assert seed_runtime.DEFAULT_SAVE_TARGET != PRODUCT_DEFAULT


def test_a_real_save_leaves_the_product_checkpoint_byte_identical(seed_runtime) -> None:
    from seed import Seed

    before = _sha(PRODUCT_DEFAULT)
    runtime = seed_runtime.SeedRuntime(Seed(episode_id="default-write-isolation"))
    written = runtime.save()

    assert written == seed_runtime.DEFAULT_SAVE_TARGET
    assert written.is_file(), "the save itself must still succeed, elsewhere"
    assert _sha(PRODUCT_DEFAULT) == before
    assert "resolve_save_target" in seed_runtime.SeedRuntime.save.__code__.co_names
