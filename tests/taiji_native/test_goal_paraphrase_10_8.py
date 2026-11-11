"""Contract guard for 10.8: paraphrases avoid the action template but keep the names."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "verify_taiji_goal_paraphrase_10_8.py"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from taiji.goal_paraphrase import STYLES, action_sentence, paraphrase_goal  # noqa: E402
from taiji.native_goal_world import OBJECTS, PLACES  # noqa: E402


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


VERIFY = _load("verify_taiji_goal_paraphrase_10_8", SCRIPT)


def test_paraphrases_drop_the_template_and_keep_names():
    obj, place = OBJECTS[0], PLACES[0]
    action = action_sentence(obj, place)
    texts = [paraphrase_goal(obj, place, style) for style in STYLES]
    assert len(set(texts)) == len(STYLES)
    for text in texts:
        assert text != action
        assert "把" not in text
        assert obj in text and place in text


def test_overlap_edges():
    assert VERIFY.overlap("把蓝杯放到窗台", "把蓝杯放到窗台") == 1.0
    assert VERIFY.overlap("甲乙", "丙丁") == 0.0


def test_named_entities_is_unique():
    goal = paraphrase_goal(OBJECTS[1], PLACES[2], "state")
    objects, places = VERIFY.named_entities(goal)
    assert objects == [OBJECTS[1]]
    assert places == [PLACES[2]]
