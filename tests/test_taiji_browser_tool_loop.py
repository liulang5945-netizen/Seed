"""The browser environment driven through Taiji's real tool loop.

Adapter here is the real class with no checkpoint: the point is the wiring, not
a trained policy. Each test asserts one boundary:

- default (no ``feed_world_state``): reward and sensation reach Taiji, the
  environment's world state does not — today's behavior, unchanged.
- opted in: the page object and its affordances enter Taiji's cognitive world,
  and the world tick is Taiji's clock, not the environment's step count.
"""

import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("playwright", reason="the browser environment needs Playwright")

from taiji import (  # noqa: E402
    GenerationController,
    Goal,
    GoalPlanner,
    PlanningCandidate,
    TSKV8Adapter,
    WorldAction,
)
from taiji.environment_browser import BrowserEnvironment  # noqa: E402

PAGE = b"""<!doctype html><html><head><title>Taiji probe</title></head><body>
<h1>Page body</h1><a href="/next.html">next</a><button>go</button>
</body></html>"""


class _Site(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - http.server API
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(PAGE)))
        self.end_headers()
        self.wfile.write(PAGE)

    def log_message(self, *_args: object) -> None:
        """Keep pytest output clean."""


@pytest.fixture(scope="module")
def site() -> str:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Site)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    yield f"http://{host}:{port}"
    server.shutdown()
    server.server_close()


def _ready_adapter() -> TSKV8Adapter:
    """An adapter with a pending motor intent, as the tool path requires."""
    adapter = TSKV8Adapter()
    adapter.attach_generation_controller(GenerationController())
    adapter.attach_goal_planner(GoalPlanner())
    adapter.set_goals((Goal("stay-informed", "read the page", priority=1.0),))
    adapter.observe(97, learn=False)
    adapter.plan_actions(
        (
            PlanningCandidate(
                candidate_id="look",
                action=WorldAction(action_id="look", kind="lookup_page", tick=1),
                predicted_reward=0.8,
                success_probability=0.9,
                expected_progress=0.8,
            ),
        ),
    )
    # plan_actions only persists the plan; act() is what leaves a pending intent
    # for the tool organ to render.
    adapter.act(
        (10, 11),
        sample=False,
        procedural_action_kinds=("lookup_page", "idle"),
        use_plan=True,
    )
    return adapter


def _call(adapter: TSKV8Adapter, tool_name: str, parameters: dict[str, object]):
    """A tool call for `tool_name` carrying `parameters`.

    ``generate_tool_call`` renders the pending intent and takes no parameters —
    the environment's parameters come from the intent, so a test supplies them
    by replacing the rendered call's parameters.
    """
    return replace(adapter.generate_tool_call(tool_name=tool_name), parameters=parameters)


def test_default_path_keeps_the_world_out_of_taiji(site: str) -> None:
    """Unchanged behavior: the tool result settles, the world does not."""
    adapter = _ready_adapter()
    with BrowserEnvironment() as env:
        outcome = adapter.execute_tool_call(env, call=_call(adapter, "goto", {"url": f"{site}/a"}))
    assert outcome.reward == 1.0
    assert adapter._cognitive_state.world.entities == ()


def test_opted_in_puts_the_page_into_taijis_world(site: str) -> None:
    adapter = _ready_adapter()
    with BrowserEnvironment() as env:
        outcome = adapter.execute_tool_call(
            env, call=_call(adapter, "goto", {"url": f"{site}/a"}), feed_world_state=True
        )

    assert outcome.reward == 1.0
    world = adapter._cognitive_state.world
    assert any(entity.startswith("page:") for entity in world.entities), world.entities
    # affordances arrived as affordances, not as text
    assert {aff.action_kind for aff in world.affordances} >= {"link", "button"}
    # the world tick is Taiji's clock, one past its own
    assert world.tick == outcome.tick


def test_failing_action_settles_negative_and_stays_learnable(site: str) -> None:
    adapter = _ready_adapter()
    with BrowserEnvironment() as env:
        outcome = adapter.execute_tool_call(
            env, call=_call(adapter, "click", {"index": 999}), feed_world_state=True
        )

    assert outcome.reward == -1.0
    assert outcome.success is False
    # the failure reason rode along in the world state
    reasons = [
        dict(event.attributes).get("reason", "")
        for event in adapter._cognitive_state.world.events
        if event.kind == "action_failed"
    ]
    assert reasons and reasons[0]
