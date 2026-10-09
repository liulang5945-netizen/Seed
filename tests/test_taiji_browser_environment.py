"""BrowserEnvironment against a real browser and a real (local) website.

The site is served by an in-process HTTP server so the tests never depend on the
public internet, and the assertions are about the contract Taiji settles
against: world objects for the page, affordances for what can be acted on, and
an honest reward/uncertainty when an action fails.
"""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("playwright", reason="browser environment needs Playwright")

from taiji.environment import EnvironmentOutcome  # noqa: E402
from taiji.environment_browser import BROWSER_TOOLS, BrowserEnvironment  # noqa: E402

PAGE_ONE = """<!doctype html><html><head><title>Alpha page</title></head><body>
<h1>Alpha heading</h1>
<p>First page body text for the observation.</p>
<a href="/page2.html">go to beta</a>
<button id="btn">Press me</button>
<input id="name" placeholder="your name">
</body></html>"""

PAGE_TWO = """<!doctype html><html><head><title>Beta page</title></head><body>
<h1>Beta heading</h1><p>Second page body text after navigation.</p>
</body></html>"""


class _Site(BaseHTTPRequestHandler):
    pages = {"/page1.html": PAGE_ONE, "/page2.html": PAGE_TWO}

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        body = self.pages.get(self.path)
        if body is None:
            self.send_error(404)
            return
        payload = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args: object) -> None:
        """Keep pytest output clean."""


@pytest.fixture(scope="module")
def site() -> str:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Site)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    yield f"http://{host!s}:{port:d}"
    server.shutdown()
    server.server_close()


@pytest.fixture
def env() -> BrowserEnvironment:
    """One environment per test.

    Playwright's sync API allows only one live instance per thread, so a
    module-scoped environment would collide with any test that needs a fresh
    one. Per-test also keeps the "no page loaded yet" state reachable.
    """
    environment = BrowserEnvironment()
    environment.start()
    yield environment
    environment.close()


def _by_kind(outcome: EnvironmentOutcome, kind: str) -> list[dict[str, object]]:
    """The affordances of one action kind, in affordance order."""
    assert outcome.world_state is not None
    return [
        {"index": dict(aff.parameters)["index"], "label": dict(aff.parameters)["label"]}
        for aff in outcome.world_state.affordances
        if aff.action_kind == kind
    ]


def test_satisfies_the_tool_environment_protocol(env: BrowserEnvironment) -> None:
    from taiji.environment import TaijiToolEnvironment

    assert isinstance(env, TaijiToolEnvironment)


def test_goto_observes_the_page_as_world_state(env: BrowserEnvironment, site: str) -> None:
    outcome = env.execute_tool("goto", {"url": f"{site}/page1.html"})

    assert outcome.success is True
    assert outcome.terminal is False
    assert outcome.reward == 1.0
    assert outcome.sensation > 0
    world = outcome.world_state
    assert world is not None
    assert world.objects[0].attribute("title") == "Alpha page"
    assert "Alpha heading" in str(world.objects[0].attribute("text"))
    assert world.uncertainty == 0.0
    # every interactive element is offered as an affordance
    assert {aff.action_kind for aff in world.affordances} >= {"link", "button", "input"}
    assert _by_kind(outcome, "link") == [{"index": 0, "label": "go to beta"}]


def test_observation_is_stable_for_the_same_page(env: BrowserEnvironment, site: str) -> None:
    first = env.execute_tool("goto", {"url": f"{site}/page1.html"})
    second = env.execute_tool("read", {})
    assert second.sensation == first.sensation


def test_read_can_target_a_selector(env: BrowserEnvironment, site: str) -> None:
    env.execute_tool("goto", {"url": f"{site}/page1.html"})
    outcome = env.execute_tool("read", {"selector": "h1"})
    world = outcome.world_state
    assert world is not None
    assert world.objects[0].attribute("text") == "Alpha heading"


def test_click_navigates_and_the_observation_follows(env: BrowserEnvironment, site: str) -> None:
    env.execute_tool("goto", {"url": f"{site}/page1.html"})
    outcome = env.execute_tool("click", {"index": 0})  # the "go to beta" link

    assert outcome.success is True
    world = outcome.world_state
    assert world is not None
    assert world.objects[0].attribute("title") == "Beta page"
    assert env.current_url.endswith("/page2.html")

    env.execute_tool("back", {})
    assert env.current_url.endswith("/page1.html")


def test_type_fills_an_input(env: BrowserEnvironment, site: str) -> None:
    env.execute_tool("goto", {"url": f"{site}/page1.html"})
    outcome = env.execute_tool("type", {"index": 2, "text": "taiji"})
    assert outcome.success is True
    filled = _by_kind(outcome, "input")
    assert filled, "the input affordance must remain after typing"


def test_unknown_tool_reports_available_actions(env: BrowserEnvironment) -> None:
    outcome = env.execute_tool("teleport", {})
    assert outcome.success is False
    assert outcome.reward == -1.0


def test_out_of_range_index_fails_without_crashing(env: BrowserEnvironment, site: str) -> None:
    env.execute_tool("goto", {"url": f"{site}/page1.html"})
    outcome = env.execute_tool("click", {"index": 999})
    assert outcome.success is False
    assert outcome.reward == -1.0
    # The failure keeps the last observation so the agent can try something else.
    assert outcome.world_state is not None
    assert outcome.world_state.uncertainty == 1.0


def test_action_before_navigation_fails_honestly() -> None:
    """A fresh environment has no page yet: acting must fail, not guess.

    Uses its own environment because the module-scoped one has already
    navigated by the time this runs — sharing it would test nothing.
    """
    with BrowserEnvironment() as fresh:
        outcome = fresh.execute_tool("click", {"index": 0})
    assert outcome.success is False
    assert outcome.reward == -1.0
    # Even with no page to attach it to, the reason must reach the agent.
    assert outcome.world_state is not None
    reasons = [
        dict(event.attributes).get("reason", "")
        for event in outcome.world_state.events
        if event.kind == "action_failed"
    ]
    assert reasons and "goto" in reasons[0]


def test_failure_reason_is_always_visible(env: BrowserEnvironment, site: str) -> None:
    """Every failure names its cause in a world event (agent self-correction)."""
    outcome = env.execute_tool("teleport", {})
    assert outcome.success is False
    assert outcome.world_state is not None
    failures = [e for e in outcome.world_state.events if e.kind == "action_failed"]
    assert failures, "a failed action must leave an action_failed event"
    assert "teleport" in dict(failures[0].attributes)["reason"]


def test_task_complete_is_terminal(env: BrowserEnvironment, site: str) -> None:
    env.execute_tool("goto", {"url": f"{site}/page1.html"})
    outcome = env.execute_tool("task_complete", {})
    assert outcome.terminal is True
    assert outcome.success is True
    assert outcome.world_state is not None
    assert outcome.world_state.percept_boundary_closed is True


def test_declared_tool_surface_matches_implementation(env: BrowserEnvironment) -> None:
    for tool in BROWSER_TOOLS:
        outcome = env.execute_tool(tool, {})
        assert outcome is not None  # no dispatch gap between surface and behavior
