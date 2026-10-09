"""Browser environment for Taiji: a real web page as an action surface.

The environment implements :class:`taiji.environment.TaijiToolEnvironment`, so a
Taiji tool call drives a real browser and the page becomes an
:class:`~taiji.contracts.WorldState` it can act on: one world object per page,
one affordance per interactive element, one event per navigation. Observations
are deliberately structural (affordances with ``target_id``/``parameters``)
rather than a serialized DOM blob — Taiji's world contracts already express
"what can be done to what", which is what an agent needs from a page.

Design notes:

- **No import-time side effects.** Playwright is imported lazily inside
  :meth:`BrowserEnvironment.start`, so importing this module never starts a
  browser. A missing Playwright surfaces as a failed tool call, not an import
  error at agent start.
- **Browser channel is configurable.** The default prefers a locally installed
  Chrome/Edge over Playwright's own download, so no extra browser payload is
  needed; pass ``executable_path`` explicitly to pin one.
- **The browser is reused across calls.** ``adapter.execute_tool_call`` is
  synchronous, so this environment is synchronous too; a Playwright sync
  context must not be created per action.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from .contracts import WorldAffordance, WorldEvent, WorldObject, WorldState
from .environment import EnvironmentOutcome

__all__ = [
    "BrowserEnvironment",
    "BrowserObservation",
    "BROWSER_TOOLS",
    "DEFAULT_USER_AGENT",
    "LOCAL_BROWSER_CANDIDATES",
]

# Attribution the agent's actions are accountable to. Matches the native search
# route's convention: the agent should be identifiable in target site logs.
DEFAULT_USER_AGENT = "TaijiBrowser/0.1 (+taiji agent)"

# Action kinds this environment affords, in a stable order. `available_actions`
# carries the indices into this tuple; `action_kinds` names them.
BROWSER_TOOLS: tuple[str, ...] = ("goto", "read", "click", "type", "back", "task_complete")

# Where a locally installed browser usually lives. Ordered by preference.
LOCAL_BROWSER_CANDIDATES: tuple[str, ...] = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)


@runtime_checkable
class _SyncPage(Protocol):
    """The slice of Playwright's page API this environment relies on.

    ``url`` is a property in Playwright's sync API while ``title`` is a method;
    the two differ, and calling either wrongly fails at runtime, not import.
    """

    @property
    def url(self) -> str: ...

    def title(self) -> str: ...
    def goto(self, url: str, **kwargs: Any) -> Any: ...
    def go_back(self, **kwargs: Any) -> Any: ...
    def inner_text(self, selector: str = "body") -> str: ...
    def query_selector_all(self, selector: str) -> Sequence[Any]: ...
    def evaluate(self, expression: str) -> Any: ...


class BrowserObservation:
    """What one page looks like to Taiji: identity, text, and what can be done."""

    __slots__ = ("url", "title", "text", "elements", "sensation")

    def __init__(
        self,
        url: str,
        title: str,
        text: str,
        elements: tuple[dict[str, Any], ...],
        sensation: int,
    ) -> None:
        self.url = url
        self.title = title
        self.text = text
        #: One entry per interactive element: ``index``/``kind``/``label``/``value``.
        self.elements: tuple[dict[str, Any], ...] = elements
        #: Stable per-observation token id. Same page content => same sensation,
        #: so Taiji can tell "nothing changed" from "the page moved on".
        self.sensation = sensation


def _sensation_of(url: str, title: str, text: str) -> int:
    """Stable non-negative token for a page observation."""
    digest = hashlib.sha256(f"{url}\x00{title}\x00{text}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _find_local_browser() -> str | None:
    """An installed Chrome/Edge executable, or ``None`` to use Playwright's own."""
    override = os.environ.get("TAIJI_BROWSER_PATH")
    if override is not None and override.strip():
        return override.strip()
    for candidate in LOCAL_BROWSER_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None


class BrowserEnvironment:
    """A real browser page, driven by Taiji tool calls.

    Satisfies :class:`~taiji.environment.TaijiToolEnvironment`: every
    ``execute_tool`` call runs one action and returns the causal outcome, which
    ``adapter.execute_tool_call`` settles back into Taiji's learning loop.
    """

    def __init__(
        self,
        *,
        executable_path: str | None = None,
        headless: bool = True,
        navigation_timeout_ms: int = 15_000,
        action_timeout_ms: int = 5_000,
        max_text_chars: int = 4_000,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self._executable_path = executable_path
        self._headless = headless
        self._navigation_timeout_ms = navigation_timeout_ms
        self._action_timeout_ms = action_timeout_ms
        self._max_text_chars = max_text_chars
        self._user_agent = user_agent
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: _SyncPage | None = None
        self._observation: BrowserObservation | None = None
        self._tick = 0

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch the browser. Idempotent; raises if Playwright is unusable."""
        if self._page is not None:
            return
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover - exercised via fake import
            raise RuntimeError(
                "browser environment needs Playwright: pip install playwright"
            ) from exc
        self._playwright = sync_playwright().start()
        launch: dict[str, Any] = {"headless": self._headless}
        executable = self._executable_path or _find_local_browser()
        if executable is not None:
            launch["executable_path"] = executable
        self._browser = self._playwright.chromium.launch(**launch)
        self._context = self._browser.new_context(user_agent=self._user_agent)
        self._context.set_default_timeout(self._action_timeout_ms)
        self._context.set_default_navigation_timeout(self._navigation_timeout_ms)
        self._page = self._context.new_page()

    def close(self) -> None:
        """Tear the browser down. Safe to call when never started."""
        for closer in (
            getattr(self._context, "close", None),
            getattr(self._browser, "close", None),
            getattr(self._playwright, "stop", None),
        ):
            if closer is None:
                continue
            try:
                closer()
            except Exception:  # pragma: no cover - teardown must not raise
                pass
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None

    def __enter__(self) -> BrowserEnvironment:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def observation(self) -> BrowserObservation | None:
        """The last page observation, or ``None`` before the first navigation."""
        return self._observation

    @property
    def current_url(self) -> str:
        """The page's current URL (empty before the first navigation)."""
        return self._observation.url if self._observation is not None else ""

    # ── the environment contract ─────────────────────────────────────────────

    def execute_tool(self, tool_name: str, parameters: Mapping[str, Any]) -> EnvironmentOutcome:
        """Run one action and report its causal outcome for Taiji to settle."""
        self._tick += 1
        try:
            self.start()
        except Exception as exc:
            return self._failure(f"browser unavailable: {exc}", terminal=False)
        assert self._page is not None  # start() succeeded

        params = dict(parameters or {})
        try:
            if tool_name == "goto":
                return self._act_goto(self._page, params)
            if tool_name == "read":
                return self._act_read(self._page, params)
            if tool_name == "click":
                return self._act_click(self._page, params)
            if tool_name == "type":
                return self._act_type(self._page, params)
            if tool_name == "back":
                return self._act_back(self._page)
            if tool_name == "task_complete":
                # Terminal by contract: the agent declares the task finished.
                return self._outcome(self._observation, reward=1.0, success=True, terminal=True)
        except Exception as exc:
            return self._failure(f"{tool_name} failed: {exc}", terminal=False)
        return self._failure(
            f"unknown tool {tool_name!r}; available: {', '.join(BROWSER_TOOLS)}",
            terminal=False,
        )

    # ── actions ──────────────────────────────────────────────────────────────

    def _act_goto(self, page: _SyncPage, params: Mapping[str, Any]) -> EnvironmentOutcome:
        url = str(params.get("url", "")).strip()
        if not url:
            return self._failure("goto requires a url parameter", terminal=False)
        page.goto(url, wait_until="load")
        return self._outcome(self._observe(page), reward=1.0, success=True)

    def _act_read(self, page: _SyncPage, params: Mapping[str, Any]) -> EnvironmentOutcome:
        if self._observation is None:
            return self._failure("read before any navigation: goto first", terminal=False)
        selector = str(params.get("selector", "")).strip()
        text = page.inner_text(selector) if selector else self._observation.text
        observation = BrowserObservation(
            url=self._observation.url,
            title=self._observation.title,
            text=text,
            elements=self._observation.elements,
            sensation=_sensation_of(self._observation.url, self._observation.title, text),
        )
        return self._outcome(observation, reward=1.0, success=True)

    def _act_click(self, page: _SyncPage, params: Mapping[str, Any]) -> EnvironmentOutcome:
        if (missing := self._require_page()) is not None:
            return missing
        element = self._element_at(page, params)
        if element is None:
            return self._failure(
                f"no interactive element at index {params.get('index')!r}", terminal=False
            )
        element.click()
        # A click may navigate; re-observe so Taiji sees the post-action page.
        return self._outcome(self._observe(page), reward=1.0, success=True)

    def _act_type(self, page: _SyncPage, params: Mapping[str, Any]) -> EnvironmentOutcome:
        if (missing := self._require_page()) is not None:
            return missing
        text = str(params.get("text", ""))
        element = self._element_at(page, params)
        if element is None:
            return self._failure(
                f"no interactive element at index {params.get('index')!r}", terminal=False
            )
        element.fill(text)
        return self._outcome(self._observation, reward=1.0, success=True)

    def _require_page(self) -> EnvironmentOutcome | None:
        """A failure outcome when no page is loaded yet, else ``None``.

        Without this guard a page-less click reports "no interactive element at
        index 0" — technically true, but it hides the actual cause and sends an
        agent looking for a button that cannot exist before the first goto.
        """
        if self._observation is None:
            return self._failure("no page loaded yet: goto a url first", terminal=False)
        return None

    def _act_back(self, page: _SyncPage) -> EnvironmentOutcome:
        page.go_back(wait_until="load")
        return self._outcome(self._observe(page), reward=1.0, success=True)

    def _element_at(self, page: _SyncPage, params: Mapping[str, Any]) -> Any | None:
        """The interactive element an action names, or ``None`` when out of range.

        Indices come from the affordances in the last world state, so an agent
        addresses elements by the handles Taiji was given rather than by CSS.
        """
        raw = params.get("index")
        if self._observation is None:
            return None
        try:
            index = int(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if not 0 <= index < len(self._observation.elements):
            return None
        handles = page.query_selector_all(_INTERACTIVE_SELECTOR)
        return handles[index] if index < len(handles) else None

    # ── observation ──────────────────────────────────────────────────────────

    def _observe(self, page: _SyncPage) -> BrowserObservation:
        url = page.url
        title = page.title()
        text = page.inner_text("body").strip()[: self._max_text_chars]
        elements: list[dict[str, Any]] = []
        for position, handle in enumerate(page.query_selector_all(_INTERACTIVE_SELECTOR)):
            try:
                label = (handle.inner_text() or handle.get_attribute("aria-label") or "").strip()
                value = (handle.get_attribute("value") or "").strip()
            except Exception:  # pragma: no cover - detached node
                continue
            elements.append(
                {
                    "index": position,
                    "kind": _element_kind(handle),
                    "label": label[:120],
                    "value": value[:120],
                }
            )
        observation = BrowserObservation(
            url=url,
            title=title,
            text=text,
            elements=tuple(elements),
            sensation=_sensation_of(url, title, text),
        )
        self._observation = observation
        return observation

    # ── outcome construction ─────────────────────────────────────────────────

    def _outcome(
        self,
        observation: BrowserObservation | None,
        *,
        reward: float,
        success: bool,
        terminal: bool = False,
    ) -> EnvironmentOutcome:
        """Package an observation as the world state Taiji settles against."""
        if observation is None:
            return EnvironmentOutcome(
                sensation=0, reward=reward, success=success, terminal=terminal
            )
        objects = (
            WorldObject(
                object_id=f"page:{observation.url}",
                attributes=(
                    ("url", observation.url),
                    ("title", observation.title),
                    ("text", observation.text),
                ),
                tags=("web-page",),
            ),
        )
        affordances = tuple(
            WorldAffordance(
                affordance_id=f"{element['kind']}@{element['index']}",
                action_kind=str(element["kind"]),
                target_id=f"page:{observation.url}",
                parameters=(("index", element["index"]), ("label", element["label"])),
            )
            for element in observation.elements
        )
        events = (
            WorldEvent(
                event_id=f"navigate:{self._tick}",
                kind="page_observed",
                tick=self._tick,
                object_id=objects[0].object_id,
                attributes=(("url", observation.url), ("title", observation.title)),
            ),
        )
        world = WorldState(
            tick=self._tick,
            entities=(objects[0].object_id,),
            objects=objects,
            events=events,
            affordances=affordances,
            uncertainty=0.0 if success else 1.0,
            # A closed percept boundary requires the ids that identify what was
            # perceived; the contract rejects a closed boundary without them.
            percept_event_id=events[0].event_id,
            percept_assembly_id=f"assembly:{self._tick}",
            percept_boundary_closed=terminal,
        )
        return EnvironmentOutcome(
            sensation=observation.sensation,
            reward=reward,
            terminal=terminal,
            success=success,
            world_state=world,
            action_kinds=tuple(str(element["kind"]) for element in observation.elements),
        )

    def _failure(self, message: str, *, terminal: bool) -> EnvironmentOutcome:
        """A failed action: negative reward, uncertainty 1.0, reason visible.

        The reason always travels in a ``page_observed``-style failure event,
        including before the first navigation — an agent that cannot read why
        its action failed cannot correct itself, so the cause is never dropped
        on the floor just because there is no page to attach it to.
        """
        observation = self._observation
        failure_event = WorldEvent(
            event_id=f"failure:{self._tick}",
            kind="action_failed",
            tick=self._tick,
            attributes=(("reason", message),),
        )
        world = WorldState(
            tick=self._tick,
            entities=(f"page:{observation.url}",) if observation is not None else (),
            events=(failure_event,),
            affordances=(),
            uncertainty=1.0,
            percept_event_id=failure_event.event_id,
            percept_assembly_id=f"assembly:{self._tick}",
            percept_boundary_closed=terminal,
        )
        return EnvironmentOutcome(
            sensation=observation.sensation if observation is not None else 0,
            reward=-1.0,
            terminal=terminal,
            success=False,
            world_state=world,
        )


# The handles an agent may act on. Ordered: page-wide interactive elements
# first (links, buttons, inputs) — the same ordering the affordance indices use.
_INTERACTIVE_SELECTOR = "a[href], button, input, textarea, select"

# Tag names the affordance layer distinguishes. Anything else falls back to the
# tag itself, so a new element type still gets an honest handle rather than
# being silently lumped in with another kind.
_ELEMENT_KINDS = {
    "a": "link",
    "button": "button",
    "input": "input",
    "textarea": "textinput",
    "select": "select",
}


def _element_kind(handle: Any) -> str:
    """The action kind one interactive element affords."""
    try:
        tag = str(handle.evaluate("el => el.tagName.toLowerCase()") or "").strip()
    except Exception:  # pragma: no cover - detached node
        return "element"
    kind = _ELEMENT_KINDS.get(tag)
    if kind is not None:
        return kind
    # A submit/image input is a button wearing an input's clothes; the agent
    # should address it as the button it is.
    if tag == "input":
        try:
            input_type = str(handle.evaluate("el => (el.type || '').toLowerCase()") or "")
        except Exception:  # pragma: no cover - detached node
            return "input"
        if input_type in {"submit", "button", "image", "reset"}:
            return "button"
    return tag or "element"
