"""
API 请求/响应数据模型（Pydantic）
"""

from pydantic import BaseModel


class WorkbenchIntentRequest(BaseModel):
    """A Taiji-owned action intent submitted to the native workbench bridge."""

    intent_id: str
    kind: str
    parameters: dict = {}
    snapshot_id: str
    source_goal_id: str | None = None
    expected_outcome: str = ""
    confidence: float = 0.0
    tick: int = 0
    approval_token: str = ""
    mcp_registry_snapshot_id: str = ""


class ChatRequest(BaseModel):
    prompt: str
    system_prompt: str = "你是Seed，一个独立的AI生命体。你用自己的大脑思考，用工具探索世界。"
    history: list[list[str]] = []


class ChatWorkbenchRequest(ChatRequest):
    """Chat prompt plus one already-formed Taiji workbench intent.

    The request deliberately carries a structured intent instead of asking the
    language surface to infer a tool from prose. Natural-language task
    planning remains a separate Taiji-owned capability and cannot be silently
    implemented by this transport boundary.
    """

    intent: WorkbenchIntentRequest


class FileSaveRequest(BaseModel):
    name: str
    content: str


class CodeRunRequest(BaseModel):
    code: str


class WorkbenchLoopPreflightRequest(BaseModel):
    """Bounded, non-executing admission request for a native workbench loop."""

    loop_id: str
    intents: list[WorkbenchIntentRequest] = []
    max_steps: int = 8
    max_budget_units: float = 32.0
    on_failure: str = "stop"
    checkpoint_boundary: str = "after_each_step"


class WorkbenchLoopExecuteRequest(WorkbenchLoopPreflightRequest):
    """Execution request bound to a previously accepted loop preflight."""

    preflight_id: str


class RAGSearchRequest(BaseModel):
    query: str
    top_k: int = 5


class CreateProjectRequest(BaseModel):
    type: str = "empty"
