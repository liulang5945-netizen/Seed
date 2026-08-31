"""
API 请求/响应数据模型（Pydantic）
"""

from typing import Any

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
    structural_evidence: dict[str, Any] | None = None


class TaijiWorkbenchTaskRequest(BaseModel):
    """Request Taiji to select one current candidate for the read-only Gate."""

    snapshot_id: str
    novelty: float = 0.0
    resource_budget: float = 1.0


class TaijiWorkbenchExecuteTaskRequest(TaijiWorkbenchTaskRequest):
    """Request execution after Taiji-owned read-only admission."""

    learn: bool = False


class TaijiWorkbenchProjectionRequest(BaseModel):
    """Structured workspace evidence used to project capabilities into Taiji."""

    snapshot_id: str
    parameter_bindings: dict[str, dict] = {}


class TaijiWorkbenchReprojectionRequest(BaseModel):
    """Re-project the latest current-tick Workbench evidence into Taiji."""

    snapshot_id: str


class TaijiWorkbenchSuccessorLoopRequest(BaseModel):
    """Run a bounded Taiji-owned successor graph through the read-only Gate."""

    snapshot_id: str
    loop_id: str
    max_steps: int = 8
    max_budget_units: float = 32.0
    novelty: float = 0.0
    resource_budget: float = 1.0
    learn: bool = False
    expected_portfolio_revision: int | None = None


class TaijiWorkbenchRecoveryHandoffRequest(BaseModel):
    """Start a new successor loop from externally refreshed workspace evidence."""

    parent_loop_id: str
    recovery_loop_id: str
    snapshot_id: str
    max_steps: int = 8
    max_budget_units: float = 32.0
    novelty: float = 0.0
    resource_budget: float = 1.0
    learn: bool = False


class TaijiWorkbenchRecoveryBranchRequest(BaseModel):
    """Register one additional compatible recovery evidence branch."""

    parent_loop_id: str
    recovery_loop_id: str
    snapshot_id: str
    expected_revision: int | None = None


class TaijiWorkbenchRecoveryBranchSelectRequest(BaseModel):
    """Select and execute one active recovery branch."""

    parent_loop_id: str
    branch_id: str
    recovery_loop_id: str
    snapshot_id: str
    max_steps: int = 8
    max_budget_units: float = 32.0
    novelty: float = 0.0
    resource_budget: float = 1.0
    learn: bool = False
    expected_revision: int | None = None


class TaijiWorkbenchRecoveryPortfolioMaintainRequest(BaseModel):
    """Run lifecycle and capacity maintenance for a recovery portfolio."""

    parent_loop_id: str
    snapshot_id: str
    expected_revision: int | None = None


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


class TaskInterpretationRequest(BaseModel):
    """Natural-language task evidence request with no execution authority."""

    prompt: str
    history: list[list[str]] = []
    constraints: list[str] = []


class TaskPlanningRequest(TaskInterpretationRequest):
    """Natural-language planning request bound to current structured affordances."""

    snapshot_id: str
    parameter_bindings: dict[str, dict] = {}
    novelty: float = 0.0
    resource_budget: float = 1.0


class TaskDecompositionRequest(BaseModel):
    """Semantic step evidence with no tool or capability execution binding."""

    steps: list[dict[str, Any]] = []
    confidence: float | None = None
    ambiguity: float | None = None
    status: str = "resolved"
    provenance: str = "taiji.semantic"


class SemanticProviderEvidenceRequest(BaseModel):
    """Provider semantic evidence submitted to Taiji for validation only."""

    prompt: str
    evidence: dict[str, Any]


class TaskSequencePlanningRequest(BaseModel):
    """Ground admitted semantic steps against current Workbench affordances."""

    snapshot_id: str
    parameter_bindings: list[dict[str, dict]] | None = None
    novelty: float = 0.0
    resource_budget: float = 1.0


class NaturalLanguageWorkbenchTaskRequest(BaseModel):
    """Run a bounded Taiji-owned task from semantic evidence, not an intent."""

    prompt: str
    semantic_evidence: dict[str, Any]
    snapshot_id: str
    parameter_bindings: list[dict[str, dict]] = []
    loop_id: str
    max_steps: int = 1
    max_budget_units: float = 1.0
    novelty: float = 0.0
    resource_budget: float = 1.0
    learn: bool = False


class LanguagePlanningRequest(BaseModel):
    """Plan a language selection from current Taiji task evidence and file evidence."""

    snapshot_id: str
    path: str
    lsp_language_id: str | None = None
    novelty: float = 0.0
    resource_budget: float = 1.0


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
