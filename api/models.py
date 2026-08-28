"""
API 请求/响应数据模型（Pydantic）
"""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    prompt: str
    system_prompt: str = (
        "你是Seed，一个独立的AI生命体。你用自己的大脑思考，用工具探索世界。"
    )
    history: list[list[str]] = []
    engine: str = "taiji"  # "taiji"（思维）| "agent"（行动）
    agent_max_iterations: int | None = 10
    agent_temperature: float | None = 0.7


class TrainRequest(BaseModel):
    dataset: str = ""
    datasets: list[str] = []
    lora_r: int = 8
    lora_alpha: int = 32
    epochs: int = 3
    learning_rate: float = 0.0002
    batch_size: int = 4


class FileSaveRequest(BaseModel):
    name: str
    content: str


class CodeRunRequest(BaseModel):
    code: str


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


class WorkbenchLoopPreflightRequest(BaseModel):
    """Bounded, non-executing admission request for a native workbench loop."""

    loop_id: str
    intents: list[WorkbenchIntentRequest] = []
    max_steps: int = 8
    max_budget_units: float = 32.0
    on_failure: str = "stop"
    checkpoint_boundary: str = "after_each_step"


class RAGSearchRequest(BaseModel):
    query: str
    top_k: int = 5


class CreateProjectRequest(BaseModel):
    type: str = "empty"


class GGUFExportRequest(BaseModel):
    """GGUF 导出请求"""

    model_dir: str
    quant: str = "Q4_K_M"


class TaijiTrainRequest(BaseModel):
    """Seed原生模型微调请求"""

    num_epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 1e-4
    max_length: int = 512
    save_steps: int = 50
    log_steps: int = 5
    extra_react_data: list[dict] | None = None
    extra_conv_data: list[dict] | None = None
    keep_checkpoints: int = 3  # 保留最近 N 个中间 checkpoint + best
