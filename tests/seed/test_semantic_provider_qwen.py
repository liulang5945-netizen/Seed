"""Tests for the model-backed semantic provider integration edge."""

from collections.abc import Mapping

import pytest

from seed.semantic_provider import (
    QwenSemanticEvidenceProvider,
    SemanticProviderArtifact,
    load_qwen_semantic_provider_from_environment,
)
from taiji import InputFrame, SemanticProviderRequest, language_provider_content_digest


class _FakeGenerator:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
        assert max_tokens > 0
        assert temperature >= 0.0
        self.prompts.append(prompt)
        return self.response


def _request() -> SemanticProviderRequest:
    frame = InputFrame(
        input_id="qwen-semantic-test",
        modality="text",
        payload="读取 README.md 并确认内容".encode(),
        source="tests.semantic_provider_qwen",
        timestamp=3,
    )
    return SemanticProviderRequest.from_frame(frame, constraints=("只读",))


def _artifact(tmp_path) -> SemanticProviderArtifact:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
    digest = language_provider_content_digest(model_dir)
    return SemanticProviderArtifact.from_model_dir(
        model_dir,
        expected_model_digest=digest,
    )


def test_qwen_adapter_returns_only_content_addressed_semantic_evidence(tmp_path) -> None:
    generator = _FakeGenerator(
        '{"goal_description":"读取目标文件并确认内容",'
        '"constraints":["保持可恢复"],'
        '"semantic_steps":[{"description":"读取 README.md",'
        '"semantic_slots":{"operation":"read","path":"README.md"},'
        '"expected_outcome":"获得文件内容"}],'
        '"confidence":0.9,"ambiguity":0.1}'
    )
    provider = QwenSemanticEvidenceProvider(generator, _artifact(tmp_path))

    proposal = provider.propose(_request())

    assert proposal.provider_id == provider.provider_id
    assert proposal.input_id == "qwen-semantic-test"
    assert proposal.semantic_steps[0]["semantic_slots"]["path"] == "README.md"
    assert "tool" not in proposal.to_payload()
    assert "parameter_bindings" not in proposal.to_payload()
    assert "README.md" in generator.prompts[0]
    assert isinstance(provider.checkpoint(), Mapping)
    assert provider.checkpoint()["artifact_digest"] == provider.artifact.artifact_digest


def test_qwen_adapter_accepts_one_fenced_json_object(tmp_path) -> None:
    generator = _FakeGenerator(
        "```json\n"
        '{"goal_description":"检查文件", "semantic_steps":[], '
        '"confidence":0.4, "ambiguity":0.4}\n'
        "```"
    )
    proposal = QwenSemanticEvidenceProvider(generator, _artifact(tmp_path)).propose(_request())

    assert proposal.goal_description == "检查文件"
    assert proposal.confidence == 0.4


def test_qwen_adapter_normalizes_bounded_singular_constraint_alias(tmp_path) -> None:
    generator = _FakeGenerator(
        '{"goal_description":"检查文件", "constraint":"保持只读", '
        '"semantic_steps":[], "confidence":0.4, "ambiguity":0.4}'
    )
    proposal = QwenSemanticEvidenceProvider(generator, _artifact(tmp_path)).propose(_request())

    assert set(proposal.constraints) == {"只读", "保持只读"}


def test_qwen_adapter_normalizes_only_bounded_language_operation_aliases(tmp_path) -> None:
    generator = _FakeGenerator(
        '{"goal_description":"判断语言", "semantic_steps":['
        '{"description":"判断 README.md 语言", '
        '"semantic_slots":{"operation":"resolve_language","path":"README.md"}}],'
        '"confidence":0.9,"ambiguity":0.1}'
    )
    provider = QwenSemanticEvidenceProvider(generator, _artifact(tmp_path))

    proposal = provider.propose(_request())

    assert proposal.semantic_steps[0]["semantic_slots"]["operation"] == "resolve-language"


def test_qwen_adapter_rejects_execution_fields_before_taiji_admission(tmp_path) -> None:
    generator = _FakeGenerator(
        '{"goal_description":"越权", "tool":"workspace.read", '
        '"confidence":0.9, "ambiguity":0.1}'
    )
    provider = QwenSemanticEvidenceProvider(generator, _artifact(tmp_path))

    with pytest.raises(ValueError, match="unsupported fields"):
        provider.propose(_request())


def test_model_digest_is_an_explicit_allowlist(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="not allowlisted"):
        SemanticProviderArtifact.from_model_dir(
            model_dir,
            expected_model_digest="0" * 64,
        )


def test_environment_opt_in_requires_complete_explicit_binding(monkeypatch) -> None:
    monkeypatch.delenv("TAIJI_SEMANTIC_PROVIDER_MODEL_DIR", raising=False)
    monkeypatch.delenv("TAIJI_SEMANTIC_PROVIDER_MODEL_DIGEST", raising=False)
    assert load_qwen_semantic_provider_from_environment() is None

    monkeypatch.setenv("TAIJI_SEMANTIC_PROVIDER_MODEL_DIR", "C:/models/qwen")
    with pytest.raises(ValueError, match="requires both"):
        load_qwen_semantic_provider_from_environment()
