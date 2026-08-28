"""Data-driven programming-language evidence for the native workbench.

Programming-language selection is a workbench concern, not a Monaco concern.
The registry below is an extensible vocabulary of evidence rules.  It does
not choose a task, invoke a runner, or become part of Taiji cognition; it
produces a provenance-carrying assessment that Taiji and the IDE can share.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

PROGRAMMING_LANGUAGE_CONTRACT_FORMAT = "seed-programming-language-v1"
PROGRAMMING_LANGUAGE_CONTRACT_VERSION = 1
LANGUAGE_CONFIDENCE_THRESHOLD = 0.72

_EVIDENCE_STRENGTHS = {
    "extension": 0.14,
    "shebang": 0.90,
    "content": 0.55,
    "manifest": 0.72,
    "lsp": 0.85,
    "neighbor": 0.22,
    "toolchain": 0.18,
}


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ProgrammingLanguageDefinition:
    """One registry entry; evidence rules are metadata, not selection code."""

    language_id: str
    label: str
    editor_language_id: str
    extensions: tuple[str, ...] = ()
    shebangs: tuple[str, ...] = ()
    content_patterns: tuple[str, ...] = ()
    manifest_files: tuple[str, ...] = ()
    toolchain_commands: tuple[str, ...] = ()
    runner_id: str = ""
    lsp_id: str = ""

    def __post_init__(self) -> None:
        if not self.language_id.strip():
            raise ValueError("programming language id cannot be empty")
        if not self.label.strip() or not self.editor_language_id.strip():
            raise ValueError("programming language label/editor id cannot be empty")
        for pattern in self.content_patterns:
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)

    def to_payload(self, *, include_rules: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "language_id": self.language_id,
            "label": self.label,
            "editor_language_id": self.editor_language_id,
            "runner_id": self.runner_id or None,
            "lsp_id": self.lsp_id or None,
            "extensions": list(self.extensions),
        }
        if include_rules:
            payload.update(
                {
                    "shebangs": list(self.shebangs),
                    "content_patterns": list(self.content_patterns),
                    "manifest_files": list(self.manifest_files),
                    "toolchain_commands": list(self.toolchain_commands),
                }
            )
        return payload


@dataclass(frozen=True)
class ProgrammingLanguageEvidence:
    source: str
    language_id: str
    strength: float
    detail: str

    def __post_init__(self) -> None:
        if self.source not in {
            "extension",
            "shebang",
            "content",
            "manifest",
            "lsp",
            "neighbor",
            "toolchain",
            "user_override",
            "taiji_selection",
        }:
            raise ValueError(f"unsupported programming language evidence source: {self.source}")
        if not 0.0 <= float(self.strength) <= 1.0:
            raise ValueError("programming language evidence strength must be in [0, 1]")

    def to_payload(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "language_id": self.language_id,
            "strength": round(float(self.strength), 6),
            "detail": self.detail,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ProgrammingLanguageEvidence:
        return cls(
            source=str(payload["source"]),
            language_id=str(payload["language_id"]),
            strength=float(payload.get("strength", 0.0)),
            detail=str(payload.get("detail", "")),
        )


@dataclass(frozen=True)
class ProgrammingLanguageAssessment:
    """A content-bound language decision shared by runtime and the IDE."""

    path: str
    file_digest: str
    programming_language_id: str
    editor_language_id: str
    confidence: float
    provenance: tuple[ProgrammingLanguageEvidence, ...]
    candidate_scores: tuple[tuple[str, float], ...]
    registry_revision: str
    capability_revision: int
    user_override: str | None = None
    selection_source: str = "evidence"
    runner_id: str | None = None
    lsp_id: str | None = None
    toolchain_commands: tuple[str, ...] = ()
    available_toolchains: tuple[str, ...] = ()

    @property
    def selection_state(self) -> str:
        if self.selection_source == "user_override":
            return "user_override"
        if self.selection_source == "taiji_selection":
            return "taiji_selection"
        if self.programming_language_id == "plaintext":
            return "unknown"
        if self.confidence < LANGUAGE_CONFIDENCE_THRESHOLD:
            return "ambiguous"
        return "resolved"

    def to_payload(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "file_digest": self.file_digest,
            "programming_language_id": self.programming_language_id,
            "editor_language_id": self.editor_language_id,
            "confidence": round(float(self.confidence), 6),
            "selection_state": self.selection_state,
            "selection_source": self.selection_source,
            "user_override": self.user_override,
            "runner_id": self.runner_id,
            "lsp_id": self.lsp_id,
            "toolchain_commands": list(self.toolchain_commands),
            "available_toolchains": list(self.available_toolchains),
            "explanation": {
                "kind": "programming_language_selection",
                "selected_language": self.programming_language_id,
                "selection_state": self.selection_state,
                "confidence": round(float(self.confidence), 6),
                "evidence": [item.to_payload() for item in self.provenance],
            },
            "provenance": [item.to_payload() for item in self.provenance],
            "candidate_scores": {
                language_id: round(float(score), 6) for language_id, score in self.candidate_scores
            },
            "registry_revision": self.registry_revision,
            "capability_revision": self.capability_revision,
            "execution_snapshot": {
                "programming_language_id": self.programming_language_id,
                "editor_language_id": self.editor_language_id,
                "runner_id": self.runner_id,
                "lsp_id": self.lsp_id,
                "toolchain_commands": list(self.toolchain_commands),
                "available_toolchains": list(self.available_toolchains),
                "available_for_language": sorted(
                    set(self.toolchain_commands).intersection(self.available_toolchains)
                ),
                "file_digest": self.file_digest,
                "registry_revision": self.registry_revision,
                "capability_revision": self.capability_revision,
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ProgrammingLanguageAssessment:
        raw_scores = payload.get("candidate_scores", {})
        scores = (
            tuple(sorted((str(key), float(value)) for key, value in raw_scores.items()))
            if isinstance(raw_scores, Mapping)
            else ()
        )
        raw_provenance = payload.get("provenance", ())
        provenance = tuple(
            ProgrammingLanguageEvidence.from_payload(item)
            for item in raw_provenance
            if isinstance(item, Mapping)
        )
        raw_execution = payload.get("execution_snapshot", {})
        execution = raw_execution if isinstance(raw_execution, Mapping) else {}
        return cls(
            path=str(payload["path"]),
            file_digest=str(payload.get("file_digest", "")),
            programming_language_id=str(payload.get("programming_language_id", "plaintext")),
            editor_language_id=str(payload.get("editor_language_id", "plaintext")),
            confidence=float(payload.get("confidence", 0.0)),
            provenance=provenance,
            candidate_scores=scores,
            registry_revision=str(payload.get("registry_revision", "")),
            capability_revision=int(payload.get("capability_revision", 0)),
            user_override=(
                None
                if payload.get("user_override") in (None, "")
                else str(payload["user_override"])
            ),
            selection_source=str(payload.get("selection_source", "evidence")),
            runner_id=(
                None
                if payload.get("runner_id", execution.get("runner_id")) in (None, "")
                else str(payload.get("runner_id", execution.get("runner_id")))
            ),
            lsp_id=(
                None
                if payload.get("lsp_id", execution.get("lsp_id")) in (None, "")
                else str(payload.get("lsp_id", execution.get("lsp_id")))
            ),
            toolchain_commands=tuple(
                str(item)
                for item in payload.get(
                    "toolchain_commands", execution.get("toolchain_commands", ())
                )
            ),
            available_toolchains=tuple(
                str(item)
                for item in payload.get(
                    "available_toolchains", execution.get("available_toolchains", ())
                )
            ),
        )


class ProgrammingLanguageRegistry:
    """Extensible evidence registry with a content-addressed rule revision."""

    def __init__(
        self,
        definitions: Iterable[ProgrammingLanguageDefinition],
    ) -> None:
        self._definitions = tuple(definitions)
        ids = [definition.language_id for definition in self._definitions]
        if len(ids) != len(set(ids)):
            raise ValueError("programming language ids must be unique")
        body = {
            "format": PROGRAMMING_LANGUAGE_CONTRACT_FORMAT,
            "version": PROGRAMMING_LANGUAGE_CONTRACT_VERSION,
            "languages": [definition.to_payload() for definition in self._definitions],
        }
        self.revision = _digest(body)

    @classmethod
    def default(cls) -> ProgrammingLanguageRegistry:
        return cls(
            (
                ProgrammingLanguageDefinition(
                    "python",
                    "Python",
                    "python",
                    runner_id="python",
                    lsp_id="pyright",
                    extensions=(".py", ".pyw"),
                    shebangs=("python", "pypy"),
                    content_patterns=(r"^\s*(from|import)\s+\w+", r"\bdef\s+\w+\s*\("),
                    manifest_files=("pyproject.toml", "requirements.txt", "setup.py", "pipfile"),
                    toolchain_commands=("python", "python3"),
                ),
                ProgrammingLanguageDefinition(
                    "javascript",
                    "JavaScript",
                    "javascript",
                    runner_id="node",
                    lsp_id="typescript-language-server",
                    extensions=(".js", ".jsx", ".mjs", ".cjs"),
                    shebangs=("node",),
                    content_patterns=(
                        r"\b(const|let|var)\s+\w+",
                        r"\b(import|export)\b.*\b(from|default)\b",
                        r"\brequire\s*\(",
                    ),
                    manifest_files=("package.json",),
                    toolchain_commands=("node", "npm"),
                ),
                ProgrammingLanguageDefinition(
                    "typescript",
                    "TypeScript",
                    "typescript",
                    runner_id="node",
                    lsp_id="typescript-language-server",
                    extensions=(".ts", ".tsx", ".mts", ".cts"),
                    content_patterns=(
                        r"\b(interface|type)\s+\w+",
                        r"\b(const|let|var)\s+\w+\s*:\s*[A-Za-z]",
                    ),
                    manifest_files=("tsconfig.json",),
                    toolchain_commands=("tsc",),
                ),
                ProgrammingLanguageDefinition(
                    "html",
                    "HTML",
                    "html",
                    lsp_id="vscode-html-language-server",
                    extensions=(".html", ".htm"),
                    content_patterns=(r"<!doctype\s+html", r"</?(html|body|div|main)\b"),
                ),
                ProgrammingLanguageDefinition(
                    "vue",
                    "Vue SFC",
                    "html",
                    runner_id="npm",
                    lsp_id="vue-language-server",
                    extensions=(".vue",),
                    content_patterns=(r"<template\b", r"<script(?:\s+lang=\"ts\")?\b"),
                    manifest_files=("package.json",),
                    toolchain_commands=("npm",),
                ),
                ProgrammingLanguageDefinition(
                    "css",
                    "CSS",
                    "css",
                    lsp_id="vscode-css-language-server",
                    extensions=(".css",),
                    content_patterns=(r"[^{}]+\{[^{}]*:[^{};]+;",),
                ),
                ProgrammingLanguageDefinition(
                    "scss",
                    "SCSS",
                    "scss",
                    lsp_id="vscode-css-language-server",
                    extensions=(".scss",),
                    content_patterns=(r"\$[A-Za-z_-][\w-]*\s*:", r"@mixin\b"),
                ),
                ProgrammingLanguageDefinition(
                    "json",
                    "JSON",
                    "json",
                    extensions=(".json",),
                    content_patterns=(r"^\s*[\[{]",),
                ),
                ProgrammingLanguageDefinition(
                    "yaml",
                    "YAML",
                    "yaml",
                    lsp_id="yaml-language-server",
                    extensions=(".yml", ".yaml"),
                    content_patterns=(r"^\s*[A-Za-z_][\w.-]*:\s*",),
                ),
                ProgrammingLanguageDefinition(
                    "xml",
                    "XML",
                    "xml",
                    extensions=(".xml",),
                    content_patterns=(r"^\s*<\?xml\b",),
                ),
                ProgrammingLanguageDefinition(
                    "java",
                    "Java",
                    "java",
                    runner_id="java",
                    lsp_id="jdtls",
                    extensions=(".java",),
                    content_patterns=(
                        r"\b(package|import)\s+[\w.]+;",
                        r"\b(public|private)\s+class\b",
                    ),
                    manifest_files=("pom.xml", "build.gradle", "settings.gradle"),
                    toolchain_commands=("java", "javac"),
                ),
                ProgrammingLanguageDefinition(
                    "go",
                    "Go",
                    "go",
                    runner_id="go",
                    lsp_id="gopls",
                    extensions=(".go",),
                    content_patterns=(r"^\s*package\s+\w+", r"\bfunc\s+\w+\s*\("),
                    manifest_files=("go.mod", "go.sum"),
                    toolchain_commands=("go",),
                ),
                ProgrammingLanguageDefinition(
                    "rust",
                    "Rust",
                    "rust",
                    runner_id="cargo",
                    lsp_id="rust-analyzer",
                    extensions=(".rs",),
                    content_patterns=(r"\bfn\s+\w+\s*\(", r"\blet\s+mut\b"),
                    manifest_files=("cargo.toml", "cargo.lock"),
                    toolchain_commands=("rustc", "cargo"),
                ),
                ProgrammingLanguageDefinition(
                    "c",
                    "C",
                    "c",
                    runner_id="cc",
                    lsp_id="clangd",
                    extensions=(".c", ".h"),
                    content_patterns=(r"#include\s*[<\"]", r"\b(printf|malloc|typedef)\s*\("),
                    manifest_files=("cmakelists.txt", "makefile"),
                    toolchain_commands=("cc", "gcc", "clang"),
                ),
                ProgrammingLanguageDefinition(
                    "cpp",
                    "C++",
                    "cpp",
                    runner_id="c++",
                    lsp_id="clangd",
                    extensions=(".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx", ".h"),
                    content_patterns=(
                        r"#include\s*[<\"]",
                        r"\bstd::",
                        r"\b(class|namespace|template)\s+\w+",
                        r"extern\s+\"C\"",
                    ),
                    manifest_files=("cmakelists.txt", "makefile"),
                    toolchain_commands=("c++", "g++", "clang++"),
                ),
                ProgrammingLanguageDefinition(
                    "csharp",
                    "C#",
                    "csharp",
                    runner_id="dotnet",
                    lsp_id="omnisharp",
                    extensions=(".cs",),
                    content_patterns=(
                        r"\b(using\s+System|namespace\s+\w+)",
                        r"\b(class|record)\s+\w+",
                    ),
                    manifest_files=(".csproj", ".sln"),
                    toolchain_commands=("dotnet",),
                ),
                ProgrammingLanguageDefinition(
                    "sql",
                    "SQL",
                    "sql",
                    extensions=(".sql",),
                    content_patterns=(
                        r"\b(select|insert|update|delete)\b.+\b(from|into|set)\b",
                        r"\bcreate\s+table\b",
                    ),
                ),
                ProgrammingLanguageDefinition(
                    "markdown",
                    "Markdown",
                    "markdown",
                    lsp_id="marksman",
                    extensions=(".md", ".markdown"),
                    content_patterns=(r"^\s{0,3}#{1,6}\s+\S", r"^```[\w+-]*\s*$"),
                ),
                ProgrammingLanguageDefinition(
                    "shell",
                    "Shell",
                    "shell",
                    runner_id="bash",
                    lsp_id="bash-language-server",
                    extensions=(".sh", ".bash", ".zsh"),
                    shebangs=("bash", "sh", "zsh", "shell"),
                    content_patterns=(r"^\s*#!/", r"\b(echo|printf|export)\s+\S"),
                    toolchain_commands=("bash", "sh"),
                ),
                ProgrammingLanguageDefinition(
                    "notebook",
                    "Jupyter Notebook",
                    "json",
                    extensions=(".ipynb",),
                    content_patterns=(r'"cells"\s*:\s*\[', r'"nbformat"\s*:'),
                ),
                ProgrammingLanguageDefinition("plaintext", "Plain text", "plaintext"),
            )
        )

    @property
    def definitions(self) -> tuple[ProgrammingLanguageDefinition, ...]:
        return self._definitions

    def get(self, language_id: str) -> ProgrammingLanguageDefinition | None:
        return next(
            (item for item in self._definitions if item.language_id == language_id),
            None,
        )

    def get_by_editor_language(
        self, editor_language_id: str
    ) -> ProgrammingLanguageDefinition | None:
        return next(
            (item for item in self._definitions if item.editor_language_id == editor_language_id),
            None,
        )

    def public_descriptors(self) -> list[dict[str, Any]]:
        return [definition.to_payload(include_rules=False) for definition in self._definitions]

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": PROGRAMMING_LANGUAGE_CONTRACT_FORMAT,
            "version": PROGRAMMING_LANGUAGE_CONTRACT_VERSION,
            "revision": self.revision,
            "languages": [definition.to_payload() for definition in self._definitions],
        }

    def available_toolchains(self) -> tuple[str, ...]:
        commands = {
            command
            for definition in self._definitions
            for command in definition.toolchain_commands
            if shutil.which(command)
        }
        return tuple(sorted(commands))

    def resolve(
        self,
        *,
        path: str,
        content: str,
        file_digest: str,
        manifest_names: Iterable[str] = (),
        lsp_language_id: str | None = None,
        neighbor_names: Iterable[str] = (),
        available_toolchains: Iterable[str] = (),
        capability_revision: int = 1,
    ) -> ProgrammingLanguageAssessment:
        manifest_set = {str(name).lower() for name in manifest_names}
        neighbor_set = {str(name).lower() for name in neighbor_names}
        toolchain_set = {str(name).lower() for name in available_toolchains}
        filename = Path(path).name.lower()
        suffix = Path(filename).suffix
        first_line = content.splitlines()[0].lower() if content.splitlines() else ""
        evidence: list[ProgrammingLanguageEvidence] = []

        def add(
            definition: ProgrammingLanguageDefinition,
            source: str,
            detail: str,
        ) -> None:
            evidence.append(
                ProgrammingLanguageEvidence(
                    source=source,
                    language_id=definition.language_id,
                    strength=_EVIDENCE_STRENGTHS[source],
                    detail=detail,
                )
            )

        for definition in self._definitions:
            if lsp_language_id == definition.language_id:
                add(definition, "lsp", "connected language service selected language")
            if suffix and suffix in definition.extensions:
                add(definition, "extension", f"filename suffix {suffix}")
            if any(
                re.search(rf"\b{re.escape(token.lower())}\d*\b", first_line)
                for token in definition.shebangs
            ):
                add(definition, "shebang", "interpreter declared by shebang")
            if any(
                re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
                for pattern in definition.content_patterns
            ):
                add(definition, "content", "content pattern matched")
            if any(
                expected.lower() == name
                or (expected.startswith(".") and name.endswith(expected.lower()))
                for expected in definition.manifest_files
                for name in manifest_set
            ):
                add(definition, "manifest", "project manifest is present")
            if any(Path(name).suffix.lower() in definition.extensions for name in neighbor_set):
                add(definition, "neighbor", "neighboring file supports language")

        # JSON has a structural signal stronger than a leading-bracket regex.
        json_definition = self.get("json")
        if json_definition is not None and suffix == ".json":
            try:
                json.loads(content)
            except (TypeError, ValueError):
                pass
            else:
                add(json_definition, "content", "content parses as JSON")

        local_sources = {"extension", "shebang", "content", "lsp"}
        local_candidates = {item.language_id for item in evidence if item.source in local_sources}
        if local_candidates:
            evidence = [
                item
                for item in evidence
                if item.source != "manifest" or item.language_id in local_candidates
            ]

        candidate_ids = {item.language_id for item in evidence}
        manifest_candidates = {item.language_id for item in evidence if item.source == "manifest"}
        for definition in self._definitions:
            if (
                definition.language_id in candidate_ids
                and any(
                    command.lower() in toolchain_set for command in definition.toolchain_commands
                )
                and (len(candidate_ids) == 1 or definition.language_id in manifest_candidates)
            ):
                add(definition, "toolchain", "declared toolchain is available")

        source_scores: dict[tuple[str, str], float] = {}
        for item in evidence:
            key = (item.language_id, item.source)
            source_scores[key] = max(source_scores.get(key, 0.0), item.strength)
        scores: dict[str, float] = {}
        for (language_id, _source), strength in source_scores.items():
            scores[language_id] = scores.get(language_id, 0.0) + strength
        scores.pop("plaintext", None)

        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if ordered:
            top_id, top_score = ordered[0]
            second_score = ordered[1][1] if len(ordered) > 1 else 0.0
            margin = max(0.0, top_score - second_score)
            confidence = min(0.99, top_score + margin * 0.35)
            selected = self.get(top_id)
            assert selected is not None  # candidate ids come from this registry
        else:
            selected = self.get("plaintext")
            assert selected is not None
            top_id = selected.language_id
            confidence = 0.0

        provenance = tuple(item for item in evidence if item.language_id == top_id)
        return ProgrammingLanguageAssessment(
            path=path,
            file_digest=file_digest,
            programming_language_id=top_id,
            editor_language_id=selected.editor_language_id,
            confidence=confidence,
            provenance=provenance,
            candidate_scores=tuple((key, value) for key, value in ordered),
            registry_revision=self.revision,
            capability_revision=capability_revision,
            runner_id=selected.runner_id or None,
            lsp_id=selected.lsp_id or None,
            toolchain_commands=selected.toolchain_commands,
            available_toolchains=tuple(sorted(toolchain_set)),
        )

    def select(
        self,
        assessment: ProgrammingLanguageAssessment,
        language_id: str,
        *,
        source: str,
    ) -> ProgrammingLanguageAssessment:
        definition = self.get(language_id)
        if definition is None:
            raise ValueError(f"unknown programming language: {language_id}")
        if source not in {"user_override", "taiji_selection"}:
            raise ValueError(f"unsupported programming language selection source: {source}")
        evidence = assessment.provenance + (
            ProgrammingLanguageEvidence(
                source=source,
                language_id=language_id,
                strength=1.0,
                detail="explicit reversible language selection",
            ),
        )
        return replace(
            assessment,
            programming_language_id=definition.language_id,
            editor_language_id=definition.editor_language_id,
            confidence=1.0,
            provenance=evidence,
            user_override=language_id if source == "user_override" else None,
            selection_source=source,
            runner_id=definition.runner_id or None,
            lsp_id=definition.lsp_id or None,
            toolchain_commands=definition.toolchain_commands,
        )
