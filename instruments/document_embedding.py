"""M5.S3 document embedding with preregistered anchoring.

Wraps a locally cached sentence-transformer model (loaded through plain
``transformers`` with mean pooling - no new pip dependency) behind a
versioned, digest-anchored interface.  The model weights never enter a
Taiji checkpoint; only the model id, revision, and config digest are
recorded so an embedding is reproducible from the preregistration.

Relocated 2026-09-13 from ``taiji/document_embedding.py`` (DEBT-A1/A2):
the HuggingFace transformers dependency must stay outside the native
substrate, so this instrument now lives in the top-level ``instruments``
package and imports the digest helper from taiji one-way.  The checkpoint
payload format is unchanged.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from taiji.internalization import content_digest

DOCUMENT_EMBEDDER_FORMAT = "taiji-document-embedder-v1"
DOCUMENT_EMBEDDER_VERSION = 1
DEFAULT_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EXPECTED_DIMENSION = 384


class DocumentEmbedder:
    """Deterministic, locally-cached text embedder (CPU, no autograd)."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        batch_size: int = 32,
        max_length: int = 256,
        device: torch.device | str = "cpu",
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        if not math.isfinite(float(batch_size)) or int(batch_size) <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.model_id = str(model_id)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()
        self.revision = self._resolve_revision()
        self.config_digest = content_digest(self._config_payload())
        dimension = int(getattr(self.model.config, "hidden_size", 0))
        if dimension != EXPECTED_DIMENSION:
            raise ValueError(
                f"preregistered embedder dimension is {EXPECTED_DIMENSION}, got {dimension}"
            )
        self.dimension = dimension

    def _config_payload(self) -> dict[str, Any]:
        config = self.model.config
        return {
            "model_type": str(getattr(config, "model_type", "")),
            "hidden_size": int(getattr(config, "hidden_size", 0)),
            "num_hidden_layers": int(getattr(config, "num_hidden_layers", 0)),
            "vocab_size": int(getattr(config, "vocab_size", 0)),
            "max_position_embeddings": int(getattr(config, "max_position_embeddings", 0)),
        }

    def _resolve_revision(self) -> str:
        """Resolve the cached snapshot revision for provenance anchoring."""
        try:
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(self.model_id, local_files_only=True))
            refs = path / "refs" / "main"
            if refs.is_file():
                return refs.read_text(encoding="utf-8").strip()
            return path.name
        except (ImportError, OSError, ValueError):
            return "unresolved"

    @torch.no_grad()
    def embed(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("embed requires at least one text")
        outputs: list[torch.Tensor] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-12)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            outputs.append(pooled.detach().cpu())
        return torch.cat(outputs, dim=0)

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": DOCUMENT_EMBEDDER_FORMAT,
            "version": DOCUMENT_EMBEDDER_VERSION,
            "model_id": self.model_id,
            "revision": self.revision,
            "config_digest": self.config_digest,
            "dimension": self.dimension,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "pooling": "mean",
            "normalized": True,
        }
