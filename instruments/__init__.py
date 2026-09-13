"""Shared measurement instruments for preregistered Taiji evaluations.

Instruments live outside ``taiji/`` on purpose: the native substrate must not
import HuggingFace transformers, even transitively (DEBT-A1/DEBT-A2 resolution,
2026-09-13).  These tools may depend on taiji one-way for digest helpers;
the dependency is never allowed to point back into the substrate.
"""

from .document_embedding import (
    DEFAULT_MODEL_ID,
    DOCUMENT_EMBEDDER_FORMAT,
    DOCUMENT_EMBEDDER_VERSION,
    EXPECTED_DIMENSION,
    DocumentEmbedder,
)

__all__ = [
    "DEFAULT_MODEL_ID",
    "DOCUMENT_EMBEDDER_FORMAT",
    "DOCUMENT_EMBEDDER_VERSION",
    "EXPECTED_DIMENSION",
    "DocumentEmbedder",
]
