"""Dependency manifests for the Seed platform and its optional Legacy plugin."""

from __future__ import annotations

import os
from typing import Mapping


# These packages are required to show the desktop shell and start the Seed API.
# Keep the manifest independent from the frozen NeuroPlex/Transformer baseline.
CORE_DEPENDENCIES: dict[str, str] = {
    "PyQt6": "PyQt6",
    "PyQt6.QtWebEngineWidgets": "PyQt6-WebEngine",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "starlette": "starlette",
    "pydantic": "pydantic",
    "multipart": "python-multipart",
    "torch": "torch>=2.0.0",
    "numpy": "numpy",
    "requests": "requests",
    "tqdm": "tqdm",
    "cryptography": "cryptography",
}


# This manifest is intentionally opt-in.  ``pyproject.toml`` remains the
# install-time source of truth; this list only supports the desktop bootstrap
# when a user explicitly asks to enable the frozen Legacy plugin.
LEGACY_DEPENDENCIES: dict[str, str] = {
    "sentencepiece": "sentencepiece",
    "transformers": "transformers",
    "peft": "peft",
    "accelerate": "accelerate",
    "bitsandbytes": "bitsandbytes",
    "datasets": "datasets",
    "langchain": "langchain",
    "langchain_community": "langchain-community",
    "langchain_core": "langchain-core",
    "langchain_openai": "langchain-openai",
    "langchain_experimental": "langchain-experimental",
    "sentence_transformers": "sentence-transformers",
    "scipy": "scipy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "PyPDF2": "PyPDF2",
    "docx": "python-docx",
    "pdfminer": "pdfminer.six",
    "jieba": "jieba",
    "bs4": "beautifulsoup4",
    "duckduckgo_search": "duckduckgo-search",
}


def legacy_requested() -> bool:
    """Return whether desktop bootstrap may install the optional Legacy set."""

    mode = os.environ.get("SEED_ENABLE_LEGACY", "0").strip().lower()
    return mode in {"1", "true", "yes", "on", "enabled"}


def dependency_manifest(*, include_legacy: bool = False) -> Mapping[str, str]:
    """Return the dependency manifest for the requested runtime profile."""

    manifest = dict(CORE_DEPENDENCIES)
    if include_legacy:
        manifest.update(LEGACY_DEPENDENCIES)
    return manifest
