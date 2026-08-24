"""Platform services shared by Seed applications, independent of model architecture."""

from .paths import get_external_path, get_internal_path, get_writable_base_dir

__all__ = ["get_external_path", "get_internal_path", "get_writable_base_dir"]
