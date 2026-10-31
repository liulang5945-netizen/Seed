"""Native content core v3: the R5c copy-expression candidate (``new_core_v3``).

R5b closed with selection at zero errors and the entire gap in copy-expression
(0.435, mechanism = training-signal, 100% guided_wrong).  The approved R5c
contract freezes mechanism C as a training-side pointer auxiliary CE with zero
new modules: the architecture is byte-identical to ``new_core_v2``; only the
loss computation gains a declared auxiliary term.  This module adds one
method (the auxiliary CE over the copy pointer distribution at the
conditioned state) plus a distinct bundle identity; every forward path,
parameter and checkpoint tensor is inherited unchanged from v2.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch

from .internalization import content_digest
from .native_content_core import (
    NativeContentCoreConfigV1,
    NativeContentStoreV1,
)
from .native_content_core_v2 import (
    NativeContentBundleV2,
    NativeContentCoreV2,
    NativeContentReadCacheV2,
)

NATIVE_CORE_BUNDLE_FORMAT_V3 = "taiji-native-content-bundle-v3"


class NativeContentCoreV3(NativeContentCoreV2):
    """The v2 core plus the declared pointer auxiliary-CE training helper."""

    def __init__(
        self,
        config: NativeContentCoreConfigV1 | None = None,
        *,
        bundle_id: str = "native-content-core-v3",
    ):
        super().__init__(config, bundle_id=bundle_id)

    def pointer_auxiliary_ce(
        self,
        state: Any,
        cache: Any,
        value: bytes,
        *,
        weight: float = 0.3,
    ) -> torch.Tensor | None:
        """The frozen auxiliary CE at the conditioned state, or None when excluded.

        Excluded when the value does not occur exactly once in the selected
        record's span (the frozen target rule covers 91.67% of train fact
        turns; zero/multiple occurrences are skipped).  Returns the raw
        auxiliary CE; the caller applies the declared weight 0.3.
        """

        if not isinstance(cache, NativeContentReadCacheV2):
            raise ValueError("v3 pointer auxiliary requires a store-bound v2 read cache")
        hidden = state.hidden if state.decoder_hidden is None else state.decoder_hidden
        read, scores = self._read(cache, hidden)
        del read
        selected = cache.content_spans[int(torch.argmax(scores))]
        if not selected:
            return None
        value_bytes = bytes(value)
        hits = [
            index
            for index in range(len(selected) - len(value_bytes) + 1)
            if bytes(selected[index : index + len(value_bytes)]) == value_bytes
        ]
        if len(hits) != 1:
            return None
        start = hits[0]
        target = torch.zeros(len(selected), dtype=torch.float32, device=hidden.device)
        target[start : start + len(value_bytes)] = 1.0
        tokens = torch.tensor(list(selected), dtype=torch.long, device=hidden.device)
        keys = self.copy_key(self.core.embedding(tokens))
        positions = keys @ self.copy_query(hidden) / (self.config.embedding_width**0.5)
        weights = torch.softmax(positions, dim=0)
        return -(target * torch.log(weights.clamp_min(1e-12))).sum()


class NativeContentBundleV3(NativeContentBundleV2):
    """A complete recoverable v3 entry: v2 architecture plus v3 identity."""

    FORMAT = "taiji-native-content-bundle-v3"

    @classmethod
    def create(
        cls,
        config: NativeContentCoreConfigV1 | None = None,
        *,
        bundle_id: str = "native-content-core-v3",
        cursor: Mapping[str, Any] | None = None,
    ) -> NativeContentBundleV3:
        model = NativeContentCoreV3(config, bundle_id=bundle_id)
        return cls(
            model=model, store=NativeContentStoreV1(()), cursor=copy.deepcopy(dict(cursor or {}))
        )

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> NativeContentBundleV3:
        expected = {
            "format",
            "version",
            "bundle_id",
            "config",
            "parameter_revision",
            "parameters",
            "store",
            "cursor",
            "task_state",
            "checkpoint_digest",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("native content bundle payload schema differs")
        if (payload["format"], payload["version"]) != (cls.FORMAT, 1):
            raise ValueError("unsupported native content bundle version")
        if (
            content_digest(
                {name: value for name, value in payload.items() if name != "checkpoint_digest"}
            )
            != payload["checkpoint_digest"]
        ):
            raise ValueError("native content bundle checkpoint digest differs")
        model = NativeContentCoreV3(
            NativeContentCoreConfigV1(**payload["config"]), bundle_id=payload["bundle_id"]
        )
        model.load_state_dict(payload["parameters"], strict=True)
        model.parameter_revision = int(payload["parameter_revision"])
        store = NativeContentStoreV1.from_payload(payload["store"])
        stamp = content_digest(
            {
                "bundle_id": model.bundle_id,
                "parameter_revision": model.parameter_revision,
                "records": [item.record_sha256 for item in store.records],
            }
        )
        cursor = dict(payload["cursor"])
        cursor.setdefault("bundle_stamp", stamp)
        return cls(model=model, store=store, cursor=cursor)
