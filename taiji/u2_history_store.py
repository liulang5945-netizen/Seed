"""Bounded, hash-linked pages preserving the existing raw-operation chain."""

from __future__ import annotations

from .u2_history import CausalHistory, assert_values_equal, detached_tree, value_digest

PAGED_SCHEMA = "taiji-u2-paged-history-v1"


class PagedCausalHistory(CausalHistory):
    def __init__(self, root, *, page_entries=96, max_pages=16, max_bytes=16 * 1024 * 1024):
        if (
            type(page_entries) is not int
            or not 1 <= page_entries <= 96
            or type(max_pages) is not int
            or not 1 <= max_pages <= 16
        ):
            raise ValueError("invalid_paged_history_budget")
        self.page_entries, self.max_pages = page_entries, max_pages
        self.max_entries, self.max_bytes = page_entries * max_pages, max_bytes
        self._core = CausalHistory(root, max_entries=page_entries, max_bytes=max_bytes)
        self._core.max_entries = page_entries * max_pages

    @classmethod
    def from_history(cls, history):
        result = cls(history.root)
        for node in history.iter_entries():
            result = result.append(node["payload"])
        assert_values_equal(result.reference(), history.reference())
        return result

    @property
    def entry_count(self):
        return self._core.entry_count

    @property
    def total_bytes(self):
        return self._core.total_bytes

    @property
    def root(self):
        return self._core.root

    @property
    def complete(self):
        return self._core.complete

    def require_complete(self):
        self._core.require_complete()

    def reference(self):
        return self._core.reference()

    def validate_reference(self, reference):
        return self._core.validate_reference(reference)

    def operations_since(self, reference):
        return self._core.operations_since(reference)

    def iter_entries(self):
        yield from self._core.iter_entries()

    def append(self, payload):
        result = object.__new__(PagedCausalHistory)
        result.page_entries, result.max_pages = self.page_entries, self.max_pages
        result.max_entries, result.max_bytes = self.max_entries, self.max_bytes
        result._core = self._core.append(payload)
        return result

    def export(self):
        pages: list[dict] = []
        previous = value_digest(self.root)
        entries = self._core._entries
        for start in range(0, self.entry_count, self.page_entries):
            body = {
                "index": len(pages),
                "start": start,
                "previous": previous,
                "entries": entries[start : start + self.page_entries],
            }
            previous = value_digest(body)
            pages.append({**body, "digest": previous})
        return detached_tree(
            {
                "schema": PAGED_SCHEMA,
                "root": self.root,
                "pages": pages,
                "page_entries": self.page_entries,
                "max_pages": self.max_pages,
                "max_bytes": self.max_bytes,
                "head": self.reference()["head"],
                "entry_count": self.entry_count,
                "total_bytes": self.total_bytes,
                "pages_head": previous,
            }
        )

    @classmethod
    def restore(cls, payload) -> PagedCausalHistory:
        required = {
            "schema",
            "root",
            "pages",
            "page_entries",
            "max_pages",
            "max_bytes",
            "head",
            "entry_count",
            "total_bytes",
            "pages_head",
        }
        if set(payload) != required or payload["schema"] != PAGED_SCHEMA:
            raise ValueError("incomplete_paged_history")
        result = cls(
            payload["root"],
            page_entries=payload["page_entries"],
            max_pages=payload["max_pages"],
            max_bytes=payload["max_bytes"],
        )
        if len(payload["pages"]) > result.max_pages:
            raise ValueError("paged_history_budget_stop")
        previous = value_digest(result.root)
        for index, page in enumerate(payload["pages"]):
            if set(page) != {"index", "start", "previous", "entries", "digest"}:
                raise ValueError("unregistered_history_page")
            body = {key: value for key, value in page.items() if key != "digest"}
            if (
                page["index"] != index
                or page["start"] != result.entry_count
                or page["previous"] != previous
                or value_digest(body) != page["digest"]
            ):
                raise ValueError("history_unavailable_or_page_identity_mismatch")
            count = len(page["entries"])
            if not 1 <= count <= result.page_entries or (
                index < len(payload["pages"]) - 1 and count != result.page_entries
            ):
                raise ValueError("invalid_history_page_size")
            for node in page["entries"]:
                result = result.append(node["payload"])
                assert_values_equal(node, result._core._entries[-1], "history.page.node")
            previous = page["digest"]
        if (
            previous != payload["pages_head"]
            or result.reference() != {"entries": payload["entry_count"], "head": payload["head"]}
            or result.total_bytes != payload["total_bytes"]
        ):
            raise ValueError("history_unavailable_or_archive_summary_mismatch")
        return result
