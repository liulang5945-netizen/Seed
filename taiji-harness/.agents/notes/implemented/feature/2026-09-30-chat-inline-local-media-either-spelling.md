# Agent Note: Chat inline local media forwards absolute paths in either spelling

Status: implemented

English | [中文](2026-09-30-chat-inline-local-media-either-spelling.zh.md)

## Problem

`AssistantMarkdown` resolved an authored image destination only when it began with `/`, so a Windows absolute path (`E:\...`) rendered as inert text in chat while the same document previewed correctly in the Sidebar. The two surfaces asked different questions about the same string: the Sidebar used the platform-aware `isAbsoluteWorkspacePath`, chat used a POSIX-only prefix test. The page-stable vocabulary in [the local-media display note](2026-09-07-session-prose-local-media-display.md) listed Windows-style paths among the intentionally inert shapes; this note supersedes that list entry, and that note retains renderer ownership, the mount-relative resolution rule, and its rationale.

## Decision

`localPathMediaUrl` now mirrors the order already used by `markdownImageUrl`: reject an empty value, a value containing NUL, and a destination starting with two separators of either kind; require `isAbsoluteWorkspacePath`, which accepts POSIX absolute and Windows drive forms; then require an HTTP(S) document base. UNC destinations stay out of scope in both surfaces, so a shared document renders the same image set in chat and in the Sidebar.

The renderer cannot learn the Host platform in the browser, so it does not decide: an absolute path is forwarded and the Host accepts or rejects it. `GET /api/file` gates on `node:path` `isAbsolute` plus the composed provider read policy, which is why a drive path succeeds on a Windows Host and returns 400 on a POSIX Host where it previously produced no request at all. The visible result for a POSIX user is unchanged - a broken image plus its alt text - and the additional cost is one rejected request.

## Enforcement

`assistant-markdown-path-images.client.spec.tsx` pins both directions: the drive form resolves for a root and a mounted base, while empty, double-separator, relative, and non-HTTP transports stay inert, so the widening cannot silently absorb a shape that was never a local file. `markdown-images.e2e.ts` authors the destination with the platform separator, keeping the requested key identical to the assertion key on either host. The `ui-deliverables` README pair states the remaining page limitation (`dsh-app:` stays inert).
