# Agent Note: ARIA path normalization admits both spellings

Status: implemented

English | [中文](2026-09-29-aria-path-normalization-admits-both-spellings.zh.md)

## Problem

Browser lanes that compare captured accessibility trees normalize the workspace path in the captured text to a `{{cwd}}` token. The normalizer split the snapshot on the separator-native path string, but an ARIA snapshot escapes each backslash, so on Windows the split never matched and the path survived in the captured text with only its trailing basename tokenized. The committed goldens were recorded where paths contain no backslashes and therefore carry the bare `{{cwd}}` token, so the same lane read as a golden mismatch on Windows while passing on Linux — reported as an unexplained divergence rather than a normalization defect.

## Decision

`normalizeAria` in `apps/web/tests/scaffold.ts` replaces the escaped spelling first and the native spelling second, mapping both to `{{cwd}}`. On POSIX the two strings are identical, so the second replacement is a no-op and captured output is byte-identical to what Linux already produced; on Windows the first replacement now reaches the path. Verification ran against the committed golden without editing it: the affected lane passes in replay mode, and two lanes whose goldens were not re-recorded still pass, which is the evidence that the added step changes nothing on POSIX.

Refreshed goldens stay Linux-shaped because the token, not the host path, is what the baseline holds. Recording a golden before this fix would have baked a Windows-only spelling into a committed baseline and made the lane fail on the next platform, which is the same failure mode an earlier recorded fixture showed for path separators.

## Alternatives considered

- **Re-record the affected golden.** It silences one lane and records a host-specific snapshot into a committed baseline, moving the failure to every other platform instead of removing it. Rejected for the same reason a separator-sensitive fixture cannot be refreshed on one machine.
- **Normalize by forward slashes before comparing.** Shorter, but it rewrites separators the page genuinely renders and weakens the assertion for every path-shaped row in the tree, not only the workspace root.
- **Skip the lane on Windows.** Removes the only platform where the defect is observable, and the repository reserves platform skips for cases where the feature itself is unavailable.

## Consequences

The lane class that reads a workspace path out of an accessibility tree is now platform-neutral, so a golden recorded on any platform validates on every platform. The cost is a standing rule for future normalization work: when a captured representation can escape a character that the host string contains literally, both spellings must be admitted by the same token replacement, and the POSIX case must be shown to be a no-op rather than assumed. No shipped wire format, captured payload, or committed golden changed beyond that rule.
