# Agent Note: Retired package ids stay in the Session format converter

Status: implemented

English | [中文](2026-09-29-retired-plugin-identity-in-v3-migration.zh.md)

## Problem

Renaming the first-party packages left the Session format converter unable to read Sessions recorded before the rename. `session-format-v3-to-v4` resolves a recorded `plugin` source to the producer's current kind through a historical identity table, and the table carried only the post-rename specifier. Every format-v3 Session whose system message named the retired specifier therefore resolved to an opaque `plugin:<retired>` kind, and the replay admission check in `packages/core/session/src/index.ts` rejects a `system/message` whose source kind is not `system-prompt`, so the whole Session failed to load. Measured over the recorded corpus in this checkout: 375 non-expected JSONL fixtures contain 213 system messages, 192 of them with the retired specifier and all 192 in `*.v3.jsonl` files.

## Decision

The historical table admits both spellings. `sources.ts` names the two system-prompt producer ids in one frozen list, and both map to `system-prompt` for the `system` role and to `runtime-context` otherwise, so a Session recorded before the rename converts exactly as one recorded after it. Two assertions in `sources.spec.ts` pin each arm through the retired id, because a role-sensitive mapping that is pinned only for the current id silently regresses on the next rename.

The table is read-only history and does not follow the current package list. An identity that no longer exists as a package still has to resolve, because the data that names it outlives the package. Renaming a first-party package therefore requires sweeping the producer tables in the format converters as its own step, not as part of the mechanical specifier replace.

## Alternatives considered

- **Re-record the corpus.** Recording needs a model, and it rewrites every scenario fixture to erase a resolution bug in the reader. It also could not help a user who already owns a pre-rename Session log, which is the case the converter exists to serve.
- **Rewrite the retired specifier inside recorded fixtures.** Rejected: recorded system and user messages are model-visible text, and this repository pins that text verbatim. Editing it would falsify what the model actually received while making the reader look correct.
- **Accept the opaque kind for system messages.** Rejected: the admission check exists so a replayed system message is attributable to the prompt producer. Weakening it would let genuinely foreign system messages through and would change what the replay guarantee means.

## Consequences

Sessions recorded before the rename load again, and format-v3 fixtures stop failing during load in the browser suites that replay them. The change is confined to the converter's identity resolution, so no shipped wire format, recorded payload, or event vocabulary changed.

The follow-up cost is a rule rather than code: any future rename of a first-party package must add the retired specifier to the producer tables and pin one assertion per role-sensitive arm. `verify-session-format-catalog` covers the catalog shape but does not detect a table that lost a retired name, so the sweep is not yet machine-enforced.
