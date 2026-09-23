# Agent Note: First-use eligibility ignores Session history

Status: implemented

English | [中文](2026-09-23-first-use-eligibility-ignores-session-history.zh.md)

## Problem

A registry with no Workspace at all and Session history on disk left the product unusable. The Client prepared the default Workspace only while the Session list was empty, the Host only while live, persisted, and archived Sessions were all absent, and a host whose directory picker is unavailable offered no other route to a Workspace. Ordinary deletion reaches that state: deleting a Workspace keeps its Session logs by contract, so the registry empties while history stays.

## Decision

Eligibility is "the registry holds no Workspace". Live, persisted, and archived Sessions never veto preparation, and the Client asks without first requiring an empty Session list. Sessions that cannot join the new Workspace — a different working directory, or none at all — stay ungrouped, and the archived set is untouched. A registry that already holds a Workspace still refuses, so the default Workspace stays a single recovery rather than a second registration, and request names still never rename an existing one.

Removing the veto also removed the Session rechecks that guarded directory preparation. The registry mutation queue already serializes every Workspace write, so no Workspace can appear between the eligibility check and the commit, and a Session starting concurrently is an ordinary ungrouped Session rather than a reason to abandon an eligible recovery.

## Alternatives considered

- Keeping the veto and relying on the directory picker leaves the dead end in place wherever the picker is unavailable.
- Resetting the initialization marker so the next start re-derives Workspaces from persisted headers would resurrect a registration the user deleted, because deletion keeps Session logs by contract.
- Relaxing only the Client precondition would change nothing: the Host refuses the same histories.

## Consequences

A first start can create the default Workspace while ungrouped history exists, and the sidebar lists those Sessions outside any project exactly as before. The [Workspace registry](../../../../packages/workspace/workspace/README.md#first-use-workspace) owns the current eligibility contract. Registry and Client tests cover the persisted, live, and archived cases, re-eligibility once the registry empties, and initialization without the Session store peer.

The eligibility clause of [First-use default Workspace](2026-09-20-default-workspace.md) is superseded; its directory resolution, durable identity, and recovery behavior still hold.