# Agent Note: Startup never opens a Session

Status: implemented

English | [中文](2026-09-29-startup-never-opens-a-session.zh.md)

## Problem

Startup restoration selected a Session on the user's behalf, so the first screen of every installation was an open Conversation rather than the choice point the Hero offers. Anything that needed to observe the empty state had to suppress that creation, and a suppression done over the wire changes what the assembled surface renders, so comparison lanes could not use it.

## Decision

Startup restores only what already exists. It waits for both the Workspace and Session baselines, prepares the default Workspace when none is registered, and reuses a saved blank Session when the persisted selection names one that is still eligible. It never creates a Session: the first one comes from a user gesture — sending from the Hero, New Session, or opening a Workspace. [First-use default Workspace](2026-09-20-default-workspace.md) owns provisioning and eligibility and is unchanged in this respect, including its single failure notice; [Session scope and provisioning](../architecture/2026-07-25-web-client-session-scope-and-provide-channel.md) owns blank-Session reuse.

Reuse of a saved blank Session is deliberately kept, so a reload still lands on the Session the user was on. The startup failure report (`initial Session restoration failed`) now covers only that reuse path, and stays console-only: a Workspace that exists is not a failed startup, so no notice is raised and no Session row appears.

## Alternatives considered

- **Keep startup creation and pin the empty state in tests.** A refused `session/create` request did make the geometry lane deterministic, but it also removed a Session the product would have opened, so the same premise could not be used by any lane that compares rendered output. It also left the shipped first screen unchanged, which is what the reports were about.
- **Refuse creation only for the Workspace the client provisioned itself.** Smaller, and it kept the default-Workspace path quiet, but a pre-existing Workspace still opened a Session on the user's behalf, so the first screen and the sidebar rows stayed unrequested.
- **Stop reusing a saved blank Session as well.** The sidebar would then never show a Session the user did not ask for, but reload would drop the user off the Session they were reading, which is a regression in the one behavior the gesture-free path had going for it.
- **Create the Session at the first keystroke instead of the first send.** Needs a draft Session, a transfer into a real one, and coordination with the first submission; the Hero already sends through the ordinary composer pipeline once a Session exists.

## Consequences

The first screen of a new installation is the Hero with a provisioned Workspace to send from, and the Sessions tree contains exactly the Sessions that were created. Comparison lanes state their premise instead of intercepting it: the geometry lane now boots cold and asserts no current Session, with no refused request in the path.

Cost: the absence of a current Session at startup is now a reachable state for every surface that reads it, so header geometry, the composer block, and Workspace selection handle it as normal rather than as a momentary gap. Recorded startup behavior moved with it: the client service specs name the contract as "prepares the default Workspace without opening a Session", and the first-use browser scenario covers provision, Hero send, and restore-after-reload.
