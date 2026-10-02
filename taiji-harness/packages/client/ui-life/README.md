---
description: "The Life panel for the web sidebar: the Taiji runtime's organs, training controls, and host projection in one global page."
kind: "package-reference"
---

# @taiji/dsh-client-ui-life

English | [中文](README.zh.md)

## Summary

Open the **Life** entry in the Web sidebar to inspect and drive the Taiji local runtime. Six sections — source, life organs, training with checkpoints, knowledge, memory, and host — render the latest reading and carry the controls: schedule start and stop, feeding, sleeping, and playing, training start over a selectable dataset roster with upload and deletion, checkpoint resume and activation, pause, resume, stop, force release, one consolidation pass, and knowledge upload and deletion. Numbers come from the controller's snapshot stream; a refusal renders the Host's stable error code, never raw RPC text.

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Model Experience](#model-experience)
- [Known Limitations and Deferred Work](#known-limitations-and-deferred-work)
- [Dev Note](#dev-note)

-----

<a id="use-this-package"></a>
## Use this package

Mount this plugin in a web composition whose Host carries [`@taiji/dsh-api-life-controller`](../../api/life-controller/README.md) — the panel reads `ctx.life`, the Client facade that the controller's browser half installs, and declares it in its `dsh.client` manifest. It contributes one sidebar entry and one main panel and nothing else.

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

The page subscribes to the controller's identity-stable snapshot state with `useSyncExternalStore`, so a stream frame swaps one object and a panel re-render follows. The six sections project one `LifeSnapshot`: the source section names the organ that answered and lists every source that did not; the life section renders the native organ's need and drive meters or the legacy scheduler's facts and meters; training shows its state badges, the latest progress sample with the runtime's own run warnings (a corpus-drift notice on a resumed run, for one), the selectable dataset roster — grouped by directory, folded per group on click, preselected from the data ring spec while that spec still names files on disk — with its upload control and a delete-selected action, and the checkpoint roster — foldable as one block, carrying a selection column for the delete-selected action — where each row offers a resumed run and an activation, the active and the configured checkpoint cannot be selected, the active checkpoint is badged, and a drift between what answers and what the next start will use is stated in words; knowledge shows the index size and the mounted documents with their `indexed` or `pending` state, offers an upload control, and deletes the selected documents — or says the surface was not served instead of showing a blank; memory and consolidation shows the journal counts by kind, the pass counter with its latest corpus, the data-ring spec behind its readiness gate, and the latest pass report — or says the surface was not served instead of showing a blank; the host section shows health, model, seed activity, the workbench capability snapshot (count, revision, source, owner — its failure text verbatim when the snapshot reports an error), the runtime's authentication state, and memory.

Controls disable while a verb is in flight and show their outcome without touching the snapshot: an accepted verb refreshes through the stream, and a refusal raises the Host's error code — `life/conflict`, `life/unavailable`, and their siblings — as the panel's localized copy. Stopping a run, force-releasing the training lock, switching the answering model, and every delete all ask for a confirming second click; the deleting verbs carry the confirming label on their own button, and running a consolidation pass offers no such confirmation because it writes rehearsal files, never weights.

-----

**Runtime invariant:** No companion is published. This package owns a browser panel over the readings and control verbs the life controller's Client facade already carries.

-----

<a id="model-experience"></a>
## Model Experience

None, as this panel renders operator-facing readings and controls without adding model input.

#### KV Cache effect

None; the page renders for the operator and contributes nothing to any model request.

## Known Limitations and Deferred Work

<a id="known-limitations-and-deferred-work"></a>

- Starting a training run picks its datasets from the roster; parameter budget, seed, and symbol cap still use the runtime's defaults and await a form.
- The knowledge section uploads and deletes documents; an index rebuild or a full clear belongs to the runtime's own surface.
- The poll interval is displayed but not editable here; it is a property of the controller's configuration.

<a id="dev-note"></a>
### Dev Note

<details>
<summary>Working context for maintainers — click to expand</summary>

The panel keeps no copy of a reading: every cell projects the controller's current snapshot frame, and an accepted verb refreshes through the stream rather than updating local state first. A section the runtime did not serve is stated in words instead of rendering zeros, because a stale reading presented as current is the failure this panel exists to avoid.

</details>
