from __future__ import annotations

import torch

from taiji import (
    TSKV8Adapter,
    WorkspaceCandidate,
    WorkspaceRouter,
    WorkspaceRoutingExample,
    WorkspaceSelection,
    WorkspaceState,
)


def _candidates() -> tuple[WorkspaceCandidate, ...]:
    return (
        WorkspaceCandidate("relevant", torch.tensor([1.0, 0.0]), source="percept"),
        WorkspaceCandidate("distractor-a", torch.tensor([0.0, 1.0]), source="memory"),
        WorkspaceCandidate("distractor-b", torch.tensor([-1.0, 0.0]), source="memory"),
    )


def test_workspace_router_learns_capacity_limited_selection() -> None:
    candidates = _candidates()
    router = WorkspaceRouter(feature_dim=2, capacity=1, seed=7)
    loss = router.fit(
        (
            WorkspaceRoutingExample(candidates=candidates, relevant_ids=("relevant",)),
            WorkspaceRoutingExample(
                candidates=tuple(reversed(candidates)), relevant_ids=("relevant",), tick=1
            ),
        ),
        epochs=120,
        learning_rate=0.2,
    )

    selection = router.route(candidates, tick=2)

    assert loss < 0.2
    assert selection.selected_ids == ("relevant",)
    assert selection.capacity == 1
    assert torch.equal(selection.broadcast, candidates[0].features)


def test_workspace_lesions_are_explicit_and_measurable() -> None:
    candidates = _candidates()
    router = WorkspaceRouter(feature_dim=2, capacity=1, seed=3)
    router.fit((WorkspaceRoutingExample(candidates, ("relevant",)),), epochs=100, learning_rate=0.2)

    no_workspace = router.route(candidates, tick=0, mode="none")
    random_workspace = router.route(candidates, tick=0, mode="random", random_seed=4)

    assert no_workspace.selected_ids == ()
    assert torch.equal(no_workspace.broadcast, torch.zeros(2))
    assert len(random_workspace.selected_ids) == 1
    assert random_workspace.mode == "random"
    assert random_workspace.selected_ids != ("relevant",)


def test_workspace_contract_and_native_checkpoint_round_trip() -> None:
    candidates = _candidates()
    router = WorkspaceRouter(feature_dim=2, capacity=2, seed=11)
    selection = router.route(candidates, tick=1)
    workspace = WorkspaceState(
        tick=1,
        focus=selection.selected_ids,
        broadcast=selection.broadcast,
        capacity=selection.capacity,
        candidates=candidates,
        selection=selection,
    )
    restored = WorkspaceState.from_payload(workspace.to_payload())

    assert restored.selection is not None
    assert restored.selection.selected_ids == selection.selected_ids
    assert torch.equal(restored.broadcast, workspace.broadcast)
    assert tuple(candidate.candidate_id for candidate in restored.candidates) == (
        "relevant",
        "distractor-a",
        "distractor-b",
    )

    model = TSKV8Adapter()
    model.attach_workspace_router(router)
    model.observe(97, learn=False, workspace_candidates=candidates)
    checkpoint = model.native_checkpoint()
    restored_model = TSKV8Adapter.from_native_checkpoint(checkpoint)
    restored_state = restored_model.cognitive_snapshot()

    assert restored_model._workspace_router is not None
    assert restored_model._workspace_router.fit_updates == router.fit_updates
    assert restored_state.workspace.selection is not None
    assert restored_state.workspace.selection.candidate_ids == tuple(
        candidate.candidate_id for candidate in candidates
    )


def test_workspace_selection_payload_round_trip() -> None:
    selection = WorkspaceSelection(
        tick=3,
        mode="none",
        candidate_ids=("a",),
        selected_ids=(),
        scores=(0.0,),
        broadcast=torch.zeros(2),
        capacity=1,
    )
    restored = WorkspaceSelection.from_payload(selection.to_payload())

    assert restored.mode == "none"
    assert restored.candidate_ids == ("a",)
    assert torch.equal(restored.broadcast, selection.broadcast)
