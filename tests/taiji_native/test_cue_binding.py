import torch

from taiji import CueBindingBank


def test_cue_binding_allocates_matches_and_reads_without_mutation() -> None:
    bank = CueBindingBank(capacity=2, pattern_dim=4, match_threshold=0.9)
    first = torch.tensor([1.0, 0.0, 0.0, 0.0])
    second = torch.tensor([0.0, 1.0, 0.0, 0.0])

    allocated = bank.route(first, learn=True)
    matched = bank.route(first, learn=True)
    before = bank.to_payload()
    read = bank.route(first, learn=False)

    assert allocated.allocated is True
    assert matched.allocated is False
    assert read.slot_index == allocated.slot_index
    assert bank.match_count == 1
    assert torch.equal(before["prototypes"], bank.prototypes)
    assert torch.equal(before["visits"], bank.visits)
    assert bank.route(second, learn=True).slot_index != allocated.slot_index


def test_cue_binding_releases_and_round_trips_checkpoint() -> None:
    bank = CueBindingBank(capacity=2, pattern_dim=3, match_threshold=0.8)
    bank.route(torch.tensor([1.0, 0.0, 0.0]), learn=True)
    bank.route(torch.tensor([0.0, 1.0, 0.0]), learn=True)
    payload = bank.to_payload()

    restored = CueBindingBank(capacity=2, pattern_dim=3, match_threshold=0.8)
    restored.load_payload(payload)
    assert restored.occupied_count == 2
    assert restored.route(torch.tensor([1.0, 0.0, 0.0]), learn=False).slot_index == 0

    restored.release(0)
    assert restored.occupied_count == 1
    assert restored.route(torch.tensor([1.0, 0.0, 0.0]), learn=False).slot_index is None
