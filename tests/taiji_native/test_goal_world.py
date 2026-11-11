"""10.1 oracles: the task environment must not be solvable by position or template."""

from __future__ import annotations

from taiji.native_goal_world import (
    OBJECTS,
    PLACES,
    SHORTCUT_POLICIES,
    build_fixtures,
    encode_actions,
    legal_moves,
    optimal,
    run_policy,
    score,
    step,
)

SEED = 20261101


def test_legal_moves_exclude_the_current_place():
    state = build_fixtures()[0].initial
    moves = legal_moves(state)
    assert len(moves) == len(OBJECTS) * (len(PLACES) - 1)
    for obj, place in moves:
        assert place != state.place_of(obj)


def test_action_ids_are_reshuffled_every_step():
    state = build_fixtures()[0].initial
    moves = legal_moves(state)
    first = encode_actions(moves, seed=SEED, step_index=0)
    second = encode_actions(moves, seed=SEED, step_index=1)
    assert set(first) == set(second)
    assert first != second, "the same moves must not keep their ids across steps"
    payloads_first = {(item["object"], item["place"]) for item in first.values()}
    payloads_second = {(item["object"], item["place"]) for item in second.values()}
    assert payloads_first == payloads_second
    id_of = {(item["object"], item["place"]): key for key, item in second.items()}
    assert any(
        id_of[payload] != key
        for key, item in first.items()
        for payload in [(item["object"], item["place"])]
    )


def test_optimal_scores_are_reachable_and_pay_the_step_cost():
    for fixture in build_fixtures():
        best = optimal(fixture.initial, fixture.goal)
        assert best["score"] > 0.0
        assert best["steps"] == len(best["moves"])
        replay = fixture.initial
        for move in best["moves"]:
            replay = step(replay, move)
        assert abs(score(replay, fixture.goal) - best["score"]) <= 1e-12
        assert fixture.goal.satisfied(replay)


def test_swapping_the_goal_changes_the_optimal_moves():
    fixtures = {fixture.fixture_id: fixture for fixture in build_fixtures()}
    first = optimal(
        fixtures["f1-cup-to-sill-from-table"].initial, fixtures["f1-cup-to-sill-from-table"].goal
    )
    second = optimal(
        fixtures["f2-seal-to-table-from-box"].initial, fixtures["f2-seal-to-table-from-box"].goal
    )
    assert (
        fixtures["f1-cup-to-sill-from-table"].initial.where
        == fixtures["f2-seal-to-table-from-box"].initial.where
    )
    assert tuple(first["moves"]) != tuple(second["moves"])


def test_no_shortcut_policy_solves_every_fixture():
    fixtures = build_fixtures()
    for name, policy in SHORTCUT_POLICIES.items():
        verdicts = []
        for fixture in fixtures:
            best = optimal(fixture.initial, fixture.goal)
            result = run_policy(policy, fixture.initial, fixture.goal, seed=SEED)
            verdicts.append(abs(result["score"] - best["score"]) <= 1e-12)
        assert not all(verdicts), f"{name} solved every fixture"
        assert any(not value for value in verdicts), name


def test_scoring_depends_only_on_the_terminal_state():
    fixture = build_fixtures()[0]
    first = step(step(fixture.initial, (OBJECTS[0], PLACES[1])), (OBJECTS[1], PLACES[0]))
    second = step(step(fixture.initial, (OBJECTS[1], PLACES[0])), (OBJECTS[0], PLACES[1]))
    assert first.where == second.where
    assert first.steps == second.steps
    assert abs(score(first, fixture.goal) - score(second, fixture.goal)) <= 1e-12
