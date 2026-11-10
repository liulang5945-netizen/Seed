"""Machine checks for the fairness of goal->action fixtures.

Why this module exists: the whole 10.7 selection line was voided by 14.1/14.2 because
its fixture put the correct answer into the prompt -- the goal string and the correct
action sentence were byte-identical -- and the model answered by copying. That failure
was found by hand, one experiment late. These rules turn the same lesson into a check
that can be run against a fixture before any training or readout is spent on it.

A sample is normalised to:

    goal              the text shown to the model
    action            the sentence that counts as correct
    entities          surface names the fixture can talk about
    answer_entities   the names the correct action depends on
    records           the glossary/state lines the mapping may be read from
    split             "train", "dev" or None

Rules report observed values as well as pass/fail, and a rule with no applicable input
reports ``not_applicable`` rather than silently passing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

RULES = (
    "goal_not_identical_to_action",
    "goal_names_no_answer_entity",
    "goal_shares_no_bigram_with_action",
    "goal_carries_no_action_template",
    "mapping_only_reachable_through_records",
    "split_holds_out_entities_and_novel_goals",
    "control_is_fixed_point_free_permutation",
)

ACTION_TEMPLATES = ("把", "放到")


def bigrams(text: str) -> set[str]:
    return {text[index : index + 2] for index in range(len(text) - 1)}


def _row(rule: str, passed: bool | None, observed: dict) -> dict:
    return {"rule": rule, "passed": passed, "observed": observed}


def check_goal_not_identical_to_action(samples: list[Mapping]) -> dict:
    hits = [
        {"index": index, "goal": sample["goal"]}
        for index, sample in enumerate(samples)
        if sample["goal"].strip() == sample["action"].strip()
    ]
    return _row(
        "goal_not_identical_to_action",
        not hits,
        {"samples": len(samples), "identical": len(hits), "examples": hits[:3]},
    )


def check_goal_names_no_answer_entity(samples: list[Mapping], entities: Iterable[str]) -> dict:
    names = tuple(entities)
    hits = [
        {
            "index": index,
            "goal": sample["goal"],
            "named": [name for name in names if name in sample["goal"]],
        }
        for index, sample in enumerate(samples)
        if any(name in sample["goal"] for name in names)
    ]
    return _row(
        "goal_names_no_answer_entity",
        not hits,
        {"entities": len(names), "samples_naming_an_entity": len(hits), "examples": hits[:3]},
    )


def check_goal_shares_no_bigram_with_action(samples: list[Mapping]) -> dict:
    worst: set[str] = set()
    worst_index: int | None = None
    total = 0
    for index, sample in enumerate(samples):
        shared = bigrams(sample["goal"]) & bigrams(sample["action"])
        total += len(shared)
        if len(shared) > len(worst):
            worst, worst_index = shared, index
    return _row(
        "goal_shares_no_bigram_with_action",
        total == 0,
        {
            "samples": len(samples),
            "shared_bigram_total": total,
            "worst_index": worst_index,
            "worst_shared": sorted(worst)[:8],
        },
    )


def check_goal_carries_no_action_template(samples: list[Mapping]) -> dict:
    hits = [
        {
            "index": index,
            "goal": sample["goal"],
            "templates": [t for t in ACTION_TEMPLATES if t in sample["goal"]],
        }
        for index, sample in enumerate(samples)
        if any(template in sample["goal"] for template in ACTION_TEMPLATES)
    ]
    return _row(
        "goal_carries_no_action_template",
        not hits,
        {
            "templates": list(ACTION_TEMPLATES),
            "samples_with_template": len(hits),
            "examples": hits[:3],
        },
    )


def check_mapping_only_reachable_through_records(samples: list[Mapping]) -> dict:
    missing: list[dict] = []
    in_goal: list[dict] = []
    for index, sample in enumerate(samples):
        records = "".join(sample.get("records") or ())
        absent = [name for name in sample["answer_entities"] if name not in records]
        if absent:
            missing.append({"index": index, "absent_from_records": absent})
        in_goal.extend(
            {"index": index, "name": name}
            for name in sample["answer_entities"]
            if name in sample["goal"]
        )
    passed = not missing and not in_goal
    return _row(
        "mapping_only_reachable_through_records",
        passed,
        {
            "samples": len(samples),
            "answer_names_missing_from_records": len(missing),
            "examples_missing": missing[:3],
            "answer_names_also_in_goal": len(in_goal),
            "examples_in_goal": in_goal[:3],
        },
    )


def check_split(splits: list[Mapping]) -> dict:
    if not any(sample.get("split") for sample in splits):
        return _row(
            "split_holds_out_entities_and_novel_goals",
            None,
            {"note": "fixture publishes no train/dev labels; rule not exercised"},
        )
    train = [s for s in splits if s.get("split") == "train"]
    holdout = [s for s in splits if s.get("split") and s["split"] != "train"]

    def task_of(sample: Mapping) -> tuple:
        signature = sample.get("records_signature")
        if signature is None:
            signature = "|".join(sample.get("records", ()))
        return (sample["goal"], signature)

    train_goals = {s["goal"] for s in train}
    # Novelty is measured on the whole task, not on the goal text alone: the 10.21 fixture
    # reuses goal strings across worlds on purpose and changes the mapping per world, so a
    # holdout example is new when its (goal, records) pair is new. The goal-only fraction is
    # still reported because "480 samples, 20 distinct goals" was the 10.16 memorisation trap.
    train_tasks = {task_of(s) for s in train}
    # The axis is the fixture's own declared holdout dimension. The 10.21 fixture holds
    # out objects and reuses places on purpose, so gating on places there would be a
    # false alarm; the 10.7f face declared a place holdout and is judged on that.
    train_entities = {name for s in train for name in s.get("holdout_entities", ())}
    novel = [s for s in holdout if task_of(s) not in train_tasks]
    novel_goals_only = [s for s in holdout if s["goal"] not in train_goals]
    leaked = sorted(
        {name for s in holdout for name in s.get("holdout_entities", ()) if name in train_entities}
    )
    passed = bool(holdout) and bool(novel) and not leaked
    return _row(
        "split_holds_out_entities_and_novel_goals",
        passed,
        {
            "train": len(train),
            "holdout": len(holdout),
            "holdout_tasks_unseen_in_train": len(novel),
            "novel_task_fraction": len(novel) / len(holdout) if holdout else 0.0,
            "holdout_goals_unseen_in_train": len(novel_goals_only),
            "novel_goal_only_fraction": len(novel_goals_only) / len(holdout) if holdout else 0.0,
            "entities_shared_between_train_and_holdout": leaked,
        },
    )


def check_control(
    mapping: Iterable[tuple[str, str]] | None, real: Iterable[tuple[str, str]] | None
) -> dict:
    """A control glossary has to re-map every descriptor to a *different* entity while
    staying a bijection onto the same entity set. A rotation that leaves one descriptor its
    own real entity, or that duplicates an entity, cannot carry a permutation contrast."""
    if mapping is None or real is None:
        return _row(
            "control_is_fixed_point_free_permutation",
            None,
            {"note": "fixture exposes no control mapping to check"},
        )
    pairs = list(mapping)
    truth = dict(real)
    sources = [key for key, _ in pairs]
    targets = [value for _, value in pairs]
    bijective = len(set(sources)) == len(sources) and len(set(targets)) == len(targets)
    same_domain = set(sources) == set(truth)
    same_codomain = set(targets) == set(truth.values())
    fixed = [key for key, value in pairs if value == truth.get(key)]
    return _row(
        "control_is_fixed_point_free_permutation",
        same_domain and same_codomain and bijective and not fixed,
        {
            "pairs": len(pairs),
            "same_domain": same_domain,
            "same_codomain": same_codomain,
            "bijective": bijective,
            "assignments_unchanged_from_real": fixed,
        },
    )


def check_fixture(
    samples: list[Mapping],
    *,
    entities: Iterable[str],
    splits: list[Mapping] | None = None,
    control: Iterable[tuple[str, str]] | None = None,
    real_mapping: Iterable[tuple[str, str]] | None = None,
) -> list[dict]:
    """Run every declared rule; the list order is fixed so reports stay comparable."""
    return [
        check_goal_not_identical_to_action(samples),
        check_goal_names_no_answer_entity(samples, entities),
        check_goal_shares_no_bigram_with_action(samples),
        check_goal_carries_no_action_template(samples),
        check_mapping_only_reachable_through_records(samples),
        check_split(splits if splits is not None else samples),
        check_control(control, real_mapping),
    ]
