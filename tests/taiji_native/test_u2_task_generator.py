"""Generator-entry shape, capsule roundtrip and family-case regressions."""

from taiji.u2_task_contract import validate_case
from taiji.u2_task_generator import generate, load_capsule, rebuild, save_capsule


def test_a_family_shape_and_capsule_roundtrip(tmp_path):
    sample = generate("A", 124)
    assert sample.envelope["schema"] == "taiji-u2-task-sample-v1"
    assert sample.envelope["family"] == "A"
    assert sample.envelope["seed"] == 124
    shapes = {
        key: value["shape"]
        for key, value in sample.envelope["model_observation"]["observations"].items()
    }
    assert shapes == {"rgb": [16, 48], "audio": [7, 33], "proprio": [1, 16]}
    assert sample.case is not None
    validate_case(sample.case)
    target = sample.case.target.decode()
    assert target not in sample.case.goal_raw.decode()
    assert target not in sample.case.request.raw.decode()
    path = tmp_path / "sample.capsule"
    assert save_capsule(sample, path) > 0
    loaded = load_capsule(path)
    assert loaded["envelope_digest"] == sample.envelope["envelope_digest"]
    assert rebuild(sample)["equal"] is True


def test_c_family_case_and_choice():
    sample = generate("C", 124)
    assert sample.case is not None
    validate_case(sample.case)
    utilities = sample.task["utilities"]
    assert sample.task["choice"] == max(utilities, key=utilities.get)
    assert sample.case.target.decode() == sample.task["truth"]
