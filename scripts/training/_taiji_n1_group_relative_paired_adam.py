"""PLAN69: the frozen bounded Adam proposal with explicit unchanged control arms."""

from scripts.training import _taiji_n1_group_relative_adam as bounded

runner = bounded.runner
POLICY = bounded.POLICY


def train_step(model, optimizer, item, session, arm, second, *, relative_limit=0.001, audit=False):
    if arm not in runner.ARMS:
        raise ValueError("frozen paired arm required")
    original_step = optimizer.step
    observations = []

    def proposed_step():
        returned, measurement = bounded.bounded_step(
            model, optimizer, original_step, relative_limit=relative_limit, audit=audit
        )
        observations.append(measurement)
        return returned

    optimizer.step = proposed_step
    try:
        loss = runner.train_step(model, optimizer, item, session, arm, second)
    finally:
        optimizer.step = original_step
    if len(observations) != 1:
        raise ValueError("exactly one unchanged same-arm Adam proposal required")
    return loss, observations[0]
