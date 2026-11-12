"""Deterministic bounded-substep disk mechanics for the frozen U2 world."""

from __future__ import annotations

import copy
import math

import numpy as np


def advance(bodies: list[dict], command, config: dict):
    bodies = copy.deepcopy(bodies)
    positions = np.array([body["position"] for body in bodies], dtype=np.float64)
    velocities = np.array([body["velocity"] for body in bodies], dtype=np.float64)
    masses = np.array([body["mass"] for body in bodies])
    radii = np.array([body["radius"] for body in bodies])
    drag = np.array([body["drag"] for body in bodies])
    target, direction = np.asarray(command[9:11]), np.asarray(command[11:13])
    intensity, pulse = float(command[13]), float(command[14])
    elapsed, count = 0.0, 0
    path = [(0.0, positions.copy())]
    sounds = []
    reaction = np.zeros(2)
    restitution = config["restitution"]

    def sound(index, impulse, when, fraction=1.0):
        amplitude = fraction * min(0.8, abs(impulse) / 0.5)
        if amplitude:
            sounds.append((when, bodies[index]["frequency"], amplitude))

    while config["tick"] - elapsed > 1e-12:
        if count >= config["max_substeps"]:
            raise ValueError("physics_substep_budget_stop")
        speed = float(np.linalg.norm(velocities, axis=1).max())
        distance = float(radii.min()) / 8
        acceleration = config["force_scale"] / float(masses.min())
        motion_step = 2 * distance / (speed + math.sqrt(speed**2 + 2 * acceleration * distance))
        h = min(config["substep_max"], config["tick"] - elapsed, motion_step)
        duty = max(0.0, min(1.0, (pulse - elapsed) / h))
        weights = np.maximum(0, 1 - np.linalg.norm(positions - target, axis=1) / (radii + 0.1))
        weights /= max(1.0, float(weights.sum()))
        force = config["force_scale"] * intensity * weights[:, None] * direction
        impulses = duty * force * h
        reaction -= impulses.sum(axis=0)
        velocities = (velocities + impulses / masses[:, None]) / (1 + drag[:, None] * h)
        positions += h * velocities
        end = elapsed + h
        for index, impulse in enumerate(impulses):
            sound(index, float(np.linalg.norm(impulse)), end)
        order = sorted(range(len(bodies)), key=lambda i: (*positions[i], radii[i]))
        for _ in range(config["contact_passes"]):
            for i in order:
                for axis in range(2):
                    lower, upper = -1 + radii[i], 1 - radii[i]
                    if positions[i, axis] < lower or positions[i, axis] > upper:
                        outward = (positions[i, axis] < lower and velocities[i, axis] < 0) or (
                            positions[i, axis] > upper and velocities[i, axis] > 0
                        )
                        positions[i, axis] = np.clip(positions[i, axis], lower, upper)
                        if outward:
                            old = velocities[i, axis]
                            velocities[i, axis] *= -restitution
                            sound(i, masses[i] * abs(old - velocities[i, axis]), end)
            for offset, i in enumerate(order):
                for j in order[offset + 1 :]:
                    delta = positions[j] - positions[i]
                    distance = float(np.linalg.norm(delta))
                    if distance <= 1e-12:
                        raise ValueError("degenerate_contact")
                    if distance > radii[i] + radii[j] + config["penetration_tolerance"]:
                        continue
                    normal = delta / distance
                    inverse = 1 / masses[i] + 1 / masses[j]
                    penetration = max(0.0, radii[i] + radii[j] - distance)
                    positions[i] -= penetration / masses[i] / inverse * normal
                    positions[j] += penetration / masses[j] / inverse * normal
                    closing = float((velocities[j] - velocities[i]) @ normal)
                    if closing < 0:
                        impulse = -(1 + restitution) * closing / inverse
                        velocities[i] -= impulse / masses[i] * normal
                        velocities[j] += impulse / masses[j] * normal
                        sound(i, impulse, end, 0.5)
                        sound(j, impulse, end, 0.5)
            penetration, closing = contact_error(positions, velocities, radii)
            if (
                penetration <= config["penetration_tolerance"]
                and closing <= config["normal_velocity_tolerance"]
            ):
                break
        else:
            raise ValueError("contact_solver_not_converged")
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise ValueError("nonfinite_physics")
        elapsed, count = end, count + 1
        path.append((elapsed, positions.copy()))
    for i, body in enumerate(bodies):
        body["position"], body["velocity"] = positions[i].tolist(), velocities[i].tolist()
    return bodies, path, sounds, reaction, count


def contact_error(positions, velocities, radii):
    penetration, closing = 0.0, 0.0
    for i in range(len(radii)):
        penetration = max(
            penetration, float(np.maximum(np.abs(positions[i]) + radii[i] - 1, 0).max())
        )
        for axis in range(2):
            if positions[i, axis] <= -1 + radii[i] + 1e-9:
                closing = max(closing, -float(velocities[i, axis]))
            if positions[i, axis] >= 1 - radii[i] - 1e-9:
                closing = max(closing, float(velocities[i, axis]))
        for j in range(i + 1, len(radii)):
            delta = positions[j] - positions[i]
            distance = float(np.linalg.norm(delta))
            penetration = max(penetration, float(radii[i] + radii[j] - distance))
            if distance <= radii[i] + radii[j] + 1e-9:
                if distance <= 1e-12:
                    return float("inf"), float("inf")
                closing = max(closing, -float((velocities[j] - velocities[i]) @ (delta / distance)))
    return penetration, closing
