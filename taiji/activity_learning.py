"""R029 activity-policy learning on the R025/R027 activity loop.

The core subject decides activities through its own policy head. This module adds
a declared value head, P_R probe reward, semi-Markov eligibility traces and
critic updates. It is an engineering instrument: no sealed data, no product
default and no claim that the policy has learned useful autonomy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn

from taiji.living_core import LivingCoreReference, LivingState
from taiji.native_activity import (
    ACTIVITIES,
    ActivityEnvironment,
    ActivityPolicy,
    build_features,
    fixed_threshold_choice,
    legal_mask,
    run_round,
)
from taiji.state_migration import CareerJournal, resume_career

GAMMA = 0.99
TRACE_LAMBDA = 0.0
POLICY_LR = 1.0
CRITIC_LR = 1e-1
EPSILON = 0.5
KAPPA = 1.0
INTERACT_REWARD = 5e-2
COMPUTE_COST = 1e-3
RETENTION_PENALTY = 0.1
HIDDEN = 16


class ActivityCritic(nn.Module):
    def __init__(self, width: int, seed: int = 20261031) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.input = nn.Linear(2 * width + 19, HIDDEN)
            self.output = nn.Linear(HIDDEN, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value: torch.Tensor = self.output(torch.tanh(self.input(features))).squeeze(-1)
        return value


@dataclass
class ActivityLearner:
    policy: ActivityPolicy
    critic: ActivityCritic
    eligibility: dict[str, torch.Tensor]
    critic_optimizer: torch.optim.Optimizer
    exposures: int = 0

    @classmethod
    def build(cls, width: int, dtype=torch.float64) -> ActivityLearner:
        policy = ActivityPolicy(width).to(dtype)
        critic = ActivityCritic(width).to(dtype)
        eligibility = {
            name: torch.zeros_like(parameter) for name, parameter in policy.named_parameters()
        }
        optimizer = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)
        return cls(policy, critic, eligibility, optimizer)

    def state_dict(self) -> dict:
        return {
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "eligibility": {
                name: value.detach().clone() for name, value in self.eligibility.items()
            },
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "exposures": self.exposures,
        }

    def load_state_dict(self, data: dict) -> None:
        self.policy.load_state_dict(data["policy"])
        self.critic.load_state_dict(data["critic"])
        for name, value in data["eligibility"].items():
            self.eligibility[name].copy_(value)
        self.critic_optimizer.load_state_dict(data["critic_optimizer"])
        self.exposures = int(data["exposures"])


def probe_nll(
    model: LivingCoreReference,
    state: LivingState,
    journal: CareerJournal,
    provider=None,
) -> float:
    with torch.no_grad():
        _, steps = resume_career(model, state, journal, 0, provider, retain_outputs=False)
        terms = [probabilities[byte] for probabilities, byte in steps if byte is not None]
        return float(-torch.stack(terms).log().mean().detach())


def masked_log_probabilities(policy: ActivityPolicy, features: torch.Tensor, env):
    logits = policy(features)
    mask = torch.tensor([bool(legal_mask(env)[name]) for name in ACTIVITIES], dtype=logits.dtype)
    masked = logits.masked_fill(mask == 0, float("-inf"))
    return torch.log_softmax(masked, dim=-1)


def activity_reward(
    model: LivingCoreReference,
    before_state: LivingState,
    after_state: LivingState,
    reward_journal: CareerJournal,
    retention_journal: CareerJournal,
    step_delta: int,
    provider=None,
    actual_reward: float = 0.0,
) -> dict:
    phi_before = -probe_nll(model, before_state, reward_journal, provider)
    phi_after = -probe_nll(model, after_state, reward_journal, provider)
    retention_before = probe_nll(model, before_state, retention_journal, provider)
    retention_after = probe_nll(model, after_state, retention_journal, provider)
    compute = COMPUTE_COST * step_delta
    retention_change = abs(retention_after - retention_before)
    reward = (
        actual_reward
        + KAPPA * (phi_after - phi_before)
        - compute
        - RETENTION_PENALTY * retention_change
    )
    return {
        "actual_reward": float(actual_reward),
        "phi_before": phi_before,
        "phi_after": phi_after,
        "retention_before": retention_before,
        "retention_after": retention_after,
        "compute_cost": compute,
        "retention_penalty": RETENTION_PENALTY * retention_change,
        "reward": float(reward),
    }


def train_activity_episode(
    model: LivingCoreReference,
    state: LivingState,
    env: ActivityEnvironment,
    learner: ActivityLearner,
    reward_journal: CareerJournal,
    retention_journal: CareerJournal,
    *,
    rounds: int,
    journal: list,
    last_event: torch.Tensor,
    prediction_error_value: float = 0.0,
    provider=None,
    max_exposures: int = 32,
    arrivals=None,
):
    rows = []
    for index in range(rounds):
        if learner.exposures >= max_exposures:
            rows.append({"round": index, "status": "exposure_budget_stop"})
            break
        round_start = time.process_time()
        goal_vector = model.encoder.goal_vector(env.goal)
        features = build_features(state, goal_vector, prediction_error_value, env).detach()
        log_probabilities = masked_log_probabilities(learner.policy, features, env)
        probabilities = log_probabilities.exp()
        legal_names = [name for name in ACTIVITIES if legal_mask(env)[name]]
        explore = bool(torch.rand((), dtype=probabilities.dtype) < EPSILON)
        if explore:
            legal_index = int(torch.randint(len(legal_names), (1,)))
            action = legal_names[legal_index]
        else:
            action_index = int(torch.multinomial(probabilities.detach(), 1))
            action = ACTIVITIES[action_index]
        action_index = ACTIVITIES.index(action)
        value = learner.critic(features)
        before_state = state
        with torch.no_grad():
            state, env, entry, last_event, prediction_error_value = run_round(
                model,
                state,
                env,
                learner.policy,
                journal,
                last_event=last_event,
                prediction_error_value=prediction_error_value,
                model_action=action,
                gate_provider=provider,
                arrivals=arrivals,
            )
        if entry["activity"] != action:
            raise RuntimeError("training action diverged from activity journal")
        next_goal = model.encoder.goal_vector(env.goal)
        next_features = build_features(state, next_goal, prediction_error_value, env).detach()
        next_value = learner.critic(next_features)
        terminal = index == rounds - 1
        actual_reward = INTERACT_REWARD if entry.get("external_trigger") else 0.0
        reward_fields = activity_reward(
            model,
            before_state,
            state,
            reward_journal,
            retention_journal,
            entry["step_delta"],
            provider,
            actual_reward,
        )
        target = reward_fields["reward"]
        if not terminal:
            target += GAMMA * float(next_value.detach())
        td_error = target - float(value.detach())
        log_probability = log_probabilities[action_index]
        gradients = torch.autograd.grad(log_probability, list(learner.policy.parameters()))
        with torch.no_grad():
            for (name, parameter), gradient in zip(
                learner.policy.named_parameters(), gradients, strict=True
            ):
                trace = learner.eligibility[name]
                trace.mul_(GAMMA * TRACE_LAMBDA).add_(gradient)
                parameter.add_(POLICY_LR * td_error * trace)
        critic_loss = 0.5 * (torch.as_tensor(target, dtype=value.dtype) - value) ** 2
        learner.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        learner.critic_optimizer.step()
        learner.exposures += 1
        rows.append(
            {
                "round": index,
                "status": "updated",
                "activity": entry["activity"],
                "chosen_by": entry["chosen_by"],
                "exploration": explore,
                "step_delta": entry["step_delta"],
                "triggers": entry["triggers"],
                "imagined": entry["imagined"],
                "store_size_after": entry["store_size_after"],
                "reward": reward_fields["reward"],
                "td_error": float(td_error),
                "value": float(value.detach()),
                "next_value": float(next_value.detach()),
                "critic_loss": float(critic_loss.detach()),
                "policy_grad_norm": float(torch.stack([g.norm() for g in gradients]).norm()),
                "eligibility_norm": float(
                    torch.stack([value.norm() for value in learner.eligibility.values()]).norm()
                ),
                "exposures": learner.exposures,
                "round_cpu_seconds": time.process_time() - round_start,
                **reward_fields,
            }
        )
    return state, env, last_event, prediction_error_value, rows


def evaluate_activity_policy(
    model: LivingCoreReference,
    state: LivingState,
    env: ActivityEnvironment,
    policy: ActivityPolicy,
    reward_journal: CareerJournal,
    retention_journal: CareerJournal,
    *,
    rounds: int,
    journal: list,
    last_event: torch.Tensor,
    prediction_error_value: float = 0.0,
    provider=None,
    control: bool = False,
    arrivals=None,
):
    """No-update P_R evaluation of a greedy model policy or the fixed control."""
    rows = []
    cumulative = 0.0
    chooser = fixed_threshold_choice if control else None
    for index in range(rounds):
        round_start = time.process_time()
        before_state = state
        with torch.no_grad():
            state, env, entry, last_event, prediction_error_value = run_round(
                model,
                state,
                env,
                policy,
                journal,
                last_event=last_event,
                prediction_error_value=prediction_error_value,
                chooser=chooser,
                gate_provider=provider,
                arrivals=arrivals,
            )
        actual_reward = INTERACT_REWARD if entry.get("external_trigger") else 0.0
        reward_fields = activity_reward(
            model,
            before_state,
            state,
            reward_journal,
            retention_journal,
            entry["step_delta"],
            provider,
            actual_reward,
        )
        cumulative += reward_fields["reward"]
        rows.append(
            {
                "round": index,
                "activity": entry["activity"],
                "chosen_by": entry["chosen_by"],
                "step_delta": entry["step_delta"],
                "triggers": entry["triggers"],
                "imagined": entry["imagined"],
                "store_size_after": entry["store_size_after"],
                "round_cpu_seconds": time.process_time() - round_start,
                **reward_fields,
            }
        )
    return state, env, last_event, prediction_error_value, rows, cumulative
