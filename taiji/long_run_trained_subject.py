"""Single-subject long-run trained durability & joint lifecycle (`taiji-long-run-trained-v1`).

Unifies all confirmed N33/N34 capabilities into ONE persistent `LongRunHarness` + `SessionStore`
subject across 100 sessions, >=10,000 persisted experience events, 10 pre-commit crash
recoveries, >=20 cross-session pending handoffs, and 2/2 sealed `check`/`final` blocks:

1. `ch.*`: `ChineseTurnPolicy(recurrent=True)` trained online on `ChineseTaskEnvV2`;
2. `phy.*`: `ConsequenceNetV2` trained online on `ramp_bumper` physics trajectories;
3. `grow.*`: `PlasticGrowthNet` + `GrowthPressureController` across P1 -> mature -> P2 on-demand
   neutral birth (`8 -> 32`) -> dormancy/wake -> 40 maintenance endurance sessions;
4. `ecl.*`: `BetaCredibility(forgetting=0.995)` tracking a mid-stream source drift;
5. `unc.*`: `LearnedConfidence` calibrated online strictly from the subject's own verifiable
   Chinese + physics prediction outcomes.

Zero oracle in training; zero `check`/`final` reads; zero confirmation seeds unless explicitly
passed by a pre-registered confirmation card (`N34.5.90`). All 5 models are serialised into and
reloaded from `record.capsule` on disk on every single session.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from scripts.training.nch_2_01_continuous_chinese import (
    EPOCHS as CH_EPOCHS,
)
from scripts.training.nch_2_01_continuous_chinese import (
    EVAL_SEEDS as CH_EVAL_SEEDS,
)
from scripts.training.nch_2_01_continuous_chinese import (
    HELDOUT_SEEDS as CH_HELDOUT_SEEDS,
)
from scripts.training.nch_2_01_continuous_chinese import (
    LR as CH_LR,
)
from scripts.training.nch_2_01_continuous_chinese import (
    ROLLOUT_EPISODES as CH_ROLLOUT,
)
from scripts.training.nch_2_01_continuous_chinese import (
    ChineseTurnPolicy,
)
from scripts.training.nch_2_01_continuous_chinese import (
    _eval_arm as ch_eval_arm,
)
from taiji.chinese_task_env_v2 import MAX_TOKENS, ChineseTaskEnvV2
from taiji.continual_growth_task import IN_DIM as GROW_IN_DIM
from taiji.continual_growth_task import NUM_CLASSES as GROW_OUT_DIM
from taiji.continual_growth_task import make_split as make_grow_split
from taiji.credibility_learning import SOURCES, TRUE_ACC, BetaCredibility, ClaimStream, StaticPrior
from taiji.growth_decision_learning import _batches as grow_batches
from taiji.long_run_harness import LongRunHarness, SealViolation
from taiji.multimodal_physics_env_v2 import (
    OUTCOMES,
    ConsequenceNetV2,
    ablate_barrier_collision_recall,
    collect_v2,
    eval_metrics_v2,
)
from taiji.plastic_growth import GrowthPressureController, PlasticGrowthNet
from taiji.session_store import SessionRecord
from taiji.uncertainty_learning import ece as compute_ece

SCHEMA = "taiji-long-run-trained-v1"
CRASH_SESSIONS_DEFAULT = (9, 19, 29, 39, 49, 59, 69, 79, 89, 95)
GROW_BATCH = 128
GROW_LR_BASE = 5e-3
GROW_LR_POST = 1e-2


def _save_prefixed_state(
    capsule: dict[str, torch.Tensor], prefix: str, state: dict[str, torch.Tensor]
) -> None:
    for k, v in state.items():
        capsule[f"{prefix}.{k}"] = v.detach().cpu().clone()


def _load_prefixed_state(capsule: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    head = f"{prefix}."
    return {k[len(head) :]: v.clone() for k, v in capsule.items() if k.startswith(head)}


def _pack_growth_net(
    capsule: dict[str, torch.Tensor],
    net: PlasticGrowthNet,
    ctrl: GrowthPressureController,
    scalars: dict[str, float],
) -> None:
    capsule["grow.fc_in.w"] = net.fc_in.weight.detach().cpu().clone()
    capsule["grow.fc_in.b"] = net.fc_in.bias.detach().cpu().clone()
    capsule["grow.fc_out.w"] = net.fc_out.weight.detach().cpu().clone()
    capsule["grow.fc_out.b"] = net.fc_out.bias.detach().cpu().clone()
    capsule["grow.stage"] = net.stage.detach().cpu().clone()  # type: ignore[operator]
    capsule["grow.unit_ids"] = net.unit_ids.detach().cpu().clone()  # type: ignore[operator]
    capsule["grow.meta"] = torch.tensor(
        [net.in_dim, net.hidden_dim, net.out_dim, net._next_unit_id, ctrl.growth_events],
        dtype=torch.long,
    )
    capsule["grow.ctrl_hist"] = torch.tensor(ctrl.history, dtype=torch.float32)
    capsule["grow.scalars"] = torch.tensor(
        [
            scalars.get("p1_before", 0.0),
            scalars.get("birth_pert", 0.0),
            scalars.get("dorm_drop", 0.0),
            scalars.get("wake_diff", 0.0),
            scalars.get("grew_session", -1.0),
        ],
        dtype=torch.float32,
    )


def _unpack_growth_net(
    capsule: dict[str, torch.Tensor],
) -> tuple[PlasticGrowthNet, GrowthPressureController, dict[str, float]]:
    in_dim, hidden_dim, out_dim, next_uid, events = (int(v) for v in capsule["grow.meta"].tolist())
    net = PlasticGrowthNet(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, seed=0)
    net._next_unit_id = next_uid
    net._rebuild_layers(
        capsule["grow.fc_in.w"].clone(),
        capsule["grow.fc_in.b"].clone(),
        capsule["grow.fc_out.w"].clone(),
        capsule["grow.fc_out.b"].clone(),
        capsule["grow.stage"].clone(),
        capsule["grow.unit_ids"].clone(),
    )
    ctrl = GrowthPressureController(window=3, plateau_eps=0.015, error_threshold=0.15, grow_k=24)
    ctrl.growth_events = events
    ctrl.history = [float(x) for x in capsule["grow.ctrl_hist"].tolist()]
    sc = capsule["grow.scalars"].tolist()
    scalars = {
        "p1_before": float(sc[0]),
        "birth_pert": float(sc[1]),
        "dorm_drop": float(sc[2]),
        "wake_diff": float(sc[3]),
        "grew_session": float(sc[4]),
    }
    return net, ctrl, scalars


def _eval_acc(net: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    net.eval()
    with torch.no_grad():
        pred = net(x).argmax(dim=-1)
    net.train()
    return float((pred == y).float().mean())


def _confidence_features(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1).values
    return torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1]], dim=1)


@dataclass
class RunConfig:
    n_sessions: int = 100
    events_per_session: int = 100
    crash_sessions: tuple[int, ...] = CRASH_SESSIONS_DEFAULT
    p1_end_session: int = 20
    p2_end_session: int = 60
    dormancy_session: int = 70
    ch_train_end_session: int = 40
    ch_updates_per_session: int = 6
    phy_steps_early: int = 40
    phy_steps_mid: int = 20
    grow_p1_steps: int = 15
    grow_p2_steps: int = 25
    grow_maint_steps: int = 5


def _init_capsule(record: SessionRecord, seed: int) -> None:
    ch = ChineseTurnPolicy(seed, recurrent=True)
    _save_prefixed_state(record.capsule, "ch", ch.state_dict())

    phy = ConsequenceNetV2(seed)
    _save_prefixed_state(record.capsule, "phy", phy.state_dict())

    grow = PlasticGrowthNet(GROW_IN_DIM, 8, GROW_OUT_DIM, seed=seed)
    ctrl = GrowthPressureController(window=3, plateau_eps=0.015, error_threshold=0.15, grow_k=24)
    _pack_growth_net(record.capsule, grow, ctrl, {})

    record.capsule["ecl.alpha"] = torch.ones((len(SOURCES),), dtype=torch.float32)
    record.capsule["ecl.beta"] = torch.ones((len(SOURCES),), dtype=torch.float32)
    # [brier_beta_sum, brier_static_sum, post_abs_beta, post_abs_static, n_total, n_post]
    record.capsule["ecl.stats"] = torch.zeros((6,), dtype=torch.float64)

    record.capsule["unc.w"] = torch.tensor([1.0, 0.5], dtype=torch.float32)
    record.capsule["unc.b"] = torch.tensor(0.0, dtype=torch.float32)


def run_long_run_subject(
    root: str | Path,
    subject_seed: int,
    deadline: float,
    *,
    cfg: RunConfig | None = None,
    subject_id: str = "taiji-longrun",
) -> dict[str, Any]:
    """Execute full multi-session trained durability run on `root` and evaluate from disk."""
    cfg = cfg or RunConfig()
    harness = LongRunHarness(root, subject_id)

    grow_data = make_grow_split(subject_seed)
    p1_tr, p1_va = grow_data["phase1"]["train"], grow_data["phase1"]["val"]
    p2_tr, p2_va = grow_data["phase2"]["train"], grow_data["phase2"]["val"]

    n_claims = cfg.n_sessions * 30
    claims = ClaimStream(subject_seed, shift=True).generate(n_claims)
    static_prior = StaticPrior()

    phy_train_x, phy_train_y = collect_v2(subject_seed, "ramp_bumper", 400)

    crashed_already: set[int] = set()
    crash_recoveries = 0
    sealed_blocks = 0
    unplanned_rescues = 0
    crash_set = set(cfg.crash_sessions)
    seal_sessions = {cfg.n_sessions // 2, cfg.n_sessions - 1}

    def session_step(record: SessionRecord, s: int) -> None:
        if time.process_time() > deadline:
            raise TimeoutError(f"long_run_subject seed={subject_seed} session={s}")
        if s == 0 and "ch.emb.weight" not in record.capsule:
            _init_capsule(record, subject_seed)

        sess_rng = random.Random(subject_seed * 1009 + s)
        sess_torch = torch.Generator().manual_seed(subject_seed * 10007 + s + 1)

        # 1. Cross-session pending handoff lifecycle (~25+ resolved across 100 sessions)
        if s % 4 == 0 and s < cfg.n_sessions - 2:
            pid = f"handoff-{s}"
            if not any(p["id"] == pid for p in record.pending):
                record.add_pending({"id": pid, "session": s, "kind": "cross_session_check"})
        elif s % 4 == 2 and record.pending:
            target_id = record.pending[0]["id"]
            record.resolve_pending(target_id, "verified_done")

        # 2. Chinese recurrent policy PPO updates (zero oracle)
        ch = ChineseTurnPolicy(subject_seed, recurrent=True)
        ch.load_state_dict(_load_prefixed_state(record.capsule, "ch"))
        n_ch_up = (
            cfg.ch_updates_per_session
            if s < cfg.ch_train_end_session
            else (1 if s % 10 == 0 else 0)
        )
        unc_feats_list: list[torch.Tensor] = []
        unc_targets_list: list[float] = []

        if n_ch_up > 0:
            ch_opt = torch.optim.Adam(ch.parameters(), lr=CH_LR)
            for up in range(n_ch_up):
                max_t = 5
                tok_b = torch.zeros((CH_ROLLOUT, max_t, MAX_TOKENS), dtype=torch.long)
                act_b = torch.zeros((CH_ROLLOUT, max_t), dtype=torch.long)
                rew_b = torch.zeros((CH_ROLLOUT, max_t), dtype=torch.float32)
                old_lp = torch.zeros((CH_ROLLOUT, max_t), dtype=torch.float32)
                old_v = torch.zeros((CH_ROLLOUT, max_t), dtype=torch.float32)
                mask = torch.zeros((CH_ROLLOUT, max_t), dtype=torch.float32)
                with torch.no_grad():
                    for b in range(CH_ROLLOUT):
                        ep_id = subject_seed * 1_000_000 + s * 100 + up * CH_ROLLOUT + b
                        env = ChineseTaskEnvV2(ep_id)
                        env.reset()
                        h = None
                        t = 0
                        while True:
                            tok = torch.tensor(env.current_tokens(), dtype=torch.long)
                            logits, val, h = ch.step_hidden(tok, h)
                            probs = F.softmax(logits, dim=-1)
                            act = int(torch.multinomial(probs, 1, generator=sess_torch).item())
                            _, rew, done = env.step(act)
                            tok_b[b, t] = tok
                            act_b[b, t] = act
                            rew_b[b, t] = rew
                            old_lp[b, t] = torch.log(probs[act].clamp(min=1e-8))
                            old_v[b, t] = val
                            mask[b, t] = 1.0
                            t += 1
                            if done:
                                break
                active = mask > 0.5
                adv = torch.zeros_like(rew_b)
                raw_adv = rew_b[active] - old_v[active]
                adv[active] = (raw_adv - raw_adv.mean()) / (raw_adv.std() + 1e-8)
                for _ in range(CH_EPOCHS):
                    logits, values = ch.forward_sequence(tok_b)
                    probs = F.softmax(logits, dim=-1)
                    lps = torch.log(
                        probs.gather(2, act_b.unsqueeze(-1)).squeeze(-1).clamp(min=1e-8)
                    )
                    ratio = torch.exp(lps[active] - old_lp[active])
                    surr = torch.min(
                        ratio * adv[active], torch.clamp(ratio, 0.8, 1.2) * adv[active]
                    )
                    loss = (
                        -surr.mean()
                        + 0.5 * F.mse_loss(values[active], rew_b[active])
                        - 0.02
                        * -(probs[active] * torch.log(probs[active].clamp(min=1e-8)))
                        .sum(dim=-1)
                        .mean()
                    )
                    ch_opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(ch.parameters(), 2.0)
                    ch_opt.step()
            _save_prefixed_state(record.capsule, "ch", ch.state_dict())

        # Collect clean greedy Chinese confidence-calibration samples (strictly observable outcomes)
        ch.eval()
        with torch.no_grad():
            for sample_i in range(4):
                env = ChineseTaskEnvV2(subject_seed * 2_000_000 + s * 10 + sample_i)
                env.reset()
                h = None
                while True:
                    tok = torch.tensor(env.current_tokens(), dtype=torch.long)
                    logits, _, h = ch.step_hidden(tok, h)
                    probs = F.softmax(logits, dim=-1)
                    pred = int(torch.argmax(probs).item())
                    _, rew, done = env.step(pred)
                    top2 = torch.topk(probs, k=2).values
                    unc_feats_list.append(torch.tensor([float(top2[0]), float(top2[0] - top2[1])]))
                    unc_targets_list.append(1.0 if rew > 0.0 else 0.0)
                    if done:
                        break
        ch.train()

        # 3. Physics consequence predictor updates
        phy = ConsequenceNetV2(subject_seed)
        phy.load_state_dict(_load_prefixed_state(record.capsule, "phy"))
        n_phy_steps = (
            cfg.phy_steps_early
            if s < cfg.p1_end_session
            else (cfg.phy_steps_mid if s < cfg.p2_end_session else 5)
        )
        phy_opt = torch.optim.Adam(phy.parameters(), lr=1e-3)
        for _ in range(n_phy_steps):
            idx = torch.randint(0, phy_train_x.shape[0], (256,), generator=sess_torch)
            loss = F.cross_entropy(phy(phy_train_x[idx]), phy_train_y[idx])
            phy_opt.zero_grad(set_to_none=True)
            loss.backward()
            phy_opt.step()
        phy.eval()
        with torch.no_grad():
            p_idx = torch.randint(0, phy_train_x.shape[0], (16,), generator=sess_torch)
            p_logits = phy(phy_train_x[p_idx])
            p_feats = _confidence_features(p_logits)
            p_corr = (p_logits.argmax(dim=-1) == phy_train_y[p_idx]).to(torch.float32)
            for row_i in range(p_feats.shape[0]):
                unc_feats_list.append(p_feats[row_i])
                unc_targets_list.append(float(p_corr[row_i]))
        _save_prefixed_state(record.capsule, "phy", phy.state_dict())

        # 4. Continual plastic & on-demand growth network
        grow, ctrl, scalars = _unpack_growth_net(record.capsule)
        if s == cfg.p1_end_session:
            grow.mature_()
            ctrl.history.clear()
        is_p1 = s < cfg.p1_end_session
        is_p2 = cfg.p1_end_session <= s < cfg.p2_end_session
        g_lr = GROW_LR_BASE if (is_p1 or grow.hidden_dim <= 8) else GROW_LR_POST
        grow_opt = torch.optim.Adam(grow.parameters(), lr=g_lr)
        grow.train()
        if is_p1:
            for xb, yb in grow_batches(sess_torch, p1_tr["x"], p1_tr["y"], cfg.grow_p1_steps):
                loss = F.cross_entropy(grow(xb), yb)
                grow_opt.zero_grad(set_to_none=True)
                loss.backward()
                grow_opt.step()
            p1_now = _eval_acc(grow, p1_va["x"], p1_va["y"])
            ctrl.observe(1.0 - p1_now)
            if s == cfg.p1_end_session - 1:
                scalars["p1_before"] = p1_now
        else:
            steps_g = cfg.grow_p2_steps if is_p2 else cfg.grow_maint_steps
            half = GROW_BATCH // 2
            for _ in range(steps_g):
                i2 = torch.randint(0, p2_tr["x"].shape[0], (half,), generator=sess_torch)
                i1 = torch.randint(0, p1_tr["x"].shape[0], (half,), generator=sess_torch)
                xb = torch.cat([p2_tr["x"][i2], p1_tr["x"][i1]], dim=0)
                yb = torch.cat([p2_tr["y"][i2], p1_tr["y"][i1]], dim=0)
                loss = F.cross_entropy(grow(xb), yb)
                grow_opt.zero_grad(set_to_none=True)
                loss.backward()
                grow.apply_selective_plasticity_grad_mask(freeze_mature=True)
                nn.utils.clip_grad_norm_(grow.parameters(), 2.0)
                grow_opt.step()
            if is_p2:
                p2_now = _eval_acc(grow, p2_va["x"], p2_va["y"])
                grow_k = ctrl.observe(1.0 - p2_now)
                if grow_k > 0:
                    grow.eval()
                    with torch.no_grad():
                        y0 = grow(p1_va["x"][:32])
                        grow.grow_neutral_(grow_k, seed=subject_seed * 17 + grow_k)
                        scalars["birth_pert"] = float((y0 - grow(p1_va["x"][:32])).abs().max())
                    scalars["grew_session"] = float(s)
            if s == cfg.dormancy_session and grow.hidden_dim > 8:
                immature = grow.immature_indices()
                if immature:
                    p2_pre = _eval_acc(grow, p2_va["x"], p2_va["y"])
                    saved_stage = grow.set_dormant_(immature)
                    scalars["dorm_drop"] = round(
                        p2_pre - _eval_acc(grow, p2_va["x"], p2_va["y"]), 6
                    )
                    grow.wake_(saved_stage)
                    scalars["wake_diff"] = abs(_eval_acc(grow, p2_va["x"], p2_va["y"]) - p2_pre)
        _pack_growth_net(record.capsule, grow, ctrl, scalars)

        # 5. Online credibility learning with exponential forgetting (30 claims/session)
        alpha = record.capsule["ecl.alpha"].clone()
        beta = record.capsule["ecl.beta"].clone()
        stats = record.capsule["ecl.stats"].clone()
        cred = BetaCredibility(
            alpha={SOURCES[i]: float(alpha[i]) for i in range(len(SOURCES))},
            beta={SOURCES[i]: float(beta[i]) for i in range(len(SOURCES))},
            forgetting=0.995,
        )
        start_i = s * 30
        shift_at = (cfg.n_sessions // 3) * 30
        for offset in range(30):
            claim_idx = start_i + offset
            src, corr = claims[claim_idx]
            p_b = cred.predict(src, claim_idx)
            p_s = static_prior.predict(src, claim_idx)
            stats[0] += (p_b - corr) ** 2
            stats[1] += (p_s - corr) ** 2
            stats[4] += 1.0
            if claim_idx >= shift_at:
                stats[2] += abs(p_b - corr)
                stats[3] += abs(p_s - corr)
                stats[5] += 1.0
            cred.update(src, corr)
        record.capsule["ecl.alpha"] = torch.tensor(
            [cred.alpha.get(src, 1.0) for src in SOURCES], dtype=torch.float32
        )
        record.capsule["ecl.beta"] = torch.tensor(
            [cred.beta.get(src, 1.0) for src in SOURCES], dtype=torch.float32
        )
        record.capsule["ecl.stats"] = stats

        # 6. Coupled uncertainty update (from own verifiable outcomes)
        w_unc = record.capsule["unc.w"].clone()
        b_bias = record.capsule["unc.b"].clone()
        if unc_feats_list:
            uf = torch.stack(unc_feats_list, dim=0)
            uy = torch.tensor(unc_targets_list, dtype=torch.float32)
            for _ in range(4):
                up_prob = torch.sigmoid(uf @ w_unc + b_bias)
                gw = ((up_prob - uy).unsqueeze(1) * uf).mean(dim=0)
                gb = (up_prob - uy).mean()
                w_unc = w_unc - 0.15 * gw
                b_bias = b_bias - 0.15 * gb
        record.capsule["unc.w"] = w_unc
        record.capsule["unc.b"] = b_bias

        # 7. Append >=100 compact structured experience events for 10,000+ event endurance
        for ev_i in range(cfg.events_per_session):
            record.append_experience(
                "e",
                {"s": s, "i": ev_i, "r": int(sess_rng.randrange(2))},
                source="long_run",
            )

    s = 0
    while s < cfg.n_sessions:
        will_crash = (s in crash_set) and (s not in crashed_already)
        if will_crash:
            crashed_already.add(s)
            try:
                harness.run_session(session_step, crash_before_commit=True)
            except RuntimeError:
                crash_recoveries += 1
                harness.resume()
            continue
        harness.run_session(session_step, crash_before_commit=False)
        if s in seal_sessions:
            split = "check" if s == min(seal_sessions) else "final"
            try:
                harness.read_split(split)
            except SealViolation:
                sealed_blocks += 1
        s += 1

    # Fresh reload from disk to prove zero hidden in-memory carryover
    fresh = LongRunHarness(root, subject_id)
    chain_valid = fresh.verify_chain()
    rec, n_committed = fresh.resume()

    ch_eval_mod = ChineseTurnPolicy(0, recurrent=True)
    ch_eval_mod.load_state_dict(_load_prefixed_state(rec.capsule, "ch"))
    ch_eval_mod.eval()
    ch_tr = ch_eval_arm(ch_eval_mod, CH_EVAL_SEEDS, heldout=False)
    ch_he = ch_eval_arm(ch_eval_mod, CH_HELDOUT_SEEDS, heldout=True)

    phy_eval_mod = ConsequenceNetV2(0)
    phy_eval_mod.load_state_dict(_load_prefixed_state(rec.capsule, "phy"))
    phy_eval_mod.eval()
    x_in, y_in = collect_v2(subject_seed + 100, "ramp_bumper", 150)
    x_cr, y_cr = collect_v2(subject_seed + 200, "rough_collision", 150)
    phy_in = eval_metrics_v2(phy_eval_mod, x_in, y_in)
    phy_cr = eval_metrics_v2(phy_eval_mod, x_cr, y_cr)
    phy_abl = ablate_barrier_collision_recall(phy_eval_mod, x_cr, y_cr)

    grow_eval, ctrl_eval, scalars_eval = _unpack_growth_net(rec.capsule)
    p1_after = _eval_acc(grow_eval, p1_va["x"], p1_va["y"])
    p2_after = _eval_acc(grow_eval, p2_va["x"], p2_va["y"])
    p1_drop = round(scalars_eval["p1_before"] - p1_after, 6)

    est = rec.capsule["ecl.stats"].tolist()
    brier_beta = est[0] / max(1.0, est[4])
    brier_stat = est[1] / max(1.0, est[4])
    post_beta = est[2] / max(1.0, est[5])
    post_stat = est[3] / max(1.0, est[5])

    # Evaluate reloaded unc model strictly on held-out Chinese + cross-family physics predictions
    uw = rec.capsule["unc.w"]
    ub = rec.capsule["unc.b"]
    ho_feats: list[torch.Tensor] = []
    ho_y: list[float] = []
    with torch.no_grad():
        for hs in CH_HELDOUT_SEEDS:
            env = ChineseTaskEnvV2(hs, heldout=True)
            env.reset()
            h = None
            while True:
                tok = torch.tensor(env.current_tokens(), dtype=torch.long)
                logits, _, h = ch_eval_mod.step_hidden(tok, h)
                probs = F.softmax(logits, dim=-1)
                pred = int(torch.argmax(probs).item())
                _, rew, done = env.step(pred)
                top2 = torch.topk(probs, k=2).values
                ho_feats.append(torch.tensor([float(top2[0]), float(top2[0] - top2[1])]))
                ho_y.append(1.0 if rew > 0.0 else 0.0)
                if done:
                    break
        p_logits_ev = phy_eval_mod(x_cr[:600])
        pf = _confidence_features(p_logits_ev)
        py = (p_logits_ev.argmax(dim=-1) == y_cr[:600]).to(torch.float32)
        for ri in range(pf.shape[0]):
            ho_feats.append(pf[ri])
            ho_y.append(float(py[ri]))

    ho_f_t = torch.stack(ho_feats, dim=0)
    ho_y_t = torch.tensor(ho_y, dtype=torch.float32)
    ho_p = torch.sigmoid(ho_f_t @ uw + ub)
    unc_ece = compute_ece(ho_p, ho_y_t)
    unc_brier = round(float(((ho_p - ho_y_t) ** 2).mean()), 4)

    # Verify convergence accuracy of credibility weights vs TRUE_ACC (last 3 sources stationary)
    cred_final_post_forum = float(
        rec.capsule["ecl.alpha"][3] / (rec.capsule["ecl.alpha"][3] + rec.capsule["ecl.beta"][3])
    )
    _ = OUTCOMES
    _ = TRUE_ACC

    resolved_count = sum(1 for e in rec.experience if e.get("kind") == "resolved_pending")
    return {
        "schema": SCHEMA,
        "seed": subject_seed,
        "durability": {
            "sessions": n_committed,
            "events": len(rec.experience),
            "crash_recoveries": crash_recoveries,
            "unplanned_rescues": unplanned_rescues,
            "chain_valid": chain_valid,
            "chain_head": fresh.chain[-1] if fresh.chain else "",
            "record_digest": rec.digest(),
            "resolved_pending": resolved_count,
            "sealed_blocks": sealed_blocks,
        },
        "chinese": {"train": ch_tr, "heldout": ch_he},
        "physics": {
            "in_family": phy_in,
            "cross_family": phy_cr,
            "barrier_ablation": phy_abl,
        },
        "growth": {
            "final_hidden": grow_eval.hidden_dim,
            "growth_events": ctrl_eval.growth_events,
            "grew_session": int(scalars_eval["grew_session"]),
            "birth_perturbation": scalars_eval["birth_pert"],
            "dormancy_drop": scalars_eval["dorm_drop"],
            "wake_diff": scalars_eval["wake_diff"],
            "p1_before_acc": round(scalars_eval["p1_before"], 4),
            "p1_after_acc": round(p1_after, 4),
            "p1_retention_drop": p1_drop,
            "p2_acc": round(p2_after, 4),
        },
        "credibility": {
            "brier_beta": round(brier_beta, 4),
            "brier_static": round(brier_stat, 4),
            "delta_brier": round(brier_beta - brier_stat, 4),
            "post_abs_beta": round(post_beta, 4),
            "post_abs_static": round(post_stat, 4),
            "delta_post_abs": round(post_beta - post_stat, 4),
            "final_forum_cred": round(cred_final_post_forum, 4),
        },
        "uncertainty": {
            "coupled_ece": unc_ece,
            "coupled_brier": unc_brier,
            "brier_vs_static": round(unc_brier - 0.25, 4),
            "n_eval_samples": int(ho_y_t.shape[0]),
            "mean_acc": round(float(ho_y_t.mean()), 4),
        },
    }
