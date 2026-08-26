"""Native local credit assignment for the Taiji peripheral learners.

The Taiji kernel assigns credit with explicit detached delta rules (see
:mod:`taiji.sparse`).  The peripheral learners historically reached for autograd
plus ``torch.optim``, which split the system into two incompatible learning
stacks and broke the native purity contract enforced by
``scripts/training/verify_taiji_native_v7.py``.

This module supplies the exact delta rules those learners need.  Every rule is
the analytically derived derivative of its objective, so a migrated learner
performs the *same* arithmetic a plain-SGD autograd step would perform, without
ever building a graph:

``mean_squared_error_delta``
    d/dprediction of ``torch.mean((prediction - target) ** 2)``.  The ``2 / n``
    factor is retained deliberately: dropping it silently rescales the caller's
    learning rate.
``squared_error_delta``
    d/dprediction of the unreduced ``(prediction - target) ** 2``.
``logistic_error_delta``
    d/dlogit of ``binary_cross_entropy_with_logits`` under mean reduction.
``softmax_error_delta``
    d/dlogit of ``cross_entropy`` under mean reduction.

``apply_linear_delta`` then performs the plain-SGD parameter update of an
``nn.Linear`` layer from an output error, and ``freeze_parameters`` removes the
autograd flag so the native ``no_autograd_parameters`` contract becomes a real
check instead of a formality.

Multi-layer learners need two more pieces.  ``linear_gradients`` exposes the
same weight and bias gradients ``apply_linear_delta`` would consume, so a
caller can route them through a stateful rule instead of applying them
immediately; ``backproject_linear`` and ``tanh_delta`` carry an output error one
hop back through a ``Linear``/``Tanh`` pair.

Learners whose forward pass normalises or compares directions get the matching
Jacobians: ``normalize_delta`` applies the ``(I - y yᵀ) / ‖x‖`` Jacobian of
``nn.functional.normalize``, and ``cosine_similarity_delta`` differentiates
``cosine_similarity`` with respect to its left argument.  Recurrent learners get
``gru_forward_trace``, which replays an ``nn.GRU`` step by step while recording
the gate activations the backward sweep needs, and ``gru_gradients``, which
turns a hidden-state error into the four ``weight_ih``/``weight_hh``/``bias_ih``/
``bias_hh`` gradients.  The reset gate multiplies the hidden contribution to the
candidate, so the n-segment of the ``_hh`` gradients carries an extra ``reset``
factor the ``_ih`` side does not -- a detail worth stating because getting it
wrong is silent.

Two application helpers close the loop.  ``clip_gradient_norm`` mirrors
``torch.nn.utils.clip_grad_norm_`` (joint L2 norm, ``max_norm / (total + 1e-6)``
scale, applied only when it would shrink), and ``apply_sgd_step`` performs the
plain descent update for callers that already hold gradients.  Finally
``LocalAdam`` reproduces the ``torch.optim.Adam`` update equation -- including
bias correction -- on detached tensors, so migrating a learner off autograd does
not also change its optimiser and invalidate every tuned learning rate above it.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def freeze_parameters(module: nn.Module) -> None:
    """Detach a learner from autograd bookkeeping permanently."""

    for parameter in module.parameters():
        parameter.requires_grad_(False)


def mean_squared_error_delta(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return d/dprediction of ``mean((prediction - target) ** 2)``."""

    if prediction.shape != target.shape:
        raise ValueError("mean squared error delta needs matching prediction and target shapes")
    if prediction.numel() == 0:
        raise ValueError("mean squared error delta needs a non-empty prediction")
    return 2.0 * (prediction - target) / prediction.numel()


def squared_error_delta(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return d/dprediction of the unreduced ``(prediction - target) ** 2``."""

    if prediction.shape != target.shape:
        raise ValueError("squared error delta needs matching prediction and target shapes")
    return 2.0 * (prediction - target)


def logistic_error_delta(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return d/dlogit of mean binary cross entropy with logits."""

    if logits.shape != targets.shape:
        raise ValueError("logistic error delta needs matching logit and target shapes")
    if logits.numel() == 0:
        raise ValueError("logistic error delta needs non-empty logits")
    return (torch.sigmoid(logits) - targets) / logits.numel()


def softmax_error_delta(logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Return d/dlogit of mean cross entropy against integer class ``indices``."""

    if logits.ndim != 2:
        raise ValueError("softmax error delta needs a two-dimensional logit matrix")
    if indices.ndim != 1 or indices.shape[0] != logits.shape[0]:
        raise ValueError("softmax error delta needs one class index per logit row")
    rows = logits.shape[0]
    if rows == 0:
        raise ValueError("softmax error delta needs non-empty logits")
    delta = torch.softmax(logits, dim=-1)
    delta[torch.arange(rows, device=logits.device), indices] -= 1.0
    return delta / rows


@torch.no_grad()
def apply_linear_delta(
    layer: nn.Linear,
    inputs: torch.Tensor,
    error: torch.Tensor,
    learning_rate: float,
) -> None:
    """Apply the plain-SGD update of ``layer`` implied by an output ``error``.

    ``inputs`` is the ``(rows, in_features)`` activation matrix that produced
    the error and ``error`` is the ``(rows, out_features)`` derivative of the
    objective with respect to the layer output.  The update is
    ``weight -= learning_rate * errorᵀ @ inputs`` and
    ``bias -= learning_rate * error.sum(0)``, which is exactly what autograd
    plus ``torch.optim.SGD`` would compute for the same objective.
    """

    if inputs.ndim != 2 or error.ndim != 2:
        raise ValueError("linear delta needs two-dimensional inputs and error matrices")
    if inputs.shape[0] != error.shape[0]:
        raise ValueError("linear delta needs one error row per input row")
    if inputs.shape[1] != layer.in_features or error.shape[1] != layer.out_features:
        raise ValueError("linear delta shapes do not match the layer contract")
    rate = float(learning_rate)
    if rate <= 0.0:
        raise ValueError("linear delta learning_rate must be positive")
    layer.weight.add_(error.transpose(0, 1) @ inputs, alpha=-rate)
    if layer.bias is not None:
        layer.bias.add_(error.sum(0), alpha=-rate)


def _validate_linear_error(layer: nn.Linear, inputs: torch.Tensor, error: torch.Tensor) -> None:
    if inputs.ndim != 2 or error.ndim != 2:
        raise ValueError("linear delta needs two-dimensional inputs and error matrices")
    if inputs.shape[0] != error.shape[0]:
        raise ValueError("linear delta needs one error row per input row")
    if inputs.shape[1] != layer.in_features or error.shape[1] != layer.out_features:
        raise ValueError("linear delta shapes do not match the layer contract")


@torch.no_grad()
def linear_gradients(
    layer: nn.Linear, inputs: torch.Tensor, error: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Return the ``(weight, bias)`` gradients implied by an output ``error``.

    The values are identical to the ``.grad`` tensors autograd would populate,
    but they are produced without a graph so a stateful rule such as
    :class:`LocalAdam` can consume them.
    """

    _validate_linear_error(layer, inputs, error)
    weight_gradient = error.transpose(0, 1) @ inputs
    if layer.bias is None:
        return (weight_gradient,)
    return (weight_gradient, error.sum(0))


@torch.no_grad()
def backproject_linear(layer: nn.Linear, error: torch.Tensor) -> torch.Tensor:
    """Carry an output ``error`` one hop back to the layer input."""

    if error.ndim != 2 or error.shape[1] != layer.out_features:
        raise ValueError("linear backprojection needs an error row per layer output")
    return error @ layer.weight


@torch.no_grad()
def tanh_delta(error: torch.Tensor, activations: torch.Tensor) -> torch.Tensor:
    """Apply the local ``1 - tanh(x) ** 2`` derivative to a backprojected error.

    ``activations`` are the ``tanh`` *outputs*, which is what a forward pass
    already has in hand; the derivative in terms of the output is
    ``1 - activation ** 2``.
    """

    if error.shape != activations.shape:
        raise ValueError("tanh delta needs matching error and activation shapes")
    return error * (1.0 - activations * activations)


@torch.no_grad()
def normalize_delta(
    error: torch.Tensor,
    normalized: torch.Tensor,
    norms: torch.Tensor,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """Carry an error back through ``torch.nn.functional.normalize``.

    For ``y = x / ||x||`` the Jacobian is ``(I - y yᵀ) / ||x||``, so the
    backward pass removes the component of the error that lies along the output
    direction and then rescales by the original norm.  ``normalized`` is the
    forward output and ``norms`` the norm used to produce it, kept with the
    reduced dimension so it broadcasts.
    """

    if error.shape != normalized.shape:
        raise ValueError("normalize delta needs matching error and output shapes")
    projection = (normalized * error).sum(dim=dim, keepdim=True)
    return (error - normalized * projection) / norms


@torch.no_grad()
def cosine_similarity_delta(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    dim: int = -1,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return d/dleft and d/dright of ``cosine_similarity(left, right)``.

    Both derivatives follow the same shape: the *other* unit vector minus the
    similarity times this unit vector, rescaled by this side's norm.
    """

    if left.shape != right.shape:
        raise ValueError("cosine similarity delta needs matching operand shapes")
    left_norm = torch.linalg.vector_norm(left, dim=dim, keepdim=True).clamp_min(eps)
    right_norm = torch.linalg.vector_norm(right, dim=dim, keepdim=True).clamp_min(eps)
    left_unit = left / left_norm
    right_unit = right / right_norm
    similarity = (left_unit * right_unit).sum(dim=dim, keepdim=True)
    left_delta = (right_unit - similarity * left_unit) / left_norm
    right_delta = (left_unit - similarity * right_unit) / right_norm
    return left_delta, right_delta


@torch.no_grad()
def gru_forward_trace(
    cell: nn.GRU, inputs: torch.Tensor
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    """Roll a single-layer :class:`torch.nn.GRU` forward, retaining gate state.

    ``inputs`` is a ``(steps, input_size)`` matrix.  The returned hidden matrix
    is ``(steps, hidden_size)`` and matches what ``cell(inputs.unsqueeze(0))``
    produces.  Each trace entry keeps the quantities the backward sweep needs
    and a plain forward pass discards: the reset gate, the update gate, the
    candidate, the candidate's pre-activation recurrent term and the incoming
    hidden state.
    """

    if cell.num_layers != 1 or cell.bidirectional:
        raise ValueError("gru trace supports a single-direction single-layer cell")
    if inputs.ndim != 2 or inputs.shape[1] != cell.input_size:
        raise ValueError("gru trace needs a (steps, input_size) input matrix")
    hidden_size = int(cell.hidden_size)
    input_weight = cell.weight_ih_l0
    hidden_weight = cell.weight_hh_l0
    input_bias = cell.bias_ih_l0 if cell.bias else None
    hidden_bias = cell.bias_hh_l0 if cell.bias else None
    hidden = torch.zeros(hidden_size, device=inputs.device, dtype=inputs.dtype)
    states: list[torch.Tensor] = []
    trace: list[dict[str, torch.Tensor]] = []
    for step in range(inputs.shape[0]):
        row = inputs[step]
        gates_input = input_weight @ row
        gates_hidden = hidden_weight @ hidden
        if input_bias is not None:
            gates_input = gates_input + input_bias
        if hidden_bias is not None:
            gates_hidden = gates_hidden + hidden_bias
        reset = torch.sigmoid(gates_input[:hidden_size] + gates_hidden[:hidden_size])
        update = torch.sigmoid(
            gates_input[hidden_size : 2 * hidden_size] + gates_hidden[hidden_size : 2 * hidden_size]
        )
        candidate_hidden = gates_hidden[2 * hidden_size :]
        candidate = torch.tanh(gates_input[2 * hidden_size :] + reset * candidate_hidden)
        trace.append(
            {
                "input": row,
                "previous": hidden,
                "reset": reset,
                "update": update,
                "candidate": candidate,
                "candidate_hidden": candidate_hidden,
            }
        )
        hidden = (1.0 - update) * candidate + update * hidden
        states.append(hidden)
    return torch.stack(states), trace


@torch.no_grad()
def gru_gradients(
    cell: nn.GRU, trace: Sequence[dict[str, torch.Tensor]], hidden_error: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Return the GRU parameter gradients implied by a per-step hidden error.

    ``hidden_error`` is ``(steps, hidden_size)``: the derivative of the
    objective with respect to each emitted hidden state, *excluding* the
    recurrent path, which this sweep adds while walking backwards in time.

    The gate layout follows PyTorch's ``r, z, n`` packing.  One asymmetry is
    easy to miss and is handled explicitly: the candidate's recurrent bias
    ``b_hn`` enters the pre-activation multiplied by the reset gate, so its
    gradient carries the extra ``reset`` factor that ``b_in`` does not.  The
    returned tuple is ordered to match ``(weight_ih_l0, weight_hh_l0,
    bias_ih_l0, bias_hh_l0)`` and omits the biases when the cell has none.
    """

    steps = len(trace)
    if hidden_error.ndim != 2 or hidden_error.shape[0] != steps:
        raise ValueError("gru gradients need one hidden error row per traced step")
    hidden_size = int(cell.hidden_size)
    if hidden_error.shape[1] != hidden_size:
        raise ValueError("gru gradients need a hidden error width of hidden_size")
    input_gradient = torch.zeros_like(cell.weight_ih_l0)
    hidden_gradient = torch.zeros_like(cell.weight_hh_l0)
    input_bias_gradient = torch.zeros(3 * hidden_size, device=hidden_error.device)
    hidden_bias_gradient = torch.zeros(3 * hidden_size, device=hidden_error.device)
    hidden_weight = cell.weight_hh_l0
    carry = torch.zeros(hidden_size, device=hidden_error.device, dtype=hidden_error.dtype)
    for step in range(steps - 1, -1, -1):
        entry = trace[step]
        reset = entry["reset"]
        update = entry["update"]
        candidate = entry["candidate"]
        candidate_hidden = entry["candidate_hidden"]
        previous = entry["previous"]
        row = entry["input"]
        error = hidden_error[step] + carry
        candidate_error = error * (1.0 - update)
        update_error = error * (previous - candidate)
        candidate_pre = candidate_error * (1.0 - candidate**2)
        reset_error = candidate_pre * candidate_hidden
        update_pre = update_error * update * (1.0 - update)
        reset_pre = reset_error * reset * (1.0 - reset)
        packed_input = torch.cat((reset_pre, update_pre, candidate_pre))
        packed_hidden = torch.cat((reset_pre, update_pre, candidate_pre * reset))
        input_gradient += packed_input.unsqueeze(1) @ row.unsqueeze(0)
        hidden_gradient += packed_hidden.unsqueeze(1) @ previous.unsqueeze(0)
        input_bias_gradient += packed_input
        hidden_bias_gradient += packed_hidden
        carry = error * update + packed_hidden @ hidden_weight
    if cell.bias:
        return (input_gradient, hidden_gradient, input_bias_gradient, hidden_bias_gradient)
    return (input_gradient, hidden_gradient)


@torch.no_grad()
def clip_gradient_norm(
    gradients: Sequence[torch.Tensor], max_norm: float
) -> tuple[torch.Tensor, ...]:
    """Scale ``gradients`` so their joint L2 norm never exceeds ``max_norm``.

    This mirrors ``torch.nn.utils.clip_grad_norm_``: the norm is taken over the
    concatenation of every gradient, and the same scalar factor is applied to
    all of them, so relative directions are preserved.
    """

    limit = float(max_norm)
    if limit <= 0.0:
        raise ValueError("gradient clip max_norm must be positive")
    if not gradients:
        raise ValueError("gradient clipping needs at least one gradient")
    total = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(gradient) for gradient in gradients])
    )
    scale = limit / (float(total) + 1e-6)
    if scale >= 1.0:
        return tuple(gradients)
    return tuple(gradient * scale for gradient in gradients)


@torch.no_grad()
def apply_sgd_step(
    parameters: Sequence[torch.Tensor],
    gradients: Sequence[torch.Tensor],
    learning_rate: float,
) -> None:
    """Apply one plain-SGD step to arbitrary detached parameter tensors."""

    rate = float(learning_rate)
    if rate <= 0.0:
        raise ValueError("sgd step learning_rate must be positive")
    if len(parameters) != len(gradients):
        raise ValueError("sgd step needs one gradient per parameter")
    for parameter, gradient in zip(parameters, gradients, strict=True):
        if parameter.shape != gradient.shape:
            raise ValueError("sgd step gradient shape does not match its parameter")
        parameter.add_(gradient, alpha=-rate)


class LocalAdam:
    """The ``torch.optim.Adam`` update equation on detached tensors.

    Migrating a learner away from autograd should not silently also migrate it
    away from Adam: the learning rates in this repository were tuned against
    Adam's bias-corrected moment estimates, and swapping in plain SGD would
    change every convergence threshold.  This class therefore keeps the exact
    update rule and only removes the graph.
    """

    def __init__(
        self,
        parameters: Sequence[torch.Tensor],
        *,
        learning_rate: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        rate = float(learning_rate)
        if rate <= 0.0:
            raise ValueError("LocalAdam learning_rate must be positive")
        beta1, beta2 = float(betas[0]), float(betas[1])
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("LocalAdam betas must lie in [0, 1)")
        self._parameters = tuple(parameters)
        if not self._parameters:
            raise ValueError("LocalAdam needs at least one parameter")
        self._learning_rate = rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = float(eps)
        self._step = 0
        self._first = [torch.zeros_like(parameter) for parameter in self._parameters]
        self._second = [torch.zeros_like(parameter) for parameter in self._parameters]

    @torch.no_grad()
    def apply(self, gradients: Sequence[torch.Tensor]) -> None:
        """Advance every parameter by one Adam step for the given gradients."""

        if len(gradients) != len(self._parameters):
            raise ValueError("LocalAdam needs one gradient per parameter")
        self._step += 1
        bias_correction1 = 1.0 - self._beta1**self._step
        bias_correction2 = 1.0 - self._beta2**self._step
        for parameter, gradient, first, second in zip(
            self._parameters, gradients, self._first, self._second, strict=True
        ):
            if gradient.shape != parameter.shape:
                raise ValueError("LocalAdam gradient shape does not match its parameter")
            first.mul_(self._beta1).add_(gradient, alpha=1.0 - self._beta1)
            second.mul_(self._beta2).addcmul_(gradient, gradient, value=1.0 - self._beta2)
            denominator = (second / bias_correction2).sqrt_().add_(self._eps)
            parameter.addcdiv_(first, denominator, value=-self._learning_rate / bias_correction1)
