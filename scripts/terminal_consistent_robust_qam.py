"""Pathwise terminal-consistent robust Hamiltonian targets for PointObstacle QAM.

One bounded terminal Q-gradient perturbation is shared across every Flow step
of a sampled trajectory.  The resulting target is an exploratory frozen-path
min-max control update, not a global robust-policy or HJBI guarantee.
"""
from __future__ import annotations

import torch


def decompose_gradient_errors(g_learned: torch.Tensor,
                              g_reference: torch.Tensor):
    """Construct privileged magnitude-only and direction-only gradients.

    Shapes must match [..., d]. Zero vectors stay zero rather than receiving an
    arbitrary direction. This helper is diagnostic; it uses reference
    information and is not a deployable estimator.
    """
    if g_learned.shape != g_reference.shape or g_learned.ndim < 1:
        raise ValueError("learned and reference gradients must have matching [..., d] shapes")
    if not bool(torch.isfinite(g_learned).all()) or not bool(torch.isfinite(g_reference).all()):
        raise FloatingPointError("non-finite gradient in privileged decomposition")
    n_learned = g_learned.norm(dim=-1, keepdim=True)
    n_reference = g_reference.norm(dim=-1, keepdim=True)
    learned_unit = torch.where(n_learned > 1e-12,
                               g_learned / n_learned.clamp_min(1e-12),
                               torch.zeros_like(g_learned))
    reference_unit = torch.where(n_reference > 1e-12,
                                 g_reference / n_reference.clamp_min(1e-12),
                                 torch.zeros_like(g_reference))
    magnitude_only = n_reference * learned_unit
    direction_only = n_learned * reference_unit
    cosine = ((g_learned * g_reference).sum(-1) /
              (n_learned.squeeze(-1) * n_reference.squeeze(-1)).clamp_min(1e-12))
    both_nonzero = (n_learned.squeeze(-1) > 1e-12) & (n_reference.squeeze(-1) > 1e-12)
    cosine = torch.where(both_nonzero, cosine.clamp(-1., 1.), torch.zeros_like(cosine))
    return dict(magnitude_only=magnitude_only, direction_only=direction_only,
                cosine=cosine, learned_norm=n_learned.squeeze(-1),
                reference_norm=n_reference.squeeze(-1))


def solve_unit_ball_quadratic(A: torch.Tensor, B: torch.Tensor,
                              kappa: float | torch.Tensor,
                              iterations: int = 64) -> torch.Tensor:
    """Solve max_{||q||<=1} q'A - kappa/2 q'Bq, for PSD B.

    A is [batch, d], B is [batch, d, d]. A secular-equation bisection handles
    active constraints, including singular B, without requiring a hard-case
    branch because the objective is concave and any null-space linear term is
    regularized by the positive Lagrange multiplier on the boundary.
    """
    if A.ndim != 2 or B.shape != (len(A), A.shape[1], A.shape[1]):
        raise ValueError("expected A=[batch,d] and B=[batch,d,d]")
    if not bool(torch.isfinite(A).all()) or not bool(torch.isfinite(B).all()):
        raise FloatingPointError("non-finite robust quadratic coefficients")

    dtype = torch.float64
    a = A.to(dtype)
    b = B.to(dtype)
    k = torch.as_tensor(kappa, dtype=dtype, device=A.device)
    if k.ndim == 0:
        k = k.expand(len(a))
    if k.shape != (len(a),):
        raise ValueError("kappa must be scalar or one value per batch row")
    if not bool(torch.isfinite(k).all()) or bool((k < 0).any()):
        raise ValueError("kappa must be finite and nonnegative")
    b = .5 * (b + b.transpose(-1, -2))
    eigvals, eigvecs = torch.linalg.eigh(b)
    if bool((eigvals[:, 0] < -1e-9).any()):
        raise ValueError("B must be positive semidefinite")
    eigvals = eigvals.clamp_min(0.)
    coords = (eigvecs.transpose(-1, -2) @ a.unsqueeze(-1)).squeeze(-1)
    denom = k[:, None] * eigvals
    tolerance = 1e-12
    quotient = torch.where(denom > tolerance,
                           coords / denom.clamp_min(tolerance),
                           torch.zeros_like(coords))
    null_residual = torch.where(denom <= tolerance, coords, torch.zeros_like(coords))
    unconstrained_norm = quotient.norm(dim=-1)
    interior = (null_residual.norm(dim=-1) <= 1e-10) & (unconstrained_norm <= 1.)
    q_interior = (eigvecs @ quotient.unsqueeze(-1)).squeeze(-1)

    # On the boundary, q_i(lambda)=a_i/(kappa*b_i+lambda), whose norm is
    # monotone decreasing for lambda>0.  This also covers B's singular cases.
    low = torch.zeros(len(a), dtype=dtype, device=a.device)
    high = torch.ones_like(low)

    def boundary_q(lam):
        qhat = coords / (denom + lam[:, None]).clamp_min(1e-300)
        return (eigvecs @ qhat.unsqueeze(-1)).squeeze(-1)

    for _ in range(128):
        too_large = boundary_q(high).norm(dim=-1) > 1.
        if not bool(too_large.any()):
            break
        high = torch.where(too_large, high * 2., high)
    else:
        raise RuntimeError("failed to bracket robust trust-region multiplier")
    for _ in range(iterations):
        mid = .5 * (low + high)
        too_large = boundary_q(mid).norm(dim=-1) > 1.
        low = torch.where(too_large, mid, low)
        high = torch.where(too_large, high, mid)
    q_boundary = boundary_q(high)
    q = torch.where(interior[:, None], q_interior, q_boundary)
    return q.to(A.dtype)


@torch.no_grad()
def adjoint_propagators(base, c, k, y, paths):
    """Return P_j = d p_j / d p_K for the exact discrete Lean recurrence.

    The policy condition (k,y) is held fixed, matching core.lean_adjoint.
    Shape is [batch, flow_steps+1, d, d], with P_K=I.
    """
    h = (1. - c.flow_epsilon) / c.flow_steps
    t = c.flow_epsilon + h * torch.arange(c.flow_steps, dtype=paths.dtype,
                                          device=paths.device)
    n, steps_plus_one, d = paths.shape
    if steps_plus_one != c.flow_steps + 1 or d != c.dimension:
        raise ValueError("path shape/config mismatch")
    eye = torch.eye(d, dtype=paths.dtype, device=paths.device)
    eye_b = eye.expand(n, d, d)
    propagators = [None] * (c.flow_steps + 1)
    propagators[c.flow_steps] = eye_b
    for j in range(c.flow_steps - 1, -1, -1):
        basis = eye.expand(n, d, d)
        # VJP(e_m)=J^T e_m; stack these vectors as columns to obtain J^T.
        jt = torch.stack([
            base.input_vjp(k, y, paths[:, j], t[j], basis[:, m, :])
            for m in range(d)
        ], dim=-1)
        step = eye_b + h * (2. * jt - eye_b / t[j])
        propagators[j] = step @ propagators[j + 1]
    result = torch.stack(propagators, dim=1).detach()
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("non-finite Lean Adjoint propagator")
    return result


def pathwise_terminal_robust_target(base, c, k, y, paths, p_hat, epsilon):
    """Return robust costates and diagnostics for a shared per-path q*.

    epsilon bounds the terminal Q-gradient error; kappa=beta*epsilon bounds
    the terminal costate error. The sampled path and Base Jacobians are frozen.
    """
    eps = torch.as_tensor(epsilon, dtype=paths.dtype, device=paths.device)
    if eps.ndim == 0:
        eps = eps.expand(len(paths))
    if eps.shape != (len(paths),) or not bool(torch.isfinite(eps).all()) \
            or bool((eps < 0).any()):
        raise ValueError("epsilon must be a finite nonnegative scalar or [batch] tensor")
    h = (1. - c.flow_epsilon) / c.flow_steps
    t = c.flow_epsilon + h * torch.arange(c.flow_steps, dtype=paths.dtype,
                                          device=paths.device)
    nu = torch.sqrt(2. * (1. - t) / t)
    n, steps_plus_one, d = paths.shape
    P = adjoint_propagators(base, c, k, y, paths)
    x = paths[:, :-1]
    kk = k[:, None].expand(n, c.flow_steps).reshape(-1)
    yy = y[:, None, :].expand(n, c.flow_steps, d).reshape(-1, d)
    xx = x.reshape(-1, d)
    tt = t[None, :].expand(n, c.flow_steps).reshape(-1, 1)
    with torch.no_grad():
        vb = base(kk, yy, xx, tt).reshape(n, c.flow_steps, d)
        drift = 2. * vb - x / t[None, :, None]
        p_right = p_hat[:, 1:]
        nominal_optimal_drift = drift - nu[None, :, None].square() * p_right
        P_right = P[:, 1:]
        A = h * (P_right.transpose(-1, -2) @
                 nominal_optimal_drift.unsqueeze(-1)).squeeze(-1).sum(1)
        PTP = P_right.transpose(-1, -2) @ P_right
        B = h * (nu[None, :, None, None].square() * PTP).sum(1)
        kappa = c.beta * eps
        q = solve_unit_ball_quadratic(A, B, kappa)
        robust_p = p_hat + kappa[:, None, None] * (P @ q[:, None, :, None]).squeeze(-1)
        robust_drift = drift - nu[None, :, None].square() * robust_p[:, 1:]
        q_norm = q.norm(dim=-1)
        target_shift = (robust_p[:, 1:] - p_right).norm(dim=-1).mean(1)
        nominal_norm = nominal_optimal_drift.norm(dim=-1).mean(1)
        robust_norm = robust_drift.norm(dim=-1).mean(1)
        drift_ratio = robust_norm / nominal_norm.clamp_min(1e-12)
    if not bool(torch.isfinite(robust_p).all()):
        raise FloatingPointError("non-finite robust adjoint target")
    info = dict(
        q_norm_mean=float(q_norm.mean()),
        q_boundary_fraction=float((q_norm > 1. - 1e-5).float().mean()),
        epsilon_mean=float(eps.mean()),
        epsilon_median=float(eps.median()),
        epsilon_max=float(eps.max()),
        target_shift_mean=float(target_shift.mean()),
        nominal_drift_norm_mean=float(nominal_norm.mean()),
        robust_drift_norm_mean=float(robust_norm.mean()),
        robust_to_nominal_drift_ratio=float(drift_ratio.mean()),
    )
    return robust_p.detach(), q.detach(), info


def terminal_consistent_actor_step(actor, opt, base, critic, task, c, k, y, z,
                                   xi, epsilon, core):
    """One TD-QAM actor update with a per-trajectory terminal-consistent q*.

    The same q* is propagated to every internal Flow step on that path. It may
    differ across sampled paths; this is explicitly a pathwise uncertainty
    approximation, not a shared function delta(s,a_1).
    """
    paths, _ = core.rollout(actor, base, c, k, y, z, xi)
    terminal = core.terminal_label(task, critic, c, k, y, paths[:, -1], "learned")
    p_hat = core.lean_adjoint(base, c, k, y, paths, terminal)
    robust_p, q_star, info = pathwise_terminal_robust_target(
        base, c, k, y, paths, p_hat, epsilon)
    q_grad, loss = core.actor_gradient(actor, base, c, k, y, paths, robust_p)
    core.set_gradient(actor, q_grad)
    clipped = torch.nn.utils.clip_grad_norm_(actor.parameters(), c.actor_clip,
                                             error_if_nonfinite=True)
    opt.step()
    actor.zero_grad(set_to_none=True)
    info.update(matching_loss=loss, matching_gradient_norm=float(q_grad.norm()),
                pre_clip_norm=float(clipped))
    return info
