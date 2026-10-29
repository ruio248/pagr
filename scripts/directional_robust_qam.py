"""First-version terminal-gradient filtering for PointObstacle QAM.

No reference estimator or environment changes belong in this module.  The
ordinary ``core.actor_step`` remains untouched and is called by the runner.
"""
from __future__ import annotations

import torch


def critic_member_action_gradients(critic, k, y, endpoint):
    """Return genuine per-member dQ/da, shaped [batch, members, dimension]."""
    members = len(critic.members)
    if members < 2:
        raise ValueError("directional robustness requires at least two critic members")
    with torch.enable_grad():
        action = endpoint.detach().requires_grad_(True)
        values = critic(k, y, action)
        if values.shape != (len(action), members):
            raise ValueError("critic output/member shape mismatch")
        gradients = [torch.autograd.grad(values[:, m].sum(), action,
                     retain_graph=m+1 < members)[0].detach() for m in range(members)]
    result = torch.stack(gradients, dim=1)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("non-finite critic member action gradient")
    return result


def gradient_covariance(gradients):
    """Sample covariance [B,D,D] and ensemble mean [B,D]."""
    if gradients.ndim != 3 or gradients.shape[1] < 2:
        raise ValueError("gradients must have shape [B,M>=2,D]")
    if not bool(torch.isfinite(gradients).all()):
        raise FloatingPointError("non-finite member gradients")
    # float64 avoids cancellation when members nearly agree.
    g = gradients.to(torch.float64)
    mean = g.mean(dim=1)
    centered = g-mean[:, None, :]
    sigma = centered.transpose(-1, -2) @ centered/(g.shape[1]-1)
    sigma = .5*(sigma+sigma.transpose(-1, -2))
    if not bool(torch.isfinite(sigma).all()):
        raise FloatingPointError("non-finite gradient covariance")
    return mean, sigma


def directional_robust_gradient(gradients, alpha, eps=1e-12, condition_limit=1e12):
    """(I + alpha Sigma/(||mean||²+eps))^-1 mean; never invert explicitly.

    ``eps`` regularizes the dimensionless covariance denominator.  The identity
    in the solve is the matrix regularizer; no covariance jitter is needed and
    exact zero covariance therefore returns the exact ensemble mean.
    """
    if not (0 <= alpha < float("inf")) or eps <= 0:
        raise ValueError("alpha must be finite/nonnegative and eps positive")
    mean, sigma = gradient_covariance(gradients)
    if alpha == 0:
        return gradients.mean(dim=1), sigma
    d = mean.shape[-1]
    eye = torch.eye(d, dtype=mean.dtype, device=mean.device).expand(len(mean), d, d)
    normalized = sigma/(mean.square().sum(dim=-1)[:, None, None]+eps)
    system = eye+alpha*normalized
    if not bool(torch.isfinite(system).all()):
        raise FloatingPointError("non-finite robust linear system")
    condition = torch.linalg.cond(system)
    if not bool(torch.isfinite(condition).all()) or bool((condition > condition_limit).any()):
        raise FloatingPointError("ill-conditioned robust linear system")
    filtered = torch.linalg.solve(system, mean[..., None]).squeeze(-1)
    if not bool(torch.isfinite(filtered).all()):
        raise FloatingPointError("non-finite robust gradient")
    return filtered.to(gradients.dtype), sigma


def eigen_analysis(gradients, alpha, eps=1e-12):
    """Detached diagnostics only; not part of the algorithmic control path."""
    mean, sigma = gradient_covariance(gradients)
    eigenvalues, eigenvectors = torch.linalg.eigh(sigma)
    # Descending principal axes for readable output.
    eigenvalues = eigenvalues.flip(-1)
    eigenvectors = eigenvectors.flip(-1)
    projections = (eigenvectors.transpose(-1, -2) @ mean[..., None]).squeeze(-1)
    normalized_eigenvalues = eigenvalues/(mean.square().sum(-1)[:, None]+eps)
    factors = 1/(1+alpha*normalized_eigenvalues)
    return dict(eigenvalues=eigenvalues, eigenvectors=eigenvectors,
                projections=projections, shrink_factors=factors)


def directional_robust_actor_step(actor, opt, base, critic, c, k, y, z, flow,
                                  alpha, core, diagnostic=False):
    paths, _ = core.rollout(actor, base, c, k, y, z, flow)
    member = critic_member_action_gradients(critic, k, y, paths[:, -1])
    filtered, covariance = directional_robust_gradient(member, alpha)
    terminal = -c.beta*filtered
    adjoints = core.lean_adjoint(base, c, k, y, paths, terminal)
    q, loss = core.actor_gradient(actor, base, c, k, y, paths, adjoints)
    core.set_gradient(actor, q)
    clipped = torch.nn.utils.clip_grad_norm_(actor.parameters(), c.actor_clip,
                                              error_if_nonfinite=True)
    opt.step()
    actor.zero_grad(set_to_none=True)
    result = dict(matching_loss=loss, matching_gradient_norm=float(q.norm()),
                  pre_clip_norm=float(clipped))
    if diagnostic:
        mean = member.mean(1)
        norm = mean.norm(dim=-1)
        robust_norm = filtered.norm(dim=-1)
        cosine = torch.nn.functional.cosine_similarity(mean, filtered, dim=-1, eps=1e-12)
        eigen = eigen_analysis(member, alpha)
        lambda_max = eigen['eigenvalues'][:, 0]
        lambda_min = eigen['eigenvalues'][:, -1]
        # Covariance can be rank-deficient for M=3,D=2, so report a
        # denominator-regularized condition number rather than infinity.
        cond = (lambda_max+1e-12)/(lambda_min.clamp_min(0)+1e-12)
        result.update(mean_g_norm=float(norm.mean()), mean_robust_g_norm=float(robust_norm.mean()),
            mean_cov_trace=float(covariance.diagonal(dim1=-2, dim2=-1).sum(-1).mean()),
            mean_lambda_max=float(lambda_max.mean()), mean_cov_condition=float(cond.mean()),
            mean_shrink_1=float(eigen['shrink_factors'][:, 0].mean()),
            mean_shrink_2=float(eigen['shrink_factors'][:, -1].mean()),
            mean_cosine=float(cosine.mean()), fraction_cosine_below_0_9=float((cosine < .9).float().mean()),
            fraction_norm_ratio_below_0_5=float((robust_norm/(norm+1e-12)<.5).float().mean()))
        result['_eigen'] = {key: value.detach().cpu().tolist() for key, value in eigen.items()}
    return result


def global_scale_actor_step(actor, opt, base, critic, c, k, y, z, flow, scale, core):
    paths, _ = core.rollout(actor, base, c, k, y, z, flow)
    terminal = core.terminal_label(None, critic, c, k, y, paths[:, -1], 'learned')*scale
    adjoints = core.lean_adjoint(base, c, k, y, paths, terminal)
    q, loss = core.actor_gradient(actor, base, c, k, y, paths, adjoints)
    core.set_gradient(actor, q)
    clipped = torch.nn.utils.clip_grad_norm_(actor.parameters(), c.actor_clip, error_if_nonfinite=True)
    opt.step(); actor.zero_grad(set_to_none=True)
    return dict(matching_loss=loss, matching_gradient_norm=float(q.norm()),
                pre_clip_norm=float(clipped))
