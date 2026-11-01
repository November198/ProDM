"""Main ProDM Progress–Density Nonconformity implementation."""
import numpy as np


def build_template(gc_success):
    """gc_success: list/array of (T, D) 仅 ID-success 轨迹。
    返回 {mu_d, sd_d, mu_t, T}: 逐维标准化器 + per-phase 均值模板。"""
    X = np.asarray(gc_success, dtype=np.float32)        # (Ns, T, D)
    Ns, T, D = X.shape
    flat = X.reshape(-1, D)
    mu_d = flat.mean(0); sd_d = flat.std(0) + 1e-6
    Xn = (X - mu_d) / sd_d
    return {'mu_d': mu_d, 'sd_d': sd_d, 'mu_t': Xn.mean(0), 'T': T}

def _nearest_template_phase(tmpl, gc_traj):
    """Return the raw nearest-template phase index for one trajectory."""
    x = (np.asarray(gc_traj, dtype=np.float32) - tmpl['mu_d']) / tmpl['sd_d']
    if x.ndim != 2:
        raise ValueError(f'Expected a [T, D] trajectory, got {x.shape}')
    if x.shape[1] != tmpl['mu_t'].shape[1]:
        raise ValueError(
            f'Embedding dimension mismatch: trajectory D={x.shape[1]}, '
            f'template D={tmpl["mu_t"].shape[1]}'
        )
    if x.shape[0] > tmpl['T']:
        raise ValueError(
            f'Rollout length {x.shape[0]} exceeds fitted horizon {tmpl["T"]}'
        )
    distances = ((x[:, None, :] - tmpl['mu_t'][None, :, :]) ** 2).sum(-1)
    return distances.argmin(1).astype(float)

def estimate_monotone_phase(tmpl, gc_traj):
    """Nearest prototype per observation, followed by a causal running maximum."""
    raw_phase = _nearest_template_phase(tmpl, gc_traj)
    if len(raw_phase) == 0:
        return raw_phase

    phase = np.empty_like(raw_phase)
    phase[0] = raw_phase[0]
    for t in range(1, len(raw_phase)):
        current = max(raw_phase[t], phase[t - 1])
        phase[t] = current
    return phase

def build_phase_statistics(tmpl, gc_success, eps=1e-6):
    """Fit phase moments from causal monotonic matches on successful D_fit only."""
    if not gc_success or any(len(g) != tmpl['T'] for g in gc_success):
        raise ValueError('Successful fitting trajectories must share a nonempty horizon')
    phases = np.stack([estimate_monotone_phase(tmpl, g) for g in gc_success], axis=0)
    return {'phase_mu': phases.mean(axis=0), 'phase_sd': phases.std(axis=0)+eps}


def per_step_deficit_robust(tmpl, gc_traj, phase_stats, eps=1e-6):
    """Positive standardized phase lag relative to successful fitting rollouts."""
    phase = estimate_monotone_phase(tmpl, gc_traj)
    length = len(phase)
    mu = np.asarray(phase_stats['phase_mu'], dtype=float)[:length]
    sd = np.asarray(phase_stats['phase_sd'], dtype=float)[:length]
    if len(mu) != length or len(sd) != length:
        raise ValueError('Trajectory exceeds fitted phase-statistics horizon')
    return np.maximum(0.0, (mu-phase)/(sd+eps))


def build_pdn_model(gc_success, density_success):
    """Fit only on the caller's successful D_fit subset."""
    if len(gc_success) != len(density_success):
        raise ValueError('Embedding/density fitting trajectory counts differ')
    tmpl = build_template(gc_success)
    stats = build_phase_statistics(tmpl, gc_success)
    progress = np.asarray([per_step_deficit_robust(tmpl,g,stats) for g in gc_success])
    density = np.asarray(density_success)
    return {'tmpl':tmpl, 'phase_stats':stats,
            'pm':progress.mean(0), 'ps':progress.std(0)+1e-6,
            'dm':density.mean(0), 'ds':density.std(0)+1e-6}


def compute_prodm_scores(model, embeddings, density):
    """Main PDN: max(signed standardized progress, absolute standardized density)."""
    if len(embeddings) != len(density):
        raise ValueError('Embedding/density trajectory counts differ')
    scores=[]
    for g,d in zip(embeddings,density):
        p=per_step_deficit_robust(model['tmpl'],g,model['phase_stats'])
        length=min(len(p),len(d),len(model['pm']),len(model['dm']))
        if length == 0:
            raise ValueError('Empty score trajectory')
        zp=(np.asarray(p[:length])-model['pm'][:length])/model['ps'][:length]
        zd=np.abs((np.asarray(d[:length])-model['dm'][:length])/model['ds'][:length])
        scores.append(np.maximum(zp,zd))
    return scores
