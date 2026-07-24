"""Tests for class-keyed shared-environment effects."""

import numpy as np

import popstatgensim as psg
from popstatgensim.traits import realized_covariance_by_class, shared_environment_effects


def test_cluster_shared_env_is_class_pure():
    # 12 sibships of 4 -> 48 individuals; FS pairs = within-sibship pairs
    N, group_size = 48, 4
    labels = np.repeat(np.arange(N // group_size), group_size)
    effects = shared_environment_effects({'FS': 0.2}, N, cluster_labels={'FS': labels}, seed=0)
    assert list(effects) == ['c2_FS']
    e = np.asarray(effects['c2_FS'].generate_component({'N': N}))

    assert np.isclose(np.var(e), 0.2, atol=0.05)  # variance recorded to target
    # within-sibship pairs share the draw (cov ~ var); across-sibship pairs ~ 0
    within = [(i, j) for g in range(N // group_size)
              for i in range(g * group_size, (g + 1) * group_size)
              for j in range(i + 1, (g + 1) * group_size)]
    across = [(0, group_size), (1, 2 * group_size), (2, 3 * group_size)]
    assert np.mean([e[i] * e[j] for i, j in within]) > 0.1
    assert abs(np.mean([e[i] * e[j] for i, j in across])) < 0.05


def test_pairs_shared_env_kernel_is_psd_and_class_dominant():
    # a ring of "deg3-like" pairs (non-clique, overlapping) -> non-PSD indicator, gets projected
    N = 40
    pairs = [(i, (i + 2) % N) for i in range(N)] + [(i, (i + 5) % N) for i in range(N)]
    pairs = [(min(a, b), max(a, b)) for (a, b) in pairs if a != b]
    effects = shared_environment_effects({'deg3': 0.15}, N, class_pairs={'deg3': pairs}, seed=1)
    e = np.asarray(effects['deg3' and 'c2_deg3'].generate_component({'N': N}))
    assert np.isfinite(e).all()
    assert np.isclose(np.var(e), 0.15, atol=0.06)

    # realized covariance is highest on the injected pairs vs random non-pairs
    pcd = {p: ('deg3', 3) for p in pairs}
    cov = realized_covariance_by_class(e, pcd, ['deg3'])
    rng = np.random.default_rng(0)
    non = [(int(a), int(b)) for a, b in rng.integers(0, N, (400, 2)) if a < b and (a, b) not in set(pairs)]
    assert cov['deg3'] > np.mean([e[i] * e[j] for i, j in non])


def test_shared_environment_effects_public_api():
    assert hasattr(psg, 'shared_environment_effects')
    assert hasattr(psg, 'realized_covariance_by_class')
