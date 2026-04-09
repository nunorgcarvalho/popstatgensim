import numpy as np
import pytest

import popstatgensim as psg
from popstatgensim import run_EO_AM, run_HEreg, run_REML
from popstatgensim.traits import build_design_matrix_from_groups


def test_estimation_entrypoints_work_from_refactored_api():
    x = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [-1.0, 0.0],
        ],
        dtype=float,
    )
    grm = (x @ x.T) / x.shape[1]
    y = np.array([0.2, 1.0, -0.1, -1.1], dtype=float)

    out_he = run_HEreg(y=y, Rs=grm, verbose=0)
    out_reml = run_REML(y=y, Rs=grm, verbose=0, method="FS")

    assert set(out_he.keys()) == {"var_comps", "var_y", "fixed_effects", "algorithm", "log_likelihood"}
    assert set(out_reml.keys()) == {"var_comps", "var_y", "fixed_effects", "algorithm", "log_likelihood"}
    assert len(out_he["var_comps"]["est"]) == 2
    assert len(out_reml["var_comps"]["est"]) == 2
    assert np.isfinite(out_he["var_comps"]["est"]).all()
    assert np.isfinite(out_reml["var_comps"]["est"]).all()


def test_reml_warmstart_handles_design_matrix_random_effects():
    groups = np.array([0, 0, 1, 1, 2, 2], dtype=int)
    z_groups = build_design_matrix_from_groups(groups)
    grm = np.array(
        [
            [1.0, 0.2, 0.0, 0.0, 0.0, 0.0],
            [0.2, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.2, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.2],
            [0.0, 0.0, 0.0, 0.0, 0.2, 1.0],
        ],
        dtype=float,
    )
    y = np.array([1.0, 0.8, -0.2, -0.1, 0.4, 0.6], dtype=float)

    out_reml = run_REML(
        y=y,
        Rs=[grm, None],
        Zs=[None, z_groups],
        verbose=0,
    )

    assert set(out_reml.keys()) == {"var_comps", "var_y", "fixed_effects", "algorithm", "log_likelihood"}
    assert len(out_reml["var_comps"]["est"]) == 3
    assert np.isfinite(out_reml["var_comps"]["est"]).all()


def test_run_eo_am_matches_manual_even_odd_correlation():
    x = np.array(
        [
            [1.0, 2.0, 0.5, -1.0, 2.0, 1.5],
            [0.0, -1.0, 1.5, 2.0, -0.5, 0.0],
            [2.0, 1.0, -0.5, 1.0, 0.5, -1.0],
            [-1.0, 0.5, 2.0, 0.0, 1.0, 2.5],
        ],
        dtype=float,
    )
    weights = np.array([0.2, -0.1, 0.3, 0.4, -0.2, 0.1], dtype=float)
    chrom_idx = [0, 2, 4]

    out = run_EO_AM(X=x, pgs_weights=weights, chrom_idx=chrom_idx)

    odd_pgs = x[:, [0, 1, 4, 5]] @ weights[[0, 1, 4, 5]]
    even_pgs = x[:, [2, 3]] @ weights[[2, 3]]
    expected = np.corrcoef(odd_pgs, even_pgs)[0, 1]
    assert np.isclose(out, expected)


def test_run_eo_am_requires_at_least_two_chromosomes():
    x = np.arange(12, dtype=float).reshape(4, 3)
    weights = np.array([0.1, 0.2, 0.3], dtype=float)

    with pytest.raises(ValueError, match="At least two chromosomes"):
        run_EO_AM(X=x, pgs_weights=weights, chrom_idx=[0])


def test_population_get_chrom_idx_uses_r_half_and_includes_zero():
    pop = psg.Population(N=4, M=6, p_init=0.3, seed=3)
    pop.R = np.array([0.1, 0.5, 0.2, 0.5, 0.1, 0.1], dtype=float)

    np.testing.assert_array_equal(pop.get_chrom_idx(), np.array([0, 1, 3], dtype=int))
