import numpy as np

from popstatgensim import run_HEreg, run_REML
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
