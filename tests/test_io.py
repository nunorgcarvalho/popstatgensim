import numpy as np
import pytest

import popstatgensim as psg


def test_read_grm_gcta_round_trips_exported_grm(tmp_path):
    grm = np.array(
        [
            [1.0, 0.1, -0.2],
            [0.1, 0.9, 0.3],
            [-0.2, 0.3, 1.1],
        ],
        dtype=float,
    )
    prefix = tmp_path / 'toy'

    psg.io.export_GRM_GCTA(grm, M=42, output_prefix=prefix)
    observed = psg.io.read_GRM_GCTA(tmp_path / 'toy.grm.bin')

    np.testing.assert_array_equal(observed, grm.astype(np.float32))


def test_read_grm_gcta_can_read_requested_sidecars(tmp_path):
    grm = np.array(
        [
            [1.0, 0.25, 0.5],
            [0.25, 1.0, -0.1],
            [0.5, -0.1, 0.8],
        ]
    )
    prefix = tmp_path / 'toy'

    psg.io.export_GRM_GCTA(grm, M=123, output_prefix=prefix)
    observed = psg.io.read_GRM_GCTA(
        tmp_path / 'toy.grm.bin',
        read_N=True,
        read_id=True,
    )

    assert set(observed) == {'grm', 'N', 'id'}
    np.testing.assert_array_equal(observed['grm'], grm.astype(np.float32))
    np.testing.assert_array_equal(observed['N'], np.full(6, 123, dtype=np.float32))
    np.testing.assert_array_equal(
        observed['id'],
        np.array([['1', '1'], ['2', '2'], ['3', '3']]),
    )


def test_read_grm_gcta_does_not_require_expected_suffix(tmp_path):
    grm = np.array([[1.0, 0.2], [0.2, 0.75]], dtype=np.float32)
    lower_triangle = grm[np.tril_indices(2)]
    path = tmp_path / 'relationships.bin'
    path.write_bytes(lower_triangle.tobytes())

    observed = psg.io.read_GRM_GCTA(path)

    np.testing.assert_array_equal(observed, grm)


def test_read_grm_gcta_rejects_invalid_lower_triangle_size(tmp_path):
    path = tmp_path / 'bad.grm.bin'
    path.write_bytes(np.arange(5, dtype=np.float32).tobytes())

    with pytest.raises(ValueError, match='lower-triangle size'):
        psg.io.read_GRM_GCTA(path)
