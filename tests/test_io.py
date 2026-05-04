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


def test_read_table_gcta_reads_selected_columns_and_dummy_encodes(tmp_path):
    path = tmp_path / 'covar.txt'
    path.write_text(
        '\n'.join(
            [
                'fam1 iid1 1.0 B 10.0',
                'fam2 iid2 3.0 A 20.0',
                'fam3 iid3 5.0 C 30.0',
            ]
        ),
        encoding='utf-8',
    )

    observed = psg.io.read_table_GCTA(path, keep=[1, 2], discretize=2)

    np.testing.assert_array_equal(observed['iid'], np.array(['iid1', 'iid2', 'iid3']))
    np.testing.assert_array_equal(observed['fid'], np.array(['fam1', 'fam2', 'fam3']))
    np.testing.assert_array_equal(observed['columns'], np.array(['V1', 'V2=B', 'V2=C']))
    np.testing.assert_array_equal(
        observed['values'],
        np.array(
            [
                [1.0, 1.0, 0.0],
                [3.0, 0.0, 0.0],
                [5.0, 0.0, 1.0],
            ]
        ),
    )


def test_read_table_gcta_standardizes_output_columns(tmp_path):
    path = tmp_path / 'covar.txt'
    path.write_text(
        '\n'.join(
            [
                'iid1 2.0 X',
                'iid2 4.0 Y',
                'iid3 6.0 Y',
            ]
        ),
        encoding='utf-8',
    )

    observed = psg.io.read_table_GCTA(
        path,
        keep=[1, 2],
        discretize=2,
        standardize=True,
        skip_FID=True,
    )

    np.testing.assert_allclose(observed['values'].mean(axis=0), np.zeros(2), atol=1e-10)
    np.testing.assert_allclose(observed['values'].std(axis=0), np.ones(2), atol=1e-10)


def test_read_table_gcta_rejects_duplicate_iids(tmp_path):
    path = tmp_path / 'dup.txt'
    path.write_text('fam1 iid1 1\nfam2 iid1 2\n', encoding='utf-8')

    with pytest.raises(ValueError, match='duplicate IIDs'):
        psg.io.read_table_GCTA(path)


def test_read_table_gcta_rejects_missing_values(tmp_path):
    path = tmp_path / 'missing.txt'
    path.write_text('fam1 iid1 -9\nfam2 iid2 2\n', encoding='utf-8')

    with pytest.raises(ValueError, match='missing values'):
        psg.io.read_table_GCTA(path)


def test_subset_grm_by_ids_reorders_matrix():
    grm = np.array(
        [
            [1.0, 0.1, 0.2],
            [0.1, 1.0, 0.3],
            [0.2, 0.3, 1.0],
        ]
    )
    ids = np.array(['iid1', 'iid2', 'iid3'])
    target = np.array(['iid3', 'iid1'])

    observed = psg.io.subset_grm_by_ids(grm, ids, target)

    np.testing.assert_array_equal(observed, grm[np.ix_([2, 0], [2, 0])])


def test_align_samples_reorders_to_reference_and_filters_keep():
    y = np.array([10.0, 20.0, 30.0])
    y_ids = np.array(['iid2', 'iid1', 'iid3'])
    X = np.array([[1.0], [2.0], [3.0]])
    X_ids = np.array(['iid1', 'iid2', 'iid3'])
    grm = np.array(
        [
            [1.0, 0.2, 0.3],
            [0.2, 1.0, 0.4],
            [0.3, 0.4, 1.0],
        ]
    )
    grm_ids = np.array(['iid3', 'iid2', 'iid1'])

    observed = psg.io.align_samples(
        y=y,
        y_ids=y_ids,
        X=X,
        X_ids=X_ids,
        Rs=grm,
        R_ids=grm_ids,
        keep=['iid1', 'iid2'],
    )

    np.testing.assert_array_equal(observed['iid'], np.array(['iid2', 'iid1']))
    np.testing.assert_array_equal(observed['y'], np.array([10.0, 20.0]))
    np.testing.assert_array_equal(observed['X'], np.array([[2.0], [1.0]]))
    np.testing.assert_array_equal(observed['Rs'], grm[np.ix_([1, 2], [1, 2])])


def test_align_samples_warns_for_missing_keep_ids():
    y = np.array([1.0, 2.0, 3.0])
    ids = np.array(['iid1', 'iid2', 'iid3'])

    with pytest.warns(UserWarning, match='keep'):
        observed = psg.io.align_samples(y=y, y_ids=ids, keep=['iid1', 'iid9'])

    np.testing.assert_array_equal(observed['iid'], np.array(['iid1']))


def test_align_samples_fast_path_preserves_matching_order():
    y = np.array([1.0, 2.0, 3.0])
    ids = np.array(['iid1', 'iid2', 'iid3'])
    X = np.array([[5.0], [6.0], [7.0]])
    grm = np.eye(3)

    observed = psg.io.align_samples(
        y=y,
        y_ids=ids,
        X=X,
        X_ids=ids,
        Rs=grm,
        R_ids=ids,
    )

    np.testing.assert_array_equal(observed['iid'], ids)
    np.testing.assert_array_equal(observed['y'], y)
    np.testing.assert_array_equal(observed['X'], X)
    np.testing.assert_array_equal(observed['Rs'], grm)


def test_prepare_reml_inputs_aligns_by_default():
    y = np.array([10.0, 20.0])
    y_ids = np.array(['iid2', 'iid1'])
    X = np.array([[1.0], [2.0]])
    X_ids = np.array(['iid1', 'iid2'])
    grm = np.array([[1.0, 0.5], [0.5, 1.0]])
    grm_ids = np.array(['iid1', 'iid2'])

    observed = psg.io.prepare_reml_inputs(
        y=y,
        y_ids=y_ids,
        X=X,
        X_ids=X_ids,
        Rs=grm,
        R_ids=grm_ids,
    )

    np.testing.assert_array_equal(observed['iid'], np.array(['iid2', 'iid1']))
    np.testing.assert_array_equal(observed['y'], np.array([10.0, 20.0]))
    np.testing.assert_array_equal(observed['X'], np.array([[2.0], [1.0]]))
    np.testing.assert_array_equal(observed['Rs'], grm[np.ix_([1, 0], [1, 0])])


def test_prepare_reml_inputs_can_skip_alignment():
    y = np.array([10.0, 20.0])
    X = np.array([[1.0], [2.0]])
    grm = np.array([[1.0, 0.5], [0.5, 1.0]])

    observed = psg.io.prepare_reml_inputs(
        y=y,
        X=X,
        Rs=grm,
        already_aligned=True,
    )

    np.testing.assert_array_equal(observed['iid'], np.array(['1', '2']))
    np.testing.assert_array_equal(observed['y'], y)
    np.testing.assert_array_equal(observed['X'], X)
    np.testing.assert_array_equal(observed['Rs'], grm)
