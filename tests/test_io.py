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


def test_read_table_gcta_drops_missing_values_with_warning(tmp_path):
    path = tmp_path / 'missing.txt'
    path.write_text('fam1 iid1 -9\nfam2 iid2 2\nfam3 iid3 4\n', encoding='utf-8')

    with pytest.warns(UserWarning, match='Removed 1 sample'):
        observed = psg.io.read_table_GCTA(path)

    np.testing.assert_array_equal(observed['iid'], np.array(['iid2', 'iid3']))
    np.testing.assert_array_equal(observed['values'], np.array([[2.0], [4.0]]))


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


def _read_pgen_hardcalls(pgen_path, N, M):
    pgenlib = pytest.importorskip('pgenlib')
    reader = pgenlib.PgenReader(str(pgen_path).encode())
    try:
        buf = np.empty(N, dtype=np.int8)
        back = np.empty((M, N), dtype=np.int8)
        for v in range(M):
            reader.read(v, buf)
            back[v] = buf
    finally:
        reader.close()
    return back.T  # N x M


def test_export_pgen_fileset_round_trips_genotypes_and_metadata(tmp_path):
    pytest.importorskip('pgenlib')
    # N=4 individuals, M=3 variants; values are ALT-allele dosages 0..2
    G = np.array(
        [
            [0, 2, 0],
            [1, 2, 0],
            [2, 1, 1],
            [0, 0, 2],
        ],
        dtype=np.uint8,
    )
    iids = ['simB0001_1', 'simB0001_2', 'simB0001_3', 'simB0001_4']
    prefix = tmp_path / 'chr1'

    out = psg.io.export_pgen_fileset(
        prefix,
        G,
        iids=iids,
        chrom=1,
        pos=[10, 20, 30],
        sex=np.array([1, 0, 1, 0]),  # M, F, M, F (popstatgensim convention)
    )
    assert out == prefix.with_suffix('.pgen')

    back = _read_pgen_hardcalls(out, N=4, M=3)
    np.testing.assert_array_equal(back, G.astype(np.int8))

    pvar_lines = (tmp_path / 'chr1.pvar').read_text().strip().splitlines()
    assert pvar_lines[0] == '#CHROM\tPOS\tID\tREF\tALT'
    assert pvar_lines[1].split('\t') == ['1', '10', 'chr1:10', 'A', 'G']
    assert pvar_lines[3].split('\t')[:3] == ['1', '30', 'chr1:30']

    psam_lines = (tmp_path / 'chr1.psam').read_text().strip().splitlines()
    assert psam_lines[0] == '#FID\tIID\tSEX'
    # FID defaults to IID; sex maps 1(M)->1, 0(F)->2
    assert psam_lines[1].split('\t') == ['simB0001_1', 'simB0001_1', '1']
    assert psam_lines[2].split('\t') == ['simB0001_2', 'simB0001_2', '2']


def test_export_pgen_by_chrom_partitions_variants(tmp_path):
    pytest.importorskip('pgenlib')
    # 6 variants across two chromosomes: [1,1,1,2,2,2]
    G = np.array(
        [
            [0, 1, 2, 0, 1, 2],
            [2, 1, 0, 2, 1, 0],
        ],
        dtype=np.uint8,
    )
    chrom = np.array([1, 1, 1, 2, 2, 2])
    iids = ['p_1', 'p_2']

    written = psg.io.export_pgen_by_chrom(
        tmp_path, G, chrom, iids=iids, fids=['fam', 'fam'],
    )
    assert [p.name for p in written] == ['chr1.pgen', 'chr2.pgen']

    back1 = _read_pgen_hardcalls(tmp_path / 'chr1.pgen', N=2, M=3)
    back2 = _read_pgen_hardcalls(tmp_path / 'chr2.pgen', N=2, M=3)
    np.testing.assert_array_equal(back1, G[:, :3].astype(np.int8))
    np.testing.assert_array_equal(back2, G[:, 3:].astype(np.int8))

    # each per-chromosome .pvar carries only its own variants
    assert len((tmp_path / 'chr2.pvar').read_text().strip().splitlines()) == 1 + 3


def test_export_pgen_fileset_validates_shapes(tmp_path):
    with pytest.raises(ValueError, match='length N'):
        psg.io.export_pgen_fileset(tmp_path / 'x', np.zeros((3, 2)), iids=['a', 'b'])


def test_map_rel_class_maps_degrees_and_first_degree_types():
    assert psg.io.map_rel_class('sibling', 1) == 'FS'
    assert psg.io.map_rel_class('parent', 1) == 'PO'
    assert psg.io.map_rel_class('offspring', 1) == 'PO'
    assert psg.io.map_rel_class('uncle/aunt', 2) == 'deg2'
    assert psg.io.map_rel_class('1cousin', 3) == 'deg3'
    assert psg.io.map_rel_class('2cousin', 5) == 'fine_deg'
    assert psg.io.map_rel_class('2cousin', 11) == 'fine_deg'
    # self / unrelated / too-distant -> omitted
    assert psg.io.map_rel_class('self', 0) is None
    assert psg.io.map_rel_class('unspecified', 12) is None


def test_export_ibd_pairs_canonicalizes_and_filters(tmp_path):
    # IIDs deliberately chosen so lexicographic order differs from index order:
    # 'p_10' < 'p_2' as strings, matching h2est canon().
    iids = ['p_2', 'p_10', 'p_3']
    relatedness = np.array(
        [
            [1.0, 0.5, 0.02],
            [0.5, 1.0, 0.26],
            [0.02, 0.26, 1.0],
        ]
    )
    pair_class_degree = {
        (0, 1): ('FS', 1),     # p_2 & p_10 full sibs
        (1, 2): ('deg2', 2),   # p_10 & p_3
        (0, 2): (None, 20),    # unrelated -> skipped
    }
    out = psg.io.export_ibd_pairs(tmp_path / 'ibd_pairs.tsv', relatedness, iids, pair_class_degree)
    lines = out.read_text().strip().splitlines()
    assert lines[0] == 'ID1\tID2\trelatedness\trel_class\tdegree'
    # two retained pairs; canonical string order p_10 <= p_2, p_10 <= p_3
    body = [ln.split('\t') for ln in lines[1:]]
    assert len(body) == 2
    assert [r[:2] for r in body] == [['p_10', 'p_2'], ['p_10', 'p_3']]
    assert body[0][3] == 'FS' and body[0][4] == '1'
    assert float(body[0][2]) == 0.5


def test_export_ibd_pairs_min_relatedness_keeps_close_classes(tmp_path):
    iids = ['s_1', 's_2', 's_3']
    relatedness = np.array(
        [
            [1.0, 0.5, 0.0],   # s_1-s_2 FS (0.5); s_1-s_3 distant, zero IBD
            [0.5, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    pcd = {
        (0, 1): ('FS', 1),        # close class, kept even if it were zero
        (0, 2): ('fine_deg', 9),  # distant, relatedness 0 -> dropped by threshold
    }
    out = psg.io.export_ibd_pairs(tmp_path / 'ibd.tsv', relatedness, iids, pcd, min_relatedness=0.0)
    body = [ln.split('\t') for ln in out.read_text().strip().splitlines()[1:]]
    assert len(body) == 1
    assert body[0][3] == 'FS'


def test_export_pheno_writes_fid_iid_y_with_header(tmp_path):
    out = psg.io.export_pheno(tmp_path / 'pheno.tsv', [1.5, np.nan], ['s_1', 's_2'],
                              fids=['fam', 'fam'])
    lines = out.read_text().strip().splitlines()
    assert lines[0] == 'FID\tIID\ty'
    assert lines[1].split('\t') == ['fam', 's_1', '1.5']
    assert lines[2].split('\t') == ['fam', 's_2', 'NA']


def test_export_variants_base_and_effects(tmp_path):
    ids = ['chr1:1', 'chr1:2', 'chr2:1']
    base = psg.io.export_variants_base(
        tmp_path / 'variants.base.tsv',
        ids, chrom=[1, 1, 2], pos=[1, 2, 1], cm=[0.0, 0.01, 0.0],
        maf=[0.2, 0.4, 0.1], typed=[True, False, True],
    )
    blines = base.read_text().strip().splitlines()
    assert blines[0] == 'id\tchrom\tpos\tcm\tmaf\ttyped'
    assert blines[1].split('\t') == ['chr1:1', '1', '1', '0', '0.2', '1']
    assert blines[2].split('\t')[-1] == '0'  # typed=False

    eff = psg.io.export_variants_effects(
        tmp_path / 'variants.effects.tsv',
        ids, causal=[True, False, True], beta_std=[0.3, 0.0, -0.2],
    )
    elines = eff.read_text().strip().splitlines()
    assert elines[0] == 'id\tcausal\tbeta_std'  # no beta_par_std column
    assert elines[1].split('\t') == ['chr1:1', '1', '0.3']


def test_export_json_serializes_numpy(tmp_path):
    import json
    manifest = {'h2': {'h2_dir': np.float64(0.42)}, 'M_causal': np.int64(100),
                'components': np.array([0.1, 0.2])}
    out = psg.io.export_json(tmp_path / 'truth_manifest.json', manifest)
    loaded = json.loads(out.read_text())
    assert loaded['h2']['h2_dir'] == 0.42
    assert loaded['M_causal'] == 100
    assert loaded['components'] == [0.1, 0.2]


def test_prepare_reml_inputs_preserves_matching_order_when_ids_match():
    y = np.array([10.0, 20.0, 30.0])
    ids = np.array(['iid1', 'iid2', 'iid3'])
    X = np.array([[1.0], [2.0], [3.0]])
    grm = np.array(
        [
            [1.0, 0.2, 0.1],
            [0.2, 1.0, 0.3],
            [0.1, 0.3, 1.0],
        ]
    )

    observed = psg.io.prepare_reml_inputs(
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
