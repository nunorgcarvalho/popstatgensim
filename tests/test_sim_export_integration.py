"""End-to-end integration test for the sim-validation bundle path.

Exercises the whole reusable chain on a tiny stylized+AM population: generation with true-IBD
tracking -> ground-truth relatedness + relationship classes -> chr-split pgen + ibd_pairs +
phenotype + variant/effect tables. Skips if pgenlib is unavailable.
"""

import warnings

import numpy as np
import pytest

import popstatgensim as psg
from popstatgensim import GeneticEffect, Population, PopulationParams, genome
from popstatgensim.genome.genotypes import standardize_G
from popstatgensim.pedigree.ibd import compute_K_IBD
from popstatgensim.utils.stats import projection_variance


def _build_tiny_pop(seed=7):
    N, M = 120, 200
    R = genome.generate_chromosomes(M, chrs=2, meioses_per_chr=2)
    params = PopulationParams(track_pedigree=True, track_haplotypes=True,
                              keep_past_generations=1, n_offspring_dist='constant')
    pop = Population(N=N, M=M, P=2, params=params, seed=seed)
    pop.params.R = R
    pop.set_founding_haplotypes()
    pop.add_trait('y', effects={'A': GeneticEffect(var_indep=0.5, M=M, M_causal=60,
                                                   dist='normal', name='A', force_var=False)},
                  var_Eps=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pop.simulate_generations(3, AM_r=0.4, AM_trait='y', AM_type='phenotypic', verbose=False)
    pop.ped.construct_paths()
    pop.ped.assign_relationships()
    return pop


def test_flattened_cohort_truth_matches_theory_and_bundle_round_trips(tmp_path):
    pytest.importorskip('pgenlib')
    base = _build_tiny_pop()
    n_per_gen = base.N

    # analyzed cohort = current + previous generation (gives FS/PO/deg2/deg3)
    cohort = base.flatten_generations(1)
    cohort.ped.assign_relationships()
    N, M = cohort.N, cohort.M
    assert N == 2 * n_per_gen

    # ground-truth relatedness across generations must match theory
    K = compute_K_IBD(cohort.Haplos)
    assert 0.9 < np.mean(np.diag(K)) < 1.1  # self-relatedness ~ 1
    pcd = psg.io.pair_class_degree_from_pedigree(cohort.ped)
    present = {c for (c, d) in pcd.values()}
    assert {'FS', 'PO', 'deg2'} <= present  # flattening yields cross-generation classes

    def mean_k(cls):
        pairs = [(i, j) for (i, j), (c, d) in pcd.items() if c == cls]
        return np.mean([K[i, j] for i, j in pairs]) if pairs else np.nan

    assert 0.4 < mean_k('FS') < 0.6     # ~0.5
    assert 0.4 < mean_k('PO') < 0.6     # ~0.5
    assert 0.15 < mean_k('deg2') < 0.35  # ~0.25

    # identities + typed subset (N >> M_typed so the projection is meaningful)
    iids = [f'simT_{i + 1}' for i in range(N)]
    rng = np.random.default_rng(1)
    typed = np.zeros(M, dtype=bool)
    typed[rng.choice(M, size=40, replace=False)] = True

    chrom_starts = cohort.get_chrom_idx()
    chrom = np.searchsorted(chrom_starts, np.arange(M), side='right').astype(int)
    written = psg.io.export_pgen_by_chrom(tmp_path / 'genotypes', cohort.G, chrom, iids=iids)
    assert [p.name for p in written] == ['chr1.pgen', 'chr2.pgen']

    ibd = psg.io.export_ibd_pairs(tmp_path / 'ibd_pairs.tsv', K, iids, pcd, min_relatedness=0.0)
    header, *rows = ibd.read_text().strip().splitlines()
    assert header == 'ID1\tID2\trelatedness\trel_class\tdegree'
    for r in rows:  # canonical string ordering ID1 <= ID2
        c = r.split('\t')
        assert c[0] <= c[1]

    psg.io.export_pheno(tmp_path / 'phenotype.tsv', cohort.traits['y'].y, iids)

    # var_A_typed truth (projection) is a fraction of the total additive variance
    t = cohort.traits['y']
    var_A_all = float(t.var['A'])
    X_typed = standardize_G(cohort.G[:, typed].astype(float), cohort.p[typed], cohort.P)
    var_A_typed = projection_variance(t.y_['A'], X_typed)
    assert 0.0 < var_A_typed <= var_A_all + 1e-9
    assert 0.0 <= 1.0 - var_A_typed / var_A_all <= 1.0
