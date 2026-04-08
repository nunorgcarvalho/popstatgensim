import numpy as np

from popstatgensim import genome


def test_genotype_helpers_match_expected_small_example():
    haplotypes = np.array(
        [
            [[0, 1], [1, 1]],
            [[1, 1], [0, 0]],
        ],
        dtype=np.uint8,
    )
    genotypes = genome.make_G(haplotypes)

    expected_genotypes = np.array([[1, 2], [2, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(genotypes, expected_genotypes)

    freqs = genome.compute_freqs(genotypes, P=2)
    np.testing.assert_allclose(freqs, np.array([0.75, 0.5]))

    standardized = genome.standardize_G(genotypes, freqs, P=2, std_method="observed")
    grm = genome.compute_GRM(standardized)

    assert standardized.shape == (2, 2)
    np.testing.assert_allclose(grm, grm.T)
    np.testing.assert_allclose(np.diag(grm), np.ones(2))


def test_pca_runs_from_refactored_namespace():
    genotypes = np.array(
        [
            [0, 0, 1],
            [1, 0, 1],
            [2, 1, 0],
            [1, 2, 0],
        ],
        dtype=float,
    )
    freqs = genome.compute_freqs(genotypes, P=2)
    pca = genome.compute_PCA(G=genotypes, p=freqs, P=2, n_components=2)

    assert pca.scores.shape == (4, 2)
    assert pca.explained_variance_ratio.shape == (2,)


def test_genome_structure_helpers_produce_expected_shapes():
    np.random.seed(0)
    p_init = genome.draw_p_init(5, method="uniform", params=(0.2, 0.8))
    haplotypes = genome.draw_binom_haplos(p_init, N=4, P=2)
    chromosomes = genome.generate_chromosomes(6, chrs=2, meioses_per_chr=1)
    blocks = genome.generate_LD_blocks(6, N_blocks=3)

    assert p_init.shape == (5,)
    assert haplotypes.shape == (4, 5, 2)
    assert set(np.unique(haplotypes)).issubset({0, 1})
    assert chromosomes.shape == (6,)
    assert blocks.shape == (6,)
