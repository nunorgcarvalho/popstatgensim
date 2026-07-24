"""Vertical slice for the RelReg simulation-validation effort.

Builds ONE tiny stylized (paradigm B) population with LD across two chromosomes, attaches an
additive trait that BOTH drives assortative mating during the population stage AND is the trait
analyzed in the final generation, simulates AM over several generations, then writes the complete
two-level export bundle described in ``docs/architecture/sim_export_contract.md`` and validates it
(plink2 reads every pfile; ground-truth relatedness matches theory).

This is the end-to-end plumbing proof: generation -> export -> (h2est ingests). In production
simmgr orchestrates generation + identity; this script exercises the same popstatgensim exporters
standalone. Run:

    python examples/sim_validation_vertical_slice.py --out /path/to/bundle_root
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from popstatgensim import GeneticEffect, Population, PopulationParams, genome, io
from popstatgensim.genome.genotypes import standardize_G
from popstatgensim.pedigree.ibd import compute_K_IBD
from popstatgensim.utils.stats import projection_variance


def _commit_hash(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ['git', '-C', str(repo), 'rev-parse', '--short', 'HEAD'],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return 'unknown'


def build_population(*, N, M, chrs, meioses_per_chr, m_causal, var_A, var_Eps,
                     n_gen, am_r, seed, cohort_generations):
    """Stylized (paradigm B) population + AM-sorting additive trait, simulated to (near-)equilibrium."""
    R = genome.generate_chromosomes(M, chrs=chrs, meioses_per_chr=meioses_per_chr)
    # retain enough past generations to flatten the analyzed cohort (>=1 also covers VT depth-1)
    params = PopulationParams(track_pedigree=True, track_haplotypes=True,
                              keep_past_generations=max(1, cohort_generations),
                              n_offspring_dist='constant')
    pop = Population(N=N, M=M, P=2, params=params, seed=seed)
    pop.params.R = R                     # 2-chromosome recombination map (LD within, break between)
    pop.set_founding_haplotypes()        # unique founder haplotype IDs -> enables true-IBD tracking
    effect_A = GeneticEffect(var_indep=var_A, M=M, M_causal=m_causal, dist='normal',
                             name='A', force_var=False)
    pop.add_trait('y', effects={'A': effect_A}, var_Eps=var_Eps)
    # AM sorts on the phenotype 'y' every generation (population stage); 'y' is also the analyzed trait
    pop.simulate_generations(n_gen, AM_r=am_r, AM_trait='y', AM_type='phenotypic', verbose=False)
    return pop


def build_cohort(pop, cohort_generations):
    """The analyzed cohort = the current generation flattened with `cohort_generations` past ones.

    ``flatten_generations`` pools the genotypes/haplotypes (preserving cross-generation founder
    haplotype IDs so true-IBD works across generations) AND rebuilds the cross-generation pedigree,
    so parent-offspring (PO) and avuncular (deg2) pairs appear alongside FS / cousins.
    """
    cohort = pop.flatten_generations(cohort_generations)
    cohort.ped.assign_relationships()
    return cohort


def apply_shared_env(cohort, pcd, class_var, seed):
    """Add class-keyed shared-environment components to the cohort's trait, in place.

    `class_var` maps relationship class -> target variance (e.g. {'FS': 0.15, 'deg3': 0.10}).
    Full-sibs are injected as an exact sibship-cluster effect; deg3 (and other non-partition
    classes) via the PSD-projected pair-indicator kernel. Returns the list of added component names.
    """
    from popstatgensim.traits import shared_environment_effects
    cv = {c: v for c, v in class_var.items() if v and v > 0}
    if not cv:
        return []
    cluster_labels, class_pairs = {}, {}
    for cls in cv:
        if cls == 'FS':
            cluster_labels['FS'] = np.asarray(cohort.relations['full_sibs'])
        else:
            class_pairs[cls] = [(i, j) for (i, j), (c, d) in pcd.items() if c == cls]
    effects = shared_environment_effects(cv, cohort.N, cluster_labels=cluster_labels,
                                         class_pairs=class_pairs, seed=seed)
    for eff in effects.values():
        cohort.traits['y'].add_effect(eff)
    return list(effects)


def assemble_and_write(cohort, out_root: Path, *, base_pop, pop_alias, m_typed, seed, am_r,
                       n_gen, cohort_generations, commit, pcd=None):
    """Write the full two-level bundle and return a summary dict for validation.

    ``cohort`` is the analyzed (flattened multi-generation) population whose genotypes / phenotype /
    IBD / variants are exported; ``base_pop`` is the underlying generative Population that is pickled
    as the reusable artifact (the cohort is ``base_pop.flatten_generations(cohort_generations)``).
    """
    pop = cohort
    N, M, P = pop.N, pop.M, pop.P
    trait = pop.traits['y']

    # ----- per-variant population-level truth -----
    chrom_starts = pop.get_chrom_idx()
    chrom = np.searchsorted(chrom_starts, np.arange(M), side='right').astype(int)  # 1-based labels
    bps = getattr(pop, 'BPs', None)
    pos = (np.asarray(bps, dtype=int) if bps is not None else np.arange(M)) + 1
    p = np.asarray(pop.p, dtype=float)
    maf = np.minimum(p, 1.0 - p)
    R = np.asarray(pop.params.R, dtype=float)
    cm = np.zeros(M)
    for c in np.unique(chrom):
        idx = np.flatnonzero(chrom == c)
        r_c = R[idx].copy()
        r_c[0] = 0.0  # no recombination distance before a chromosome's first variant
        cm[idx] = 100.0 * np.cumsum(r_c)
    variant_ids = [f'chr{c}:{pp}' for c, pp in zip(chrom, pos)]

    # typed subset (the genotyping "array") — random, sparsified so an appreciable fraction is untyped
    rng = np.random.default_rng(seed + 1)
    typed = np.zeros(M, dtype=bool)
    typed[rng.choice(M, size=m_typed, replace=False)] = True

    # ----- identities -----
    iids = [f'{pop_alias}_{i + 1}' for i in range(N)]
    fids = [pop_alias] * N
    sex = np.asarray(pop.traits['sex'].y).astype(int)  # 0=F / 1=M (popstatgensim convention)

    # ----- ground truth: relatedness + relationship classes -----
    K = compute_K_IBD(pop.Haplos)
    if pcd is None:
        pcd = io.pair_class_degree_from_pedigree(pop.ped)

    # ----- trait truth -----
    Vp = float(np.var(trait.y))
    var_components = {k: float(v) for k, v in trait.var.items()}
    # shared-environment components (names 'c2_<class>'): record VC + realized per-class covariance
    from popstatgensim.traits import realized_covariance_by_class
    shared_env = {}
    for comp in trait.y_:
        if comp.startswith('c2_'):
            shared_env[comp] = {
                'var': float(trait.var[comp]),
                'realized_cov_by_class': realized_covariance_by_class(
                    np.asarray(trait.y_[comp]), pcd, ['FS', 'PO', 'deg2', 'deg3']),
            }
    var_A_all = float(trait.var['A'])
    X_typed = standardize_G(pop.G[:, typed].astype(float), p[typed], P)
    var_A_typed = projection_variance(trait.y_['A'], X_typed)
    try:
        spouse_corr = float(base_pop.past[1].get_spouse_corr('y'))
    except Exception:
        spouse_corr = float('nan')
    j_causal = np.asarray(trait.effects['A'].j_causal, dtype=int)
    beta_std_all = np.asarray(trait.effects['A'].effects_standardized, dtype=float)
    causal_mask = np.zeros(M, dtype=bool)
    causal_mask[j_causal] = True

    # ================= write bundle =================
    pop_dir = out_root / pop_alias
    geno_dir = pop_dir / 'genotypes'
    trait_run_id = f'{pop_alias}_t_y_r1'
    trait_dir = pop_dir / 'traits' / trait_run_id

    # population-level
    io.export_pgen_by_chrom(geno_dir, pop.G, chrom, iids=iids, variant_ids=variant_ids,
                            pos=pos, fids=fids, sex=sex)
    io.export_variants_base(pop_dir / 'variants.tsv', variant_ids, chrom, pos, cm, maf, typed)
    # threshold out the large tail of zero-IBD distant pairs (== absent → 0 on the consumer
    # side); close classes are retained regardless for the confound matrices
    io.export_ibd_pairs(pop_dir / 'ibd_pairs.tsv', K, iids, pcd, min_relatedness=0.0)
    # pickle the underlying generative population (the reusable artifact); the analyzed cohort is
    # base_pop.flatten_generations(cohort_generations)
    with (pop_dir / 'population.pkl').open('wb') as fh:
        pickle.dump(base_pop, fh)
    (pop_dir / f'SIM_{pop_alias}.txt').write_text('\n'.join(iids) + '\n', encoding='utf-8')

    class_counts = Counter(v[0] for v in pcd.values())
    mean_rel_by_class = {}
    for cls in io.sim_export.H2EST_REL_CLASSES:
        pairs = [(i, j) for (i, j), (c, d) in pcd.items() if c == cls]
        mean_rel_by_class[cls] = float(np.mean([K[i, j] for i, j in pairs])) if pairs else None

    # pedigree edges: IID -> (parent IIDs) for cohort members whose parents are in the cohort
    parents_rel = np.asarray(pop.relations['parents'])
    pedigree_edges = {}
    for i in range(N):
        par = [int(parents_rel[i, k]) for k in range(parents_rel.shape[1])]
        par_iids = [iids[pj] if (0 <= pj < N) else None for pj in par]
        if any(pi is not None for pi in par_iids):
            pedigree_edges[iids[i]] = par_iids

    io.export_json(pop_dir / 'population.meta.json', {
        'pop_alias': pop_alias, 'paradigm': 'B', 'seed': int(seed),
        'N_cohort': int(N), 'N_per_generation': int(base_pop.N), 'M': int(M),
        'n_chromosomes': int(len(chrom_starts)),
        'cohort_generations': int(cohort_generations),
        'base_generation': int(base_pop.t), 'keep_past_generations': int(base_pop.keep_past_generations),
        'AM_r_target': float(am_r), 'AM_trait': 'y', 'n_generations': int(n_gen),
        'M_typed': int(typed.sum()), 'M_causal': int(len(j_causal)),
        'popstatgensim_commit': commit,
    })
    io.export_json(pop_dir / 'truth.json', {
        'realized_spouse_corr': spouse_corr,
        'relationship_class_counts': {k: int(v) for k, v in class_counts.items()},
        'mean_relatedness_by_class': mean_rel_by_class,
        'pedigree_edges': pedigree_edges,
    })

    # trait-level
    io.export_pheno(trait_dir / 'phenotype.tsv', trait.y, iids, fids=fids)
    # per-individual genetic-value decomposition (optional diagnostic)
    comp_names = list(trait.y_.keys())
    with (trait_dir / 'genetic_values.tsv').open('w', encoding='utf-8') as fh:
        fh.write('FID\tIID\t' + '\t'.join(comp_names) + '\ty\n')
        for k in range(N):
            vals = '\t'.join(f'{float(trait.y_[c][k]):.10g}' for c in comp_names)
            fh.write(f'{fids[k]}\t{iids[k]}\t{vals}\t{float(trait.y[k]):.10g}\n')
    io.export_json(trait_dir / 'truth.json', {
        'trait_name': 'y',
        'variance_components': var_components,
        'Vp': Vp,
        'h2': {
            'h2_dir': var_A_all / Vp,
            'h2_SNP_plim': var_A_typed / Vp,
            'var_A_allcausal': var_A_all,
            'var_A_typed': var_A_typed,
            'untyped_fraction': 1.0 - var_A_typed / var_A_all,
        },
        'am': {'AM_r_target': float(am_r), 'spouse_corr_realized': spouse_corr,
               'AM_trait': 'y', 'n_generations': int(n_gen)},
        'shared_env': shared_env,
        'causal': {'ids': [variant_ids[j] for j in j_causal],
                   'beta_std': [float(beta_std_all[j]) for j in j_causal]},
        'provenance': {'popstatgensim_commit': commit, 'paradigm': 'B', 'seed': int(seed),
                       'N': int(N), 'M': int(M), 'M_causal': int(len(j_causal)),
                       'M_typed': int(typed.sum()), 'generation_analyzed': int(pop.t)},
    })

    return {
        'pop_dir': pop_dir, 'geno_dir': geno_dir, 'trait_dir': trait_dir,
        'n_chrom': int(len(chrom_starts)), 'class_counts': dict(class_counts),
        'mean_rel_by_class': mean_rel_by_class, 'Vp': Vp, 'var_A_all': var_A_all,
        'var_A_typed': var_A_typed, 'spouse_corr': spouse_corr, 'shared_env': shared_env,
        'M_typed': int(typed.sum()), 'M_causal': int(len(j_causal)),
    }


def validate_bundle(summary, plink2='plink2'):
    """Read every pfile back with plink2 and sanity-check the ground-truth relatedness."""
    import shutil
    import tempfile
    ok = True
    geno_dir = summary['geno_dir']
    plink2_bin = shutil.which(plink2) or str(Path.home() / 'my_software' / 'plink2')
    with tempfile.TemporaryDirectory() as chk_dir:  # keep plink2 check outputs out of the bundle
        for pvar in sorted(geno_dir.glob('*.pvar')):
            prefix = pvar.with_suffix('')
            out_chk = str(Path(chk_dir) / prefix.name)
            res = subprocess.run([plink2_bin, '--pfile', str(prefix), '--freq', '--out', out_chk],
                                 capture_output=True, text=True)
            status = 'OK' if res.returncode == 0 else 'FAIL'
            if res.returncode != 0:
                ok = False
            print(f'  plink2 read {prefix.name}: {status}')
    fs = summary['mean_rel_by_class'].get('FS')
    if fs is None or not (0.4 <= fs <= 0.6):
        print(f'  WARNING: FS mean relatedness {fs} outside expected [0.4, 0.6]')
        ok = False
    else:
        print(f'  FS mean relatedness {fs:.3f} within expected ~0.5  OK')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='bundle root directory')
    ap.add_argument('--N', type=int, default=300)   # per generation; cohort ~ N*(cohort_generations+1)
    ap.add_argument('--M', type=int, default=400)
    ap.add_argument('--chrs', type=int, default=2)
    ap.add_argument('--meioses-per-chr', type=int, default=2)
    ap.add_argument('--m-causal', type=int, default=120)
    ap.add_argument('--m-typed', type=int, default=120)
    ap.add_argument('--var-A', type=float, default=0.5)
    ap.add_argument('--var-Eps', type=float, default=0.5)
    ap.add_argument('--n-gen', type=int, default=6)
    ap.add_argument('--am-r', type=float, default=0.4)
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--pop-alias', default='simB0001')
    ap.add_argument('--cohort-generations', type=int, default=1,
                    help='past generations flattened into the analyzed cohort (1 = current + parents, '
                         'giving FS/PO/deg2/deg3)')
    ap.add_argument('--shared-env-fs', type=float, default=0.0,
                    help='full-sib shared-environment variance (c2_FS); 0 = off')
    ap.add_argument('--shared-env-deg3', type=float, default=0.0,
                    help='3rd-degree shared-environment variance (c2_deg3); 0 = off')
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    commit = _commit_hash(repo)
    np.random.seed(args.seed)

    print(f'Building stylized population (N={args.N}/gen, M={args.M}, {args.chrs} chr, '
          f'{args.n_gen} gens, AM_r={args.am_r})...')
    pop = build_population(N=args.N, M=args.M, chrs=args.chrs,
                           meioses_per_chr=args.meioses_per_chr, m_causal=args.m_causal,
                           var_A=args.var_A, var_Eps=args.var_Eps, n_gen=args.n_gen,
                           am_r=args.am_r, seed=args.seed, cohort_generations=args.cohort_generations)
    print(f'Flattening cohort: current + {args.cohort_generations} past generation(s)...')
    cohort = build_cohort(pop, args.cohort_generations)
    pcd = io.pair_class_degree_from_pedigree(cohort.ped)
    class_var = {'FS': args.shared_env_fs, 'deg3': args.shared_env_deg3}
    added = apply_shared_env(cohort, pcd, class_var, seed=args.seed + 7)
    if added:
        print(f'Added shared-environment components: {added}')
    summary = assemble_and_write(cohort, Path(args.out), base_pop=pop, pop_alias=args.pop_alias,
                                 m_typed=args.m_typed, seed=args.seed, am_r=args.am_r,
                                 n_gen=args.n_gen, cohort_generations=args.cohort_generations,
                                 commit=commit, pcd=pcd)
    print(f'\nBundle written to {summary["pop_dir"]}  (cohort N={cohort.N})')
    print(f'  chromosomes: {summary["n_chrom"]}   M_causal={summary["M_causal"]}  M_typed={summary["M_typed"]}')
    print(f'  relationship classes: {summary["class_counts"]}')
    print(f'  Vp={summary["Vp"]:.3f}  h2_dir={summary["var_A_all"]/summary["Vp"]:.3f}  '
          f'h2_SNP_plim={summary["var_A_typed"]/summary["Vp"]:.3f}  '
          f'untyped_frac={1 - summary["var_A_typed"]/summary["var_A_all"]:.3f}')
    if summary['shared_env']:
        for comp, info in summary['shared_env'].items():
            print(f'  {comp}: var={info["var"]:.3f}  realized_cov_by_class={info["realized_cov_by_class"]}')
    print(f'  realized spouse corr={summary["spouse_corr"]:.3f} (target {args.am_r})')
    print('\nValidating bundle...')
    ok = validate_bundle(summary)
    print('\nVERTICAL SLICE:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
