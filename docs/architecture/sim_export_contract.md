# Simulation → h2est export contract

**Status: v0.3 — FULLY CONVERGED (h2est + simmgr, 2026-07-23). No open items.**
A working vertical-slice bundle exists on disk (see §9). Owner: popstatgensim worker. Consumers:
h2est pipeline (analyzer + `simulations/` bridge) + simmgr (two-stage generation orchestrator).

Defines the on-disk bundle `popstatgensim` writes per simulated population so the **real** h2est
pipeline can estimate on it and score each estimator against known truth (RelReg simulation
validation; thesisMgr task-20260722-175800 / -175802 / -175801).

`[LOCKED]` = agreed by all parties.

---

## 0. Settled ground rules
- popstatgensim generates DATA only; the real h2est pipeline estimates. No estimators reimplemented.
- **Ground-truth IBD fed to methods** (not hap-IBD/IBDkin); inferred IBD deferred.
- h2est **reuses its IBD build (task-034030)**: popstatgensim emits ONE per-pair truth table (§4);
  h2est's "truth-mode" adapter derives every LOCKED matrix (R_IBD, R_IBD_4to11, d_IBD_leq3,
  d_IBD_FS, C_FS, C_PO, C_eq2, C_eq3, R_IBD_leq3, …) from it. popstatgensim emits no matrices/segments.
- **Subset trick**: ALL causal-eligible variants simulated + exported; h2est builds the SNP GRM
  (`R_oo`) on the `typed` subset; ground-truth IBD (all variants) gives total direct-h². Typed subset
  is a **population-level** flag, never an omission.
- Two DGP paradigms feed the SAME format: (A) real-founder [UKB-governed]; (B) stylized [the slice].

---

## 1. Bundle layout — two-level  **[LOCKED]**
Filenames match simmgr's generation manifest so its index resolves directly. simmgr mints a
content-hash `population_id` (dir name + dedup) and a short stable `pop_alias` (e.g. `simBe269f8`)
used in IIDs.

```
<population_id>/                         # POPULATION-level, built once, reused across trait runs
  genotypes/
    chr1.pgen chr1.pvar chr1.psam
    chr2.pgen chr2.pvar chr2.psam        # one plink2 pfile-set per chromosome (ALL variants)
  variants.tsv                           # id, chrom, pos, cm, maf, typed              (§2)
  ibd_pairs.tsv                          # ground-truth per-pair relatedness           (§4)
  population.pkl                         # full Population incl. past gens, version-stamped (§7)
  population.meta.json                   # pop_alias, population_id, paradigm, seed, N, M,
                                         # generation_analyzed, M_typed, M_causal, commit, AM params
  truth.json                             # population-level realized truth: spouse corr, per-class
                                         # relatedness summary
  SIM_<population_id>.txt                # sample list (one IID/line) — simmgr owns; slice also emits
  traits/
    <trait_run_id>/
      phenotype.tsv                      # FID IID y (header, NA missing)              (§3)
      genetic_values.tsv                 # per-individual y_ decomposition (diagnostic)
      truth.json                         # trait truth: VCs, h2 defs, var_A_typed, causal, am (§6)
```
Analyzed cohort = individuals h2est estimates on. Default = the **single most-recent generation**
(see §8 cohort note). genotypes/pheno/ibd/variants all cover exactly that cohort (same N, IID set,
order). The pickle may carry ancestors.

---

## 2. Genotypes + per-variant table  **[LOCKED]**
- Chr-split plink2 pfiles via in-process `pgenlib.PgenWriter` (`io.plink.export_pgen_by_chrom`);
  biallelic ALT-dosage from `pop.G`; REF=allele 0, ALT=allele 1. Chromosomes from `R==0.5` breaks
  (`Population.get_chrom_idx`); the `structure.py` chr-break bug is fixed.
- `.psam`: `#FID IID SEX` (SEX 1=male/2=female). `.pvar`: `#CHROM POS ID REF ALT`.
- **`variants.tsv`** (population-level; `io.export_variants_base`): `id chrom pos cm maf typed`.
  `cm = 100 × cumulative per-variant recomb` for B (real map for A) — completeness only; the
  truth-IBD build uses no cM map. h2est builds `R_oo` on `typed==1`, ONCE per population.
- Per-trait causal effects are NOT in `variants.tsv` (they are per-trait) — they live in the trait
  `truth.json` (§6).

---

## 3. Phenotype  **[LOCKED]**
`phenotype.tsv`, tab-sep, header, missing token `NA`: `FID   IID   y`. `y` = `Trait.y` of the
analyzed generation. In the AM slice the same trait drives AM (population stage) and is analyzed.
h2est SIM_pheno reader: fid_column=FID, iid_column=IID.

---

## 4. Ground-truth IBD — one scalar per-pair table  **[LOCKED]**
`ibd_pairs.tsv` (population-level; `io.export_ibd_pairs`), tab-sep, header:
```
ID1   ID2   relatedness   rel_class   degree
```
- `relatedness` = realized π = 2 × kinship = `pedigree.ibd.compute_K_IBD` (exact truth).
- `ID1 <= ID2` by Python string `<=` (matches h2est `canon()`).
- `rel_class` from `Pedigree.path_to_relationship` via `io.map_rel_class`: full_sibling→`FS`,
  parent_offspring→`PO`, 2nd→`deg2`, 3rd→`deg3`, degrees 4–11→`fine_deg`. `degree` = int 1..11.
- **Thresholding (scaling)**: the writer takes `min_relatedness`; the slice uses `0.0`, emitting only
  pairs with realized π > 0 (plus the close classes FS/PO/deg2/deg3 always, for the confound
  matrices). The zero-π distant tail is omitted — equivalent to absent → 0 on h2est's side — so the
  table does not scale as N². (Slice: ~54k rows for N=1000 vs ~500k unthresholded.)
- No N×N, no segments, no cM map.

---

## 5. Sample-ID namespacing  **[LOCKED]**
- simmgr mints `pop_alias` (short, e.g. `simBe269f8`) ↔ content-hash `population_id`, and passes the
  alias to the exporter's `pop_id`/`pop_alias` param.
- **IID = `<pop_alias>_<i>`** (i=1..N, globally unique); **FID = `<pop_alias>`**.
- Identical IID strings + order across `.psam`, `phenotype.tsv`, `ibd_pairs.tsv`, `SIM_*.txt`, truth
  JSONs. h2est keys on IID; sample-fingerprint = sha1 of sorted unique IIDs.

---

## 6. Truth JSONs  **[LOCKED]**
Population-level `truth.json`: realized spouse correlation, per-class pair counts + mean relatedness.

Trait-level `truth.json` (`io.export_json`):
- `variance_components`: realized VCs per component (`Trait.var`): `A`, `A_par`/`VT`, `popstrat`, `Eps`.
- `Vp`: total phenotypic variance.
- `h2`:
  - `h2_dir` = `var_A_allcausal / Vp`, `var_A_allcausal` = `Trait.var['A']` (exact).
  - `h2_SNP_plim` = `var_A_typed / Vp`, where **`var_A_typed` = Var(OLS projection of the true genetic
    value `y_['A']` onto the standardized typed genotypes)** (`utils.projection_variance`). This IS
    the plim of GREML-on-typed / relreg1's R_oo channel (h2est owns this definition). ⚠ In-sample
    projection: pick **N ≫ M_typed** or it inflates toward 1.
  - `untyped_fraction` = `1 − var_A_typed / var_A_allcausal`.
  - popstatgensim ALSO exports the raw substrate (all-causal `beta_std`, `typed`/`causal` flags,
    genotypes) so h2est can recompute `var_A_typed` and absorb estimand refinements (e.g. AM-corrected).
- `am`: `AM_r` target + realized spouse corr (`pop.past[1].get_spouse_corr`) + AM trait name + n gens.
  (h2est owns the AM-inflation correction of h2_dir; these numbers let it predict the inflation.)
- `causal`: causal variant ids + `beta_std` (+ `beta_par_std` if simulated).
- `provenance`: commit, paradigm, seed, N, M, M_causal, M_typed, generation_analyzed.

**Per-estimator estimand targets (h2est forms the ratios):** GREML→h2_SNP; relreg1→h2_SNP+h2_dir+
h2_scale; relreg2/4→h2_dir; relreg5→h2_dir+h2_scale; sibreg→h2_dir. `h2_scale − h2_dir ≈ confound
magnitude` (VT+popstrat+AM), which the sim sets.

---

## 7. Persistence & governance  **[LOCKED]**
- `population.pkl`: full `Population` incl. `pop.past`, version-stamped. simmgr verified plain pickle
  round-trips past gens + pedigree + traits (global RNG doesn't travel; simmgr reseeds stage 2).
  `keep_past_generations` = deepest ancestral effect: **VT → 1** (the slice); higher-order IGE → k
  (not yet implemented).
- **Governance**: paradigm-A (real-founder) bundles live under scratch, under UKB rules, never logged.
  Paradigm-B is synthetic.

---

## 8. Analyzed-cohort scope  **[LOCKED — multi-generation default]**
The analyzed cohort is the current generation **flattened with `cohort_generations` past generations**
via `Population.flatten_generations(k)` — which pools genotypes/haplotypes (preserving cross-generation
founder haplotype IDs so true-IBD works across generations) AND rebuilds the cross-generation Pedigree.
Default `k=1` (current + parents) gives the full near-relative spectrum, verified against theory:
**FS ≈ 0.50, PO ≈ 0.51, deg2 (avuncular) ≈ 0.26, deg3 (1st cousins) ≈ 0.13** — so `C_FS`, `C_PO`,
`C_eq2`, `R_IBD*`, `d_IBD*` are all populated and every RelReg variant is exercised. Notes:
- Each cohort member keeps its OWN realized phenotype (join concatenates, does not regenerate); the
  causal architecture + chromosome map survive the join.
- **Half-relatives are not produced** (monogamous pairing) — `deg2` is entirely avuncular. Half-sibs
  would need multi-mate pairing (a future feature).
- **Uneven per-generation subsampling** of the cohort is a future parameter (default = all individuals
  of both generations).
- `population.pkl` is the base generative Population; the cohort = `base_pop.flatten_generations(k)`
  (`cohort_generations` recorded in `population.meta.json`).

## 9. Status
Vertical slice (stylized B + AM, **multi-generation cohort**) implemented and PASSING end-to-end on
branch `sim-validation` (`examples/sim_validation_vertical_slice.py`): chr-split pgen (plink2-validated),
ibd_pairs with FS/PO/deg2/deg3, phenotype, variants, truth JSONs (+ pedigree_edges), pickle. Durable
handoff bundles on scratch:
- `/n/scratch/users/n/nur479/sim_validation/vertical_slice_v1/simB0001/` — **current**, multi-gen cohort
  (N=600 cohort = 300/gen × 2 gens, M=400, 2 chr, AM_r=0.4; FS≈0.51 PO≈0.51 deg2≈0.26 deg3≈0.13;
  h2_dir≈0.52, h2_SNP_plim≈0.29, untyped_fraction≈0.43).
- `/n/scratch/users/n/nur479/sim_validation/vertical_slice_v0/simB0001/` — single-generation (kept for
  reference; no PO/deg2).
Next: real-founder paradigm (A, UKB-governed) + population-structure/VT confounds + per-Population RNG.
