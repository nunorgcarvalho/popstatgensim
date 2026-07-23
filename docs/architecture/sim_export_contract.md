# Simulation → h2est export contract

**Status: v0.2 — converged with the h2est + simmgr workers over inter-agent comms
(2026-07-23).** Nearly everything is LOCKED; one small ownership item remains (Var_typed, §6).
Owner: popstatgensim worker (simulator + exporters). Consumers: h2est pipeline (analyzer +
`simulations/` bridge) + simmgr (two-stage generation orchestrator).

This document defines the on-disk bundle that `popstatgensim` writes for **one simulated
population** so that the **real h2est pipeline** can estimate heritability/confounding on it and
score each estimator against known truth (simulation-based validation of the RelReg estimators;
thesisMgr task-20260722-175800 / -175802 / -175801).

`[LOCKED]` = agreed by all parties. `[OPEN]` = still to settle.

---

## 0. Settled ground rules (from the parent task; not re-litigated)
- popstatgensim generates DATA only; the **real** h2est pipeline estimates. No estimators are
  reimplemented in popstatgensim.
- **Ground-truth IBD is fed to the methods** (not hap-IBD/IBDkin). Inferred IBD is deferred.
- h2est **reuses its existing IBD-table build (task-034030)**: popstatgensim emits ONE per-pair
  truth table (§4); h2est's build derives every LOCKED relationship matrix (R_IBD, R_IBD_4to11,
  d_IBD_leq3, d_IBD_FS, C_FS, C_PO, C_eq2, C_eq3, R_IBD_leq3, …) from it via a "truth-mode"
  adapter (h2est task-175802 design-choice #3). popstatgensim does NOT emit matrices or segments.
- **Subset trick** for SNP-h² vs direct-h²: ALL causal-eligible variants are simulated + exported;
  the SNP GRM (`R_oo`) is built by h2est on the **typed subset** (a `typed` flag, never realized
  by dropping variants — IBD needs all of them). The typed subset is a **population-level**
  property (the "array"), shared across trait runs.
- Two DGP paradigms feed the SAME bundle format: (A) real-founder forward-sim [UKB-governed],
  (B) stylized from-scratch [vertical slice]. Only the front-end differs.

---

## 1. Bundle layout — two-level (population vs trait)  **[LOCKED]**
Split so an analysis-only trait can reuse a population without regenerating genotypes. simmgr
mints identities: a content-hash `population_id` (drives dedup/reuse) with a short stable alias
`pop_id` (e.g. `simB0001`) that goes into IIDs.

```
<population_id>/                         # POPULATION-level, built once, reused
  genotypes/
    chr1.pgen  chr1.pvar  chr1.psam
    chr2.pgen  chr2.pvar  chr2.psam      # one plink2 pfile-set per chromosome (all variants)
  variants.base.tsv                      # id, chrom, pos, cm, maf, typed   (§2)
  ibd_pairs.tsv                          # ground-truth per-pair relatedness (§4) — population-level
  population.pkl                         # full Population incl. past gens, version-stamped (§7)
  meta.json                              # pop_id, population_id, paradigm(A/B), seed, N, M,
                                         # generation_analyzed, popstatgensim commit hash, params
  traits/
    <trait_run_id>/                      # TRAIT-level, one per trait run
      pheno.tsv                          # analyzed-cohort phenotype (§3)
      truth_manifest.json                # ground-truth for scoring (§6)
      variants.effects.tsv               # id, causal, beta_std, beta_par_std (§2)
```
- **Analyzed cohort** = the individuals h2est estimates on. Default = most recent generation;
  `meta.json.generation_analyzed` records the index. genotypes / pheno / ibd / variants all cover
  exactly this cohort (same N, same IID set + order). The pickle may carry ancestors too.
- For the **AM vertical slice**, the structuring (AM) trait is the first `trait_run` under its
  population (pop×trait coincide), realized during forward-sim and saved in the pickle.

---

## 2. Genotypes + per-variant truth  **[LOCKED]**
- Chr-split plink2 pfiles written in-process with `pgenlib.PgenWriter` (biallelic hardcalls from
  `pop.G`, ALT-dosage 0..2). **Built + tested + committed** (`io.plink.export_pgen_by_chrom`,
  branch `sim-validation`, commit 73e9b68). All causal-eligible variants written; typed subset is
  a flag, not an omission. Chromosomes from `R == 0.5` breaks (`Population.get_chrom_idx`); the
  `structure.py` chr-break bug (missed chr1↔chr2) is fixed (commit b4d905d).
- `.psam`: `#FID IID SEX`, FID=`<pop_id>`, IID=`<pop_id>_<i>` (SEX 1=male/2=female).
- `.pvar`: `#CHROM POS ID REF ALT`. REF=allele 0, ALT=allele 1, so `pop.G` is the ALT dosage.
- **`variants.base.tsv`** (population-level, tab-sep, header): `id chrom pos cm maf typed`.
  `id` matches `.pvar` ID; `cm = 100 × cumulative per-variant recomb` for B (real map for A) —
  kept for completeness, NOT on the critical path (h2est truth-IBD uses no cM map).
- **`variants.effects.tsv`** (trait-level, tab-sep, header): `id causal beta_std beta_par_std`.
  `beta_std` = true per-standardized-allele direct effect (`GeneticEffect.effects_standardized`),
  0 if non-causal; `beta_par_std` for the parental/IGE effect if simulated.

---

## 3. Phenotype  **[LOCKED]**
`pheno.tsv`, tab-sep, WITH header, missing token `NA`:
```
FID   IID   y
```
`y` = `Trait.y` of the analyzed generation. In the AM slice the same trait is the AM-sorting trait
(population stage) AND the analyzed phenotype. h2est's SIM_pheno source reader: fid_column=FID,
iid_column=IID.

---

## 4. Ground-truth IBD — ONE scalar per-pair table  **[LOCKED]**
`ibd_pairs.tsv` (population-level, tab-sep, header). h2est derives all matrices from this.
```
ID1   ID2   relatedness   rel_class   degree
```
- `relatedness` = realized π = 2 × kinship = `pedigree.ibd.compute_K_IBD` (exact truth).
- `ID1 <= ID2` by **Python string `<=`** (plain lexicographic on the namespaced IID strings) — must
  match h2est's `canon()`.
- `rel_class` from `Pedigree.path_to_relationship`: full_sibling→`FS`, parent_offspring→`PO`,
  2nd-degree→`deg2`, 3rd→`deg3`, degrees 4–11→`fine_deg` (integer in `degree`).
- `degree` = integer 1..11.
- **Related pairs only** (degree ≤ 11); all other pairs absent → default 0 (h2est convention).
  No N×N, no hap-IBD segments, no cM map (segments only matter for deferred inferred-IBD).

---

## 5. Sample-ID namespacing  **[LOCKED]**
- `pop_id` = short stable alias minted by simmgr (↔ content-hash `population_id` in its registry).
- **IID = `<pop_id>_<i>`** (i in 1..N, globally unique); **FID = `<pop_id>`**.
- Identical IID strings + order across `.psam`, `pheno.tsv`, `ibd_pairs.tsv`, and the manifest.
- h2est keys on IID; sample-fingerprint = sha1 of sorted unique IIDs → uniqueness + consistency is
  all that matters. Canonical pair order in `ibd_pairs.tsv` = string `<=` (§4).

---

## 6. Truth-manifest — for scoring  **[LOCKED fields; one OPEN ownership item]**
`truth_manifest.json` (trait-level). popstatgensim reports:
- `variance_components`: realized VCs per component (`Trait.var`): `A` (direct), `A_par`/`VT`,
  `popstrat`, `Eps`, plus total phenotypic variance `Vp`.
- `h2`:
  - `h2_dir` = `Var_allcausal / Vp`, with **`Var_allcausal` reported exactly as `Trait.var['A']`**.
  - `h2_SNP` = additive var tagged by the typed subset / Vp — see OPEN item below.
  - `untyped_fraction` = `1 − Var_typed / Var_allcausal`.
- `am`: `AM_r` target + realized spouse correlation (`Population.get_spouse_corr`) + AM trait name +
  n generations to equilibrium. (h2est owns the AM-inflation correction of the h2_dir estimand; it
  needs these truthful numbers to predict the expected inflation.)
- `provenance`: popstatgensim commit, paradigm (A/B), seed, N, M, M_causal, M_typed.
- `pedigree_edges`: IID → (parent_IID_1, parent_IID_2); founders have missing parents. (Recommended;
  not strictly blocking slice 1 since per-pair rel_class already drives the matrices.)
- `y_decomposition` (per-individual `Trait.y_` by IID): optional diagnostic (plim-vs-finite-N).

**Per-estimator estimand targets (from h2est methods doc 09 §2.4; for reference — h2est forms the
ratios):** GREML→h2_SNP; relreg1→h2_SNP+h2_dir+h2_scale (joint); relreg2/relreg4→h2_dir; relreg5→
h2_dir+h2_scale; sibreg→h2_dir; relreg3→future. `h2_scale − h2_dir ≈ confound magnitude` (VT +
popstrat + AM), which the sim SETS — so reporting VT var, popstrat var, AM_r + spouse corr lets
h2est predict the expected h2_scale bias.

**[OPEN — Var_typed ownership]** Under LD, "additive variance tagged by the typed subset" is
estimand-model dependent (typed non-causal tag untyped causal). Since h2est owns the GREML estimand
definition, default: **popstatgensim exports the pieces (all-causal `beta_std`, per-variant
`typed`/`causal` flags, genotypes) and h2est computes `Var_typed`.** popstatgensim can additionally
ship a diagnostic `Var_typed = R²(A ~ X_typed)·Vp` (projection of the true genetic value onto the
typed standardized genotypes). Awaiting h2est's pick.

---

## 7. Persistence & governance  **[LOCKED]**
- `population.pkl`: full `Population` incl. `pop.past` generations, version-stamped with the
  popstatgensim commit. simmgr verified plain `pickle` round-trips past gens + pedigree + traits
  (the global RNG does not travel; simmgr reseeds in stage 2). `keep_past_generations` = deepest
  ancestral effect: **VT → 1** (the slice); explicit higher-order IGE → k (not yet implemented).
- **Governance:** paradigm-**A** (real-founder) bundles live under **scratch**, under UKB data
  rules, **never logged** (no individual-level data anywhere). Paradigm-B bundles are synthetic.

---

## 8. Remaining open item
Only §6 **Var_typed ownership** (h2est to pick: h2est computes it, or popstatgensim also ships the
projection diagnostic). Everything else is LOCKED. popstatgensim is building the remaining exporters
(ibd_pairs, pheno, variants.*, manifest, meta, pickle) + the first stylized+AM bundle against v0.2.
