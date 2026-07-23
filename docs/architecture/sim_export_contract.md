# Simulation → h2est export contract

**Status: v0.1 — PROVISIONAL, pending h2est sign-off.**
Owner: popstatgensim worker (simulator + exporters). Consumer: h2est pipeline
(analyzer + `simulations/` bridge). Orchestrator: simmgr.

This document defines the on-disk bundle that `popstatgensim` writes for **one simulated
population** so that the **real h2est pipeline** can estimate heritability/confounding on it
and score each estimator against known truth. It is the shared interface for the
simulation-based validation of the RelReg estimators (thesisMgr task-20260722-175800 /
-175802 / -175801).

Sections marked **[SETTLED — popstatgensim]** are decisions popstatgensim will build to
unless h2est objects. Sections marked **[OPEN — needs h2est]** are questions only the h2est
worker can answer (they depend on what the reused IBD-build task-034030 and the scoring code
actually consume). h2est: please reply per numbered OPEN item.

---

## 0. Scope & settled ground rules (from the parent task; not re-litigated here)
- popstatgensim generates DATA only; the **real** h2est pipeline estimates. No estimators
  are reimplemented in popstatgensim.
- **Ground-truth IBD is fed to the methods** (not hap-IBD/IBDkin). Inferred IBD is deferred.
- h2est **reuses its existing IBD-table build (task-034030)** from truth-formatted inputs —
  popstatgensim exports truth in whatever format that build already consumes (see §4).
- **SNP-h² vs direct-h² tagging via the "subset trick"**: ALL causal-eligible variants are
  simulated and exported; the SNP GRM (`R_oo`) is built by h2est on a random **typed subset**;
  ground-truth IBD (all variants) gives total direct-h². The typed subset is flagged in the
  export, never realized by dropping variants (IBD needs all of them). See §2/§5.
- Two DGP paradigms feed the SAME bundle format: (A) real-founder forward-sim [UKB-governed],
  (B) stylized from-scratch [vertical slice]. **Only the front-end differs; this contract is
  identical for both.**

---

## 1. Bundle layout  **[SETTLED — popstatgensim]**
One directory per simulated population:

```
<pop_id>/
  genotypes/
    chr1.pgen  chr1.pvar  chr1.psam
    chr2.pgen  chr2.pvar  chr2.psam
    ...                                  # one plink2 pfile-set per chromosome
  variants.tsv                           # per-variant truth + typed/causal flags (§2)
  pheno.tsv                              # analyzed-cohort phenotype (§3)
  ibd/                                   # ground-truth IBD (§4) — format TBD by h2est
    ibd_segments.tsv                     # hap-IBD-like segments (proposed)
    ibd_pairs.tsv                        # scalar genome-wide relatedness per pair (proposed)
  truth_manifest.json                    # ground-truth for scoring (§6)
  population.pkl                         # full Population incl. past gens, version-stamped (§7)
  meta.json                             # pop_id, paradigm (A/B), generation analyzed, seed,
                                         # popstatgensim commit hash, param summary
```

- **Analyzed cohort** = the individuals h2est estimates on. Default = the **most recent
  generation**. `meta.json.generation_analyzed` records which generation index it is.
- Genotypes/pheno/ibd/variants all cover **exactly the analyzed cohort** (same N, same IID
  set, same order). The manifest + pickle may carry additional (ancestral) individuals.

---

## 2. Genotypes — chr-split plink2 pfiles  **[SETTLED — popstatgensim]**
- Written **in-process** with `pgenlib.PgenWriter` (biallelic hardcalls from `pop.G`, dosages
  0/1/2). No intermediate VCF.
- **All causal-eligible variants are written** (the full simulated set). The typed subset is
  NOT realized by omission — it is a flag (see `variants.tsv`).
- One pfile-set per chromosome. Stylized (B) chromosomes are defined by the `R == 0.5`
  recombination breaks (`Population.get_chrom_idx`); real-founder (A) chromosomes come from the
  real map. **NOTE:** popstatgensim will fix the `structure.py` chr-break bug (break currently
  set only from the 3rd chromosome) so 2-chromosome stylized genomes split correctly.
- `.psam`: columns `IID SEX` (+ `FID` if h2est wants it — see OPEN Q7). IIDs namespaced per §5.
  `SEX` from `pop.traits['sex']` (0=F/1=M → plink 2/1; unknown handling per plink2).
- `.pvar`: standard `#CHROM POS ID REF ALT`. Per-variant truth (typed/causal/effect) is kept
  OUT of `.pvar` and put in `variants.tsv` (below) to keep pfiles clean.

### `variants.tsv` — per-variant truth (tab-separated, header row)  **[SETTLED — popstatgensim]**
One row per variant (all chromosomes), keyed by `.pvar` `ID`:

| column        | meaning |
|---------------|---------|
| `id`          | variant ID, matches `.pvar` ID (e.g. `chr{c}:{pos}`) |
| `chrom`       | chromosome index (1-based) |
| `pos`         | within-chromosome position (bp for A; synthetic bp for B) |
| `cm`          | genetic-map position in cM (real map for A; derived from `R` for B — see §4 note) |
| `maf`         | minor-allele frequency in the analyzed cohort |
| `typed`       | 1 if in the typed subset used for `R_oo`, else 0 |
| `causal`      | 1 if it carries a nonzero true direct effect, else 0 |
| `beta_std`    | true per-standardized-allele direct effect (`GeneticEffect.effects_standardized`); 0 if non-causal |
| `beta_par_std`| true per-standardized-allele parental/IGE effect (`A_par`), if simulated; else absent/0 |

h2est builds `R_oo` from variants with `typed==1`; ground-truth IBD (§4) uses all variants.

---

## 3. Phenotype  **[SETTLED — popstatgensim, pending Q7 on ID columns]**
`pheno.tsv`, whitespace/tab-delimited, one row per analyzed individual:

```
FID   IID   y
```
- `y` = `Trait.y` of the analyzed generation (the realized phenotype).
- The SAME trait is used both as the AM-sorting trait during the population stage and as the
  analyzed phenotype in the latest generation (per the vertical-slice design).
- **[OPEN — Q7]** Does h2est's pheno reader want `FID IID y` (GCTA/plink style, current
  `io.gcta.export_trait` emits `IID IID y`), or IID-only, or a named header? Confirm exact
  format + missing-value token.

---

## 4. Ground-truth IBD  **[OPEN — needs h2est: this is the key fork]**
popstatgensim can produce truth-IBD in two shapes, both from the tracked haplotype-ID tensor
(`pop.Haplos`, enabled by `track_haplotypes=True`):

**(a) Scalar genome-wide relatedness per pair** — from `pedigree.ibd.compute_K_IBD(Haplos)`,
which returns the realized coefficient of **relatedness** (= 2 × kinship) for every pair.
Proposed `ibd_pairs.tsv`:
```
IID_i   IID_j   relatedness   kinship   rel_class   degree
```
(`rel_class`/`degree` from `Pedigree.path_to_relationship`, e.g. `full_sibling`, degree 1.)

**(b) Per-segment IBD (hap-IBD-like)** — from `pedigree.ibd.IBD_tensor_to_segments`, giving
per-haplotype-pair IBD1 segments. Proposed `ibd_segments.tsv` mirroring hap-IBD output:
```
IID_i   hap_i   IID_j   hap_j   chrom   start_cM   end_cM   [start_bp   end_bp]
```

**OPEN questions for h2est:**
1. **Which does the task-034030 IBD-build actually consume as input** — a per-pair scalar
   table (a), a hap-IBD-style segment file (b), or something else? Please paste/point to the
   exact columns + delimiter that build reads, so popstatgensim emits a drop-in.
2. If segments (b): confirm **units** (cM vs bp), whether **IBD2** is encoded as overlapping
   IBD1 segments or a separate state, and whether both haplotype orientations are expected.
3. For the stylized genome (B), there is no natural cM map. Proposed convention: **cM position
   = 100 × cumulative per-variant recombination rate** along a chromosome (so `R` integrates to
   a Morgan map), and hap-IBD-style segment endpoints reported on that derived map. **Does that
   satisfy the 034030 build**, or does it need a specific map file / `--map` companion?
4. Does the build key pairs by **IID** only (both haplotypes implied) or does it need explicit
   haplotype indices? Does it expect only related pairs (relatedness > 0) or the full N×N set?

---

## 5. Sample-ID namespacing  **[SETTLED — popstatgensim, pending Q7]**
Synthetic within-population indices `1..N` collide across populations, and h2est's
`io.alignment` keys everything on **IID** (with duplicate-IID rejection). Therefore:
- `pop_id` is a short population identifier assigned by the generator/simmgr (e.g. `simB0001`).
- **IID = `<pop_id>_<i>`** for i in `1..N` (globally unique).
- **FID = `<pop_id>`** (all individuals of one population share a FID).
- The same IID strings are used identically across `.psam`, `pheno.tsv`, `ibd/*`, and the
  manifest, in a consistent order.
- **[OPEN — Q7]** Confirm h2est keys on IID alone and this scheme is compatible with the
  034030 build + scoring join.

---

## 6. Truth-manifest — for scoring  **[OPEN — needs h2est on columns]**
`truth_manifest.json`. popstatgensim can populate any of the following from the Trait/Pedigree
objects; h2est should confirm the minimal set the scoring code needs and the exact
naming/shape. Proposed contents:
- `variance_components`: realized VCs per component (`Trait.var`) AND target VCs
  (`Trait.var_initial`) — e.g. `A`, `A_par`, `VT`, `popstrat`, `Eps`.
- `h2`: true heritability under each definition popstatgensim exposes
  (`get_h2('additive_variance')`, `'additive_covariance'`, `'additive_effects'`), so the
  estimand-vs-target comparison is unambiguous about which "truth" is meant.
- `y_decomposition`: per-individual latent components (`Trait.y_`) as a table keyed by IID,
  columns = component names + total `y` (this is the per-individual ground-truth breakdown).
- `pedigree_edges`: IID → (parent_IID_1, parent_IID_2) for the analyzed cohort (and ancestors
  if requested); founders have missing parents.
- `causal`: list of causal variant IDs + true `beta_std` (redundant with `variants.tsv`;
  include a pointer instead if h2est prefers).
- `am`: `AM_r` target and realized spouse correlation (`Population.get_spouse_corr`), plus the
  AM trait name and n generations to equilibrium.
- `provenance`: popstatgensim commit hash, paradigm (A/B), seed, generation analyzed, N, M,
  M_causal, M_typed.

**OPEN questions for h2est:**
5. Which of the above does the scoring code require, and under what JSON keys / table schema?
   In particular, is the per-individual `y_decomposition` needed, or only aggregate VCs/h²?
6. Which **h² definition** is the canonical estimand target for each RelReg estimator
   (`relreg1–5`, `sibreg`, GREML), so popstatgensim reports the matching truth prominently?

---

## 7. Persistence & governance  **[SETTLED — popstatgensim]**
- `population.pkl`: the full `Population` incl. `pop.past` retained generations, version-stamped
  with the popstatgensim commit. `keep_past_generations` is set to the deepest ancestral effect
  in use: **VT → 1; explicit higher-order IGE → k** (higher-order IGE is not yet implemented).
- **Governance:** paradigm-**A** (real-founder) bundles are derived from real UKB founders →
  they live under **scratch**, are treated under UKB data rules, and are **never logged**
  (no individual-level data in tasks/agent logs). Paradigm-B bundles are fully synthetic.

---

## 8. Open items summary (h2est: please answer Q1–Q7)
1. task-034030 IBD-build input format (scalar table vs segments vs other) — exact columns.
2. If segments: units (cM/bp), IBD2 encoding, haplotype orientations.
3. Stylized-genome cM-map convention acceptable? Need a `--map` companion?
4. IBD keyed by IID only vs explicit haplotypes; related pairs only vs full N×N.
5. Truth-manifest minimal required fields + JSON/table schema; is per-individual
   `y_decomposition` needed?
6. Canonical h² estimand target per estimator.
7. Pheno/`.psam` ID columns (`FID IID y`?) + IID-only keying + missing-value token.

popstatgensim will build the generation + AM + pgen-writer path against the SETTLED sections
now, treating the OPEN sections as provisional until h2est replies (via the coordination task).
