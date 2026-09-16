"""On-demand relationship finding, and constructive per-class enumeration.

``relate(A, B)`` is a **best-first search**, and the property that makes it exact is
worth stating because it removes the usual trade-off between a cheap heuristic and
exhaustive enumeration:

    Every factor in a path's product lies in (0, 1]. Each net meiosis multiplies by
    (1 + rho_g)/2 < 1, and every case of eq:chain-weight is <= 1 for rho_y, rho_g < 1.
    So a partial path's product is an UPPER BOUND on anything it can complete to --
    the admissibility condition for best-first search in -log space.

    Therefore the FIRST complete path popped is the dominant path, guaranteed. There is
    no need to enumerate the rest.

Truncation is a **floor on the phenotypic correlation** (default 0.005) rather than a
generation-count horizon, so it is stated in the units of the quantity being
approximated. ``\\assumptionid{No-inbreeding}`` is explicitly relaxed in the writeup to
"a couple generations back", which is what sanctions this.

Two jobs are kept deliberately separate (D-0007):

* **Labeling** -- a parameter-free rule (shortest by total steps, fixed tie-breaks), so
  labels are stable and reproducible.
* **Prediction** -- scoring by the actual product at *measured* ``rho_y``, ``rho_g``.

They usually agree. When they do not, that is a case where
``\\approximationid{Dominant-path}`` is doing real work, so the disagreement is
**reported and flagged, never silently resolved**. A precomputed global ordering of
signatures would not be sound: within a step type monotonicity holds, but across types
the ordering moves with the parameters -- full sibs carry ``(1+rho_g)/2`` while step
sibs carry ``rho_g((1+rho_y)/2)^2``, and which is larger depends on ``rho_y`` and
``rho_g``.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Iterator, Optional, Tuple

import numpy as np

from .graph import PedigreeGraph
from .paths import PathSignature, predicted_corr, predicted_cov

#: Conventional names for the signatures that have them. Everything else gets
#: ``PathSignature.systematic_name()`` rather than a single 'unspecified' catch-all.
KNOWN_SIGNATURES = {
    # (sum_N_lin, sum_N_o, sorted mating chains, sorted proband cases) -> name
    (0, 0, (), ('sink', 'sink')): 'self',
    (1, 0, (), ('sink', 'source')): 'parent-offspring',
    (2, 0, (), ('sink', 'source')): 'grandparent',
    (3, 0, (), ('sink', 'source')): 'great-grandparent',
    (2, 2, ((1, 2),), ('sink', 'sink')): 'full siblings',
    (2, 2, ((2, 2),), ('sink', 'sink')): 'half siblings',
    (2, 2, ((3, 2),), ('sink', 'sink')): 'step siblings',
    (2, 2, ((4, 2),), ('sink', 'sink')): 'step-step siblings',
    (3, 2, ((1, 2),), ('sink', 'sink')): 'avuncular (full)',
    (3, 2, ((2, 2),), ('sink', 'sink')): 'avuncular (half)',
    (4, 2, ((1, 2),), ('sink', 'sink')): 'first cousins',
    (0, 0, ((1, 0),), ('terminal_mating', 'terminal_mating')): 'parent--parent (mates)',
    (2, 0, ((1, 0),), ('source', 'source')): 'parents of a married couple',
}


@dataclass
class RelateResult:
    """What ``relate`` reports about one pair.

    ``name`` is the conventional name when the signature has one, else the systematic
    name. ``label_name`` is the parameter-free shortest-path label. ``label_agrees``
    says whether the two picked the same path -- a False there marks a pair where the
    dominant path under the measured parameters is NOT the shortest one, i.e. where
    ``Dominant-path`` is doing real work.
    """

    A: int
    B: int
    signature: Optional[PathSignature]
    name: str
    cov: float
    corr: float
    label_name: Optional[str] = None
    label_agrees: bool = True
    n_expanded: int = 0
    all_paths: Optional[list] = None

    @property
    def related(self) -> bool:
        return self.signature is not None

    @property
    def systematic_name(self) -> str:
        return self.signature.systematic_name() if self.signature else 'unrelated'


def signature_name(signature: PathSignature) -> str:
    """Conventional name if the signature has one, else its systematic name."""
    return KNOWN_SIGNATURES.get(signature.key, signature.systematic_name())


def _endpoints(graph: PedigreeGraph, proband: int, max_steps: int):
    """Every lineal-chain endpoint reachable from ``proband``, with its orientation.

    A path is (A's lineal chain) + mating chain + (B's lineal chain), and a proband may
    sit at EITHER end of its own lineal chain. So both directions are enumerated:

    * walking **up** reaches an ancestor -- the proband is the youngest on that chain,
      a **sink**, and the chain descends FROM the mating chain, contributing 1 to ``N_o``.
    * walking **down** reaches a descendant -- the proband is the oldest, a **source**,
      and the chain ascends INTO the mating chain, contributing 0 to ``N_o``.
    * ``n_lin = 0`` means the proband is itself on the mating chain, with no lineal
      chain of its own: a **terminal mating** proband.

    Walking down matters: without it, "parents of a married couple" (A -> i -- j <- B)
    is unreachable, because that path descends from both probands rather than ascending.

    Yields ``(node, n_lin, direction)`` with direction in ``{'up', 'down', 'self'}``.
    """
    yield int(proband), 0, 'self'

    seen_up = {int(proband)}
    frontier = [int(proband)]
    for depth in range(1, max_steps + 1):
        nxt = []
        for node in frontier:
            for p in graph.parents_of(node):
                p = int(p)
                if p in seen_up:
                    continue
                seen_up.add(p)
                nxt.append(p)
                yield p, depth, 'up'
        if not nxt:
            break
        frontier = nxt

    seen_dn = {int(proband)}
    frontier = [int(proband)]
    for depth in range(1, max_steps + 1):
        nxt = []
        for node in frontier:
            for c in graph.offspring_of(node):
                c = int(c)
                if c in seen_dn:
                    continue
                seen_dn.add(c)
                nxt.append(c)
                yield c, depth, 'down'
        if not nxt:
            break
        frontier = nxt


def _chain_contains_parent_pair(graph: PedigreeGraph, members: set,
                                endpoint: int, proband: int, n_lin: int) -> bool:
    """The writeup's containment rule for an out-edge of a mating chain.

    "When a mating chain has an out-edge it must contain BOTH parents of the
    second-oldest individual of the lineal chain it points to." An out-edge represents
    a mate *pair*; an in-edge a single individual.

    This is what distinguishes half sibs from full sibs. For half sibs A (parents P, Q1)
    and B (parents P, Q2), the candidate joining Q1 to P via one mating would otherwise
    pass and be scored as a FULL sib pair -- but that chain {Q1, P} does not contain Q2,
    B's other parent, so it is correctly rejected, leaving the true chain Q1 -- P -- Q2
    at N_mu = 2.
    """
    if n_lin <= 0:
        return True
    # second-oldest individual on the chain from `endpoint` down to `proband`
    if n_lin == 1:
        child = int(proband)
    else:
        # walk up from the proband to depth n_lin - 1
        node = int(proband)
        for _ in range(n_lin - 1):
            ps = [int(x) for x in graph.parents_of(node) if int(x) in members or True]
            if not ps:
                return False
            # follow the branch that leads to `endpoint`
            nxt = None
            for cand in ps:
                anc = graph.ancestors(cand, max_up=n_lin)
                if int(endpoint) in anc:
                    nxt = cand
                    break
            if nxt is None:
                return False
            node = nxt
        child = node
    parents = [int(x) for x in graph.parents_of(child)]
    return len(parents) == 2 and all(p in members for p in parents)


def _candidate_signatures(graph: PedigreeGraph, A: int, B: int,
                          max_up: int, max_mating_chains: int):
    """Yields every valid candidate ``(signature, kind)`` for the pair.

    Loops are nested **ancestors-first** (D-0007): enumerate A's chain endpoints, then
    B's, then check each pair for a mating chain. The mating-graph step is the cheap one
    -- degree is ~kappa -- so it belongs innermost.
    """
    ends_A = list(_endpoints(graph, A, max_up))
    ends_B = list(_endpoints(graph, B, max_up))

    # If one proband is a lineal ancestor of the other, the relationship IS lineal and
    # the lineal chain is its correct representation. Routing it through the ancestor's
    # own mating edge instead would double-count the same meiosis and (because the
    # terminal-mating correction multiplies by 1/h^2) badly overstate the covariance.
    # NOTE: this rule is our reading, not something the writeup states explicitly.
    anc_A = graph.ancestors(A, max_up=max_up)
    anc_B = graph.ancestors(B, max_up=max_up)
    lineal_pair = (B in anc_A) or (A in anc_B)

    for node_a, n_a, dir_a in ends_A:
        for node_b, n_b, dir_b in ends_B:
            # --- lineal / common-ancestor, no mating chain ---
            if node_a == node_b:
                if n_a == 0 or n_b == 0:
                    if n_a == 0 and n_b == 0:
                        yield PathSignature(0, 0, (), 'sink', 'sink'), 'self'
                    else:
                        case_a = 'source' if n_a == 0 else 'sink'
                        case_b = 'source' if n_b == 0 else 'sink'
                        yield PathSignature(n_a + n_b, 0, (), case_a, case_b), 'lineal'
                continue

            if lineal_pair or max_mating_chains < 1:
                continue

            chain = graph.mating_chain(node_a, node_b)
            if chain is None or chain['N_mu'] == 0:
                continue
            members = set(chain['members'])

            # the containment rule applies to each DESCENDING lineal chain (an out-edge)
            if dir_a == 'up' and not _chain_contains_parent_pair(
                    graph, members, node_a, A, n_a):
                continue
            if dir_b == 'up' and not _chain_contains_parent_pair(
                    graph, members, node_b, B, n_b):
                continue

            # N_o counts out-edges: only chains that DESCEND from the mating chain
            N_o = int(dir_a == 'up') + int(dir_b == 'up')
            sum_N_lin = n_a + n_b

            def case_for(n, d):
                if n == 0:
                    return 'terminal_mating'
                return 'sink' if d == 'up' else 'source'

            rounds = (chain['round_ids'],) if chain['round_ids'] else None
            yield (PathSignature(sum_N_lin, N_o, ((chain['N_mu'], N_o),),
                                 case_for(n_a, dir_a), case_for(n_b, dir_b),
                                 mating_round_ids=rounds),
                   'mating')


def relate(graph: PedigreeGraph, A: int, B: int,
           rho_y: float, rho_g: float, V_A: float = 1.0, V_Y: float = 1.0,
           floor: float = 0.005, max_up: int = 4,
           max_mating_chains: int = 1,
           rho_y_by_round=None,
           return_all: bool = False) -> RelateResult:
    """Finds the dominant path between ``A`` and ``B`` and predicts their covariance.

    Best-first: candidates are expanded in order of decreasing predicted magnitude, so
    the first complete path popped is the dominant one, exactly (see the module
    docstring for why this is sound rather than heuristic).

    Parameters:
        graph: The :class:`PedigreeGraph`.
        A, B: Individual indices.
        rho_y, rho_g: Mate correlations. Pass **measured** values (D-0009).
        V_A, V_Y: Additive and phenotypic variance, measured.
        floor: Prune any path whose predicted PHENOTYPIC CORRELATION falls below this.
            Stated in correlation units rather than as a generation horizon. Report
            unrelated if nothing clears it.
        max_up: How many generations to walk up from each proband.
        max_mating_chains: Number of mating chains a path may cross. Default 1 covers
            every named class in the writeup's table. Per D-0010 this is a parameter
            with **no hard ceiling**, not a cap -- the correlation floor already bounds
            the search.
        rho_y_by_round: Optional mapping from matching-round index to that round's
            realized ``rho_y``. Round k > 1 matches in a smaller pool and attenuates
            slightly, so a chain crossing several rounds is better predicted by the
            product of their realized correlations than by a power of the pooled one.
            Without this the pooled ``rho_y`` is used, which is the writeup's own form.
        return_all: Opt-in diagnostic. Keep popping until the queue is exhausted and
            return every candidate above the floor, so the dominant path can be
            compared against the sum of all of them -- a direct measure of how tight
            ``Dominant-path``'s bound is. Off by default: the normal path stops at the
            first complete path.

    Returns:
        :class:`RelateResult`.
    """
    A, B = int(A), int(B)
    heap = []
    seen = set()
    n_expanded = 0

    for sig, kind in _candidate_signatures(graph, A, B, max_up, max_mating_chains):
        corr = predicted_corr(sig, V_A, V_Y, rho_y, rho_g,
                              rho_y_by_round=rho_y_by_round)
        if corr < floor:
            continue
        key = (sig.key, sig.mating_round_ids)
        if key in seen:
            continue
        seen.add(key)
        # -log of the magnitude: best-first pops the largest |corr| first
        priority = -corr
        heapq.heappush(heap, (priority, len(heap), sig, kind))

    if not heap:
        return RelateResult(A=A, B=B, signature=None, name='unrelated',
                            cov=0.0, corr=0.0, n_expanded=0,
                            all_paths=[] if return_all else None)

    all_paths = [] if return_all else None
    best = None
    while heap:
        neg_corr, _, sig, kind = heapq.heappop(heap)
        n_expanded += 1
        if best is None:
            best = sig
            if not return_all:
                break
        if return_all:
            all_paths.append((sig, -neg_corr))

    # Parameter-free label: shortest by total steps, with fixed tie-breaks, so labels
    # are stable across parameter values. Scored separately from the prediction.
    def label_key(s: PathSignature):
        total_mu = sum(N_mu for N_mu, _ in s.mating_chains)
        return (s.sum_N_lin + total_mu, s.sum_N_lin, total_mu,
                s.n_mating_chains, s.systematic_name())

    all_sigs = [s for (_, _, s, _) in heap] + [best]
    label_sig = min(all_sigs, key=label_key)

    cov = predicted_cov(best, V_A, rho_y, rho_g, phenotypic=True,
                        rho_y_by_round=rho_y_by_round)
    corr = cov / V_Y if V_Y else float('nan')

    return RelateResult(
        A=A, B=B, signature=best, name=signature_name(best),
        cov=float(cov), corr=float(corr),
        label_name=signature_name(label_sig),
        label_agrees=(label_sig.key == best.key),
        n_expanded=n_expanded,
        all_paths=all_paths,
    )


# ---------------------------------------------------------------------------
# Constructive per-class enumeration
# ---------------------------------------------------------------------------

VALID_CLASS_SPECS = (
    'full_sibs', 'half_sibs', 'step_sibs', 'parent_offspring',
    'grandparent', 'avuncular', 'first_cousins', 'parent_parent',
    'married_couple_parents',
)


def enumerate_class(graph: PedigreeGraph, spec: str,
                    limit: Optional[int] = None) -> Iterator[Tuple[int, int]]:
    """Enumerates pairs of a given relationship class directly from local structure.

    For verification you never want all ``N^2`` pairs -- you want a sample per class. So
    the problem is inverted: each class is generated constructively in
    ``O(number of instances)`` rather than by classifying every pair.

    Parameters:
        graph: The :class:`PedigreeGraph`.
        spec: One of :data:`VALID_CLASS_SPECS`.
        limit: Stop after this many pairs.

    Yields:
        ``(A, B)`` index pairs.
    """
    spec = str(spec).strip().lower()
    if spec not in VALID_CLASS_SPECS:
        raise ValueError(f'`spec` must be one of {list(VALID_CLASS_SPECS)}, got {spec!r}.')

    count = 0

    def emit(a, b):
        return int(a), int(b)

    if spec == 'parent_parent':
        for i, j in graph.mate_edges:
            yield emit(i, j)
            count += 1
            if limit and count >= limit:
                return
        return

    if spec == 'parent_offspring':
        for child in range(graph.N):
            for p in graph.parents_of(child):
                yield emit(p, child)
                count += 1
                if limit and count >= limit:
                    return
        return

    if spec == 'grandparent':
        for child in range(graph.N):
            for p in graph.parents_of(child):
                for gp in graph.parents_of(int(p)):
                    yield emit(gp, child)
                    count += 1
                    if limit and count >= limit:
                        return
        return

    if spec == 'full_sibs':
        # group by mate-pair id: sharing it is exactly sharing both parents
        pid = graph.mate_pair_id
        order = np.argsort(pid, kind='stable')
        sorted_pid = pid[order]
        start = 0
        while start < len(order):
            if sorted_pid[start] < 0:
                start += 1
                continue
            end = start
            while end < len(order) and sorted_pid[end] == sorted_pid[start]:
                end += 1
            members = order[start:end]
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    yield emit(members[a], members[b])
                    count += 1
                    if limit and count >= limit:
                        return
            start = end
        return

    if spec == 'half_sibs':
        # an individual with >= 2 mates: cross offspring across different mate pairs
        for i in range(graph.N):
            mates = graph.mates_of(i)
            if len(mates) < 2:
                continue
            kids_by_mate = []
            own = set(int(k) for k in graph.offspring_of(i))
            for m in mates:
                shared = [k for k in own
                          if int(m) in [int(x) for x in graph.parents_of(k)]]
                kids_by_mate.append(shared)
            for a_i in range(len(kids_by_mate)):
                for b_i in range(a_i + 1, len(kids_by_mate)):
                    for ka in kids_by_mate[a_i]:
                        for kb in kids_by_mate[b_i]:
                            yield emit(ka, kb)
                            count += 1
                            if limit and count >= limit:
                                return
        return

    if spec == 'step_sibs':
        # walk 3 mating edges; offspring of the two end mate pairs are step sibs
        for i, j in graph.mate_edges:
            # i--j is the middle mating; step sibs are kids of i's OTHER mate pair
            # and kids of j's OTHER mate pair
            for other_i in graph.mates_of(int(i)):
                if int(other_i) == int(j):
                    continue
                for other_j in graph.mates_of(int(j)):
                    if int(other_j) == int(i):
                        continue
                    kids_a = [k for k in graph.offspring_of(int(i))
                              if int(other_i) in [int(x) for x in graph.parents_of(k)]]
                    kids_b = [k for k in graph.offspring_of(int(j))
                              if int(other_j) in [int(x) for x in graph.parents_of(k)]]
                    for ka in kids_a:
                        for kb in kids_b:
                            if ka == kb:
                                continue
                            yield emit(ka, kb)
                            count += 1
                            if limit and count >= limit:
                                return
        return

    if spec in ('avuncular', 'first_cousins'):
        # up to parents, across their full sibs, then down
        for child in range(graph.N):
            own_parents = {int(x) for x in graph.parents_of(child)}
            for p in graph.parents_of(child):
                pid = graph.mate_pair_id[int(p)]
                if pid < 0:
                    continue
                sibs = np.flatnonzero(graph.mate_pair_id == pid)
                for s in sibs:
                    # Skip either parent. If the two parents are themselves full sibs
                    # (which the mate graph permits unless the close-relative constraint
                    # is on), the other parent appears here as an "aunt/uncle" and its
                    # offspring include `child` itself.
                    if int(s) in own_parents:
                        continue
                    if spec == 'avuncular':
                        yield emit(s, child)
                        count += 1
                        if limit and count >= limit:
                            return
                    else:
                        for cousin in graph.offspring_of(int(s)):
                            if int(cousin) == int(child):
                                continue
                            yield emit(cousin, child)
                            count += 1
                            if limit and count >= limit:
                                return
        return

    if spec == 'married_couple_parents':
        # the parents of one member of a mate pair vs the parents of the other
        for i, j in graph.mate_edges:
            for pa in graph.parents_of(int(i)):
                for pb in graph.parents_of(int(j)):
                    yield emit(pa, pb)
                    count += 1
                    if limit and count >= limit:
                        return
        return
