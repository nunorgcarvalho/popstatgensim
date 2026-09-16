"""Local-only pedigree graph: O(N) per generation, relationships computed on demand.

The previous ``Pedigree``/``PedPath`` layer materialized a dict over all related pairs
and encoded a relationship as a tuple of signed meiosis steps. That had two fatal
problems for this effort (D-0007):

1. **Mating steps were unrepresentable.** A signed-meiosis encoding has no mating edge,
   so step-siblings, parent--parent and parents-of-a-married-couple came out as
   *unrelated* -- five of the eleven rows of the writeup's relationship table were
   invisible.
2. **It did not scale.** At N = 10,000 over 4 generations the all-pairs dict is ~10^9
   entries. That is the wrong data structure, not a tuning problem.

This module stores only local structure and computes relationships on demand:

    parents[i]     = (mother, father)
    offspring[i]   = [...]          (CSR, the reverse index)
    mates[i]       = [...]          (CSR, from the mate edge list)
    generation[i]  = t

Dropping old generations then works naturally: because the state is purely local,
dropping generation ``t - k`` truncates the ancestor walk, and a path through a dropped
generation is simply unreachable and reports unrelated.

Deliberately no networkx: at N in the thousands with per-query traversals its
Python-object overhead dominates, and plain numpy/CSR adjacency with hand-written
traversals is both faster and one fewer dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


def _build_csr(sources: np.ndarray, targets: np.ndarray, N: int):
    """Builds a CSR adjacency (indptr, indices) from parallel edge endpoint arrays."""
    sources = np.asarray(sources, dtype=np.int64).ravel()
    targets = np.asarray(targets, dtype=np.int32).ravel()
    if sources.size == 0:
        return np.zeros(N + 1, dtype=np.int64), np.zeros(0, dtype=np.int32)
    order = np.argsort(sources, kind='stable')
    indices = targets[order]
    counts = np.bincount(sources[order], minlength=N)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return indptr, indices


@dataclass
class PedigreeGraph:
    """Local pedigree structure for one or more generations, flattened into one index space.

    Individuals are numbered ``0 .. N-1``. Generation 0 is the OLDEST retained
    generation, so ``generation[i]`` increases toward the present.

    Attributes:
        N: Total number of individuals.
        parents: ``(N, 2)`` int32; ``-1`` where the parent is outside the retained set.
        generation: ``(N,)`` int32 generation index.
        mate_edges: ``(U, 2)`` int32 mate pairs.
        mate_round_id: ``(U,)`` int32, which matching round each mate edge came from.
        mate_pair_id: ``(N,)`` int32; the mate pair an individual descends from, or -1.
            This is ``relations['full_sibs']``, which generalizes correctly: two
            individuals are full sibs iff they share a mate-pair id, and half sibs iff
            they share exactly one parent but not a mate-pair id.
    """

    N: int
    parents: np.ndarray
    generation: np.ndarray
    mate_edges: np.ndarray
    mate_round_id: np.ndarray
    mate_pair_id: np.ndarray

    def __post_init__(self):
        self.parents = np.asarray(self.parents, dtype=np.int32).reshape(self.N, 2)
        self.generation = np.asarray(self.generation, dtype=np.int32).ravel()
        self.mate_edges = np.asarray(self.mate_edges, dtype=np.int32).reshape(-1, 2)
        self.mate_round_id = np.asarray(self.mate_round_id, dtype=np.int32).ravel()
        self.mate_pair_id = np.asarray(self.mate_pair_id, dtype=np.int32).ravel()

        # reverse index: offspring of each individual
        valid = self.parents >= 0
        src = np.concatenate([self.parents[:, 0][valid[:, 0]],
                              self.parents[:, 1][valid[:, 1]]])
        dst = np.concatenate([np.flatnonzero(valid[:, 0]), np.flatnonzero(valid[:, 1])])
        self._off_indptr, self._off_indices = _build_csr(src, dst, self.N)

        # mate adjacency, symmetric
        if self.mate_edges.shape[0] > 0:
            msrc = np.concatenate([self.mate_edges[:, 0], self.mate_edges[:, 1]])
            mdst = np.concatenate([self.mate_edges[:, 1], self.mate_edges[:, 0]])
            medge = np.concatenate([np.arange(self.mate_edges.shape[0]),
                                    np.arange(self.mate_edges.shape[0])])
            order = np.argsort(msrc, kind='stable')
            self._mate_indices = mdst[order].astype(np.int32, copy=False)
            self._mate_edge_of = medge[order].astype(np.int32, copy=False)
            counts = np.bincount(msrc[order], minlength=self.N)
            self._mate_indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        else:
            self._mate_indptr = np.zeros(self.N + 1, dtype=np.int64)
            self._mate_indices = np.zeros(0, dtype=np.int32)
            self._mate_edge_of = np.zeros(0, dtype=np.int32)

    # -- local lookups, all O(degree) -------------------------------------------------

    def offspring_of(self, i: int) -> np.ndarray:
        return self._off_indices[self._off_indptr[i]:self._off_indptr[i + 1]]

    def mates_of(self, i: int) -> np.ndarray:
        return self._mate_indices[self._mate_indptr[i]:self._mate_indptr[i + 1]]

    def mate_edges_of(self, i: int) -> np.ndarray:
        """Edge ids of ``i``'s mate edges, parallel to :meth:`mates_of`."""
        return self._mate_edge_of[self._mate_indptr[i]:self._mate_indptr[i + 1]]

    def parents_of(self, i: int) -> np.ndarray:
        p = self.parents[i]
        return p[p >= 0]

    def n_mates(self) -> np.ndarray:
        return np.diff(self._mate_indptr).astype(np.int32, copy=False)

    # -- construction -----------------------------------------------------------------

    @classmethod
    def from_population(cls, pop, generations: Optional[int] = None) -> 'PedigreeGraph':
        """Builds the graph from a Population and its retained past generations.

        Walks ``pop.past`` back as far as it goes (or ``generations`` steps), and
        flattens into a single index space with the OLDEST retained generation first.
        Parent indices that point outside the retained set become ``-1``, which is
        exactly the truncation D-0007 relies on: a path through a dropped generation is
        unreachable and reports unrelated.
        """
        chain = [pop]
        past = getattr(pop, 'past', None) or []
        k = 1
        while k < len(past) and past[k] is not None:
            chain.append(past[k])
            k += 1
            if generations is not None and len(chain) > generations:
                break
        chain = list(reversed(chain))  # oldest first

        sizes = [p.N for p in chain]
        offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        N = int(offsets[-1])

        parents = np.full((N, 2), -1, dtype=np.int32)
        generation = np.zeros(N, dtype=np.int32)
        mate_pair_id = np.full(N, -1, dtype=np.int32)
        edge_blocks, round_blocks = [], []
        next_pair = 0

        for g, p in enumerate(chain):
            lo = int(offsets[g])
            generation[lo:lo + p.N] = g

            # mate edges live on the generation that mated
            e = np.asarray(p.relations.get('mate_edges', np.zeros((0, 2))),
                           dtype=np.int32).reshape(-1, 2)
            if e.shape[0] > 0:
                edge_blocks.append(e + lo)
                r = np.asarray(p.relations.get('mate_round_id', np.zeros(e.shape[0])),
                               dtype=np.int32).ravel()
                round_blocks.append(r if r.shape[0] == e.shape[0]
                                    else np.zeros(e.shape[0], dtype=np.int32))

            # parents point into the PREVIOUS generation in the chain
            par = np.asarray(p.relations.get('parents', np.full((p.N, 2), -1)),
                             dtype=np.int32).reshape(-1, 2)
            if g > 0 and par.shape[0] == p.N:
                prev_lo = int(offsets[g - 1])
                prev_N = sizes[g - 1]
                ok = (par >= 0) & (par < prev_N)
                shifted = np.full_like(par, -1)
                shifted[ok] = par[ok] + prev_lo
                parents[lo:lo + p.N] = shifted

            # mate-pair ids ('full_sibs'), renumbered globally
            fam = np.asarray(p.relations.get('full_sibs', np.full(p.N, -1)),
                             dtype=np.int32).ravel()
            if fam.shape[0] == p.N and np.any(fam >= 0):
                valid = fam >= 0
                _, inverse = np.unique(fam[valid], return_inverse=True)
                block = np.full(p.N, -1, dtype=np.int32)
                block[valid] = inverse + next_pair
                next_pair = int(block[valid].max()) + 1
                mate_pair_id[lo:lo + p.N] = block

        mate_edges = (np.concatenate(edge_blocks, axis=0).astype(np.int32, copy=False)
                      if edge_blocks else np.zeros((0, 2), dtype=np.int32))
        mate_round_id = (np.concatenate(round_blocks).astype(np.int32, copy=False)
                         if round_blocks else np.zeros(0, dtype=np.int32))

        return cls(N=N, parents=parents, generation=generation,
                   mate_edges=mate_edges, mate_round_id=mate_round_id,
                   mate_pair_id=mate_pair_id)

    # -- traversal used by the search -------------------------------------------------

    def ancestors(self, i: int, max_up: int = 4) -> dict:
        """Maps ancestor index -> number of meioses up from ``i`` (0 for ``i`` itself).

        Keeps the SHORTEST distance to each ancestor. Under ``No-inbreeding`` an
        ancestor is reached only one way, so the distinction rarely matters; keeping the
        shortest is the right call when it does, since it is the dominant path.
        """
        out = {int(i): 0}
        frontier = [int(i)]
        for depth in range(1, max_up + 1):
            nxt = []
            for node in frontier:
                for p in self.parents_of(node):
                    p = int(p)
                    if p not in out:
                        out[p] = depth
                        nxt.append(p)
            if not nxt:
                break
            frontier = nxt
        return out

    def mating_chain(self, a: int, b: int, max_len: int = 6) -> Optional[dict]:
        """Finds the mating chain between ``a`` and ``b``, or None if there is none.

        Under ``No-mating-loops`` the mating graph is a forest, so this chain is
        **unique** when it exists and a BFS finds it exactly. Returns a dict with
        ``N_mu``, the ``members`` along the chain, and the ``round_ids`` of the edges
        crossed -- the rounds matter because round k > 1 can realize a different
        ``rho_y``.
        """
        a, b = int(a), int(b)
        if a == b:
            return {'N_mu': 0, 'members': (a,), 'round_ids': ()}
        # BFS over mate edges
        prev = {a: (None, None)}
        frontier = [a]
        for _ in range(max_len):
            nxt = []
            for node in frontier:
                mates = self.mates_of(node)
                edges = self.mate_edges_of(node)
                for m, e in zip(mates, edges):
                    m = int(m)
                    if m in prev:
                        continue
                    prev[m] = (node, int(e))
                    if m == b:
                        # reconstruct
                        members, rounds = [b], []
                        cur = b
                        while prev[cur][0] is not None:
                            p, e_id = prev[cur]
                            rounds.append(int(self.mate_round_id[e_id])
                                          if e_id < len(self.mate_round_id) else 0)
                            members.append(int(p))
                            cur = p
                        members.reverse()
                        rounds.reverse()
                        return {'N_mu': len(rounds), 'members': tuple(members),
                                'round_ids': tuple(rounds)}
                    nxt.append(m)
            if not nxt:
                break
            frontier = nxt
        return None

    def mate_graph_is_forest(self) -> bool:
        """Whether the mating graph is acyclic, i.e. whether chain uniqueness holds."""
        parent = np.arange(self.N, dtype=np.int32)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, j in self.mate_edges:
            ri, rj = find(int(i)), find(int(j))
            if ri == rj:
                return False
            parent[rj] = ri
        return True
