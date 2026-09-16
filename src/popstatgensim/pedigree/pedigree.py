"""Unpickling shim for the removed ``PedPath`` pedigree layer.

The all-pairs pedigree layer is **gone**. It stored a relationship as a ``PedPath`` --
a tuple of signed meiosis steps -- in a dict over every related pair, and it had two
problems that no amount of tuning fixes (D-0007):

1. **A signed-meiosis encoding has no mating edge.** Step-siblings, parent--parent and
   parents-of-a-married-couple were therefore recorded as *unrelated*: five of the
   eleven rows of the ``relative_covariance`` writeup's relationship table were
   structurally invisible.
2. **It did not scale.** The path dict is ~10^9 entries at N = 10,000 over 4
   generations, and building it cost 35-64x an entire simulation step at N = 200-800.

It is replaced by :mod:`popstatgensim.pedigree.graph` (local-only state, ``O(N)`` per
generation) plus :mod:`popstatgensim.pedigree.relate` (relationships computed on demand,
and a signature-to-name lookup that replaces the old ``REL_TYPES`` table).

A note so it is not re-litigated later: the old ``full_half`` proxy in
``relative_types`` -- inferring full vs half siblings from ``up_last in {1, 2}`` versus
``3`` -- was found to be **sound in principle** (step code 3 means the path runs through
both parents, 1 or 2 through a single parent). It had simply never been exercised,
because monogamous mating makes every sibling link run through both parents. It was
replaced for *representational* reasons, not because it was wrong: the encoding cannot
express a mating step at all, so step and step-step siblings have no representation in
it regardless of whether the proxy works.

## Why this shim exists

``Population.__init__`` used to assign ``self.ped = Pedigree(self.N)`` unconditionally --
for **every** population, whether or not ``track_pedigree`` was set. So every
``Population`` pickled before the removal embeds a reference to this class, and deleting
it outright would make those files unloadable. Since the popstatgenwriteups generate/
verify split keeps ``Population`` bundles on scratch, that would invalidate real compute.

Unpickling does not call ``__init__``, so this empty class is enough to restore such a
file. The restored ``.ped`` attribute carries whatever the old object held and **nothing
reads it** -- the live code path never touches it. New populations do not create one at
all; use ``pop.pedigree_graph``.

This shim can be deleted once no pre-removal bundles remain.
"""

from typing import Tuple

#: Compact key for a pair of individuals, kept for old pickles.
PedKey2 = Tuple[int, int]


class Pedigree:
    """Empty stand-in so pre-removal ``Population`` pickles still load.

    Carries no behaviour. Anything reaching for the old path machinery should use
    ``pop.pedigree_graph`` with :func:`popstatgensim.pedigree.relate.relate` and
    :func:`popstatgensim.pedigree.relate.enumerate_class` instead.
    """

    __slots__ = ('__dict__',)

    def __init__(self, *args, **kwargs):
        raise TypeError(
            'The PedPath pedigree layer has been removed; Pedigree exists only so that '
            'Populations pickled before the removal can still be loaded. Use '
            'pop.pedigree_graph (a PedigreeGraph) with pedigree.relate() and '
            'pedigree.enumerate_class() instead.'
        )

    def __repr__(self):
        return '<Pedigree (removed layer; unpickling shim only)>'
