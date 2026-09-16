"""Legacy relationship-type lookup.

.. deprecated::
    Superseded by the signature-based naming in ``pedigree.relate``
    (``KNOWN_SIGNATURES`` plus ``PathSignature.systematic_name()``), per D-0007.

    Two reasons, not one. The obvious one is that ``full_half`` was inferred from
    ``up_last in {1,2}`` vs ``3`` -- a proxy that had never been exercised, because
    monogamy makes every sibling connection run through both parents, so ``up_last``
    was always 3 and the ``half`` branch was dead code. That proxy is in fact sound
    in principle (step code 3 = both parents = full; 1 or 2 = one parent = half).

    The deeper reason is that the encoding cannot represent a mating STEP at all, so
    step siblings and step-step siblings have no representation here regardless of
    whether the full/half proxy works. The replacement distinguishes the four
    sibling classes by ``N_mu`` within a single mating chain (1, 2, 3, 4), which is
    the writeup's own vocabulary.
"""

# Note: a path from A-->B is classified as the relationship of B with respect to A. So, e.g., if A-->B is [+1], then that path is classified as 'parent', because B is the parent of A.
# within each attribute, the keys are:
# - sigs: dictionary of signature constraints to match. If value is a single integer, it is treated as an exact match. If value is a list, it is treated as a set of possible values. If value is a tuple of length 2, it is treated as a range (inclusive). A missing signature attribute means it is not constrained.
# - fh: Whether distinguishing full vs half relatives is relevant. Default (e.g. if missing) is True. Only applicable to 'relationship' attribute.
REL_TYPES = {
    'direction' : { # direct ancestors vs descendants
        'direct_ancestor':      { 'sigs': {'ups': (1,9999), 'downs': 0} },
        'direct_descendant':    { 'sigs': {'ups': 0, 'downs': (1,9999)} },
        'N/A':                  { 'sigs': {} }
    },
    'full_half' : { # full vs half relatives
        'full': { 'sigs': {'up_last': 3 } },
        'half': { 'sigs': {'up_last': (1,2)} },
        'N/A':  { 'sigs': {'up_last':-9} }
    },
    'parental_line' : { # side of family
        'maternal': { 'sigs': {'up_first': 1} },
        'paternal': { 'sigs': {'up_first': 2} },
        'both':     { 'sigs': {'up_first': 3} },
        'N/A':      { 'sigs': {'up_first':-9} } # for self
    },
    'relationship' : { # broad relationship types
        # deg 0
        'self':         { 'sigs': {'ups': 0, 'downs': 0}, 'fh': False },
        # deg 1
        'parent':       { 'sigs': {'ups': 1, 'downs': 0}, 'fh': False },
        'offspring':    { 'sigs': {'ups': 0, 'downs': 1}, 'fh': False },
        'sibling':      { 'sigs': {'ups': 1, 'downs': 1} },
        # deg 2
        'uncle/aunt':   { 'sigs': {'ups': 2, 'downs': 1} },
        'nephew/niece': { 'sigs': {'ups': 1, 'downs': 2} },
        'grandparent':  { 'sigs': {'ups': 2, 'downs': 0}, 'fh': False },
        'grandchild':   { 'sigs': {'ups': 0, 'downs': 2}, 'fh': False },
        'double_1cousin': { 'sigs': {'ups': 2, 'downs': 2, 'up_first': 3, 'up_last': 3} },
        # deg 3
        '1cousin': { 'sigs': {'ups': 2, 'downs': 2} },
        'g1_grandparent': { 'sigs': {'ups': 3, 'downs': 0}, 'fh': False },
        'g1_grandchild': { 'sigs': {'ups': 0, 'downs': 3}, 'fh': False },
        'g1_uncle/aunt': { 'sigs': {'ups': 3, 'downs': 1} },
        'g1_nephew/niece': { 'sigs': {'ups': 1, 'downs': 3} },
        # deg 4
        '1cousin_1removed': { 'sigs': {'ups': 2, 'downs': 3} },
        'parent_1cousin': { 'sigs': {'ups': 3, 'downs': 2} },
        'g2_grandparent': { 'sigs': {'ups': 4, 'downs': 0}, 'fh': False },
        'g2_grandchild': { 'sigs': {'ups': 0, 'downs': 4}, 'fh': False },
        'g2_uncle/aunt': { 'sigs': {'ups': 4, 'downs': 1} },
        'g2_nephew/niece': { 'sigs': {'ups': 1, 'downs': 4} },
        # deg 5
        '2cousin': { 'sigs': {'ups': 3, 'downs': 3} },
        # Catch-all, always leave at the end
        'unspecified': {'sigs': {} }
    }
}
