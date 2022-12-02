def jaccard_h(s1, s2):
    """ Jaccard similarity score.  (1 - jaccard) is a valid distance metric """
    a = set(s1)
    b = set(s2)
    if len(a) == 0 and len(b) == 0:
        return 0
    return len(a & b) / len(a | b)


# N-gram function courtesy of Peter Norvig
def ngrams(seq, n):
    """List all the (overlapping) ngrams in a sequence."""
    return [seq[i:i + n] for i in range(1 + len(seq) - n)]

