import pandas as pd
import numpy as np

def smith_waterman_similarity(s1,
                              s2,
                              match=5,
                              mismatch=-5,
                              gap_start=-5,
                              gap_continue=-1,
                              norm="mean"):
    """Smith-Waterman string comparison.
    An implementation of the Smith-Waterman string comparison algorithm
    described in Christen, Peter (2012).
    Parameters
    ----------
    s1 : label, pandas.Series
        Series or DataFrame to compare all fields.
    s2 : label, pandas.Series
        Series or DataFrame to compare all fields.
    match : float
        The value added to the match score if two characters match.
        Greater than mismatch, gap_start, and gap_continue. Default: 5.
    mismatch : float
        The value added to the match score if two characters do not match.
        Less than match. Default: -5.
    gap_start : float
        The value added to the match score upon encountering the start of
        a gap. Default: -5.
    gap_continue : float
        The value added to the match score for positions where a previously
        started gap is continuing. Default: -1.
    norm : str
        The name of the normalization metric to be used. Applied by dividing
        the match score by the normalization metric multiplied by match. One
        of "min", "max",or "mean". "min" will use the minimum string length
        as the normalization metric. "max" and "mean" use the maximum and
        mean string length respectively. Default: "mean""
    Returns
    -------
    pandas.Series
        A pandas series with similarity values. Values equal or between 0
        and 1.
    """

    # Assert that match is greater than or equal to mismatch, gap_start, and
    # gap_continue.
    assert match >= max(mismatch, gap_start, gap_continue), \
        "match must be greater than or equal to mismatch, " \
        "gap_start, and gap_continue"

    if len(s1) != len(s2):
        raise ValueError('Arrays or Series have to be same length.')

    if len(s1) == len(s2) == 0:
        return []

    concat = pd.Series(list(zip(s1, s2)))

    def sw_apply(t):
        """
        sw_apply(t)
        A helper function that is applied to each pair of records
        in s1 and s2. Assigns a similarity score to each pair,
        between 0 and 1. Used by the pandas.apply method.
        Parameters
        ----------
        t : pandas.Series
            A pandas Series containing two strings to be compared.
        Returns
        -------
        Float
            A similarity score between 0 and 1.
        """
        str1 = t[0]
        str2 = t[1]

        def compute_score():
            """
            compute_score()
            The helper function that produces the non-normalized
            similarity score between two strings. The scores are
            determined using the Smith-Waterman dynamic programming
            algorithm. The scoring scheme is determined from the
            parameters provided to the parent smith_waterman_similarity()
            function.
            Returns
            -------
            Float
                A score 0 or greater. Indicates similarity between two strings.
            """

            # Initialize the score matrix with 0s

            m = [[0] * (1 + len(str2)) for i in range(1 + len(str1))]

            # Initialize the trace matrix with empty lists
            trace = [[[] for _ in range(1 + len(str2))]
                     for _ in range(1 + len(str1))]

            # Initialize the highest seen score to 0
            highest = 0

            # Iterate through the matrix
            for x in range(1, 1 + len(str1)):
                for y in range(1, 1 + len(str2)):
                    # Calculate Diagonal Score
                    if str1[x - 1] == str2[y - 1]:
                        # If characters match, add the match score to the
                        # diagonal score
                        diagonal = m[x - 1][y - 1] + match
                    else:
                        # If characters do not match, add the mismatch score
                        # to the diagonal score
                        diagonal = m[x - 1][y - 1] + mismatch

                    # Calculate the Left Gap Score
                    if "H" in trace[x - 1][y]:
                        # If cell to the left's score was calculated based on
                        # a horizontal gap, add the gap continuation penalty
                        # to the left score.
                        gap_horizontal = m[x - 1][y] + gap_continue
                    else:
                        # Otherwise, add the gap start penalty to the left
                        # score
                        gap_horizontal = m[x - 1][y] + gap_start

                    # Calculate the Above Gap Score
                    if "V" in trace[x][y - 1]:
                        # If above cell's score was calculated based on a
                        # vertical gap, add the gap continuation penalty to
                        # the above score.
                        gap_vertical = m[x][y - 1] + gap_continue
                    else:
                        # Otherwise, add the gap start penalty to the above
                        # score
                        gap_vertical = m[x][y - 1] + gap_start

                    # Choose the highest of the three scores
                    score = max(diagonal, gap_horizontal, gap_vertical)

                    if score <= 0:
                        # If score is less than 0, boost to 0
                        score = 0
                    else:
                        # If score is greater than 0, determine whether it was
                        # calculated based on a diagonal score, horizontal gap,
                        # or vertical gap. Store D, H, or V in the trace matrix
                        # accordingly.
                        if score == diagonal:
                            trace[x][y].append("D")
                        if score == gap_horizontal:
                            trace[x][y].append("H")
                        if score == gap_vertical:
                            trace[x][y].append("V")

                    # If the cell's score is greater than the highest score
                    # previously present, record the score as the highest.
                    if score > highest:
                        highest = score

                    # Set the cell's score to score
                    m[x][y] = score

            # After iterating through the entire matrix, return the highest
            # score found.
            return highest

        def normalize(score):
            """
            normalize(score)
            A helper function used to normalize the score produced by
            compute_score() to a score between 0 and 1. The method for
            normalization is determined by the norm argument provided
            to the parent, smith_waterman_similarity function.
            Parameters
            ----------
            score : Float
                The score produced by the compute_score() function.
            Returns
            -------
            Float
                A normalized score between 0 and 1.
            """
            if norm == "min":
                # Normalize by the shorter string's length
                return score / (min(len(str1), len(str2)) * match)
            if norm == "max":
                # Normalize by the longer string's length
                return score / (max(len(str1), len(str2)) * match)
            if norm == "mean":
                # Normalize by the mean length of the two strings
                return 2 * score / ((len(str1) + len(str2)) * match)
            else:
                warnings.warn(
                    'Unrecognized longest common substring normalization. '
                    'Defaulting to "mean" method.')
                return 2 * score / ((len(str1) + len(str2)) * match)

        try:
            if len(str1) == 0 or len(str2) == 0:
                return 0
            return normalize(compute_score())

        except Exception as err:
            if pd.isnull(t[0]) or pd.isnull(t[1]):
                return np.nan
            else:
                raise err

    return concat.apply(sw_apply)
