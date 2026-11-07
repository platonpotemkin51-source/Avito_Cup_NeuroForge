"""Competition binary NDCG with pessimistic ties; standard library only."""
import math

def ndcg(targets, scores, cutoff=None):
    targets, scores = list(targets), list(scores)
    if not targets or len(targets) != len(scores):
        raise ValueError("Nonempty equal-length arrays required")
    if any(y not in (0, 1) for y in targets) or any(not math.isfinite(s) for s in scores):
        raise ValueError("Binary targets and finite scores required")
    if cutoff is not None and (not isinstance(cutoff, int) or cutoff < 1):
        raise ValueError("Positive integer cutoff required")
    ranked = sorted(zip(targets, scores), key=lambda pair: (-pair[1], pair[0]))
    limit = min(cutoff or len(targets), len(targets))
    dcg = sum(ranked[i][0] / math.log2(i + 2) for i in range(limit))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(sum(targets), limit)))
    return dcg / ideal if ideal else 1.0

def validate_rows(rows):
    result = {}
    for row in rows:
        key = (row['feed_id'], row['item_id'])
        if any(not isinstance(k, int) or isinstance(k, bool) or not -(2**31) <= k < 2**31 for k in key):
            raise ValueError("Keys must be int32-compatible integers")
        if key in result:
            raise ValueError("Duplicate prediction key")
        score = row['score']
        if score is None or not math.isfinite(score):
            raise ValueError("Nonfinite or missing score")
        result[key] = float(score)
    if not result:
        raise ValueError("Empty predictions")
    return result
