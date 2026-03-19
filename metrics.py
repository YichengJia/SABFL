# metrics.py
# Lightweight metrics for intent classification and explanation generation.

from typing import List, Sequence, Tuple, Union
import math
from collections import Counter, defaultdict
import numpy as np

TextOrTokens = Union[str, List[str]]


def tri_objective_score(
    accuracy: float,
    communication_mb: float,
    latency_sec: float,
    comm_budget_mb: float,
    latency_budget_sec: float,
    w_acc: float = 0.6,
    w_comm: float = 0.2,
    w_lat: float = 0.2,
) -> float:
    """
    Accuracy-Communication-Latency composite score in [0,1] (approximately).

    score = w_acc * acc + w_comm * comm_term + w_lat * lat_term
    where comm_term = max(0, 1 - comm / budget),
          lat_term  = max(0, 1 - lat / budget).
    """
    comm_term = max(0.0, 1.0 - float(communication_mb) / max(float(comm_budget_mb), 1e-9))
    lat_term = max(0.0, 1.0 - float(latency_sec) / max(float(latency_budget_sec), 1e-9))
    return float(w_acc * float(accuracy) + w_comm * comm_term + w_lat * lat_term)


def within_budgets(
    communication_mb: float,
    latency_sec: float,
    comm_budget_mb: float,
    latency_budget_sec: float,
) -> bool:
    """Return whether a run satisfies both communication and latency budgets."""
    return float(communication_mb) <= float(comm_budget_mb) and float(latency_sec) <= float(latency_budget_sec)

def macro_f1(preds: Sequence[int], golds: Sequence[int], num_classes: int = None) -> float:
    """
    Macro-F1 for multi-class classification.

    preds/golds: list of integer class ids (same label space).
    num_classes: if provided, use labels 0..num_classes-1; otherwise union of preds/golds.
    """
    assert len(preds) == len(golds)
    if num_classes is not None:
        labels = list(range(num_classes))
    else:
        labels = sorted(set(golds) | set(preds))

    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for p, g in zip(preds, golds):
        if p == g:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1

    f1s = []
    for c in labels:
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        recall    = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s)/len(f1s) if f1s else 0.0


def _ngrams(tokens: List[str], n: int) -> Counter:
    if n <= 0:
        return Counter()
    L = len(tokens)
    if L < n:
        return Counter()
    return Counter(tuple(tokens[i:i+n]) for i in range(L - n + 1))


def _ensure_tokenized(items: List[TextOrTokens], lowercase: bool = True) -> List[List[str]]:
    out: List[List[str]] = []
    for x in items:
        if isinstance(x, str):
            s = x.strip()
            if lowercase:
                s = s.lower()
            out.append(s.split())
        else:
            # assume already a list of tokens
            out.append([t.lower() for t in x] if lowercase else list(x))
    return out


def corpus_bleu(
    references: List[TextOrTokens],
    hypotheses: List[TextOrTokens],
    max_n: int = 4,
    smooth: bool = True,
    lowercase: bool = True
) -> float:
    """
    Corpus-level BLEU-N (default BLEU-4) with simple add-one smoothing.
    Accepts either strings (auto-tokenized by whitespace) or token lists.

    references: list[str|List[str]]
    hypotheses: list[str|List[str]] (same length as references)
    """
    assert len(references) == len(hypotheses), "refs & hyps length mismatch"
    if len(references) == 0:
        return 0.0

    refs = _ensure_tokenized(references, lowercase=lowercase)
    hyps = _ensure_tokenized(hypotheses, lowercase=lowercase)

    weights = [1.0/max_n]*max_n if max_n > 0 else []

    # Modified precision with (optional) add-one smoothing
    p_ns = []
    for n in range(1, max_n+1):
        clipped_count = 0
        total_count = 0
        for ref, hyp in zip(refs, hyps):
            ref_ngrams = _ngrams(ref, n)
            hyp_ngrams = _ngrams(hyp, n)
            total_count += sum(hyp_ngrams.values())
            if hyp_ngrams:
                for ng, cnt in hyp_ngrams.items():
                    clipped_count += min(cnt, ref_ngrams.get(ng, 0))
        if smooth:
            # add-one smoothing on counts
            p = (clipped_count + 1.0) / (total_count + 1.0)
        else:
            p = (clipped_count / total_count) if total_count > 0 else 0.0
        # protect from exactly 0 to avoid -inf
        p_ns.append(max(p, 1e-12))

    # Brevity penalty
    ref_len = sum(len(r) for r in refs)
    hyp_len = sum(len(h) for h in hyps)
    if hyp_len == 0:
        return 0.0
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)

    # Geometric mean in log-space
    s = 0.0
    for w, p in zip(weights, p_ns):
        s += w * math.log(p)
    bleu = bp * math.exp(s)
    # Clip to [0,1]
    return max(0.0, min(1.0, float(bleu)))


def pareto_front_mask(
    accuracy: Sequence[float],
    communication_mb: Sequence[float],
    latency_sec: Sequence[float],
) -> List[bool]:
    """
    Mark Pareto-optimal points under:
      - maximize accuracy
      - minimize communication
      - minimize latency
    """
    acc = np.asarray(accuracy, dtype=np.float64)
    comm = np.asarray(communication_mb, dtype=np.float64)
    lat = np.asarray(latency_sec, dtype=np.float64)
    n = len(acc)
    mask = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dominates = (
                (acc[j] >= acc[i]) and
                (comm[j] <= comm[i]) and
                (lat[j] <= lat[i]) and
                ((acc[j] > acc[i]) or (comm[j] < comm[i]) or (lat[j] < lat[i]))
            )
            if dominates:
                mask[i] = False
                break
    return mask.tolist()
