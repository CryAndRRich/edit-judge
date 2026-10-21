"""Pure binary metrics and measured latency summaries."""
import math


def metrics(truth, labels):
    if len(truth) != len(labels) or not truth:
        raise ValueError('Nonempty aligned vectors required')
    if any(x not in (0, 1) for x in list(truth)+list(labels)):
        raise ValueError('Binary labels required')
    tp = sum(y == 1 and p == 1 for y, p in zip(truth, labels))
    tn = sum(y == 0 and p == 0 for y, p in zip(truth, labels))
    fp = sum(y == 0 and p == 1 for y, p in zip(truth, labels))
    fn = sum(y == 1 and p == 0 for y, p in zip(truth, labels))
    div = lambda a, b: a/b if b else 0.0
    return dict(n=len(truth), TP=tp, TN=tn, FP=fp, FN=fn, accuracy=(tp+tn)/len(truth),
                precision=div(tp, tp+fp), recall=div(tp, tp+fn), F1=div(2*tp, 2*tp+fp+fn), FPR=div(fp, fp+tn))


def percentile(values, q):
    values = sorted(values)
    index = (len(values)-1)*q
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi]-values[lo])*(index-lo)


def latency(values, prefix='e2e'):
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('Positive finite measured latency required')
    mean = sum(values)/len(values)
    return {f'{prefix}_mean_ms': mean, f'{prefix}_p50_ms': percentile(values, .5),
            f'{prefix}_p95_ms': percentile(values, .95), f'{prefix}_throughput_images_s': 1000/mean}


def majority_vote(labels):
    if len(labels) != 3 or any(len(x) != len(labels[0]) for x in labels):
        raise ValueError('Exactly three aligned models required')
    return [int(sum(x) >= 2) for x in zip(*labels)]
