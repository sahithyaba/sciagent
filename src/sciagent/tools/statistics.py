import numpy as np
from scipy import stats

def summary_statistics(series):
    s = series.dropna()
    return {
        "n": int(s.size),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)),
        "median": float(s.median()),
        "min": float(s.min()),
        "max": float(s.max()),
    }

def pearson_correlation(x, y):
    mask = x.notna() & y.notna()
    r, p = stats.pearsonr(x[mask], y[mask])
    return {"r": float(r), "p_value": float(p), "n": int(mask.sum())}

def welch_t_test(a, b):
    a, b = a.dropna(), b.dropna()
    result = stats.ttest_ind(a, b, equal_var=False)
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
    }
