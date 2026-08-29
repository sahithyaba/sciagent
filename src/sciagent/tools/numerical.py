import numpy as np

def z_score(series):
    values = series.astype(float)
    mean = values.mean()
    std = values.std(ddof=0)
    if std == 0:
        return np.zeros(len(values))
    return (values - mean) / std

def detect_zscore_anomalies(series, threshold=3.0):
    z = z_score(series)
    return {
        "threshold": float(threshold),
        "anomaly_count": int((np.abs(z) > threshold).sum()),
        "anomaly_indices": np.flatnonzero(np.abs(z) > threshold).tolist(),
    }
