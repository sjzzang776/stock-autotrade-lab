"""지속 호몰로지 특징 (Gidea & Katz 2018 방식).

점구름 = 최근 window 거래일 × 종목 로그수익률 (각 날짜가 종목 수 차원의 점 하나).
"""
import numpy as np
from ripser import ripser


def landscape_norms(diagram, layers=5, grid=100):
    d = diagram[np.isfinite(diagram[:, 1])] if len(diagram) else diagram
    if len(d) == 0:
        return 0.0, 0.0
    b, de = d[:, 0], d[:, 1]
    t = np.linspace(b.min(), de.max(), grid)
    tents = np.maximum(0.0, np.minimum(t[None, :] - b[:, None], de[:, None] - t[None, :]))
    k = min(layers, len(d))
    lam = -np.sort(-tents, axis=0)[:k]
    dt = t[1] - t[0] if grid > 1 else 0.0
    l1 = float(np.abs(lam).sum() * dt)
    l2 = float(np.sqrt((lam ** 2).sum() * dt))
    return l1, l2


def persistence_entropy(diagram):
    d = diagram[np.isfinite(diagram[:, 1])] if len(diagram) else diagram
    life = d[:, 1] - d[:, 0] if len(d) else np.array([])
    life = life[life > 0]
    if len(life) == 0:
        return 0.0
    p = life / life.sum()
    return float(-(p * np.log(p)).sum())


def total_persistence(diagram):
    d = diagram[np.isfinite(diagram[:, 1])] if len(diagram) else diagram
    return float((d[:, 1] - d[:, 0]).sum()) if len(d) else 0.0


def window_features(points, layers=5, grid=100):
    """points: (window, n_assets) → [H1 L1, H1 L2, H1 엔트로피, H0 총 지속성]"""
    dgms = ripser(points, maxdim=1)["dgms"]
    l1, l2 = landscape_norms(dgms[1], layers, grid)
    return np.array([l1, l2, persistence_entropy(dgms[1]), total_persistence(dgms[0])])


def rolling_features(logret, window=50, layers=5, grid=100):
    """logret: (T, n) 로그수익률, NaN은 0 처리. t일 값은 t일까지(포함) 데이터만 사용. 앞 window-1일은 NaN."""
    x = np.nan_to_num(logret, nan=0.0)
    out = np.full((len(x), 4), np.nan)
    for t in range(window - 1, len(x)):
        out[t] = window_features(x[t - window + 1:t + 1], layers, grid)
    return out


def trailing_percentile(series, window=252):
    """각 시점 값이 직전 window개(현재 포함) 중 몇 분위인지 [0,1]. 과거만 사용."""
    out = np.full(len(series), np.nan)
    for t in range(len(series)):
        if np.isnan(series[t]):
            continue
        h = series[max(0, t - window + 1):t + 1]
        h = h[~np.isnan(h)]
        if len(h) >= 20:
            out[t] = (h < series[t]).sum() / (len(h) - 1)
    return out


def heat_to_scale(heat, threshold=0.65, max_reduction=0.5):
    """과열점수 → 비중 배율. threshold 이하 1.0, 1.0에서 1 - max_reduction."""
    h = np.nan_to_num(np.asarray(heat, dtype=float), nan=0.0)
    over = np.clip((h - threshold) / (1.0 - threshold), 0.0, 1.0)
    return 1.0 - max_reduction * over
