import numpy as np


def relative(values, bench, regime_days=21, regime_band=0.03):
    """시장(KOSPI200) 대비 평가 — 수익률만 보지 않고 시장 흐름 속에서 얼마나 벌고 얼마나 막았나 (사용자 기준 2026-10-16).
    values, bench: 같은 날짜의 일별 평가금액·지수 (첫 값 = 시작)
      excess_return   총수익 − 시장 총수익
      beta / alpha    시장 민감도 / 시장 몫을 뺀 연환산 초과수익
      info_ratio      일별 초과수익 평균 / 추적오차 (연환산)
      up_capture      시장이 오른 날 시장 상승분 중 따라간 비율 (1 이상이면 더 오름)
      down_capture    시장이 내린 날 시장 하락분 중 같이 맞은 비율 (낮을수록 방어 잘함)
      bull/bear/flat  약 1개월 구간을 시장 수익 ±3% 로 나눠 구간별 포트·시장 평균 수익, 하락장 방어율
      mdd_ratio       포트 MDD / 시장 MDD"""
    v, b = np.asarray(values, float), np.asarray(bench, float)
    ok = np.isfinite(v) & np.isfinite(b) & (v > 0) & (b > 0)
    v, b = v[ok], b[ok]
    if len(v) < 22:
        return {}
    rp, rb = v[1:] / v[:-1] - 1, b[1:] / b[:-1] - 1
    var = rb.var(ddof=1)
    beta = np.cov(rp, rb, ddof=1)[0, 1] / var if var > 0 else np.nan
    te = (rp - rb).std(ddof=1)
    up, dn = rb > 0, rb < 0
    mdd = lambda x: float((1 - x / np.maximum.accumulate(x)).max())
    out = {"excess_return": float(v[-1] / v[0] - b[-1] / b[0]), "beta": float(beta),
           "alpha": float((rp.mean() - beta * rb.mean()) * 252),
           "info_ratio": float((rp - rb).mean() / te * np.sqrt(252)) if te > 0 else np.nan,
           "up_capture": float(rp[up].mean() / rb[up].mean()) if up.any() else np.nan,
           "down_capture": float(rp[dn].mean() / rb[dn].mean()) if dn.any() else np.nan,
           "mdd_ratio": mdd(v) / mdd(b) if mdd(b) > 0 else np.nan}
    seg_p, seg_b = [], []
    for s in range(0, len(v) - regime_days, regime_days):
        seg_p.append(v[s + regime_days] / v[s] - 1)
        seg_b.append(b[s + regime_days] / b[s] - 1)
    seg_p, seg_b = np.array(seg_p), np.array(seg_b)
    for name, m in (("bull", seg_b > regime_band), ("bear", seg_b < -regime_band),
                    ("flat", np.abs(seg_b) <= regime_band)):
        out[f"{name}_n"] = int(m.sum())
        out[f"{name}_port"] = float(seg_p[m].mean()) if m.any() else np.nan
        out[f"{name}_mkt"] = float(seg_b[m].mean()) if m.any() else np.nan
    # 하락장 방어율: 시장이 −x% 일 때 포트 손실이 그보다 얼마나 작았나 (1 = 손실 0, 0 = 시장만큼, 음수 = 더 잃음)
    out["bear_defense"] = 1 - out["bear_port"] / out["bear_mkt"] if out["bear_n"] else np.nan
    return out


def summarize(values, turnover=None, costs=None, initial=None):
    """values: 일별 평가금액 배열 (첫 값 = 시작 금액)"""
    v = np.asarray(values, dtype=float)
    r = np.diff(np.log(v))
    n = len(r)
    if n == 0:
        return {}
    years = n / 252
    total = v[-1] / v[0] - 1
    cagr = (v[-1] / v[0]) ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std(ddof=1) * np.sqrt(252) if n > 1 else np.nan
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(252) if n > 1 and r.std() > 0 else 0.0
    down = r[r < 0]
    sortino = r.mean() / down.std(ddof=1) * np.sqrt(252) if len(down) > 1 and down.std() > 0 else np.nan
    peak = np.maximum.accumulate(v)
    mdd = float((1 - v / peak).max())
    q = np.quantile(r, 0.05)
    out = {"days": n, "total_return": total, "cagr": cagr, "vol": vol, "sharpe": sharpe, "sortino": sortino,
           "mdd": mdd, "calmar": cagr / mdd if mdd > 0 else np.nan, "cvar95": float(r[r <= q].mean())}
    if turnover is not None:
        out["turnover_annual"] = float(np.sum(turnover) / years / 2)
    if costs is not None:
        profit = v[-1] - v[0]
        out["cost_total"] = float(np.sum(costs))
        out["cost_to_profit"] = float(np.sum(costs) / profit) if profit > 0 else np.nan
    return {k: float(x) for k, x in out.items()}
