"""베이스라인 전략 (제안서 7장) — 같은 환경(비용·제약)에서 실행. 섹터 과열 오버레이 유/무 비교 포함.

    python -m stockrl.baselines --start 2021-01-01 --end 2025-12-31
"""
import argparse
import json

import numpy as np
import pandas as pd

from .common import get_logger, load_config, rpath
from .env import Panel, make_env
from .metrics import summarize
from .tda import trailing_percentile

log = get_logger("baselines")


def run(env, decide):
    """decide(env, i, w_now) → 목표비중(K+1) 또는 None(유지). 반환: 평가금액·회전율·비용 시계열"""
    env.reset()
    values, turnover, costs, dates = [env.V0], [], [], [env.p.dates[env.i]]
    done = False
    while not done:
        i = env.i
        w_now = env._slot_weights(i, env._value(i))
        w = decide(env, i, w_now)
        _, _, term, trunc, info = env.step_weights(w_now if w is None else w)
        values.append(info["value"])
        turnover.append(info["turnover"])
        costs.append(info["costs"])
        dates.append(info["date"])
        done = term or trunc
    return {"dates": dates, "values": values, "turnover": turnover, "costs": costs}


def _valid(env, i):
    tix = env.p.S[i]
    return (tix >= 0) & ~np.isnan(env.p.C[i, np.where(tix >= 0, tix, 0)])


def _month_start(env, i):
    return i == env.start_i or env.p.dates[i][:7] != env.p.dates[i - 1][:7]


def equal_weight(env, i, w_now):
    return equal_weight_now(env, i) if _month_start(env, i) else None


def momentum_12_1(env, i, w_now, top=10):
    if not _month_start(env, i):
        return None
    C, tix = env.p.C_ffill, env.p.S[i]
    v = _valid(env, i)
    mom = np.full(env.p.K, -np.inf)
    if i >= 252:
        r = C[i - 21, np.where(tix >= 0, tix, 0)] / C[i - 252, np.where(tix >= 0, tix, 0)] - 1
        mom[v] = np.nan_to_num(r[v], nan=-np.inf)
    w = np.zeros(env.p.K + 1)
    pick = np.argsort(-mom)[:top]
    pick = pick[np.isfinite(mom[pick])]
    w[pick] = 1.0 / top
    w[-1] = 1 - w[:-1].sum()
    return w


def ma_cross(env, i, w_now):
    C, tix = env.p.C_ffill, env.p.S[i]
    v = _valid(env, i)
    cols = np.where(tix >= 0, tix, 0)
    ma20 = np.nanmean(C[i - 19:i + 1, cols], axis=0)
    ma60 = np.nanmean(C[i - 59:i + 1, cols], axis=0)
    w = np.zeros(env.p.K + 1)
    w[:-1][v & (ma20 > ma60)] = 1.0 / env.p.K
    w[-1] = 1 - w[:-1].sum()
    return w


def make_tda_filter(panel, q=0.9, window=252):
    pct = trailing_percentile(panel.mkt_h1_l1_raw, window)

    def decide(env, i, w_now):
        if pct[i] >= q:
            w = np.zeros(env.p.K + 1)
            w[-1] = 1.0
            return w
        if _month_start(env, i) or pct[i - 1] >= q:  # 월초 또는 현금화 해제 직후 동일가중 복귀
            return equal_weight_now(env, i)
        return None
    return decide


def equal_weight_now(env, i):
    v = _valid(env, i)
    w = np.zeros(env.p.K + 1)
    w[:-1][v] = 1.0 / env.p.K
    w[-1] = 1 - w[:-1].sum()
    return w


def make_ridge_rank(panel, start, horizon=5, top=10, lam=10.0):
    """선형(릿지) 랭크 모델: 특징 → 향후 horizon일 수익 순위. start 이전에 결과가 확정된 데이터로만 학습 (룩어헤드 없음).
    horizon일마다 상위 top 종목 균등 보유. RL 이 최소한 이것보다 나아야 의미가 있다."""
    lo = int(np.searchsorted(panel.dates, str(start)))  # start 당일 또는 그 다음 거래일 인덱스
    keep = [j for j, n in enumerate(panel.stock_feat_names) if str(n) != "ridge_score"]
    C, X, S = panel.C_ffill, panel.X[:, :, keep], panel.S
    xs, ys = [], []
    for i in range(260, lo - horizon):
        tix = S[i][S[i] >= 0]
        y = C[i + horizon, tix] / C[i, tix] - 1
        ok = np.isfinite(y)
        if ok.sum() >= 10:
            xs.append(X[i, tix[ok]])
            ys.append(pd.Series(y[ok]).rank(pct=True).to_numpy() - 0.5)
    Xtr, ytr = np.vstack(xs), np.concatenate(ys)
    w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ ytr)

    def decide(env, i, w_now):
        if (i - env.start_i) % horizon:
            return None
        tix = env.p.S[i]
        v = _valid(env, i)
        score = np.full(env.p.K, -np.inf)
        score[v] = env.p.X[i, tix[v]][:, keep] @ w
        out = np.zeros(env.p.K + 1)
        out[np.argsort(-score)[:top]] = 1.0 / top
        out[-1] = 1 - out[:-1].sum()
        return out
    return decide


def buy_hold_k200(panel, start, end):
    lo, hi = panel.index_range(start, end)
    k = panel.K200[lo:hi + 1]
    return {"dates": list(panel.dates[lo:hi + 1]), "values": list(k / k[0] * 1e8)}


def run_all(panel, cfg, start, end):
    out = {}
    bh = buy_hold_k200(panel, start, end)
    out["kospi200_buyhold"] = summarize(bh["values"])
    strategies = {"equal_weight": equal_weight, "momentum_12_1": momentum_12_1, "ma_cross_20_60": ma_cross,
                  "tda_regime_filter": make_tda_filter(panel), "ridge_rank_top10": make_ridge_rank(panel, start)}
    curves = {"kospi200_buyhold": bh}
    for name, fn in strategies.items():
        for overlay in (False, True):
            env = make_env(panel, cfg, start, end, random_start=False, heat_overlay=overlay)
            res = run(env, fn)
            key = name + ("+heat" if overlay else "")
            out[key] = summarize(res["values"], res["turnover"], res["costs"])
            curves[key] = res
            log.info("%-26s sharpe %.2f  cagr %6.1f%%  mdd %5.1f%%", key, out[key]["sharpe"],
                     100 * out[key]["cagr"], 100 * out[key]["mdd"])
    return out, curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2025-12-31")
    a = ap.parse_args()
    cfg = load_config(a.config)
    panel = Panel(rpath(cfg["data"]["processed"]))
    out, curves = run_all(panel, cfg, a.start, a.end)
    od = rpath(cfg["output_dir"]) / "baselines"
    od.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).T.to_csv(od / f"summary_{a.start}_{a.end}.csv")
    with open(od / f"curves_{a.start}_{a.end}.json", "w") as f:
        json.dump({k: {"dates": [str(d) for d in v["dates"]], "values": [float(x) for x in v["values"]]}
                   for k, v in curves.items()}, f)
    log.info("저장: %s", od)


if __name__ == "__main__":
    main()
