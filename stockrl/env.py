"""한국 주식 포트폴리오 거래 환경 (제안서 4장).

t일 장 마감 후 30슬롯+현금 목표비중 결정 → t+1일 시가에 정수 주문으로 리밸런싱 → t+1일 종가로 평가.
보유는 종목 단위로 관리하므로 연초 유니버스에서 빠진 종목은 다음 결정에서 자동 매도된다.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .tda import heat_to_scale

POS_FEATS = 3  # 현재 비중, 미실현 손익률, 보유일수/252


class Panel:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.dates = z["dates"]
        self.tickers = z["tickers"]
        self.names = z["names"]
        self.stock_feat_names = z["stock_feat_names"]
        self.mkt_names = z["mkt_names"]
        self.X, self.M = z["X"], z["M"]
        self.O, self.C, self.VOL = z["O"], z["C"], z["VOL"]
        self.S, self.SS = z["S"], z["SS"]
        self.sector_names, self.heat = z["sector_names"], z["heat"]
        self.K200, self.mkt_h1_l1_raw = z["K200"], z["mkt_h1_l1_raw"]
        self.T, self.N, self.F = self.X.shape
        self.K = self.S.shape[1]
        self.Fm = self.M.shape[1]
        # 종가 결측(거래정지·상장 전)은 직전 종가로 평가
        self.C_ffill = self.C.copy()
        for i in range(1, self.T):
            m = np.isnan(self.C_ffill[i])
            self.C_ffill[i, m] = self.C_ffill[i - 1, m]

    def index_range(self, start, end):
        d = self.dates
        idx = np.where((d >= str(start)) & (d <= str(end)))[0]
        return int(idx[0]), int(idx[-1])


def make_env(panel, cfg, start, end, **kw):
    """전체 설정(cfg)에서 환경 생성: env 설정 + 섹터 과열 파라미터 + 시장/종목 관측 길이"""
    ecfg = {**cfg["env"], "heat_threshold": cfg["tda"]["sector"]["threshold"],
            "heat_max_reduction": cfg["tda"]["sector"]["max_reduction"]}
    f = cfg["features"]
    return TradingEnv(panel, ecfg, start, end, f["lookback"], stock_lookback=f.get("stock_lookback"), **kw)


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, panel, cfg, start, end, lookback=60, random_start=True, heat_overlay=False, seed=None,
                 stock_lookback=None):
        """lookback: 시장 시계열 길이, stock_lookback: 종목 시계열 길이 (기본 = lookback)"""
        super().__init__()
        self.p, self.cfg = panel, cfg
        self.L = lookback
        self.Ls = stock_lookback or lookback
        self.random_start = random_start
        self.heat_overlay = heat_overlay
        lo, hi = panel.index_range(start, end)
        self.lo = max(lo, max(self.L, self.Ls) + 252)  # z-score·윈도우 워밍업
        self.hi = hi
        if self.hi - self.lo < 2:
            raise ValueError(f"구간이 너무 짧음: {start}~{end}")
        self.ep_len = cfg["episode_len"] if random_start else self.hi - self.lo
        K = panel.K
        self.observation_space = spaces.Dict({
            "stocks": spaces.Box(-10, 10, (K, self.Ls, panel.F), np.float32),
            "pos": spaces.Box(-10, 10, (K, POS_FEATS), np.float32),
            "market": spaces.Box(-10, 10, (lookback, panel.Fm), np.float32),
            "weights": spaces.Box(0, 1, (K + 1,), np.float32),
        })
        self.action_space = spaces.Box(-10, 10, (K + 1,), np.float32)
        self.buy_cost = cfg["fee"] + cfg["slippage"]
        self.sell_cost = cfg["fee"] + cfg["sell_tax"] + cfg["slippage"]
        self._seed = seed

    # ---------- 상태 ----------
    def _value(self, i):
        return self.cash + float(np.nansum(self.shares * self.p.C_ffill[i]))

    def _slot_weights(self, i, V):
        tix = self.p.S[i]
        w = np.zeros(self.p.K + 1)
        m = tix >= 0
        w[:-1][m] = self.shares[tix[m]] * np.nan_to_num(self.p.C_ffill[i, tix[m]]) / V
        w[-1] = self.cash / V
        return w

    def _obs(self):
        i, p, L, Ls = self.i, self.p, self.L, self.Ls
        tix = p.S[i]
        m = tix >= 0
        stocks = np.zeros((p.K, Ls, p.F), np.float32)
        stocks[m] = p.X[i - Ls + 1:i + 1][:, tix[m]].transpose(1, 0, 2)
        V = self._value(i)
        w = self._slot_weights(i, V)
        pos = np.zeros((p.K, POS_FEATS), np.float32)
        held = m & (self.shares[np.where(m, tix, 0)] > 0)
        t_h = tix[held]
        pos[held, 0] = w[:-1][held]
        pos[held, 1] = np.clip(p.C_ffill[i, t_h] / self.avg_cost[t_h] - 1, -1, 1)
        pos[held, 2] = np.minimum(self.hold_days[t_h] / 252.0, 1.0)
        return {"stocks": stocks, "pos": pos, "market": p.M[i - L + 1:i + 1].astype(np.float32),
                "weights": w.astype(np.float32)}

    # ---------- 행동 → 목표비중 ----------
    def target_weights(self, w, i):
        """w: 슬롯 K개 + 현금 비중(합 1). 빈 슬롯·가격 없는 종목 0, 종목당 상한, 과열 섹터 축소 → 나머지 현금."""
        tix = self.p.S[i]
        stock_w = np.asarray(w, dtype=np.float64)[:-1].copy()
        stock_w[(tix < 0) | np.isnan(self.p.C[i, np.where(tix >= 0, tix, 0)])] = 0.0
        fc = self.cfg.get("fixed_cash")
        if fc is not None and stock_w.sum() > 0:
            # 현금 비중 고정 → 현금 타이밍을 없애고 종목 선택만 학습
            stock_w = stock_w / stock_w.sum() * (1.0 - fc)
        stock_w = np.minimum(stock_w, self.cfg["max_weight"])
        if self.heat_overlay:
            sec = self.p.SS[i]
            h = np.where(sec >= 0, self.p.heat[np.where(sec >= 0, sec, 0), i], 0.0)
            stock_w *= heat_to_scale(h, self.cfg.get("heat_threshold", 0.65), self.cfg.get("heat_max_reduction", 0.5))
        return np.append(stock_w, 1.0 - stock_w.sum())

    # ---------- 체결 ----------
    def _rebalance(self, target, i):
        """i일 결정 → i+1일 시가 체결. 반환: 거래금액 합계, 비용 합계"""
        p, j = self.p, i + 1
        tix = p.S[i]
        open_px = np.where(np.isnan(p.O[j]), p.C_ffill[i], p.O[j])
        prev = p.C_ffill[i]
        tradable = (p.VOL[j] > 0) & ~np.isnan(p.O[j])
        up_lock = open_px >= prev * (1 + self.cfg["limit_pct"])
        dn_lock = open_px <= prev * (1 - self.cfg["limit_pct"])
        V = self.cash + float(np.nansum(self.shares * open_px))
        cur_w = np.nan_to_num(self.shares * open_px / V)

        tw = np.zeros(p.N)
        m = tix >= 0
        tw[tix[m]] = target[:-1][m]
        band = self.cfg["no_trade_band"]
        traded = costs = 0.0

        # 매도 먼저 (목표 0은 밴드와 무관하게 전량 매도)
        for n in np.where(self.shares > 0)[0]:
            if not tradable[n] or dn_lock[n]:
                continue
            if tw[n] > 0 and abs(tw[n] - cur_w[n]) < band:
                continue
            tgt = np.floor(tw[n] * V / open_px[n])
            q = self.shares[n] - tgt
            if q > 0:
                amt = q * open_px[n]
                c = amt * self.sell_cost
                self.cash += amt - c
                self.shares[n] -= q
                traded += amt
                costs += c
                if self.shares[n] == 0:
                    self.hold_days[n] = 0
                    self.avg_cost[n] = np.nan
        # 매수: 현금 부족하면 비례 축소
        buys = {}
        for n in np.where(tw > 0)[0]:
            if not tradable[n] or up_lock[n]:
                continue
            # band_on_entry=False: 무거래 밴드는 보유 종목 비중 조정에만. True(예전 동작)면 신규 진입도 막혀
            # 목표 1.7% < 밴드 3% 같은 경우 영원히 못 삼
            if abs(tw[n] - cur_w[n]) < band and (self.cfg.get("band_on_entry", True) or self.shares[n] > 0):
                continue
            q = np.floor(tw[n] * V / open_px[n]) - self.shares[n]
            if q > 0:
                buys[n] = q
        need = sum(q * open_px[n] * (1 + self.buy_cost) for n, q in buys.items())
        scale = min(1.0, self.cash / need) if need > 0 else 1.0
        for n, q in buys.items():
            q = np.floor(q * scale)
            if q <= 0:
                continue
            amt = q * open_px[n]
            c = amt * self.buy_cost
            if amt + c > self.cash:
                continue
            old = self.shares[n]
            self.avg_cost[n] = (np.nan_to_num(self.avg_cost[n]) * old + amt) / (old + q)
            self.shares[n] += q
            self.cash -= amt + c
            traded += amt
            costs += c
        return traded, costs, V

    # ---------- gym ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed if seed is not None else self._seed)
        if self.random_start:
            self.i = int(self.np_random.integers(self.lo, max(self.lo + 1, self.hi - self.ep_len)))
        else:
            self.i = self.lo
        self.start_i = self.i
        self.cash = float(self.cfg["initial_cash"])
        self.shares = np.zeros(self.p.N)
        self.avg_cost = np.full(self.p.N, np.nan)
        self.hold_days = np.zeros(self.p.N)
        self.V0 = self.V = self.peak = self.cash
        self.dd = 0.0
        self.A = self.B = 0.0
        return self._obs(), {}

    def step(self, action):
        z = np.asarray(action, dtype=np.float64)
        z = np.exp(z - z.max())
        return self.step_weights(z / z.sum())

    def step_weights(self, w):
        i = self.i
        V_prev = self._value(i)
        w_prev = self._slot_weights(i, V_prev)
        target = self.target_weights(w, i)
        traded, costs, V_open = self._rebalance(target, i)
        self.i = i + 1
        self.hold_days[self.shares > 0] += 1
        V = self._value(self.i)
        ret = np.log(V / V_prev)
        self.peak = max(self.peak, V)
        dd = 1 - V / self.peak
        turnover = float(np.abs(target - w_prev).sum())
        # (공개본) 강화학습 보상 설계는 비공개 — 백테스트에는 보상이 쓰이지 않음
        reward = ret
        self.dd = dd
        terminated = V < self.V0 * self.cfg["stop_value_ratio"]
        truncated = (self.i - self.start_i >= self.ep_len) or (self.i >= self.hi)
        info = {"date": self.p.dates[self.i], "value": V, "ret": ret, "turnover": turnover,
                "traded": traded, "costs": costs, "cash_w": self.cash / V}
        return self._obs(), float(reward), bool(terminated), bool(truncated), info
