"""사용자가 돌린 백테스트 기록·우수 결과 선별·적용 가능성 검토 (~/stockrl_data/db/web.db).

우수 판정 = RL 합격 기준과 같은 잣대: 정보비율 ≥ 0.3, 하락장 손실 비율 ≤ 0.9, MDD 비율 ≤ 1.0, 기간 3년 이상.
검토 = 기간을 해마다 쪼개 다시 돌려 '여러 해에 걸쳐' 시장을 이기는지(한 해 대박인지), 회전율·비용, 시도 횟수(과적합) 확인."""
import json
import sqlite3
from datetime import datetime

import pandas as pd

from ..common import DB_DIR

DB = DB_DIR / "web.db"
SCHEMA = """CREATE TABLE IF NOT EXISTS backtests (
    id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT UNIQUE, first_ts TEXT, last_ts TEXT, runs INTEGER, user TEXT,
    strategy TEXT, start TEXT, end TEXT, params_json TEXT,
    total_return REAL, cagr REAL, mdd REAL, sharpe REAL, turnover REAL,
    excess REAL, info_ratio REAL, beta REAL, up_capture REAL, down_capture REAL, mdd_ratio REAL,
    years REAL, good INTEGER, review_json TEXT, review_ts TEXT, verdict TEXT);"""
GOOD = {"min_info_ratio": 0.3, "max_down_capture": 0.9, "max_mdd_ratio": 1.0, "min_years": 3.0}


def con():
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB, timeout=10)
    c.executescript(SCHEMA)
    c.row_factory = sqlite3.Row
    return c


def is_good(m, r, years):
    return bool(years >= GOOD["min_years"] and (r.get("info_ratio") or -9) >= GOOD["min_info_ratio"]
                and (r.get("down_capture") if r.get("down_capture") is not None else 9) <= GOOD["max_down_capture"]
                and (r.get("mdd_ratio") if r.get("mdd_ratio") is not None else 9) <= GOOD["max_mdd_ratio"])


def save(user, params, res):
    m, r = res["metrics"], res["relative"]
    years = len(res["dates"]) / 252
    # 실행자(admin/guest)별로 따로 저장 → guest 가 찾은 좋은 설정을 따로 모아 볼 수 있게
    key = json.dumps({**params, "_by": "guest" if user == "guest" else "admin"}, sort_keys=True)
    now = datetime.now().isoformat(timespec="seconds")
    c = con()
    row = c.execute("SELECT id FROM backtests WHERE key=?", (key,)).fetchone()
    if row:
        c.execute("UPDATE backtests SET runs = runs + 1, last_ts = ? WHERE id = ?", (now, row["id"]))
        c.commit()
        return row["id"]
    cur = c.execute(
        """INSERT INTO backtests (key, first_ts, last_ts, runs, user, strategy, start, end, params_json, total_return, cagr,
           mdd, sharpe, turnover, excess, info_ratio, beta, up_capture, down_capture, mdd_ratio, years, good)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (key, now, now, 1, user, params["strategy"], params["start"], params["end"], json.dumps(params, sort_keys=True),
         m.get("total_return"),
         m.get("cagr"), m.get("mdd"), m.get("sharpe"), m.get("turnover_annual"), r.get("excess_return"),
         r.get("info_ratio"), r.get("beta"), r.get("up_capture"), r.get("down_capture"), r.get("mdd_ratio"), years,
         int(is_good(m, r, years))))
    c.commit()
    return cur.lastrowid


def listing(limit=60, who="all"):
    c = con()
    where = {"all": "", "guest": "WHERE user = 'guest'", "admin": "WHERE user != 'guest'"}[who]
    rows = [dict(r) for r in c.execute(
        f"SELECT * FROM backtests {where} ORDER BY good DESC, COALESCE(info_ratio, -99) DESC, last_ts DESC LIMIT ?",
        (limit,))]
    total = c.execute("SELECT COUNT(*), COALESCE(SUM(runs), 0) FROM backtests").fetchone()
    for r in rows:
        r["params"] = json.loads(r["params_json"])
        r["review"] = json.loads(r["review_json"]) if r["review_json"] else None
    return {"rows": rows, "n_settings": total[0], "n_runs": total[1]}


def review(bt_id, run_backtest):
    """적용 가능성 검토: 해마다 쪼개 다시 돌림 → 연도별 시장 대비, 회전율, 시도 횟수"""
    c = con()
    row = c.execute("SELECT * FROM backtests WHERE id=?", (bt_id,)).fetchone()
    if not row:
        raise KeyError(bt_id)
    p = json.loads(row["params_json"])
    y0, y1 = int(p["start"][:4]), int(p["end"][:4])
    yearly = []
    for y in range(y0, y1 + 1):
        s, e = max(p["start"], f"{y}-01-01"), min(p["end"], f"{y}-12-31")
        try:
            r = run_backtest(p["strategy"], s, e, p["cap"], p["heat"], p["threshold"], p["max_reduction"],
                             p.get("rules", ""))
        except ValueError:
            continue
        if len(r["dates"]) < 60:   # 석 달 미만 조각은 판단에서 뺌
            continue
        b = r["bench"]
        yearly.append({"year": y, "ret": r["metrics"].get("total_return"), "mkt": b[-1] / b[0] - 1,
                       "excess": r["relative"].get("excess_return"), "info_ratio": r["relative"].get("info_ratio"),
                       "down_capture": r["relative"].get("down_capture")})
    n_tries = c.execute("SELECT COALESCE(SUM(runs), 0) FROM backtests").fetchone()[0]
    wins = sum(1 for y in yearly if (y["excess"] or 0) > 0)
    win_rate = wins / len(yearly) if yearly else 0.0
    checks = [
        ("기간 3년 이상", (row["years"] or 0) >= 3, f"{row['years']:.1f}년"),
        ("해마다 시장을 이긴 비율 ≥ 67%", win_rate >= 0.667, f"{wins}/{len(yearly)}년"),
        ("정보비율 ≥ 0.3", (row["info_ratio"] or -9) >= 0.3, f"{row['info_ratio']:.2f}" if row["info_ratio"] is not None else "—"),
        ("하락장 손실 비율 ≤ 0.9", (row["down_capture"] if row["down_capture"] is not None else 9) <= 0.9,
         f"{row['down_capture']:.2f}" if row["down_capture"] is not None else "—"),
        ("MDD 가 시장보다 크지 않음", (row["mdd_ratio"] if row["mdd_ratio"] is not None else 9) <= 1.0,
         f"{row['mdd_ratio']:.2f}" if row["mdd_ratio"] is not None else "—"),
        ("회전율 연 6회 이하 (비용·체결 현실성)", (row["turnover"] or 0) <= 6, f"연 {row['turnover'] or 0:.1f}회"),
    ]
    passed = sum(ok for _, ok, _ in checks)
    if passed == len(checks):
        verdict = "적용 후보"
    elif passed >= len(checks) - 2 and win_rate >= 0.5:
        verdict = "보류"
    else:
        verdict = "부적합"
    notes = []
    if n_tries >= 20:
        notes.append(f"지금까지 설정 {n_tries}번 시도 — 가장 좋은 것만 고르면 우연히 좋아 보일 수 있음(과적합). "
                     "적용 전 다른 기간(최근 데이터)과 모의투자로 다시 확인")
    if p["strategy"] != "kospi200":
        notes.append("적용하려면: 모의투자 장기몫 설정(config/train.yaml paper.core)에 반영 → 최소 1~3개월 모의로 시장 대비 확인 → 실전")
    res = {"yearly": yearly, "checks": [{"name": n, "ok": bool(ok), "value": v} for n, ok, v in checks],
           "passed": passed, "total": len(checks), "verdict": verdict, "notes": notes}
    c.execute("UPDATE backtests SET review_json=?, review_ts=?, verdict=? WHERE id=?",
              (json.dumps(res, ensure_ascii=False), datetime.now().isoformat(timespec="seconds"), verdict, bt_id))
    c.commit()
    return res
