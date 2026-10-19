"""투자 현황 웹사이트 (gcp-free 에서 상시 실행).

    uvicorn stockrl.web.app:app --host 0.0.0.0 --port 8080
페이지: / 전체 현황, /market/{kr,us,crypto} 시장별, /paper 모의투자, /strategy 전략 설명, /backtest 백테스트
접근: HTTP Basic 인증 — .env 의 WEB_USER / WEB_PASSWORD (없으면 전부 거절). 한투 API 는 부르지 않는다(매매와 한도 공유)."""
import os
import secrets
import time
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .. import common  # noqa: F401  (.env 로드)
from . import data, store

HERE = Path(__file__).parent
app = FastAPI(title="주식자동매매 현황", docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
tpl = Jinja2Templates(directory=HERE / "templates")
def _fmt(f):
    def g(v):
        try:
            return f(float(v))
        except (TypeError, ValueError):
            return "—"          # 값 없음(아직 기록 전)·Undefined
    return g


tpl.env.filters["won"] = _fmt(lambda v: f"{v:,.0f}원")
tpl.env.filters["pct"] = _fmt(lambda v: f"{v * 100:+.2f}%")
tpl.env.filters["num"] = _fmt(lambda v: f"{v:.2f}")
security = HTTPBasic()

MARKETS = {
    "kr": {"name": "한국주식", "status": "모의투자 운용 중", "broker": "한국투자증권 (모의)", "bench": "KOSPI200"},
    "us": {"name": "미국주식", "status": "데이터 수집 중", "broker": "한국투자증권 해외주식 (예정)", "bench": "S&P500 · 나스닥100"},
    "crypto": {"name": "암호화폐", "status": "데이터 수집 중", "broker": "바이낸스 (예정)", "bench": "BTC 보유 · 시총 상위 바스켓"},
}


GUEST = ("guest", "guest")   # 공개 저장소 README 에 적는 체험 계정 (2026-10-19 사용자 요청) — 모든 페이지 보기 허용


def _via_tunnel(request):
    """Cloudflare 터널(HTTPS)로 들어왔거나 서버 안에서 온 요청인지. 80번 직접 접속은 평문 HTTP"""
    return bool(request.headers.get("cf-ray")) or (request.client and request.client.host in ("127.0.0.1", "::1"))


def auth(request: Request, cred: HTTPBasicCredentials = Depends(security)):
    user, pw = os.getenv("WEB_USER", "admin"), os.getenv("WEB_PASSWORD", "")
    eq = lambda a, b: secrets.compare_digest(a.encode(), b.encode())
    if eq(cred.username, GUEST[0]) and eq(cred.password, GUEST[1]):
        return "guest"
    if bool(pw) and eq(cred.username, user) and eq(cred.password, pw):
        if not _via_tunnel(request):
            # 관리자 비밀번호가 암호화 없이 오가지 않게: 관리자는 HTTPS(터널) 주소로만
            raise HTTPException(status.HTTP_403_FORBIDDEN, "관리자 계정은 HTTPS 주소로만 로그인할 수 있습니다")
        return cred.username
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, headers={"WWW-Authenticate": "Basic"})


_RATE = {}


def guest_limit(request, user, kind, n, window=600):
    """guest 는 IP 별로 window 초에 n 번까지 (e2-micro 보호)"""
    if user != "guest":
        return
    ip = request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "?")
    now = time.time()
    q = [t for t in _RATE.get((ip, kind), []) if now - t < window]
    if len(q) >= n:
        raise HTTPException(429, f"guest 는 {window // 60}분에 {n}번까지 실행할 수 있습니다. 잠시 후 다시 해 주세요")
    _RATE[(ip, kind)] = q + [now]


import threading
import time

# e2-micro 는 디스크가 느리고 메모리가 작아 매 요청마다 DB 를 다시 읽으면 수십 초(2026-10-19 실측 `/` 37초)
# → 백그라운드에서 1분마다 미리 읽어 메모리에 두고, 화면은 그걸 바로 보여 준다 (최대 1분 늦음)
_CACHE = {}
_JOBS = {"paper": data.paper, "live": data.live, "alerts": data.alerts, "us": lambda: data.glob_status("us"),
         "crypto": lambda: data.glob_status("crypto")}


def cached(key):
    if key not in _CACHE:
        _CACHE[key] = _JOBS[key]()
    return _CACHE[key]


def _warm():
    try:
        from ..backtest import data as bt_data
        bt_data()                     # 백테스트 패널·시총 미리 읽기 (첫 백테스트가 느리지 않게)
    except Exception:
        pass
    while True:
        for k, f in _JOBS.items():
            try:
                _CACHE[k] = f()
            except Exception:
                pass
        time.sleep(60)


@app.on_event("startup")
def _start_warm():
    threading.Thread(target=_warm, daemon=True).start()


def page(request, name, **ctx):
    return tpl.TemplateResponse(request, name, {"markets": MARKETS, **ctx})


@app.get("/", response_class=HTMLResponse)
def overview(request: Request, _=Depends(auth)):
    # 실전 계좌는 아직 없음 → real=None (실전 시작하면 data.real() 로 채움)
    return page(request, "overview.html", active="overview", p=cached("paper"), live=cached("live"), real=None)


@app.get("/market/{m}", response_class=HTMLResponse)
def market(request: Request, m: str, _=Depends(auth)):
    if m not in MARKETS:
        raise HTTPException(404)
    return page(request, "market.html", active=m, m=m, info=MARKETS[m], p=cached("paper") if m == "kr" else None, live=cached("live") if m == "kr" else None,
                g=None if m == "kr" else cached(m))


@app.get("/paper", response_class=HTMLResponse)
def paper(request: Request, _=Depends(auth)):
    return page(request, "paper.html", active="paper", p=cached("paper"), live=cached("live"), alerts=cached("alerts"))


@app.get("/strategy", response_class=HTMLResponse)
def strategy(request: Request, _=Depends(auth)):
    return page(request, "strategy.html", active="strategy")


@app.get("/backtest", response_class=HTMLResponse)
def backtest_page(request: Request, _=Depends(auth)):
    from ..backtest import INDICATORS, STRATEGIES, date_range
    lo, hi = date_range()
    ind = {k: {"name": v[0], "default": v[1], "unit": v[2]} for k, v in INDICATORS.items()}
    return page(request, "backtest.html", active="backtest", strategies=STRATEGIES, lo=lo, hi=hi, indicators=ind)


@app.get("/api/backtest")
def backtest_api(strategy: str = "cap", start: str = "2021-01-01", end: str = "2026-12-31",
                 cap: float = Query(0.30, ge=0.05, le=1.0), heat: bool = False,
                 threshold: float = Query(0.65, ge=0.0, le=1.0), max_reduction: float = Query(0.5, ge=0.0, le=1.0),
                 rules: str = Query("", max_length=4000), request: Request = None, user=Depends(auth)):
    import json as _json
    guest_limit(request, user, "backtest", 20)
    from ..backtest import STRATEGIES, parse_rules, run_backtest
    if strategy not in STRATEGIES:
        raise HTTPException(400, "알 수 없는 전략")
    params = {"strategy": strategy, "start": start[:10], "end": end[:10], "cap": round(cap, 2), "heat": heat,
              "threshold": round(threshold, 2), "max_reduction": round(max_reduction, 2), "rules": ""}
    if strategy == "custom":
        try:
            r = parse_rules(rules)
        except (ValueError, TypeError) as e:
            raise HTTPException(400, f"규칙 형식 오류: {e}")
        params["rules"] = _json.dumps(r, sort_keys=True, ensure_ascii=False)   # 같은 규칙은 같은 키로 저장
    elif strategy != "cap":
        params["cap"] = 0.30          # 상한은 시총가중·직접 규칙에만 쓰임 → 같은 결과가 따로 저장되지 않게
    if not heat:
        params["threshold"], params["max_reduction"] = 0.65, 0.5
    try:
        r = run_backtest(**params)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return {**r, "saved_id": store.save(user, params, r)}


@app.get("/api/indicators")
def indicators_api(_=Depends(auth)):
    from ..backtest import INDICATORS
    return {k: {"name": v[0], "default": v[1], "unit": v[2]} for k, v in INDICATORS.items()}


@app.get("/api/backtests")
def backtests_api(who: str = "all", _=Depends(auth)):
    return store.listing(who=who if who in ("all", "guest", "admin") else "all")


@app.post("/api/backtests/{bt_id}/review")
def review_api(bt_id: int, request: Request, user=Depends(auth)):
    guest_limit(request, user, "review", 5)
    from ..backtest import run_backtest
    try:
        return store.review(bt_id, run_backtest)
    except KeyError:
        raise HTTPException(404)


@app.get("/health")
def health():
    return {"ok": True}
