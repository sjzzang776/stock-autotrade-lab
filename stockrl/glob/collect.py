"""미국주식·암호화폐 데이터·뉴스 수집 (2026-10-19 시작, gcp-free 에서 cron) → ~/stockrl_data/db/global.db

    python -m stockrl.glob.collect crypto-universe          # 바이낸스 USDT 현물 거래대금 상위 30 (스테이블·레버리지 토큰 제외)
    python -m stockrl.glob.collect crypto-daily              # 일봉 전체 역사 (이어받기)
    python -m stockrl.glob.collect crypto-minute --months 12 # 1분봉 (월별·일별 압축 파일, 이어받기)
    python -m stockrl.glob.collect crypto-futures --months 12  # 무기한 선물 펀딩비(월별)·미결제약정·롱숏비(일별 5분)
    python -m stockrl.glob.collect us-daily                  # S&P500+나스닥100 일봉 (수정주가 포함, 가능한 전체 역사)
    python -m stockrl.glob.collect us-minute                 # 나스닥100+지수 1분봉 최근 7일 (매일 돌려서 쌓음)
    python -m stockrl.glob.collect news                      # 야후 파이낸스(종목별)·코인텔레그래프·코인데스크 RSS
    python -m stockrl.glob.collect sec                       # SEC 종목별 최근 공시 ~1000건 (8-K 항목·10-K/Q·내부자 4)
    python -m stockrl.glob.collect sec-current               # SEC 실시간 8-K 피드 → 우리 종목만

바이낸스 본 API(api/fapi.binance.com)는 미국 IP(gcp-free = us-west1)를 막음(451) → 공개 시세 data-api.binance.vision 과
압축 아카이브 data.binance.vision 만 씀(시세 수집엔 충분, 실거래는 한국 IP 필요). SEC 공시는 .env SEC_CONTACT(연락처 이메일, 사용자 승인 2026-10-19)로 수집.
"""
import argparse
import csv
import io
import json
import os
import re
import sqlite3
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET

import pandas as pd
import requests

from ..common import DB_DIR, get_logger, rpath

log = get_logger("glob")
DB = DB_DIR / "global.db"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) stockrl-research"}
BV = "https://data-api.binance.vision/api/v3"
ARCH = "https://data.binance.vision/data"
STABLE = {"USDC", "FDUSD", "TUSD", "DAI", "BUSD", "USDP", "EUR", "EURI", "AEUR", "USDE", "XUSD", "BFUSD", "USD1", "PAXG",
          "WBTC", "WBETH", "BNSOL"}
US_INDEX = ["^GSPC", "^NDX", "^DJI", "^RUT", "^VIX", "SPY", "QQQ", "TLT", "GLD", "UUP"]
SCHEMA = """
CREATE TABLE IF NOT EXISTS crypto_universe (symbol TEXT PRIMARY KEY, base TEXT, quote_volume REAL, rank INTEGER, updated TEXT);
CREATE TABLE IF NOT EXISTS crypto_daily (symbol TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,
    quote_volume REAL, trades INTEGER, taker_buy_quote REAL, PRIMARY KEY (symbol, date));
CREATE TABLE IF NOT EXISTS crypto_minute (symbol TEXT, ts INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL,
    quote_volume REAL, trades INTEGER, taker_buy_quote REAL, PRIMARY KEY (symbol, ts)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS crypto_funding (symbol TEXT, ts INTEGER, rate REAL, PRIMARY KEY (symbol, ts));
CREATE TABLE IF NOT EXISTS crypto_metrics (symbol TEXT, ts INTEGER, open_interest REAL, open_interest_value REAL,
    top_ls_account REAL, top_ls_position REAL, ls_account REAL, taker_ls_vol REAL, PRIMARY KEY (symbol, ts)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS us_daily (ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, adj_close REAL,
    volume REAL, PRIMARY KEY (ticker, date));
CREATE TABLE IF NOT EXISTS us_minute (ticker TEXT, ts INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (ticker, ts)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS gnews (id TEXT, market TEXT, symbol TEXT, dt TEXT, source TEXT, title TEXT, summary TEXT, url TEXT,
    collected_at TEXT, PRIMARY KEY (id, symbol));
CREATE TABLE IF NOT EXISTS done (kind TEXT, key TEXT, n INTEGER, at TEXT, PRIMARY KEY (kind, key));
CREATE TABLE IF NOT EXISTS stats (key TEXT PRIMARY KEY, value TEXT, updated TEXT);
CREATE TABLE IF NOT EXISTS us_cik (ticker TEXT PRIMARY KEY, cik INTEGER, title TEXT);
CREATE TABLE IF NOT EXISTS us_filings (accession TEXT PRIMARY KEY, cik INTEGER, ticker TEXT, form TEXT, filed TEXT,
    accepted TEXT, items TEXT, title TEXT, url TEXT, collected_at TEXT);
CREATE INDEX IF NOT EXISTS us_filings_tf ON us_filings (ticker, filed);
"""
# 웹은 큰 표를 세지 않고 이 요약만 읽는다 (us_daily 수백만 행 COUNT 가 e2-micro 에서 30초 넘게 걸림, 2026-10-19)
STATS = {
    "crypto_universe": "SELECT COUNT(*) FROM crypto_universe",
    "crypto_daily": "SELECT COUNT(DISTINCT symbol), COUNT(*), MIN(date), MAX(date) FROM crypto_daily",
    "crypto_minute": "SELECT COUNT(DISTINCT symbol), COUNT(*), MIN(ts), MAX(ts) FROM crypto_minute",
    "crypto_funding": "SELECT COUNT(DISTINCT symbol), COUNT(*) FROM crypto_funding",
    "crypto_metrics": "SELECT COUNT(DISTINCT symbol), COUNT(*) FROM crypto_metrics",
    "us_daily": "SELECT COUNT(DISTINCT ticker), COUNT(*), MIN(date), MAX(date) FROM us_daily",
    "us_minute": "SELECT COUNT(DISTINCT ticker), COUNT(*), MIN(ts), MAX(ts) FROM us_minute",
    "us_filings": "SELECT COUNT(DISTINCT ticker), COUNT(*), MIN(filed), MAX(filed) FROM us_filings",
}
STATS_FOR = {"crypto-universe": ["crypto_universe"], "crypto-daily": ["crypto_daily"], "crypto-minute": ["crypto_minute"],
             "crypto-futures": ["crypto_funding", "crypto_metrics"], "us-daily": ["us_daily"], "us-minute": ["us_minute"],
             "news": [], "sec": ["us_filings"], "sec-current": ["us_filings"], "status": list(STATS)}


def write_stats(c, keys):
    now = datetime.now().isoformat(timespec="seconds")
    for k in keys:
        c.execute("INSERT OR REPLACE INTO stats VALUES (?,?,?)", (k, json.dumps(list(c.execute(STATS[k]).fetchone())), now))
    c.commit()


def con():
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB, timeout=120)
    c.execute("PRAGMA journal_mode=WAL")   # 웹(읽기)이 수집(쓰기)에 막히지 않게
    c.executescript(SCHEMA)
    return c


def get(url, params=None, timeout=30, tries=4):
    for k in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            if r.status_code == 404:
                return None
            if r.status_code in (418, 429) or r.status_code >= 500:
                time.sleep(5 * (k + 1))
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            log.warning("요청 실패(%d) %s: %s", k + 1, url[:90], e)
            time.sleep(5 * (k + 1))
    return None


def mark(c, kind, key, n):
    c.execute("INSERT OR REPLACE INTO done VALUES (?,?,?,?)", (kind, key, n, datetime.now().isoformat(timespec="seconds")))
    c.commit()


def is_done(c, kind, key):
    return c.execute("SELECT 1 FROM done WHERE kind=? AND key=?", (kind, key)).fetchone() is not None


def ms(t):
    """바이낸스 아카이브: 2025년부터 현물은 마이크로초 → 밀리초로 통일"""
    t = int(float(t))
    return t // 1000 if t > 10 ** 14 else t


def zip_rows(url):
    r = get(url, timeout=120)
    if r is None:
        return None
    z = zipfile.ZipFile(io.BytesIO(r.content))
    rows = list(csv.reader(io.TextIOWrapper(z.open(z.namelist()[0]), encoding="utf-8")))
    if rows and not rows[0][0].replace(".", "").lstrip("-").isdigit():   # 머리줄 있는 파일(선물)
        rows = rows[1:]
    return rows


# ---------------- 암호화폐 ----------------
def crypto_universe(c, n=30):
    t = get(f"{BV}/ticker/24hr").json()
    rows = [x for x in t if x["symbol"].endswith("USDT")]
    out = []
    for x in rows:
        base = x["symbol"][:-4]
        if base in STABLE or base.endswith(("UP", "DOWN", "BULL", "BEAR")):
            continue
        out.append((x["symbol"], base, float(x["quoteVolume"])))
    # 상장 1년 이상만 (토큰화 주식 SNDKB·SOXLB 같은 신규·이벤트성 종목이 거래대금 상위에 끼는 것 방지, 2026-10-19)
    cutoff = int((datetime.now(timezone.utc) - timedelta(days=365)).timestamp() * 1000)
    keep = []
    for s, b, q in sorted(out, key=lambda r: -r[2])[:120]:
        r = get(f"{BV}/klines", {"symbol": s, "interval": "1M", "startTime": 0, "limit": 1})
        k = r.json() if r is not None else []
        if k and k[0][0] <= cutoff:
            keep.append((s, b, q))
        if len(keep) >= n:
            break
        time.sleep(0.1)
    out = keep
    now = datetime.now().isoformat(timespec="seconds")
    c.execute("DELETE FROM crypto_universe")
    c.executemany("INSERT INTO crypto_universe VALUES (?,?,?,?,?)", [(s, b, q, i + 1, now) for i, (s, b, q) in enumerate(out)])
    c.commit()
    log.info("암호화폐 유니버스 %d: %s", len(out), ", ".join(s for s, _, _ in out[:12]) + " …")
    return [s for s, _, _ in out]


def symbols(c, n=None):
    s = [r[0] for r in c.execute("SELECT symbol FROM crypto_universe ORDER BY rank")]
    return s or crypto_universe(c)


def crypto_daily(c):
    for s in symbols(c):
        last = c.execute("SELECT MAX(date) FROM crypto_daily WHERE symbol=?", (s,)).fetchone()[0]
        start = int(pd.Timestamp(last or "2017-01-01", tz="UTC").timestamp() * 1000)
        n = 0
        while True:
            r = get(f"{BV}/klines", {"symbol": s, "interval": "1d", "startTime": start, "limit": 1000})
            k = r.json() if r is not None else []
            if not k:
                break
            c.executemany("INSERT OR REPLACE INTO crypto_daily VALUES (?,?,?,?,?,?,?,?,?,?)",
                          [(s, datetime.fromtimestamp(x[0] / 1000, timezone.utc).strftime("%Y-%m-%d"), float(x[1]), float(x[2]),
                            float(x[3]), float(x[4]), float(x[5]), float(x[7]), int(x[8]), float(x[10])) for x in k])
            c.commit()
            n += len(k)
            if len(k) < 1000:
                break
            start = k[-1][0] + 1
            time.sleep(0.3)
        log.info("%s 일봉 +%d", s, n)


def _months(n):
    d = date.today().replace(day=1)
    out = []
    for _ in range(n):
        d = (d - timedelta(days=1)).replace(day=1)
        out.append(d.strftime("%Y-%m"))
    return sorted(out)


def _days_this_month():
    t, d = date.today(), date.today().replace(day=1)
    out = []
    while d < t - timedelta(days=1):      # 어제까지 (아카이브는 하루 늦게 올라옴)
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _put_klines(c, s, rows):
    c.executemany("INSERT OR IGNORE INTO crypto_minute VALUES (?,?,?,?,?,?,?,?,?,?)",
                  [(s, ms(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]), float(x[7]), int(x[8]),
                    float(x[10])) for x in rows])


def crypto_minute(c, months=12, top=None):
    syms = symbols(c)[:top] if top else symbols(c)
    for s in syms:
        for m in _months(months):
            key = f"{s}:{m}"
            if is_done(c, "cm_month", key):
                continue
            rows = zip_rows(f"{ARCH}/spot/monthly/klines/{s}/1m/{s}-1m-{m}.zip")
            if rows is None:
                mark(c, "cm_month", key, 0)       # 그 달엔 상장 전
                continue
            _put_klines(c, s, rows)
            mark(c, "cm_month", key, len(rows))
            log.info("%s 1분봉 %s %d개", s, m, len(rows))
        for d in _days_this_month():
            key = f"{s}:{d}"
            if is_done(c, "cm_day", key):
                continue
            rows = zip_rows(f"{ARCH}/spot/daily/klines/{s}/1m/{s}-1m-{d}.zip")
            if rows is None:
                continue                          # 아직 안 올라옴 → 다음에 다시
            _put_klines(c, s, rows)
            mark(c, "cm_day", key, len(rows))
        c.commit()


def crypto_futures(c, months=12, metric_days=90):
    for s in symbols(c):
        for m in _months(months):
            key = f"{s}:{m}"
            if is_done(c, "funding", key):
                continue
            rows = zip_rows(f"{ARCH}/futures/um/monthly/fundingRate/{s}/{s}-fundingRate-{m}.zip")
            if rows:
                c.executemany("INSERT OR IGNORE INTO crypto_funding VALUES (?,?,?)",
                              [(s, ms(x[0]), float(x[2])) for x in rows if len(x) >= 3])
            mark(c, "funding", key, len(rows or []))
        for k in range(metric_days, 0, -1):
            d = (date.today() - timedelta(days=k)).isoformat()
            key = f"{s}:{d}"
            if is_done(c, "metrics", key):
                continue
            rows = zip_rows(f"{ARCH}/futures/um/daily/metrics/{s}/{s}-metrics-{d}.zip")
            if rows is None:
                if k > 2:
                    mark(c, "metrics", key, 0)    # 선물 없는 코인
                continue
            out = []
            for x in rows:
                try:
                    t = int(pd.Timestamp(x[0], tz="UTC").timestamp() * 1000)
                    out.append((s, t, *[float(v) if v not in ("", None) else None for v in x[2:8]]))
                except (ValueError, IndexError):
                    continue
            c.executemany("INSERT OR IGNORE INTO crypto_metrics VALUES (?,?,?,?,?,?,?,?)", out)
            mark(c, "metrics", key, len(out))
        c.commit()
        log.info("%s 선물 펀딩비·미결제약정 갱신", s)


# ---------------- 미국주식 ----------------
def us_tickers(ndx_only=False):
    u = pd.read_csv(rpath("data/raw/us_universe.csv"))
    if ndx_only:
        u = u[u.ndx == 1]
    return list(u.ticker)


def yahoo(t, interval, rng=None, start=None):
    """rng="max" 는 야후가 월봉으로 바꿔 줌(^GSPC 169행) → 일봉 전체는 start(period1) 로 요청 (2026-10-19)"""
    p = {"interval": interval, "includeAdjustedClose": "true", "events": "div,splits"}
    if start is not None:
        p.update(period1=int(pd.Timestamp(start, tz="UTC").timestamp()), period2=int(time.time()))
    else:
        p["range"] = rng
    r = get(f"https://query1.finance.yahoo.com/v8/finance/chart/{t}", p)
    if r is None:
        return None
    res = (r.json().get("chart") or {}).get("result")
    if not res or "timestamp" not in res[0]:
        return None
    q = res[0]["indicators"]["quote"][0]
    df = pd.DataFrame({"ts": res[0]["timestamp"], **{k: q.get(k) for k in ("open", "high", "low", "close", "volume")}})
    adj = res[0]["indicators"].get("adjclose")
    df["adj_close"] = adj[0]["adjclose"] if adj else df["close"]
    return df.dropna(subset=["close"])


def us_daily(c):
    for i, t in enumerate(US_INDEX + us_tickers()):
        last = c.execute("SELECT MAX(date) FROM us_daily WHERE ticker=?", (t,)).fetchone()[0]
        start = "1990-01-01" if last is None else (pd.Timestamp(last) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        df = yahoo(t, "1d", start=start)
        if df is None:
            log.warning("%s 일봉 없음", t)
            continue
        df["date"] = pd.to_datetime(df.ts, unit="s", utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
        c.executemany("INSERT OR REPLACE INTO us_daily VALUES (?,?,?,?,?,?,?,?)",
                      [(t, r.date, r.open, r.high, r.low, r.close, r.adj_close, r.volume) for r in df.itertuples()])
        c.commit()
        if i % 50 == 0:
            log.info("미국 일봉 %d개 종목째 (%s %d행)", i + 1, t, len(df))
        time.sleep(0.4)


def us_minute(c):
    for t in US_INDEX + us_tickers(ndx_only=True):
        df = yahoo(t, "1m", "7d")
        if df is None:
            continue
        c.executemany("INSERT OR IGNORE INTO us_minute VALUES (?,?,?,?,?,?,?)",
                      [(t, int(r.ts), r.open, r.high, r.low, r.close, r.volume) for r in df.itertuples()])
        c.commit()
        time.sleep(0.4)
    log.info("미국 1분봉 갱신 (나스닥100 + 지수)")



# ---------------- SEC 공시 (2026-10-19 사용자 승인: 연락처 이메일 사용) ----------------
def sec_get(url, params=None):
    """SEC 공정 접근 정책: User-Agent 에 연락처, 초당 10건 이하"""
    contact = os.getenv("SEC_CONTACT", "")
    if not contact:
        raise RuntimeError(".env 에 SEC_CONTACT(연락처 이메일)가 없음")
    for k in range(4):
        try:
            r = requests.get(url, params=params, timeout=30,
                             headers={"User-Agent": f"stockrl research {contact}", "Accept-Encoding": "gzip, deflate"})
            if r.status_code == 404:
                return None
            if r.status_code in (403, 429) or r.status_code >= 500:
                time.sleep(10 * (k + 1))
                continue
            r.raise_for_status()
            time.sleep(0.15)
            return r
        except requests.RequestException as e:
            log.warning("SEC 요청 실패(%d) %s: %s", k + 1, url[:80], e)
            time.sleep(5 * (k + 1))
    return None


def sec_ciks(c):
    r = sec_get("https://www.sec.gov/files/company_tickers.json")
    if r is None:
        raise RuntimeError("company_tickers.json 실패")
    m = {v["ticker"].upper().replace(".", "-"): (int(v["cik_str"]), v["title"]) for v in r.json().values()}
    rows = [(t, *m[t]) for t in us_tickers() if t in m]
    c.execute("DELETE FROM us_cik")
    c.executemany("INSERT INTO us_cik VALUES (?,?,?)", rows)
    c.commit()
    log.info("SEC CIK 매핑 %d/%d", len(rows), len(us_tickers()))


def sec_filings(c):
    """종목별 최근 공시(약 1000건씩) — 8-K 항목번호(2.02 실적 등), 10-K/Q, 내부자 거래(4) 포함"""
    if not c.execute("SELECT 1 FROM us_cik LIMIT 1").fetchone():
        sec_ciks(c)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for i, (t, cik) in enumerate(c.execute("SELECT ticker, cik FROM us_cik ORDER BY ticker").fetchall()):
        r = sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
        if r is None:
            continue
        f = r.json()["filings"]["recent"]
        n = len(f["accessionNumber"])
        docs = f.get("primaryDocument", [""] * n)
        desc = f.get("primaryDocDescription", [""] * n)
        rows = [(f["accessionNumber"][k], cik, t, f["form"][k], f["filingDate"][k], f["acceptanceDateTime"][k],
                 f["items"][k], desc[k],
                 f"https://www.sec.gov/Archives/edgar/data/{cik}/{f['accessionNumber'][k].replace('-', '')}/{docs[k]}", now)
                for k in range(n)]
        c.executemany("INSERT OR IGNORE INTO us_filings VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        c.commit()
        if i % 100 == 0:
            log.info("SEC 공시 %d종목째 (%s %d건)", i + 1, t, len(rows))


def sec_current(c):
    """SEC 실시간 8-K 피드(최근 100건) → 우리 종목만"""
    ciks = dict(c.execute("SELECT cik, ticker FROM us_cik").fetchall())
    if not ciks:
        sec_ciks(c)
        ciks = dict(c.execute("SELECT cik, ticker FROM us_cik").fetchall())
    r = sec_get("https://www.sec.gov/cgi-bin/browse-edgar",
                {"action": "getcurrent", "type": "8-K", "owner": "include", "count": "100", "output": "atom"})
    if r is None:
        return
    ns = {"a": "http://www.w3.org/2005/Atom"}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for e in ET.fromstring(r.content).findall("a:entry", ns):
        title = e.findtext("a:title", "", ns)
        m = re.search(r"\((\d{10})\)", title)
        acc = re.search(r"accession-number=([\d-]+)", e.findtext("a:id", "", ns) or "")
        if not m or not acc or int(m.group(1)) not in ciks:
            continue
        cik = int(m.group(1))
        items = ",".join(re.findall(r"Item (\d+\.\d+)", e.findtext("a:summary", "", ns) or ""))
        upd = e.findtext("a:updated", "", ns)
        ln = e.find("a:link", ns)
        rows.append((acc.group(1), cik, ciks[cik], title.split(" - ")[0].strip(), upd[:10], upd, items, title,
                     ln.get("href") if ln is not None else "", now))
    c.executemany("INSERT OR IGNORE INTO us_filings VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    c.commit()
    log.info("SEC 실시간 8-K: 우리 종목 %d건", len(rows))

# ---------------- 뉴스 ----------------
def _rss(url):
    r = get(url, timeout=20)
    if r is None:
        return []
    try:
        root = ET.fromstring(r.content)
    except ET.ParseError:
        return []
    out = []
    for it in root.iter("item"):
        g = lambda k: (it.findtext(k) or "").strip()
        try:
            dt = parsedate_to_datetime(g("pubDate")).astimezone(timezone.utc).isoformat(timespec="seconds")
        except (TypeError, ValueError):
            dt = ""
        out.append({"id": g("guid") or g("link"), "dt": dt, "title": g("title"), "summary": g("description")[:2000],
                    "url": g("link")})
    return out


def news(c):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n = 0
    feeds = [("crypto", "", "cointelegraph", "https://cointelegraph.com/rss"),
             ("crypto", "", "coindesk", "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml")]
    feeds += [("us", t, "yahoo", f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=US&lang=en-US")
              for t in us_tickers(ndx_only=True)]
    for market, sym, src, url in feeds:
        items = _rss(url)
        c.executemany("INSERT OR IGNORE INTO gnews VALUES (?,?,?,?,?,?,?,?,?)",
                      [(x["id"], market, sym, x["dt"], src, x["title"], x["summary"], x["url"], now) for x in items if x["id"]])
        n += c.total_changes
        c.commit()
        time.sleep(0.5 if src == "yahoo" else 0)
    tot = c.execute("SELECT market, COUNT(*) FROM gnews GROUP BY market").fetchall()
    log.info("뉴스 수집 완료 — 누적 %s", dict(tot))


def status(c):
    q = lambda s: c.execute(s).fetchall()
    print("crypto_universe", q("SELECT COUNT(*) FROM crypto_universe"))
    print("crypto_daily", q("SELECT COUNT(DISTINCT symbol), COUNT(*), MIN(date), MAX(date) FROM crypto_daily"))
    print("crypto_minute", q("SELECT COUNT(DISTINCT symbol), COUNT(*) FROM crypto_minute"))
    print("crypto_funding", q("SELECT COUNT(DISTINCT symbol), COUNT(*) FROM crypto_funding"))
    print("crypto_metrics", q("SELECT COUNT(DISTINCT symbol), COUNT(*) FROM crypto_metrics"))
    print("us_daily", q("SELECT COUNT(DISTINCT ticker), COUNT(*), MIN(date), MAX(date) FROM us_daily"))
    print("us_minute", q("SELECT COUNT(DISTINCT ticker), COUNT(*) FROM us_minute"))
    print("gnews", q("SELECT market, source, COUNT(*) FROM gnews GROUP BY market, source"))
    print("us_filings", q("SELECT COUNT(DISTINCT ticker), COUNT(*), MIN(filed), MAX(filed) FROM us_filings"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["crypto-universe", "crypto-daily", "crypto-minute", "crypto-futures", "us-daily",
                                       "us-minute", "news", "sec", "sec-current", "status"])
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--top", type=int)
    a = ap.parse_args()
    c = con()
    {"crypto-universe": lambda: crypto_universe(c), "crypto-daily": lambda: crypto_daily(c),
     "crypto-minute": lambda: crypto_minute(c, a.months, a.top), "crypto-futures": lambda: crypto_futures(c, a.months),
     "us-daily": lambda: us_daily(c), "us-minute": lambda: us_minute(c), "news": lambda: news(c),
     "sec": lambda: sec_filings(c), "sec-current": lambda: sec_current(c),
     "status": lambda: status(c)}[a.action]()
    write_stats(c, STATS_FOR[a.action])


if __name__ == "__main__":
    main()
