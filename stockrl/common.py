import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")  # KRX_ID/PW, DART_API_KEY — pykrx 는 import 시점에 KRX 로그인을 시도한다
# 계속 쓰이는 SQLite·로그는 구글 드라이브(G:) 밖에 둔다 — 드라이브가 쓰는 중인 DB 를 동기화하다 충돌 사본을
# lost_and_found 로 빼냄(2026-10-16, minute.db 6개). 드라이브엔 하루 1번 `python -m stockrl.db backup` 사본만.
DATA_HOME = Path(os.environ.get("STOCKRL_DATA_HOME", Path.home() / "stockrl_data"))
DB_DIR = DATA_HOME / "db"
LOG_DIR = DATA_HOME / "logs"
BACKUP_DIR = ROOT / "data" / "db_backup"


def load_config(path=None):
    path = Path(path or os.environ.get("CONFIG", ROOT / "config" / "train.yaml"))
    if not path.is_absolute():
        path = ROOT / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def rpath(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def get_logger(name):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    return logging.getLogger(name)
