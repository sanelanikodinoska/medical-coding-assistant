"""
Lakebase (Postgres) connection helper.
The URL is fetched from Databricks secret scope once and cached for the lifetime
of the process — avoiding repeated SDK calls on every query.
"""

import base64, os
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_SCOPE = os.environ.get("LAKEBASE_SECRET_SCOPE", "database")
_KEY   = os.environ.get("LAKEBASE_SECRET_KEY",   "lakebase-url")

# Cached at module level — fetched once on first use
_cached_url: str | None = None

def _lakebase_url() -> str:
    global _cached_url
    if _cached_url:
        return _cached_url
    url = os.environ.get("LAKEBASE_URL")
    if not url:
        from databricks.sdk import WorkspaceClient
        secret = WorkspaceClient().secrets.get_secret(scope=_SCOPE, key=_KEY)
        url = base64.b64decode(secret.value).decode("utf-8")
    _cached_url = url
    return url

@contextmanager
def get_connection():
    conn = psycopg2.connect(_lakebase_url(), cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()

def run_query(sql, params=None):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

def run_write(sql, params=None):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()
            return cur.rowcount

def run_write_returning(sql, params=None):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()
            row = cur.fetchone()
            return dict(row) if row else {}
