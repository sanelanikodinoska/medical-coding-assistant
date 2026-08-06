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

def _lakebase_url() -> str:
    url = os.environ.get("LAKEBASE_URL")
    if url:
        return url
    from databricks.sdk import WorkspaceClient
    secret = WorkspaceClient().secrets.get_secret(scope=_SCOPE, key=_KEY)
    return base64.b64decode(secret.value).decode("utf-8")

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
