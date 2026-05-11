import aiosqlite
from app.config import settings

DB_PATH = str(settings.db_path)

CREATE_JOBS_TABLE = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id      TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    filename    TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'received',
    progress    INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""

CREATE_DOCS_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    tree_path   TEXT,
    page_count  INTEGER,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(CREATE_JOBS_TABLE)
        await db.execute(CREATE_DOCS_TABLE)
        await db.commit()

async def get_db():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        yield db