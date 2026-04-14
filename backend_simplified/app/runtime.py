import logging
import os
from pathlib import Path
from typing import Optional

from sqlmodel import Session

from app.document_parser import DocumentParser
from app.feature_matrix import precompute_feature_matrix
from app.utils import ensure_directory, get_database_url, get_engine, get_session

logger = logging.getLogger(__name__)


def get_upload_dir() -> Path:
    path = Path(os.getenv("UPLOAD_DIR", "data/uploads"))
    ensure_directory(path)
    return path


def get_extract_dir() -> Path:
    path = Path(os.getenv("EXTRACT_DIR", "data/extracted"))
    ensure_directory(path)
    return path


def get_static_dir() -> Path:
    path = Path(os.getenv("STATIC_DIR", "data"))
    ensure_directory(path)
    return path


def get_desc_dir() -> Path:
    path = Path(os.getenv("DESCRIPTOR_DIR", "data/descriptors"))
    ensure_directory(path)
    return path


def build_document_parser() -> DocumentParser:
    return DocumentParser(str(get_upload_dir()), str(get_extract_dir()))


def get_db():
    database_url = get_database_url()
    with get_session(database_url) as session:
        yield session


def bg_precompute_features(image_id: int, image_path: str):
    """Background task: precompute feature matrix for a single image."""
    from sqlmodel import Session as _Session

    try:
        engine = get_engine()
        with _Session(engine) as session:
            precompute_feature_matrix(image_id, image_path, session)
    except Exception as exc:
        logger.warning("Background precompute failed for image %s: %s", image_id, exc)


def ensure_precompute(background_tasks, image_id: int, image_path: str):
    """Always trigger precomputation: via BackgroundTasks or thread fallback."""
    if background_tasks is not None:
        background_tasks.add_task(bg_precompute_features, image_id, image_path)
        return

    import threading

    worker = threading.Thread(
        target=bg_precompute_features, args=(image_id, image_path), daemon=True
    )
    worker.start()


def resolve_static_path(static_dir: Path, file_path: Optional[str]) -> Optional[Path]:
    if not file_path:
        return None
    fp = file_path
    if fp.startswith("data/"):
        fp = fp[len("data/") :]
    return static_dir / fp


def resolve_static_file_path(file_path: Optional[str]):
    return resolve_static_path(get_static_dir(), file_path)


def migrate_db_schema(database_url: str):
    """Auto-migrate SQLite schema: add missing columns to existing tables."""
    import sqlite3

    if not database_url.startswith("sqlite"):
        return
    db_path = database_url.replace("sqlite:///", "")
    if not os.path.exists(db_path):
        return
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute("PRAGMA table_info(images)")
        existing_cols = {row[1] for row in cursor.fetchall()}
        expected_optional = {
            "colorhash": "VARCHAR",
            "dhash": "VARCHAR",
            "ahash": "VARCHAR",
            "whash": "VARCHAR",
            "extracted_from": "VARCHAR",
            "file_size": "INTEGER",
            "width": "INTEGER",
            "height": "INTEGER",
        }
        for col, col_type in expected_optional.items():
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE images ADD COLUMN {col} {col_type}")
                logger.info("DB migration: added column images.%s", col)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS similarity_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hash_a VARCHAR(32),
                hash_b VARCHAR(32),
                algorithm VARCHAR(32),
                rotation_invariant BOOLEAN DEFAULT 0,
                score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_simcache_a ON similarity_cache(hash_a)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_simcache_b ON similarity_cache(hash_b)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_simcache_algo ON similarity_cache(algorithm)"
        )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_store (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL REFERENCES images(id),
                variant_idx INTEGER NOT NULL DEFAULT 0,
                algorithm VARCHAR(32) NOT NULL,
                vector TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(image_id, variant_idx, algorithm)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fs_image ON feature_store(image_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fs_algo ON feature_store(algorithm)"
        )
        if "feature_status" not in existing_cols:
            conn.execute(
                "ALTER TABLE images ADD COLUMN feature_status VARCHAR(16) DEFAULT 'pending'"
            )
            logger.info("DB migration: added column images.feature_status")
        conn.commit()
    except Exception as e:
        logger.warning("DB migration error: %s", e)
    finally:
        conn.close()
