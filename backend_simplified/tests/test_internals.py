"""Tests for internal helper functions across runtime.py, services, and utils.py."""

import importlib
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import numpy as np
from PIL import Image as PILImage
from sqlmodel import Session, SQLModel, create_engine, select
from sqlmodel.pool import StaticPool

from app.models import Image, Project, FeatureStore, SimilarityCache


# ============================================================================
#  utils path helpers
# ============================================================================


class TestUtilsPathHelpers:
    def test_sanitize_filename_strips_directories(self):
        from app.utils import sanitize_filename

        assert sanitize_filename("../nested/demo.png") == "demo.png"

    def test_resolve_path_within_rejects_traversal(self, tmp_path):
        from app.utils import resolve_path_within

        with pytest.raises(ValueError):
            resolve_path_within(tmp_path, "../escape.txt")


# ============================================================================
#  resolve_static_file_path (runtime.py)
# ============================================================================


class TestResolveStaticPath:
    @pytest.fixture(autouse=True)
    def _patch_static_dir(self, tmp_path, monkeypatch):
        import app.runtime as runtime_mod

        monkeypatch.setattr(runtime_mod, "get_static_dir", lambda: tmp_path)
        self.static_dir = tmp_path

    def test_none_returns_none(self):
        from app.runtime import resolve_static_file_path

        assert resolve_static_file_path(None) is None

    def test_empty_returns_none(self):
        from app.runtime import resolve_static_file_path

        assert resolve_static_file_path("") is None

    def test_strips_data_prefix(self):
        from app.runtime import resolve_static_file_path

        result = resolve_static_file_path("data/uploads/img.png")
        assert result == self.static_dir / "uploads" / "img.png"

    def test_no_data_prefix(self):
        from app.runtime import resolve_static_file_path

        result = resolve_static_file_path("uploads/img.png")
        assert result == self.static_dir / "uploads" / "img.png"


# ============================================================================
#  delete_image_artifacts (services/project_service.py)
# ============================================================================


class TestDeleteImageArtifacts:
    @pytest.fixture
    def setup_env(self, tmp_path):
        engine = create_engine(
            "sqlite://",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        SQLModel.metadata.create_all(engine)
        with Session(engine) as session:
            project = Project(name="test-proj")
            session.add(project)
            session.commit()
            session.refresh(project)

            uploads_dir = tmp_path / "uploads"
            uploads_dir.mkdir()
            img_file = uploads_dir / "test_img.png"
            PILImage.new("RGB", (10, 10)).save(str(img_file))

            image = Image(
                filename="test_img.png",
                project_id=project.id,
                file_path="uploads/test_img.png",
                file_hash="abc123",
                phash="0" * 16,
            )
            session.add(image)
            session.commit()
            session.refresh(image)

            yield session, image, img_file, tmp_path

    def test_deletes_file_and_cleans(self, setup_env):
        from app.services.project_service import delete_image_artifacts

        session, image, img_file, static_dir = setup_env
        assert img_file.exists()

        delete_image_artifacts(session, image, static_dir)

        assert not img_file.exists()

    def test_handles_missing_file(self, setup_env):
        from app.services.project_service import delete_image_artifacts

        session, image, img_file, static_dir = setup_env

        img_file.unlink()
        delete_image_artifacts(session, image, static_dir)


# ============================================================================
#  migrate_db_schema (runtime.py)
# ============================================================================


class TestMigrateDbSchema:
    def test_skips_non_sqlite(self):
        from app.runtime import migrate_db_schema

        migrate_db_schema("postgresql://localhost/test")

    def test_skips_nonexistent_db(self, tmp_path):
        from app.runtime import migrate_db_schema

        db_path = str(tmp_path / "nonexistent.db")
        migrate_db_schema(f"sqlite:///{db_path}")
        assert not (tmp_path / "nonexistent.db").exists()

    def test_adds_missing_columns(self, tmp_path):
        from app.runtime import migrate_db_schema

        db_path = str(tmp_path / "migrate_test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE images (
                id INTEGER PRIMARY KEY,
                filename VARCHAR(255),
                file_path VARCHAR(500),
                file_hash VARCHAR(32),
                phash VARCHAR(64)
            )
        """)
        conn.commit()
        conn.close()

        migrate_db_schema(f"sqlite:///{db_path}")

        conn = sqlite3.connect(db_path)
        cursor = conn.execute("PRAGMA table_info(images)")
        cols = {row[1] for row in cursor.fetchall()}
        conn.close()

        for expected_col in [
            "colorhash",
            "dhash",
            "ahash",
            "whash",
            "extracted_from",
            "file_size",
            "width",
            "height",
            "feature_status",
        ]:
            assert expected_col in cols, f"Missing column: {expected_col}"

    def test_creates_similarity_cache_table(self, tmp_path):
        from app.runtime import migrate_db_schema

        db_path = str(tmp_path / "migrate_tables.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE images (
                id INTEGER PRIMARY KEY,
                filename VARCHAR, file_path VARCHAR, file_hash VARCHAR, phash VARCHAR,
                colorhash VARCHAR, dhash VARCHAR, ahash VARCHAR, whash VARCHAR,
                extracted_from VARCHAR, file_size INTEGER, width INTEGER, height INTEGER,
                feature_status VARCHAR DEFAULT 'pending'
            )
        """)
        conn.commit()
        conn.close()

        migrate_db_schema(f"sqlite:///{db_path}")

        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='similarity_cache'"
        )
        assert cursor.fetchone() is not None, "similarity_cache table should exist"

        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='feature_store'"
        )
        assert cursor.fetchone() is not None, "feature_store table should exist"
        conn.close()

    def test_idempotent(self, tmp_path):
        from app.runtime import migrate_db_schema

        db_path = str(tmp_path / "idempotent.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE images (
                id INTEGER PRIMARY KEY,
                filename VARCHAR, file_path VARCHAR, file_hash VARCHAR, phash VARCHAR
            )
        """)
        conn.commit()
        conn.close()

        migrate_db_schema(f"sqlite:///{db_path}")
        migrate_db_schema(f"sqlite:///{db_path}")


# ============================================================================
#  _initialize_app_state (main.py)
# ============================================================================


class TestInitializeAppState:
    def test_normalizes_legacy_data_paths(self, tmp_path, monkeypatch):
        db_path = tmp_path / "startup.db"
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
        monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
        monkeypatch.setenv("EXTRACT_DIR", str(tmp_path / "extracted"))
        monkeypatch.setenv("STATIC_DIR", str(tmp_path))

        import app.utils as utils_mod
        import app.main as main_mod

        importlib.reload(utils_mod)
        importlib.reload(main_mod)

        engine = utils_mod.get_engine(f"sqlite:///{db_path}")
        with Session(engine) as session:
            project = Project(name="LegacyPaths")
            session.add(project)
            session.commit()
            session.refresh(project)

            image = Image(
                filename="legacy.png",
                project_id=project.id,
                file_path="data/uploads/legacy.png",
                extracted_from="data/extracted/source.pdf",
                file_hash="legacy-hash",
                phash="abcd",
            )
            session.add(image)
            session.commit()

        main_mod._initialize_app_state()

        with Session(engine) as session:
            saved = session.exec(select(Image)).one()
            assert saved.file_path == "uploads/legacy.png"
            assert saved.extracted_from == "extracted/source.pdf"
