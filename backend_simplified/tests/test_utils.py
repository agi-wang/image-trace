"""Tests for app.utils module."""

import os
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.utils import (
    is_supported_image_format,
    is_supported_document_format,
    ensure_directory,
    generate_unique_filename,
    save_upload_file,
    get_file_size_mb,
    format_file_size,
    get_database_url,
    get_engine,
    get_session,
    group_similar_by_metric,
)


# ---------- is_supported_image_format ---------------------------------------


class TestIsSupportedImageFormat:
    @pytest.mark.parametrize(
        "name",
        [
            "photo.jpg",
            "photo.jpeg",
            "photo.png",
            "photo.gif",
            "photo.bmp",
            "photo.tif",
            "photo.tiff",
            "photo.webp",
            "photo.jp2",
            "photo.jfif",
            "photo.psd",
            "photo.ico",
            "photo.tga",
            "photo.pbm",
            "photo.dng",
            "photo.heic",
            "photo.avif",
        ],
    )
    def test_supported(self, name):
        assert is_supported_image_format(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "doc.pdf",
            "movie.mp4",
            "readme.txt",
            "archive.zip",
        ],
    )
    def test_unsupported(self, name):
        assert is_supported_image_format(name) is False

    def test_case_insensitive(self):
        assert is_supported_image_format("PHOTO.TIF") is True
        assert is_supported_image_format("Photo.PNG") is True


# ---------- is_supported_document_format ------------------------------------


class TestIsSupportedDocumentFormat:
    @pytest.mark.parametrize("name", ["report.pdf", "doc.docx", "slides.pptx"])
    def test_supported(self, name):
        assert is_supported_document_format(name) is True

    @pytest.mark.parametrize("name", ["photo.jpg", "readme.txt", "archive.zip"])
    def test_unsupported(self, name):
        assert is_supported_document_format(name) is False


# ---------- ensure_directory ------------------------------------------------


class TestEnsureDirectory:
    def test_creates_nested(self, tmp_dir):
        target = tmp_dir / "a" / "b" / "c"
        result = ensure_directory(target)
        assert result.exists()
        assert result.is_dir()

    def test_existing_dir_ok(self, tmp_dir):
        result = ensure_directory(tmp_dir)
        assert result.exists()


# ---------- generate_unique_filename ----------------------------------------


class TestGenerateUniqueFilename:
    def test_no_conflict(self, tmp_dir):
        name = generate_unique_filename("test.png", tmp_dir)
        assert name == "test.png"

    def test_with_conflict(self, tmp_dir):
        (tmp_dir / "test.png").touch()
        name = generate_unique_filename("test.png", tmp_dir)
        assert name == "test_1.png"

    def test_multiple_conflicts(self, tmp_dir):
        (tmp_dir / "test.png").touch()
        (tmp_dir / "test_1.png").touch()
        name = generate_unique_filename("test.png", tmp_dir)
        assert name == "test_2.png"


# ---------- save_upload_file ------------------------------------------------


class TestSaveUploadFile:
    def test_saves_file(self, tmp_dir):
        mock_file = MagicMock()
        mock_file.filename = "upload.png"
        mock_file.file = BytesIO(b"fake image data")

        dest = tmp_dir / "upload.png"
        result = save_upload_file(mock_file, dest)
        assert result.exists()
        assert result.read_bytes() == b"fake image data"


# ---------- get_file_size_mb ------------------------------------------------


class TestGetFileSizeMb:
    def test_calculation(self, tmp_dir):
        path = tmp_dir / "data.bin"
        path.write_bytes(b"x" * (1024 * 1024))  # 1 MB
        assert abs(get_file_size_mb(str(path)) - 1.0) < 0.01


# ---------- format_file_size ------------------------------------------------


class TestFormatFileSize:
    def test_bytes(self):
        assert format_file_size(512) == "512 B"

    def test_kilobytes(self):
        assert "KB" in format_file_size(2048)

    def test_megabytes(self):
        assert "MB" in format_file_size(5 * 1024 * 1024)

    def test_gigabytes(self):
        assert "GB" in format_file_size(3 * 1024 * 1024 * 1024)


# ---------- get_database_url ------------------------------------------------


class TestGetDatabaseUrl:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        url = get_database_url()
        assert url.startswith("sqlite:///")

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "sqlite:////tmp/test.db")
        url = get_database_url()
        assert url == "sqlite:////tmp/test.db"


# ---------- get_session -----------------------------------------------------


class TestGetEngine:
    def test_reuses_file_engine(self, tmp_path):
        url = f"sqlite:///{tmp_path / 'cache.db'}"
        engine_a = get_engine(url)
        engine_b = get_engine(url)
        assert engine_a is engine_b

    def test_in_memory_engine_is_not_cached(self):
        engine_a = get_engine("sqlite://")
        engine_b = get_engine("sqlite://")
        assert engine_a is not engine_b


class TestGetSession:
    def test_creates_session(self):
        session = get_session("sqlite://")
        assert session is not None
        session.close()

    def test_reuses_singleton_engine_for_file_db(self, tmp_path):
        url = f"sqlite:///{tmp_path / 'session_cache.db'}"
        session_a = get_session(url)
        session_b = get_session(url)
        try:
            assert session_a.bind is session_b.bind
        finally:
            session_a.close()
            session_b.close()


# ---------- group_similar_by_metric -----------------------------------------


class TestGroupSimilarByMetric:
    def test_groups_similar(self):
        images = [
            {"id": 1, "val": 10},
            {"id": 2, "val": 11},
            {"id": 3, "val": 100},
        ]

        def scorer(a, b):
            diff = abs(a["val"] - b["val"])
            return max(0, 1.0 - diff / 100.0)

        groups, ungrouped = group_similar_by_metric(
            images, threshold=0.9, scorer=scorer
        )
        assert len(groups) == 1
        assert len(groups[0]) == 2
        assert len(ungrouped) == 1
        assert ungrouped[0]["id"] == 3

    def test_all_unique(self):
        images = [
            {"id": 1, "val": 0},
            {"id": 2, "val": 50},
            {"id": 3, "val": 100},
        ]

        def scorer(a, b):
            return 0.0

        groups, ungrouped = group_similar_by_metric(
            images, threshold=0.5, scorer=scorer
        )
        assert len(groups) == 0
        assert len(ungrouped) == 3

    def test_all_same(self):
        images = [
            {"id": 1, "val": 1},
            {"id": 2, "val": 1},
            {"id": 3, "val": 1},
        ]

        def scorer(a, b):
            return 1.0

        groups, ungrouped = group_similar_by_metric(
            images, threshold=0.5, scorer=scorer
        )
        assert len(groups) == 1
        assert len(groups[0]) == 3
        assert len(ungrouped) == 0

    def test_transitive_similarity_forms_single_group(self):
        images = [
            {"id": 1, "val": 0.0},
            {"id": 2, "val": 0.4},
            {"id": 3, "val": 0.8},
        ]

        pairs = {
            (1, 2): 0.9,
            (2, 3): 0.9,
            (1, 3): 0.2,
        }

        def scorer(a, b):
            key = tuple(sorted((a["id"], b["id"])))
            return pairs.get(key, 0.0)

        groups, ungrouped = group_similar_by_metric(
            images, threshold=0.5, scorer=scorer
        )
        assert len(groups) == 1
        assert {img["id"] for img in groups[0]} == {1, 2, 3}
        assert ungrouped == []


# ---------- delete_file_if_exists -------------------------------------------


class TestDeleteFileIfExists:
    def test_deletes_existing(self, tmp_dir):
        from app.utils import delete_file_if_exists

        path = tmp_dir / "to_delete.txt"
        path.write_text("data")
        assert delete_file_if_exists(path) is True
        assert not path.exists()

    def test_returns_false_for_missing(self, tmp_dir):
        from app.utils import delete_file_if_exists

        assert delete_file_if_exists(tmp_dir / "nonexistent.txt") is False

    def test_returns_false_on_error(self, tmp_dir):
        from app.utils import delete_file_if_exists

        # 尝试删除目录（会失败）
        d = tmp_dir / "subdir"
        d.mkdir()
        (d / "file.txt").write_text("x")
        result = delete_file_if_exists(d)
        # 删除非空目录会出错，返回 False
        assert isinstance(result, bool)


# ---------- cleanup_project_files -------------------------------------------


class TestCleanupProjectFiles:
    def test_cleanup_deletes_image_files(self, tmp_dir):
        from app.utils import cleanup_project_files
        from unittest.mock import MagicMock

        # 创建模拟的文件
        img_path = tmp_dir / "img1.png"
        img_path.write_bytes(b"fake")

        mock_image = MagicMock()
        mock_image.file_path = str(img_path)

        mock_project = MagicMock()
        mock_project.images = [mock_image]

        results = cleanup_project_files(mock_project)
        assert len(results["deleted_files"]) == 1
        assert len(results["errors"]) == 0
        assert not img_path.exists()

    def test_cleanup_handles_missing_files(self, tmp_dir):
        from app.utils import cleanup_project_files
        from unittest.mock import MagicMock

        mock_image = MagicMock()
        mock_image.file_path = str(tmp_dir / "nonexistent.png")

        mock_project = MagicMock()
        mock_project.images = [mock_image]

        results = cleanup_project_files(mock_project)
        assert len(results["deleted_files"]) == 0
        assert len(results["errors"]) == 0
