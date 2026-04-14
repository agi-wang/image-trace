"""Tests for DocumentParser internal methods: _convert_to_jpeg, _render_pdf_pages."""

import io
from pathlib import Path

import pytest
from PIL import Image as PILImage

from app.document_parser import DocumentParser


# ============================================================================
#  _convert_to_jpeg
# ============================================================================

class TestConvertToJpeg:
    @pytest.fixture
    def parser(self, tmp_path):
        upload_dir = tmp_path / "uploads"
        extract_dir = tmp_path / "extracted"
        upload_dir.mkdir()
        extract_dir.mkdir()
        return DocumentParser(str(upload_dir), str(extract_dir))

    def test_rgb_image_converted(self, parser, tmp_path):
        """RGB PNG is converted to JPEG."""
        img = PILImage.new("RGB", (50, 50), color=(255, 0, 0))
        src = tmp_path / "test_rgb.png"
        img.save(str(src))

        result = parser._convert_to_jpeg(src)

        assert result.suffix == ".jpg"
        assert result.exists()
        # Original .png should be deleted
        assert not src.exists()
        # Verify it's a valid JPEG
        with PILImage.open(result) as loaded:
            assert loaded.format == "JPEG"

    def test_rgba_image_converted(self, parser, tmp_path):
        """RGBA image is converted to RGB JPEG (alpha stripped)."""
        img = PILImage.new("RGBA", (50, 50), color=(255, 0, 0, 128))
        src = tmp_path / "test_rgba.png"
        img.save(str(src))

        result = parser._convert_to_jpeg(src)

        assert result.suffix == ".jpg"
        assert result.exists()
        with PILImage.open(result) as loaded:
            assert loaded.mode == "RGB"

    def test_corrupt_file_returns_original(self, parser, tmp_path):
        """Corrupt file returns original path without error."""
        src = tmp_path / "corrupt.bmp"
        src.write_bytes(b"not a real image file")

        result = parser._convert_to_jpeg(src)

        # Should return original path as fallback
        assert result == src


# ============================================================================
#  _render_pdf_pages
# ============================================================================

class TestRenderPdfPages:
    @pytest.fixture
    def parser(self, tmp_path):
        upload_dir = tmp_path / "uploads"
        extract_dir = tmp_path / "extracted"
        upload_dir.mkdir()
        extract_dir.mkdir()
        return DocumentParser(str(upload_dir), str(extract_dir))

    def _create_simple_pdf(self, path: Path):
        """Create a minimal 1-page PDF using PyMuPDF."""
        import fitz
        doc = fitz.open()
        page = doc.new_page(width=200, height=200)
        # Draw something so it's not blank
        shape = page.new_shape()
        shape.draw_rect(fitz.Rect(10, 10, 190, 190))
        shape.finish(color=(1, 0, 0), fill=(0, 0, 1))
        shape.commit()
        doc.save(str(path))
        doc.close()

    def test_renders_single_page(self, parser, tmp_path):
        """Single-page PDF renders to JPEG."""
        pdf_path = tmp_path / "test.pdf"
        self._create_simple_pdf(pdf_path)

        results = parser._render_pdf_pages(pdf_path, "test")

        assert len(results) == 1
        result = results[0]
        assert result["extraction_method"] == "rendered"
        assert result["page_number"] == 1
        assert "phash" in result
        assert result["filename"].endswith(".jpg")

    def test_handles_corrupt_pdf(self, parser, tmp_path):
        """Corrupt PDF returns empty list without crashing."""
        pdf_path = tmp_path / "corrupt.pdf"
        pdf_path.write_bytes(b"not a pdf")

        results = parser._render_pdf_pages(pdf_path, "corrupt")

        assert results == []

    def test_multipage_pdf(self, parser, tmp_path):
        """Multi-page PDF renders all pages."""
        import fitz
        pdf_path = tmp_path / "multi.pdf"
        doc = fitz.open()
        for _ in range(3):
            page = doc.new_page(width=100, height=100)
            shape = page.new_shape()
            shape.draw_circle(fitz.Point(50, 50), 30)
            shape.finish(color=(0, 1, 0))
            shape.commit()
        doc.save(str(pdf_path))
        doc.close()

        results = parser._render_pdf_pages(pdf_path, "multi")

        assert len(results) == 3
        for i, r in enumerate(results, 1):
            assert r["page_number"] == i
