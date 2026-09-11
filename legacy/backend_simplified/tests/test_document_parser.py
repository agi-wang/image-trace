"""Tests for app.document_parser module."""

import os
from pathlib import Path

import pytest
import fitz  # PyMuPDF
from PIL import Image as PILImage

from app.document_parser import DocumentParser


# ---------- helpers ---------------------------------------------------------

def _create_test_pdf(path: Path, with_image: bool = True):
    """用 PyMuPDF 创建一个含/不含图片的 PDF 用于测试。"""
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)

    if with_image:
        # 插入一张红色小图
        img = PILImage.new("RGB", (50, 50), color=(255, 0, 0))
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        rect = fitz.Rect(10, 10, 60, 60)
        page.insert_image(rect, stream=buf.read())

    doc.save(str(path))
    doc.close()


# ---------- DocumentParser Tests --------------------------------------------

class TestDocumentParser:
    @pytest.fixture
    def parser(self, upload_dir, extract_dir):
        return DocumentParser(str(upload_dir), str(extract_dir))

    def test_init(self, parser, upload_dir, extract_dir):
        assert parser.upload_dir == upload_dir
        assert parser.extract_dir == extract_dir

    def test_guess_image_extension_png(self, parser):
        # PNG 文件头
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        ext = parser._guess_image_extension(data)
        assert ext == ".png"

    def test_guess_image_extension_jpeg(self, parser):
        # JPEG 文件头
        data = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        ext = parser._guess_image_extension(data)
        assert ext == ".jpg"

    def test_guess_image_extension_gif(self, parser):
        data = b"GIF89a" + b"\x00" * 100
        ext = parser._guess_image_extension(data)
        assert ext == ".gif"

    def test_guess_image_extension_bmp(self, parser):
        data = b"BM" + b"\x00" * 100
        ext = parser._guess_image_extension(data)
        assert ext == ".bmp"

    def test_guess_image_extension_tiff(self, parser):
        data = b"II*\x00" + b"\x00" * 100
        ext = parser._guess_image_extension(data)
        assert ext == ".tiff"

    def test_guess_image_extension_unknown_defaults_jpg(self, parser):
        """未知文件头默认返回 .jpg。"""
        data = b"\x00\x00\x00\x00" + b"\x00" * 100
        ext = parser._guess_image_extension(data)
        assert ext == ".jpg"

    def test_process_unsupported_format_raises(self, parser, upload_dir):
        """不支持的文档格式应抛出 ValueError。"""
        txt_path = upload_dir / "readme.txt"
        txt_path.write_text("hello")
        with pytest.raises(ValueError, match="不支持的文档格式"):
            parser.process_document(str(txt_path))

    def test_process_nonexistent_file_raises(self, parser):
        """不存在的文件应抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError, match="文件不存在"):
            parser.process_document("/nonexistent/file.pdf")

    def test_process_pdf_with_image(self, parser, upload_dir):
        pdf_path = upload_dir / "test.pdf"
        _create_test_pdf(pdf_path, with_image=True)
        result = parser.process_document(str(pdf_path))
        assert result["status"] == "success"
        # 应该至少提取出嵌入图片或渲染页面
        assert len(result["images"]) >= 1

    def test_process_pdf_without_image(self, parser, upload_dir):
        pdf_path = upload_dir / "empty.pdf"
        _create_test_pdf(pdf_path, with_image=False)
        result = parser.process_document(str(pdf_path))
        assert result["status"] == "success"

    def test_rel_to_base(self, parser, extract_dir):
        """相对路径转换正确。"""
        full_path = extract_dir / "sub" / "img.png"
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.touch()
        rel = parser._rel_to_base(full_path)
        assert not rel.startswith("/")
        assert "extracted" in rel

    def test_extract_images_from_pdf(self, parser, upload_dir):
        """直接测试底层 extract_images_from_document。"""
        pdf_path = upload_dir / "direct.pdf"
        _create_test_pdf(pdf_path, with_image=True)
        images = parser.extract_images_from_document(str(pdf_path))
        assert isinstance(images, list)
        assert len(images) >= 1
        # 每张图片应该有 file_path 和 phash
        for img in images:
            assert "file_path" in img
            assert "phash" in img
