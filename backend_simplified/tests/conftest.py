"""Shared test fixtures for Image Trace backend tests."""

import os
import shutil
import tempfile
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image as PILImage
from PIL import ImageDraw
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool
from fastapi.testclient import TestClient

# -- 目录 fixtures ----------------------------------------------------------


@pytest.fixture
def tmp_dir(tmp_path):
    """提供临时目录，测试结束后自动清理。"""
    return tmp_path


@pytest.fixture
def upload_dir(tmp_dir):
    d = tmp_dir / "uploads"
    d.mkdir()
    return d


@pytest.fixture
def extract_dir(tmp_dir):
    d = tmp_dir / "extracted"
    d.mkdir()
    return d


# -- 图片 fixtures ----------------------------------------------------------


@pytest.fixture
def sample_image(tmp_dir) -> Path:
    """生成一张 100×100 RGB 测试图片（PNG）。"""
    path = tmp_dir / "sample.png"
    img = PILImage.new("RGB", (100, 100), color=(128, 64, 32))
    img.save(str(path))
    return path


@pytest.fixture
def sample_tif_image(tmp_dir) -> Path:
    """生成一张 100×100 RGB TIF 测试图片。"""
    path = tmp_dir / "sample.tif"
    img = PILImage.new("RGB", (100, 100), color=(200, 100, 50))
    img.save(str(path))
    return path


@pytest.fixture
def large_image(tmp_dir) -> Path:
    """生成一张 2048×2048 的大图。"""
    path = tmp_dir / "large.png"
    img = PILImage.new("RGB", (2048, 2048), color=(64, 128, 255))
    img.save(str(path))
    return path


@pytest.fixture
def sample_image_pair(tmp_dir):
    """生成两张图片：一张相同内容（重复），一张不同内容。"""
    img_a = PILImage.new("RGB", (100, 100), color=(128, 64, 32))
    img_b = PILImage.new("RGB", (100, 100), color=(128, 64, 32))  # 同色
    img_c = PILImage.new("RGB", (100, 100), color=(0, 255, 0))  # 不同色

    path_a = tmp_dir / "pair_a.png"
    path_b = tmp_dir / "pair_b.png"
    path_c = tmp_dir / "pair_c.png"
    img_a.save(str(path_a))
    img_b.save(str(path_b))
    img_c.save(str(path_c))
    return path_a, path_b, path_c


# -- 数据库 fixtures --------------------------------------------------------


@pytest.fixture
def db_session():
    """提供内存 SQLite 会话，每个测试独立。"""
    engine = create_engine(
        "sqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


# -- FastAPI TestClient fixture ---------------------------------------------


class _SyncASGIClient:
    """httpx.AsyncClient 的同步包装，用于测试 ASGI 应用。"""

    def __init__(self, app):
        import httpx

        self._app = app
        self._transport = httpx.ASGITransport(app=app)
        self._client = httpx.AsyncClient(
            transport=self._transport, base_url="http://testserver"
        )
        self._lifespan = app.router.lifespan_context(app)
        self._run(self._lifespan.__aenter__())

    def _run(self, coro):
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def get(self, url, **kwargs):
        return self._run(self._client.get(url, **kwargs))

    def post(self, url, **kwargs):
        return self._run(self._client.post(url, **kwargs))

    def put(self, url, **kwargs):
        return self._run(self._client.put(url, **kwargs))

    def delete(self, url, **kwargs):
        return self._run(self._client.delete(url, **kwargs))

    def close(self):
        try:
            self._run(self._client.aclose())
        finally:
            if self._lifespan is not None:
                self._run(self._lifespan.__aexit__(None, None, None))
                self._lifespan = None


@pytest.fixture
def client(tmp_dir):
    """
    创建指向临时目录 + 临时文件 DB 的 TestClient。
    通过环境变量注入配置，然后重新 import app。
    """
    upload_d = tmp_dir / "uploads"
    extract_d = tmp_dir / "extracted"
    static_d = tmp_dir
    upload_d.mkdir(exist_ok=True)
    extract_d.mkdir(exist_ok=True)

    db_path = tmp_dir / "test.db"

    os.environ["UPLOAD_DIR"] = str(upload_d)
    os.environ["EXTRACT_DIR"] = str(extract_d)
    os.environ["STATIC_DIR"] = str(static_d)
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    os.environ["DESCRIPTOR_DIR"] = str(tmp_dir / "descriptors")

    # 重新加载模块以使环境变量生效
    import importlib
    import app.image_processor as ip_mod
    import app.utils as utils_mod
    import app.main as main_mod

    importlib.reload(ip_mod)
    importlib.reload(utils_mod)
    importlib.reload(main_mod)

    c = _SyncASGIClient(main_mod.app)
    yield c
    c.close()

    # 清理环境变量
    for key in [
        "UPLOAD_DIR",
        "EXTRACT_DIR",
        "STATIC_DIR",
        "DATABASE_URL",
        "DESCRIPTOR_DIR",
    ]:
        os.environ.pop(key, None)


# -- 辅助函数 ---------------------------------------------------------------


def make_upload_bytes(
    filename: str = "test.png", size=(100, 100), color=(128, 64, 32)
) -> tuple:
    """生成用于 TestClient 上传的 (filename, BytesIO, content_type) 元组。"""
    buf = BytesIO()
    img = PILImage.new("RGB", size, color=color)
    ext = filename.rsplit(".", 1)[-1].lower()
    fmt = {
        "png": "PNG",
        "jpg": "JPEG",
        "jpeg": "JPEG",
        "tif": "TIFF",
        "tiff": "TIFF",
        "bmp": "BMP",
    }.get(ext, "PNG")
    img.save(buf, format=fmt)
    buf.seek(0)
    ct = {"png": "image/png", "jpg": "image/jpeg", "tif": "image/tiff"}.get(
        ext, "image/png"
    )
    return (filename, buf, ct)


def make_pattern_upload_bytes(filename: str = "pattern.png", size=(240, 240)) -> tuple:
    """生成带有明显特征点的测试图片。"""
    buf = BytesIO()
    img = PILImage.new("RGB", size, color=(245, 245, 245))
    draw = ImageDraw.Draw(img)

    draw.rectangle((20, 20, size[0] - 20, size[1] - 20), outline=(30, 30, 30), width=4)
    draw.line((20, 20, size[0] - 20, size[1] - 20), fill=(220, 40, 40), width=5)
    draw.line((size[0] - 20, 20, 20, size[1] - 20), fill=(40, 90, 220), width=5)
    draw.ellipse((70, 50, 170, 150), outline=(30, 160, 70), width=6)
    draw.rectangle(
        (95, 95, 145, 145), fill=(250, 190, 40), outline=(20, 20, 20), width=3
    )

    ext = filename.rsplit(".", 1)[-1].lower()
    fmt = {
        "png": "PNG",
        "jpg": "JPEG",
        "jpeg": "JPEG",
        "tif": "TIFF",
        "tiff": "TIFF",
        "bmp": "BMP",
    }.get(ext, "PNG")
    img.save(buf, format=fmt)
    buf.seek(0)
    ct = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "tif": "image/tiff",
        "tiff": "image/tiff",
        "bmp": "image/bmp",
    }.get(ext, "image/png")
    return (filename, buf, ct)
