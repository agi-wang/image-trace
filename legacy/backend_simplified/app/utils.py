import logging
import os
from pathlib import Path
from typing import List, Optional, Union, Callable, Any
import shutil

logger = logging.getLogger(__name__)

from sqlmodel import Session, create_engine
from sqlmodel.pool import StaticPool

from .models import Project


def get_database_url(db_path: str = "data/database.db"):
    """获取数据库连接URL"""
    # 优先读取环境变量（便于 Docker/Compose 配置）
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        # sqlite:///relative/path.db 或 sqlite:////absolute/path.db
        if env_url.startswith("sqlite:///"):
            sqlite_path = env_url.replace("sqlite:///", "", 1)
            # relative path 会相对于当前工作目录；这里确保父目录存在
            Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        return env_url

    # 默认使用本地 SQLite 文件
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path}"


def create_db_and_tables(engine):
    """创建数据库和表"""
    from sqlmodel import SQLModel
    from .models import Project, Image

    SQLModel.metadata.create_all(engine)


_ENGINE_CACHE: dict[str, Any] = {}


def _build_engine(database_url: str):
    engine = create_engine(
        database_url,
        poolclass=StaticPool,
        connect_args={
            "check_same_thread": False,
        },
        echo=False,  # 设置为True可以看到SQL日志
    )
    create_db_and_tables(engine)
    return engine


def _is_in_memory_sqlite(database_url: str) -> bool:
    return database_url in {"sqlite://", "sqlite:///:memory:"}


def get_engine(database_url: Optional[str] = None):
    """获取应用级复用 engine；内存 SQLite 保持独立实例。"""
    resolved_url = database_url or get_database_url()
    if _is_in_memory_sqlite(resolved_url):
        return _build_engine(resolved_url)

    engine = _ENGINE_CACHE.get(resolved_url)
    if engine is None:
        engine = _build_engine(resolved_url)
        _ENGINE_CACHE[resolved_url] = engine
    return engine


def get_session(database_url: Optional[str] = None):
    """获取数据库会话"""
    engine = get_engine(database_url)
    return Session(engine)


def ensure_directory(directory: Union[str, Path]) -> Path:
    """确保目录存在"""
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def sanitize_filename(filename: str) -> str:
    """清理客户端传入文件名，阻止路径穿越。"""
    cleaned = Path(filename).name.strip().replace("\x00", "")
    if cleaned in {"", ".", ".."}:
        raise ValueError("文件名无效")
    return cleaned


def resolve_path_within(
    base_dir: Union[str, Path], relative_path: Union[str, Path]
) -> Path:
    """将相对路径解析到指定目录内，越界时抛出异常。"""
    base = Path(base_dir).resolve()
    candidate = (base / relative_path).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError("文件路径越界") from exc
    return candidate


def generate_unique_filename(
    original_filename: str, directory: Union[str, Path]
) -> str:
    """生成唯一的文件名，避免冲突"""
    directory = Path(directory)
    original_filename = sanitize_filename(original_filename)
    name, ext = os.path.splitext(original_filename)

    # 如果文件不存在，直接使用原名
    file_path = directory / original_filename
    if not file_path.exists():
        return original_filename

    # 添加序号直到找到不存在的文件名
    counter = 1
    while True:
        new_filename = f"{name}_{counter}{ext}"
        new_path = directory / new_filename
        if not new_path.exists():
            return new_filename
        counter += 1


def save_upload_file(upload_file, destination: Union[str, Path]) -> Path:
    """保存上传的文件"""
    destination = Path(destination)
    ensure_directory(destination.parent)

    # 生成唯一文件名
    unique_filename = generate_unique_filename(upload_file.filename, destination.parent)
    file_path = destination.parent / unique_filename

    # 保存文件
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return file_path


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """获取文件大小（MB）"""
    return os.path.getsize(file_path) / (1024 * 1024)


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小为可读字符串"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def is_supported_image_format(filename: str) -> bool:
    """检查是否为支持的图像格式（全品种）"""
    from .image_processor import SUPPORTED_IMAGE_EXTENSIONS

    _, ext = os.path.splitext(filename.lower())
    return ext in SUPPORTED_IMAGE_EXTENSIONS


def is_supported_document_format(filename: str) -> bool:
    """检查是否为支持的文档格式"""
    doc_extensions = {".pdf", ".docx", ".pptx"}
    _, ext = os.path.splitext(filename.lower())
    return ext in doc_extensions


def delete_file_if_exists(file_path: Union[str, Path]) -> bool:
    """删除文件（如果存在）"""
    try:
        file_path = Path(file_path)
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    except Exception as exc:
        logger.debug("File deletion failed for %s: %s", file_path, exc)
        return False


def cleanup_project_files(
    project: Project,
    upload_dir: str = "data/uploads",
    extract_dir: str = "data/extracted",
) -> dict:
    """清理项目相关的文件"""
    results = {"deleted_files": [], "errors": []}

    try:
        # 删除项目中的图像文件
        for image in project.images:
            if delete_file_if_exists(image.file_path):
                results["deleted_files"].append(image.file_path)

        # 注意：这里不删除原始上传文件，因为可能被多个项目使用
        # 可以根据需要调整这个策略

    except Exception as e:
        results["errors"].append(str(e))

    return results


def group_similar_by_metric(
    images: List[dict], threshold: float, scorer: Callable[[dict, dict], float]
) -> tuple[list[list[dict]], list[dict]]:
    """
    通用分组：基于自定义相似度 scorer 构建连通分量。
    返回 (groups, ungrouped)
    """
    n = len(images)
    if n <= 1:
        return [], images.copy()

    parent = list(range(n))
    rank = [0] * n

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for i in range(n):
        for j in range(i + 1, n):
            sim = scorer(images[i], images[j])
            if sim >= threshold:
                union(i, j)

    groups_map: dict[int, list[dict]] = {}
    for idx, image in enumerate(images):
        root = find(idx)
        groups_map.setdefault(root, []).append(image)

    groups: list[list[dict]] = []
    ungrouped: list[dict] = []
    for idx in range(n):
        members = groups_map.get(idx)
        if not members:
            continue
        if len(members) > 1:
            groups.append(members)
        else:
            ungrouped.extend(members)

    return groups, ungrouped
