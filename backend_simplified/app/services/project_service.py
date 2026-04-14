import logging
from pathlib import Path
from typing import List, Optional

from sqlmodel import Session, select
from sqlalchemy import func

from app.image_processor import (
    invalidate_feature_cache,
    invalidate_similarity_cache,
    is_image_file,
)
from app.models import AnalysisRun, FeatureStore, Image, Project, ProjectRead
from app.runtime import resolve_static_path

logger = logging.getLogger(__name__)


def project_image_counts(session: Session, project_ids: List[int]) -> dict[int, int]:
    if not project_ids:
        return {}

    statement = (
        select(Image.project_id, func.count(Image.id))
        .where(Image.project_id.in_(project_ids))
        .group_by(Image.project_id)
    )
    rows = session.exec(statement).all()
    return {int(project_id): int(count) for project_id, count in rows}


def serialize_project(project: Project, image_count: int = 0) -> ProjectRead:
    payload = project.model_dump()
    payload["image_count"] = image_count
    return ProjectRead.model_validate(payload)


def delete_image_artifacts(session: Session, image: Image, static_dir: Path) -> None:
    full_path = resolve_static_path(static_dir, image.file_path)

    if full_path:
        try:
            full_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Failed to delete image file %s: %s", full_path, exc)

        thumb_dir = static_dir / "thumbnails"
        if thumb_dir.exists():
            for thumb in thumb_dir.glob(f"{image.id}_*.jpg"):
                try:
                    thumb.unlink(missing_ok=True)
                except Exception as exc:
                    logger.warning("Failed to delete thumbnail %s: %s", thumb, exc)

        invalidate_feature_cache(str(full_path))

    invalidate_similarity_cache(session, image.file_hash)

    feature_rows = session.exec(
        select(FeatureStore).where(FeatureStore.image_id == image.id)
    ).all()
    for row in feature_rows:
        session.delete(row)


def delete_project_with_artifacts(
    session: Session, project: Project, static_dir: Path
) -> None:
    images = session.exec(select(Image).where(Image.project_id == project.id)).all()
    for image in images:
        delete_image_artifacts(session, image, static_dir)
        session.delete(image)

    runs = session.exec(
        select(AnalysisRun).where(AnalysisRun.project_id == project.id)
    ).all()
    for run in runs:
        session.delete(run)

    session.delete(project)
    session.commit()


def resolve_download_path(base_dir: Path, file_path: str) -> Path:
    normalized = (
        file_path[len("data/") :] if file_path.startswith("data/") else file_path
    )
    full_path = base_dir / normalized
    try:
        full_path.resolve().relative_to(base_dir.resolve())
    except ValueError as exc:
        raise PermissionError("访问被拒绝") from exc

    if not full_path.exists():
        raise FileNotFoundError("文件不存在")
    return full_path


def detect_media_type(full_path: Path, raw_path: str) -> str:
    if is_image_file(str(full_path)):
        return "image/jpeg"
    if raw_path.endswith(".pdf"):
        return "application/pdf"
    return "application/octet-stream"


def ensure_thumbnail(image: Image, size: int, static_dir: Path) -> Path:
    from PIL import Image as PILImage

    src_path = resolve_static_path(static_dir, image.file_path)
    if src_path is None or not src_path.exists():
        raise FileNotFoundError("Image file not found")

    thumb_dir = static_dir / "thumbnails"
    thumb_dir.mkdir(exist_ok=True)
    thumb_path = thumb_dir / f"{image.id}_{size}.jpg"

    if not thumb_path.exists():
        img = PILImage.open(str(src_path))
        img = img.convert("RGB")
        img.thumbnail((size, size), PILImage.LANCZOS)
        img.save(str(thumb_path), "JPEG", quality=85)

    return thumb_path


def normalize_legacy_paths(session: Session) -> None:
    changed = False
    images = session.exec(select(Image)).all()
    for img in images:
        if img.file_path and img.file_path.startswith("data/"):
            img.file_path = img.file_path[len("data/") :]
            changed = True
        if img.extracted_from and img.extracted_from.startswith("data/"):
            img.extracted_from = img.extracted_from[len("data/") :]
            changed = True
    if changed:
        session.commit()
