from pathlib import Path
from typing import Callable, Optional
import logging
from pathlib import Path
from typing import Callable, Optional

from sqlmodel import Session, select

logger = logging.getLogger(__name__)

from app.document_parser import DocumentParser
from app.image_processor import compute_image_features
from app.models import Image
from app.utils import (
    is_supported_document_format,
    is_supported_image_format,
    resolve_path_within,
)


class UnsupportedUploadFormatError(ValueError):
    pass


class DocumentExtractionError(ValueError):
    pass


def _relative_to_static(saved_path: Path, static_dir: Path) -> str:
    try:
        return str(saved_path.relative_to(static_dir))
    except Exception as exc:
        logger.debug(
            "Path relativization failed for %s against %s: %s",
            saved_path,
            static_dir,
            exc,
        )
        return str(saved_path)


def _persist_extracted_images(
    session: Session,
    extraction_result: dict,
    project_id: int,
) -> list[Image]:
    db_images: list[Image] = []
    for img_info in extraction_result["images"]:
        img_info["project_id"] = project_id
        db_image = Image.model_validate(img_info)
        session.add(db_image)
        db_images.append(db_image)

    session.commit()
    for db_image in db_images:
        session.refresh(db_image)

    return db_images


def process_uploaded_file(
    session: Session,
    project_id: int,
    safe_filename: str,
    saved_path: Path,
    static_dir: Path,
    doc_parser: DocumentParser,
    enqueue_precompute: Optional[Callable[[int, str], None]] = None,
) -> dict:
    saved_rel_path = _relative_to_static(saved_path, static_dir)
    result = {
        "project_id": project_id,
        "filename": safe_filename,
        "file_path": saved_rel_path,
        "file_size": saved_path.stat().st_size,
        "file_type": "unknown",
        "processed_images": [],
        "error": None,
    }

    if is_supported_image_format(safe_filename):
        result["file_type"] = "image"

        features = compute_image_features(str(saved_path))
        features.update(
            {
                "filename": safe_filename,
                "project_id": project_id,
                "file_path": saved_rel_path,
                "extracted_from": None,
            }
        )

        db_image = Image.model_validate(features)
        session.add(db_image)
        session.commit()
        session.refresh(db_image)

        result["processed_images"].append(
            {
                "id": db_image.id,
                "filename": db_image.filename,
                "file_path": db_image.file_path,
                "type": "direct_upload",
            }
        )

        if enqueue_precompute is not None:
            enqueue_precompute(db_image.id, str(saved_path))
        return result

    if is_supported_document_format(safe_filename):
        result["file_type"] = "document"
        extraction_result = doc_parser.process_document(str(saved_path))
        if extraction_result["status"] != "success":
            raise DocumentExtractionError(extraction_result["error"])

        db_images = _persist_extracted_images(session, extraction_result, project_id)
        result["processed_images"] = [
            {
                "id": db_image.id,
                "filename": db_image.filename,
                "file_path": db_image.file_path,
                "type": "extracted_from_document",
            }
            for db_image in db_images
        ]

        if enqueue_precompute is not None:
            for db_image in db_images:
                image_path = static_dir / db_image.file_path
                if image_path.exists():
                    enqueue_precompute(db_image.id, str(image_path))

        return result

    raise UnsupportedUploadFormatError(f"不支持的文件格式: {safe_filename}")


def extract_document_images(
    session: Session,
    project_id: int,
    file_path: str,
    upload_dir: Path,
    doc_parser: DocumentParser,
) -> dict:
    full_path = resolve_path_within(upload_dir, file_path)
    if not full_path.exists():
        raise FileNotFoundError("文件不存在")

    extraction_result = doc_parser.process_document(str(full_path))
    if extraction_result["status"] != "success":
        raise RuntimeError(extraction_result.get("error") or "提取失败")

    _persist_extracted_images(session, extraction_result, project_id)

    for img in extraction_result["images"]:
        if "id" in img:
            continue
        statement = select(Image).where(
            Image.file_path == img["file_path"],
            Image.project_id == project_id,
        )
        db_img = session.exec(statement).first()
        if db_img:
            img["id"] = db_img.id

    return extraction_result
