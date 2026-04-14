from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from sqlmodel import Session

from app.models import Project
from app.runtime import (
    build_document_parser,
    ensure_precompute,
    get_db,
    get_static_dir,
    get_upload_dir,
)
from app.services.upload_service import (
    DocumentExtractionError,
    UnsupportedUploadFormatError,
    extract_document_images,
    process_uploaded_file,
)
from app.utils import sanitize_filename, save_upload_file

router = APIRouter()

_cached_doc_parser = None


def _get_doc_parser():
    global _cached_doc_parser
    if _cached_doc_parser is None:
        _cached_doc_parser = build_document_parser()
    return _cached_doc_parser


@router.post("/upload")
async def upload_file(
    project_id: int = Form(...),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    session: Session = Depends(get_db),
):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")

    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名为空")

    try:
        safe_filename = sanitize_filename(file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    upload_dir = get_upload_dir()
    file_path = upload_dir / safe_filename
    saved_path = save_upload_file(file, file_path)

    try:
        return process_uploaded_file(
            session=session,
            project_id=project_id,
            safe_filename=safe_filename,
            saved_path=saved_path,
            static_dir=get_static_dir(),
            doc_parser=_get_doc_parser(),
            enqueue_precompute=lambda image_id, image_path: ensure_precompute(
                background_tasks, image_id, image_path
            ),
        )
    except UnsupportedUploadFormatError as exc:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DocumentExtractionError as exc:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as e:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"上传处理失败: {str(e)}") from e


@router.post("/extract/{file_path:path}")
async def extract_from_document(
    file_path: str, project_id: int = Form(...), session: Session = Depends(get_db)
):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")

    try:
        return extract_document_images(
            session=session,
            project_id=project_id,
            file_path=file_path,
            upload_dir=get_upload_dir(),
            doc_parser=_get_doc_parser(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=f"提取失败: {str(exc)}") from exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"提取失败: {str(e)}") from e
