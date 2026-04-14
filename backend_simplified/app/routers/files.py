from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlmodel import Session

from app.models import Image
from app.runtime import get_db, get_static_dir
from app.services.project_service import (
    detect_media_type,
    ensure_thumbnail,
    resolve_download_path,
)

router = APIRouter()


@router.get("/download/{file_path:path}")
async def download_file(file_path: str):
    static_dir = get_static_dir()
    try:
        full_path = resolve_download_path(static_dir, file_path)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    media_type = detect_media_type(full_path, file_path)
    return FileResponse(path=full_path, filename=full_path.name, media_type=media_type)


@router.get("/thumbnail/{image_id}")
async def get_thumbnail(
    image_id: int,
    size: int = 400,
    session: Session = Depends(get_db),
):
    image = session.get(Image, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    try:
        thumb_path = ensure_thumbnail(image, size, get_static_dir())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Thumbnail generation failed: {exc}"
        ) from exc

    return FileResponse(
        path=thumb_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )
