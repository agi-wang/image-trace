import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session

from .runtime import (
    get_db,
    get_extract_dir,
    get_static_dir,
    get_upload_dir,
    migrate_db_schema,
)
from .services.project_service import normalize_legacy_paths
from .utils import ensure_directory, get_database_url, get_engine
from .routers.analysis import router as analysis_router
from .routers.files import router as files_router
from .routers.projects import router as projects_router
from .routers.uploads import router as uploads_router

logger = logging.getLogger(__name__)

UPLOAD_DIR = get_upload_dir()
EXTRACT_DIR = get_extract_dir()
STATIC_DIR = get_static_dir()


def _initialize_app_state():
    ensure_directory(UPLOAD_DIR)
    ensure_directory(EXTRACT_DIR)
    ensure_directory(STATIC_DIR)

    database_url = get_database_url()
    migrate_db_schema(database_url)

    session = Session(get_engine(database_url))
    try:
        normalize_legacy_paths(session)
    finally:
        session.close()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _initialize_app_state()
    yield


app = FastAPI(
    title="Image Trace API (简化版)",
    description="简化的图像比对和文档图片提取系统",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(uploads_router)
app.include_router(analysis_router)
app.include_router(projects_router)
app.include_router(files_router)


@app.get("/")
async def root():
    return {
        "message": "Image Trace API (简化版)",
        "version": "2.0.0",
        "endpoints": {
            "projects": "/projects",
            "upload": "/upload",
            "compare": "/compare/{project_id}",
            "results": "/results/{project_id}",
            "download": "/download/{file_path:path}",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.0.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
