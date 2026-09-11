from fastapi import APIRouter, Depends, Form, HTTPException
from sqlmodel import Session, select

from app.image_processor import ALL_ALGOS, draw_feature_matches
from app.models import Image, Project
from app.runtime import (
    ensure_precompute,
    get_db,
    get_static_dir,
    resolve_static_file_path,
)
from app.services.analysis_run_service import (
    AnalysisResultCorruptedError,
    AnalysisResultNotFoundError,
    AnalysisRunNotFoundError,
    ProjectNotFoundError,
    execute_compare_analysis,
    get_analysis_run_detail,
    list_project_analysis_runs,
    load_saved_comparison_result,
    run_smart_compare,
    trigger_recompute_features,
)
from app.services.analysis_service import (
    build_match_data_response,
    build_project_report,
    compute_pairwise_matrix_payload,
)

router = APIRouter()


@router.post("/compare/{project_id}")
async def compare_project_images(
    project_id: int,
    threshold: float = Form(default=0.85),
    hash_type: str = Form(default="phash"),
    rotation_invariant: bool = Form(default=False),
    session: Session = Depends(get_db),
):
    if not 0 <= threshold <= 1:
        raise HTTPException(status_code=400, detail="阈值必须在0-1之间")

    if hash_type not in ALL_ALGOS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的比对算法: {hash_type}。支持: {', '.join(sorted(ALL_ALGOS))}",
        )

    try:
        return execute_compare_analysis(
            session=session,
            project_id=project_id,
            threshold=threshold,
            hash_type=hash_type,
            rotation_invariant=rotation_invariant,
        )
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"比对失败: {str(exc)}") from exc


@router.post("/smart_compare/{project_id}")
async def smart_compare(
    project_id: int,
    threshold: float = 0.92,
    min_agree: int = 4,
    session: Session = Depends(get_db),
):
    try:
        return run_smart_compare(
            session=session,
            project_id=project_id,
            threshold=threshold,
            min_agree=min_agree,
        )
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"智能比对失败: {str(exc)}"
        ) from exc


@router.post("/recompute_features/{project_id}")
async def recompute_features(
    project_id: int,
    session: Session = Depends(get_db),
):
    try:
        return trigger_recompute_features(
            session=session,
            project_id=project_id,
            path_resolver=resolve_static_file_path,
            enqueue=lambda image_id, image_path: ensure_precompute(
                None, image_id, image_path
            ),
        )
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/results/{project_id}")
async def get_comparison_results(
    project_id: int,
    threshold: float = 0.85,
    hash_type: str = "phash",
    session: Session = Depends(get_db),
):
    try:
        return load_saved_comparison_result(
            session=session,
            project_id=project_id,
            threshold=threshold,
            hash_type=hash_type,
        )
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AnalysisResultNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AnalysisResultCorruptedError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/analysis_runs")
async def list_analysis_runs(
    project_id: int,
    skip: int = 0,
    limit: int = 50,
    session: Session = Depends(get_db),
):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")

    return list_project_analysis_runs(
        session=session,
        project_id=project_id,
        skip=skip,
        limit=limit,
    )


@router.get("/analysis_runs/{run_id}")
@router.get("/analysis_runs/detail/{run_id}")
async def get_analysis_run(run_id: int, session: Session = Depends(get_db)):
    try:
        return get_analysis_run_detail(session, run_id)
    except AnalysisRunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/visualize_match")
async def visualize_match(
    image_a_id: int = Form(...),
    image_b_id: int = Form(...),
    hash_type: str = Form(default="orb"),
    session: Session = Depends(get_db),
):
    descriptor_algos = ["orb", "brisk", "sift", "akaze", "kaze"]
    algo = hash_type
    if algo not in descriptor_algos:
        raise HTTPException(
            status_code=400,
            detail=f"该算法不支持特征点可视化，支持: {', '.join(descriptor_algos)}",
        )

    img_a = session.get(Image, image_a_id)
    img_b = session.get(Image, image_b_id)
    if not img_a or not img_b:
        raise HTTPException(status_code=404, detail="图像不存在")

    path_a = resolve_static_file_path(img_a.file_path or "")
    path_b = resolve_static_file_path(img_b.file_path or "")
    if path_a is None or path_b is None or not path_a.exists() or not path_b.exists():
        raise HTTPException(status_code=404, detail="图像文件不存在")

    try:
        vis_path = draw_feature_matches(
            str(path_a),
            str(path_b),
            algo=algo,
            output_dir=get_static_dir() / "visualizations",
        )
        rel_path = vis_path.relative_to(get_static_dir())
        return {"file_path": str(rel_path), "media_type": "image/jpeg"}
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"生成可视化失败: {str(exc)}"
        ) from exc


@router.get("/pairwise_matrix/{project_id}")
async def pairwise_matrix(
    project_id: int,
    hash_type: str = "sift",
    rotation_invariant: bool = False,
    session: Session = Depends(get_db),
):
    if hash_type not in ALL_ALGOS:
        raise HTTPException(
            status_code=400, detail=f"Unsupported algorithm: {hash_type}"
        )

    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")

    images = session.exec(select(Image).where(Image.project_id == project_id)).all()
    static_dir = get_static_dir()
    return compute_pairwise_matrix_payload(
        session,
        images,
        hash_type,
        rotation_invariant,
        static_dir=static_dir,
    )


@router.get("/feature_status/{project_id}")
async def feature_status(project_id: int, session: Session = Depends(get_db)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")

    images = session.exec(select(Image).where(Image.project_id == project_id)).all()
    statuses = [
        {
            "id": img.id,
            "filename": img.filename,
            "status": img.feature_status or "pending",
        }
        for img in images
    ]
    total = len(statuses)
    ready = sum(1 for item in statuses if item["status"] == "ready")
    return {
        "project_id": project_id,
        "total": total,
        "ready": ready,
        "all_ready": ready == total and total > 0,
        "images": statuses,
    }


@router.get("/system_info")
async def system_info():
    return {
        "engines": ["matrix", "legacy"],
        "algorithms": list(ALL_ALGOS),
        "matrix_engine": "numpy_blas",
        "description": "Upload precomputes all features (11 types × 8 variants). Comparison uses BLAS mat@mat.T.",
    }


@router.post("/match_data")
async def get_match_data(
    image_a_id: int = Form(...),
    image_b_id: int = Form(...),
    hash_type: str = Form(default="sift"),
    session: Session = Depends(get_db),
):
    descriptor_algos = ["orb", "brisk", "sift", "akaze", "kaze"]
    if hash_type not in descriptor_algos:
        raise HTTPException(
            status_code=400, detail=f"Supports: {', '.join(descriptor_algos)}"
        )

    img_a = session.get(Image, image_a_id)
    img_b = session.get(Image, image_b_id)
    if not img_a or not img_b:
        raise HTTPException(status_code=404, detail="Image not found")

    try:
        return build_match_data_response(img_a, img_b, hash_type, get_static_dir())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Match computation failed: {exc}"
        ) from exc


@router.get("/report/{project_id}")
async def generate_report(
    project_id: int,
    hash_type: str = "sift",
    threshold: float = 0.85,
    rotation_invariant: bool = False,
    session: Session = Depends(get_db),
):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    images = session.exec(select(Image).where(Image.project_id == project_id)).all()
    static_dir = get_static_dir()
    return build_project_report(
        session,
        project,
        images,
        hash_type,
        threshold,
        rotation_invariant,
        static_dir,
    )
