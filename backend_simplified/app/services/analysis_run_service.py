import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

from sqlmodel import Session, select

from app.feature_matrix import compute_similarity_matrix_fast
from app.image_processor import (
    DESCRIPTOR_ALGOS,
    FUSION_ALGOS,
    PIXEL_ALGOS,
    calculate_descriptor_similarity,
    calculate_histogram_similarity,
    calculate_hybrid_similarity,
    calculate_similarity,
    calculate_ssim_similarity,
    calculate_template_similarity,
    compare_with_orientations,
    get_cached_descriptor,
    group_similar_images,
)
from app.models import (
    AnalysisRun,
    ComparisonResult,
    Image,
    ImageRead,
    Project,
    SimilarGroup,
)
from app.runtime import get_static_dir
from app.utils import group_similar_by_metric


SMART_ALGOS = [
    "phash",
    "dhash",
    "ahash",
    "whash",
    "ssim",
    "sift",
    "orb",
    "brisk",
    "akaze",
    "kaze",
]

_HASH_GATE_ALGOS = {"phash", "dhash", "ahash", "whash"}


class ProjectNotFoundError(LookupError):
    pass


class AnalysisRunNotFoundError(LookupError):
    pass


class AnalysisResultNotFoundError(LookupError):
    pass


class AnalysisResultCorruptedError(RuntimeError):
    pass


def require_project(session: Session, project_id: int) -> Project:
    project = session.get(Project, project_id)
    if not project:
        raise ProjectNotFoundError("项目不存在")
    return project


def _build_image_dicts(session: Session, project_id: int, hash_type: str) -> list[dict]:
    images = session.exec(select(Image).where(Image.project_id == project_id)).all()
    image_dicts = []
    for img in images:
        image_dict = {
            "id": img.id,
            "filename": img.filename,
            "file_path": img.file_path,
            "file_hash": img.file_hash,
            "phash": img.phash,
            "dhash": img.dhash,
            "ahash": img.ahash,
            "whash": img.whash,
            "colorhash": getattr(img, "colorhash", None),
            "extracted_from": img.extracted_from,
            "file_size": img.file_size,
            "width": img.width,
            "height": img.height,
            "created_at": img.created_at,
        }
        image_dicts.append(image_dict)

    if hash_type in DESCRIPTOR_ALGOS:
        for img in image_dicts:
            try:
                fp = get_static_dir() / img["file_path"]
                desc, norm = get_cached_descriptor(str(fp), img["file_hash"], hash_type)
                img["descriptor"] = desc
                img["descriptor_norm"] = norm
            except Exception as exc:
                logger.debug(
                    "Descriptor load failed for image %s [%s]: %s",
                    img["id"],
                    hash_type,
                    exc,
                )
                img["descriptor"] = None
                img["descriptor_norm"] = None

    return image_dicts


def _build_analysis_scorer(hash_type: str, rotation_invariant: bool):
    if hash_type in DESCRIPTOR_ALGOS:

        def scorer(a: dict, b: dict) -> float:
            if rotation_invariant:
                pa = str(get_static_dir() / a["file_path"])
                pb = str(get_static_dir() / b["file_path"])

                def _desc_scorer(p_a, p_b):
                    from app.image_processor import compute_descriptor

                    da, na = compute_descriptor(p_a, hash_type)
                    db, _ = compute_descriptor(p_b, hash_type)
                    return calculate_descriptor_similarity(da, db, na)

                return compare_with_orientations(pa, pb, _desc_scorer)

            return calculate_descriptor_similarity(
                a.get("descriptor"),
                b.get("descriptor"),
                a.get("descriptor_norm") or b.get("descriptor_norm") or 4,
            )

        return scorer

    if hash_type in PIXEL_ALGOS:
        pixel_fn_map = {
            "ssim": calculate_ssim_similarity,
            "histogram": calculate_histogram_similarity,
            "template": calculate_template_similarity,
        }
        pixel_fn = pixel_fn_map[hash_type]

        def scorer(a: dict, b: dict) -> float:
            try:
                pa = str(get_static_dir() / a["file_path"])
                pb = str(get_static_dir() / b["file_path"])
                if rotation_invariant:
                    return compare_with_orientations(pa, pb, pixel_fn)
                return pixel_fn(pa, pb)
            except Exception as exc:
                logger.warning(
                    "Pixel scorer failed for %s vs %s [%s]: %s",
                    a.get("filename"),
                    b.get("filename"),
                    hash_type,
                    exc,
                )
                return 0.0

        return scorer

    if hash_type in FUSION_ALGOS:

        def scorer(a: dict, b: dict) -> float:
            try:
                pa = str(get_static_dir() / a["file_path"])
                pb = str(get_static_dir() / b["file_path"])
                if rotation_invariant:

                    def _fusion_scorer(p_a, p_b):
                        from app.image_processor import compute_image_features as _cif

                        fa = _cif(p_a)
                        fb = _cif(p_b)
                        return calculate_hybrid_similarity(p_a, p_b, fa, fb)

                    return compare_with_orientations(pa, pb, _fusion_scorer)
                return calculate_hybrid_similarity(pa, pb, a, b)
            except Exception as exc:
                logger.warning(
                    "Fusion scorer failed for %s vs %s [%s]: %s",
                    a.get("filename"),
                    b.get("filename"),
                    hash_type,
                    exc,
                )
                return 0.0

        return scorer

    if rotation_invariant:
        from app.image_processor import compute_features_for_variants

        variant_cache = {}

        def scorer(a: dict, b: dict) -> float:
            file_hash_b = b["file_hash"]
            if file_hash_b not in variant_cache:
                path_b = str(get_static_dir() / b["file_path"])
                variant_cache[file_hash_b] = compute_features_for_variants(path_b)
            best = 0.0
            for variant_features in variant_cache[file_hash_b]:
                score = calculate_similarity(
                    a.get(hash_type, ""), variant_features.get(hash_type, "")
                )
                best = max(best, score)
                if best >= 0.95:
                    break
            return best

        return scorer

    return None


def _group_project_images(
    image_dicts: list[dict],
    threshold: float,
    hash_type: str,
    rotation_invariant: bool,
):
    scorer = _build_analysis_scorer(hash_type, rotation_invariant)
    if scorer is None:
        return group_similar_images(image_dicts, threshold, hash_type), None
    return group_similar_by_metric(image_dicts, threshold, scorer), scorer


def _to_image_read(project_id: int, img_dict: dict) -> ImageRead:
    return ImageRead(
        id=img_dict["id"],
        filename=img_dict["filename"],
        project_id=img_dict.get("project_id", project_id),
        file_path=img_dict["file_path"],
        file_hash=img_dict["file_hash"],
        phash=img_dict["phash"],
        dhash=img_dict.get("dhash"),
        ahash=img_dict.get("ahash"),
        whash=img_dict.get("whash"),
        colorhash=img_dict.get("colorhash"),
        extracted_from=img_dict.get("extracted_from"),
        file_size=img_dict.get("file_size"),
        width=img_dict.get("width"),
        height=img_dict.get("height"),
        created_at=img_dict["created_at"],
    )


def _persist_analysis_run(
    session: Session,
    project_id: int,
    hash_type: str,
    threshold: float,
    result: ComparisonResult,
) -> ComparisonResult:
    try:
        run = AnalysisRun(
            project_id=project_id,
            hash_type=hash_type,
            threshold=threshold,
            total_images=result.total_images,
            groups_count=len(result.groups),
            unique_count=len(result.unique_images),
            summary=json.dumps(result.model_dump(), default=str),
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        result_dict = result.model_dump()
        result_dict["run_id"] = run.id
        return ComparisonResult.model_validate(result_dict)
    except Exception as exc:
        logging.getLogger(__name__).warning("Failed to save AnalysisRun: %s", exc)
        return result


def compare_images_in_project(
    session: Session,
    project_id: int,
    threshold: float = 0.85,
    hash_type: str = "orb",
    rotation_invariant: bool = False,
) -> ComparisonResult:
    require_project(session, project_id)
    image_dicts = _build_image_dicts(session, project_id, hash_type)

    if not image_dicts:
        return ComparisonResult(
            project_id=project_id, total_images=0, groups=[], unique_images=[]
        )

    (groups, ungrouped), scorer = _group_project_images(
        image_dicts, threshold, hash_type, rotation_invariant
    )

    similar_groups = []
    for group in groups:
        if len(group) > 1:
            total_similarity = 0.0
            count = 0
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    if scorer is not None:
                        similarity = scorer(group[i], group[j])
                    else:
                        similarity = calculate_similarity(
                            group[i].get(hash_type, ""), group[j].get(hash_type, "")
                        )
                    total_similarity += similarity
                    count += 1
            avg_similarity = total_similarity / count if count > 0 else 1.0
        else:
            avg_similarity = 1.0

        similar_groups.append(
            SimilarGroup(
                group_id=len(similar_groups) + 1,
                similarity_score=avg_similarity,
                images=[_to_image_read(project_id, img_dict) for img_dict in group],
            )
        )

    unique_images = [_to_image_read(project_id, img_dict) for img_dict in ungrouped]
    result = ComparisonResult(
        project_id=project_id,
        total_images=len(image_dicts),
        groups=similar_groups,
        unique_images=unique_images,
    )
    return _persist_analysis_run(session, project_id, hash_type, threshold, result)


def execute_compare_analysis(
    session: Session,
    project_id: int,
    threshold: float,
    hash_type: str,
    rotation_invariant: bool = False,
):
    return compare_images_in_project(
        session,
        project_id,
        threshold,
        hash_type,
        rotation_invariant=rotation_invariant,
    )


def _validate_smart_compare_inputs(session: Session, project_id: int):
    require_project(session, project_id)

    images = session.exec(select(Image).where(Image.project_id == project_id)).all()
    if len(images) < 2:
        return None, {
            "total_images": len(images),
            "algorithms_used": len(SMART_ALGOS),
            "found_duplicates": False,
            "duplicate_groups": [],
            "unique_count": len(images),
            "scan_seconds": 0,
            "summary": f"项目中{'只有 1 张图片' if len(images) == 1 else '没有图片'}，无法进行查重比对",
        }

    not_ready = [img for img in images if (img.feature_status or "pending") != "ready"]
    if not_ready:
        names = ", ".join(img.filename for img in not_ready[:5])
        suffix = f" 等 {len(not_ready)} 张" if len(not_ready) > 5 else ""
        return None, {
            "total_images": len(images),
            "algorithms_used": 0,
            "found_duplicates": False,
            "duplicate_groups": [],
            "unique_count": len(images),
            "scan_seconds": 0,
            "features_pending": True,
            "summary": f"部分图片特征尚未计算完成（{names}{suffix}），请稍后重试",
        }

    return images, None


def _collect_pair_hits(
    session: Session,
    image_ids: list[int],
    threshold: float,
):
    pair_hits: dict[tuple[int, int], list[tuple[str, float]]] = {}
    n = len(image_ids)
    for algo in SMART_ALGOS:
        try:
            sim = compute_similarity_matrix_fast(
                session, image_ids, algo, rotation_invariant=True
            )
            for i in range(n):
                for j in range(i + 1, n):
                    score = float(sim[i, j])
                    if score < threshold:
                        continue
                    key = (i, j)
                    pair_hits.setdefault(key, []).append((algo, round(score, 4)))
        except Exception as exc:
            logger.debug("Smart compare algorithm %s skipped: %s", algo, exc)
            continue
    return pair_hits


def _build_duplicate_groups(
    pair_hits: dict[tuple[int, int], list[tuple[str, float]]],
    n: int,
    min_agree: int,
    image_ids: list[int],
    id_to_img: dict[int, Image],
):
    confirmed_pairs = {}
    for key, hits in pair_hits.items():
        if len(hits) < min_agree:
            continue
        if not any(algo in _HASH_GATE_ALGOS for algo, _ in hits):
            continue
        confirmed_pairs[key] = hits

    parent = list(range(n))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    for i, j in confirmed_pairs:
        union(i, j)

    groups_map: dict[int, list[int]] = defaultdict(list)
    for idx in range(n):
        groups_map[find(idx)].append(idx)

    duplicate_groups = []
    grouped_indices = set()
    for members in groups_map.values():
        if len(members) < 2:
            continue
        grouped_indices.update(members)

        group_algos = set()
        best_score = 0.0
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                key = (min(members[i], members[j]), max(members[i], members[j]))
                if key not in confirmed_pairs:
                    continue
                for algo, score in confirmed_pairs[key]:
                    group_algos.add(algo)
                    best_score = max(best_score, score)

        duplicate_groups.append(
            {
                "images": [
                    {
                        "id": id_to_img[image_ids[idx]].id,
                        "filename": id_to_img[image_ids[idx]].filename,
                        "file_path": id_to_img[image_ids[idx]].file_path,
                    }
                    for idx in members
                ],
                "confidence": round(best_score, 4),
                "matched_algorithms": sorted(group_algos),
                "matched_count": len(group_algos),
            }
        )

    duplicate_groups.sort(key=lambda group: group["confidence"], reverse=True)
    return duplicate_groups, grouped_indices


def run_smart_compare(
    session: Session,
    project_id: int,
    threshold: float = 0.92,
    min_agree: int = 4,
):
    t0 = time.time()

    images, early_response = _validate_smart_compare_inputs(session, project_id)
    if early_response is not None:
        return early_response

    image_ids = [img.id for img in images]
    id_to_img = {img.id: img for img in images}

    pair_hits = _collect_pair_hits(session, image_ids, threshold)
    duplicate_groups, grouped_indices = _build_duplicate_groups(
        pair_hits, len(image_ids), min_agree, image_ids, id_to_img
    )
    total_duplicated = len(grouped_indices)
    n = len(image_ids)
    unique_count = n - total_duplicated
    elapsed = round(time.time() - t0, 2)
    found = len(duplicate_groups) > 0
    if found:
        summary = (
            f"在 {n} 张图片中使用 {len(SMART_ALGOS)} 种特征进行比对，"
            f"发现 {len(duplicate_groups)} 组共 {total_duplicated} 张疑似相似图片"
        )
    else:
        summary = (
            f"在 {n} 张图片中使用 {len(SMART_ALGOS)} 种特征进行比对，未发现相似图片"
        )

    return {
        "total_images": n,
        "algorithms_used": len(SMART_ALGOS),
        "found_duplicates": found,
        "duplicate_groups": duplicate_groups,
        "unique_count": unique_count,
        "scan_seconds": elapsed,
        "summary": summary,
    }


def trigger_recompute_features(
    session: Session,
    project_id: int,
    path_resolver: Callable[[Optional[str]], Optional[Path]],
    enqueue: Callable[[int, str], None],
):
    require_project(session, project_id)
    images = session.exec(select(Image).where(Image.project_id == project_id)).all()
    pending = [img for img in images if (img.feature_status or "pending") != "ready"]
    if not pending:
        return {"triggered": 0, "message": "所有图片特征均已计算完成"}

    triggered = 0
    for img in pending:
        image_path = path_resolver(img.file_path)
        if image_path and image_path.exists():
            enqueue(img.id, str(image_path))
            triggered += 1

    return {
        "triggered": triggered,
        "message": f"已触发 {triggered} 张图片的特征计算",
    }


def load_saved_comparison_result(
    session: Session,
    project_id: int,
    threshold: float,
    hash_type: str,
):
    require_project(session, project_id)
    statement = (
        select(AnalysisRun)
        .where(
            AnalysisRun.project_id == project_id,
            AnalysisRun.hash_type == hash_type,
            AnalysisRun.threshold == threshold,
        )
        .order_by(AnalysisRun.id.desc())
    )
    run = session.exec(statement).first()
    if not run or not run.summary:
        raise AnalysisResultNotFoundError("当前条件下暂无已保存的分析结果")

    try:
        return ComparisonResult.model_validate(json.loads(run.summary))
    except Exception as exc:
        raise AnalysisResultCorruptedError("分析结果已损坏，请重新执行分析") from exc


def list_project_analysis_runs(
    session: Session,
    project_id: int,
    skip: int = 0,
    limit: int = 50,
):
    statement = (
        select(AnalysisRun)
        .where(AnalysisRun.project_id == project_id)
        .order_by(AnalysisRun.id.desc())
        .offset(skip)
        .limit(limit)
    )
    return session.exec(statement).all()


def get_analysis_run_detail(session: Session, run_id: int):
    run = session.get(AnalysisRun, run_id)
    if not run:
        raise AnalysisRunNotFoundError("分析记录不存在")

    parsed = None
    if run.summary:
        try:
            parsed = json.loads(run.summary)
        except Exception as exc:
            logger.warning("AnalysisRun %d summary JSON parse failed: %s", run_id, exc)
            parsed = None

    return {"run": run, "result": parsed}
