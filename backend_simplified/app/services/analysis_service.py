import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from sqlmodel import Session

from app.feature_matrix import (
    ALGO_TO_FEATURE,
    are_features_ready,
    compute_similarity_matrix_fast,
)
from app.image_processor import DESCRIPTOR_ALGOS, get_or_compute_similarity
from app.models import Image, Project
from app.runtime import resolve_static_path

logger = logging.getLogger(__name__)


def create_feature_detector(algo: str):
    import cv2

    if algo == "sift":
        return cv2.SIFT_create()
    if algo == "orb":
        return cv2.ORB_create(nfeatures=500)
    if algo == "brisk":
        return cv2.BRISK_create()
    if algo == "akaze":
        return cv2.AKAZE_create()
    if algo == "kaze":
        return cv2.KAZE_create()
    return cv2.ORB_create()


def serialize_compact_matches(keypoints_a, keypoints_b, matches):
    compact_a = []
    compact_b = []
    index_map_a = {}
    index_map_b = {}
    compact_matches = []

    for match in matches:
        src_a = int(match.queryIdx)
        src_b = int(match.trainIdx)
        if src_a >= len(keypoints_a) or src_b >= len(keypoints_b):
            continue

        if src_a not in index_map_a:
            index_map_a[src_a] = len(compact_a)
            kp = keypoints_a[src_a]
            compact_a.append({"x": round(kp.pt[0], 1), "y": round(kp.pt[1], 1)})

        if src_b not in index_map_b:
            index_map_b[src_b] = len(compact_b)
            kp = keypoints_b[src_b]
            compact_b.append({"x": round(kp.pt[0], 1), "y": round(kp.pt[1], 1)})

        compact_matches.append(
            {
                "a_idx": index_map_a[src_a],
                "b_idx": index_map_b[src_b],
                "distance": round(float(match.distance), 2),
            }
        )

    return compact_a, compact_b, compact_matches


def compute_feature_match_payload(im_a, im_b, algo: str):
    import cv2

    gray_a = cv2.cvtColor(im_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(im_b, cv2.COLOR_BGR2GRAY)
    detector = create_feature_detector(algo)

    kp_a, desc_a = detector.detectAndCompute(gray_a, None)
    kp_b, desc_b = detector.detectAndCompute(gray_b, None)

    if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) == 0:
        return {"keypoints_a": [], "keypoints_b": [], "matches": [], "score": 0.0}

    norm = cv2.NORM_L2 if algo in ("sift", "kaze") else cv2.NORM_HAMMING
    bf = cv2.BFMatcher(norm)
    raw_matches = bf.knnMatch(desc_a, desc_b, k=2)

    good_matches = []
    for match_pair in raw_matches:
        if len(match_pair) != 2:
            continue
        match, neighbor = match_pair
        if match.distance < 0.75 * neighbor.distance:
            good_matches.append(match)

    good_matches.sort(key=lambda x: x.distance)
    good_matches = good_matches[:50]

    compact_a, compact_b, compact_matches = serialize_compact_matches(
        kp_a, kp_b, good_matches
    )
    score = min(len(compact_matches) / max(len(kp_a), len(kp_b), 1), 1.0)

    return {
        "keypoints_a": compact_a,
        "keypoints_b": compact_b,
        "matches": compact_matches,
        "score": round(score, 4),
    }


def image_similarity_features(image: Image) -> dict:
    return {
        "phash": image.phash or "",
        "dhash": image.dhash or "",
        "ahash": image.ahash or "",
        "whash": image.whash or "",
        "colorhash": image.colorhash or "",
        "file_hash": image.file_hash,
    }


def compute_pairwise_matrix_payload(
    session: Session,
    images: List[Image],
    hash_type: str,
    rotation_invariant: bool = False,
    static_dir: Optional[Path] = None,
) -> dict:
    if not images:
        return {
            "names": [],
            "image_ids": [],
            "matrix": [],
            "algorithm": hash_type,
        }

    if static_dir is None:
        raise ValueError("static_dir is required when images are present")

    names = [img.filename for img in images]
    ids = [img.id for img in images]
    n = len(images)

    feature_name = ALGO_TO_FEATURE.get(hash_type)
    if feature_name and are_features_ready(session, ids):
        try:
            sim_matrix = compute_similarity_matrix_fast(
                session, ids, hash_type, rotation_invariant
            )
            matrix = [
                [round(float(sim_matrix[i, j]), 4) for j in range(n)] for i in range(n)
            ]
            return {
                "names": names,
                "image_ids": ids,
                "matrix": matrix,
                "algorithm": hash_type,
                "engine": "matrix",
            }
        except Exception as exc:
            logger.warning(
                "Matrix engine failed for [%s], falling back to legacy: %s",
                hash_type,
                exc,
            )

    paths = [
        str(resolve_static_path(static_dir, img.file_path) or "") for img in images
    ]
    features = [image_similarity_features(img) for img in images]

    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0

    for i in range(n):
        for j in range(i + 1, n):
            try:
                score = get_or_compute_similarity(
                    path_a=paths[i],
                    path_b=paths[j],
                    file_hash_a=images[i].file_hash,
                    file_hash_b=images[j].file_hash,
                    algorithm=hash_type,
                    features_a=features[i],
                    features_b=features[j],
                    rotation_invariant=rotation_invariant,
                    session=session,
                )
            except Exception as exc:
                logger.debug(
                    "Pairwise comparison failed for %s vs %s [%s]: %s",
                    images[i].filename,
                    images[j].filename,
                    hash_type,
                    exc,
                )
                score = 0.0

            matrix[i][j] = round(score, 4)
            matrix[j][i] = round(score, 4)

    return {
        "names": names,
        "image_ids": ids,
        "matrix": matrix,
        "algorithm": hash_type,
        "engine": "legacy",
    }


def _cluster_similar_images(
    images: List[Image], matrix: List[List[float]], threshold: float
):
    parent = {}
    group_scores = {}

    def find(node_id: int) -> int:
        while parent.get(node_id, node_id) != node_id:
            parent[node_id] = parent.get(parent[node_id], parent[node_id])
            node_id = parent[node_id]
        return node_id

    def union(a_id: int, b_id: int) -> None:
        root_a = find(a_id)
        root_b = find(b_id)
        if root_a != root_b:
            parent[root_a] = root_b

    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            score = matrix[i][j]
            if score >= threshold:
                union(images[i].id, images[j].id)
                group_scores[(images[i].id, images[j].id)] = score

    clusters = {}
    for image in images:
        root = find(image.id)
        clusters.setdefault(root, []).append(image)

    return clusters, group_scores


def _build_pair_match_info(
    image_a: Image, image_b: Image, hash_type: str, static_dir: Path
):
    import cv2

    pair_score = 0.0
    match_info = {
        "image_a_id": image_a.id,
        "image_b_id": image_b.id,
        "score": pair_score,
        "matches": [],
        "keypoints_a": [],
        "keypoints_b": [],
    }

    if hash_type not in DESCRIPTOR_ALGOS:
        return match_info

    try:
        path_a = resolve_static_path(static_dir, image_a.file_path)
        path_b = resolve_static_path(static_dir, image_b.file_path)
        if path_a is None or path_b is None:
            return match_info

        im_a = cv2.imread(str(path_a))
        im_b = cv2.imread(str(path_b))
        if im_a is None or im_b is None:
            return match_info

        payload = compute_feature_match_payload(im_a, im_b, hash_type)
        if payload["matches"]:
            match_info["keypoints_a"] = payload["keypoints_a"]
            match_info["keypoints_b"] = payload["keypoints_b"]
            match_info["matches"] = payload["matches"]
            match_info["image_a_size"] = {
                "width": im_a.shape[1],
                "height": im_a.shape[0],
            }
            match_info["image_b_size"] = {
                "width": im_b.shape[1],
                "height": im_b.shape[0],
            }
    except Exception as exc:
        logger.debug(
            "Feature match info failed for %s vs %s [%s]: %s",
            image_a.filename,
            image_b.filename,
            hash_type,
            exc,
        )

    return match_info


def build_match_data_response(
    image_a: Image,
    image_b: Image,
    hash_type: str,
    static_dir: Path,
):
    import cv2

    path_a = resolve_static_path(static_dir, image_a.file_path)
    path_b = resolve_static_path(static_dir, image_b.file_path)
    if path_a is None or path_b is None or not path_a.exists() or not path_b.exists():
        raise FileNotFoundError("Image file not found")

    im_a = cv2.imread(str(path_a))
    im_b = cv2.imread(str(path_b))
    if im_a is None or im_b is None:
        raise RuntimeError("Failed to read image")

    payload = compute_feature_match_payload(im_a, im_b, hash_type.lower())
    return {
        "image_a": {
            "width": im_a.shape[1],
            "height": im_a.shape[0],
            "keypoints": payload["keypoints_a"],
        },
        "image_b": {
            "width": im_b.shape[1],
            "height": im_b.shape[0],
            "keypoints": payload["keypoints_b"],
        },
        "matches": payload["matches"],
        "score": payload["score"],
    }


def build_project_report(
    session: Session,
    project: Project,
    images: List[Image],
    hash_type: str,
    threshold: float,
    rotation_invariant: bool,
    static_dir: Path,
):
    if not images:
        return {
            "project": {
                "id": project.id,
                "name": project.name,
                "description": project.description,
            },
            "generated_at": datetime.utcnow().isoformat(),
            "algorithm": hash_type,
            "threshold": threshold,
            "rotation_invariant": rotation_invariant,
            "images": [],
            "matrix": {"names": [], "image_ids": [], "values": []},
            "groups": [],
            "summary": {
                "total_images": 0,
                "similar_groups": 0,
                "unique_images": 0,
                "duplicate_rate": 0,
            },
        }

    image_meta = [
        {
            "id": img.id,
            "filename": img.filename,
            "width": img.width,
            "height": img.height,
            "file_size": img.file_size,
        }
        for img in images
    ]

    matrix_payload = compute_pairwise_matrix_payload(
        session, images, hash_type, rotation_invariant, static_dir
    )
    matrix = matrix_payload["matrix"]
    clusters, group_scores = _cluster_similar_images(images, matrix, threshold)

    groups = []
    group_id = 0
    for cluster_images in clusters.values():
        if len(cluster_images) < 2:
            continue
        group_id += 1

        pair_matches = []
        for a_idx in range(len(cluster_images)):
            for b_idx in range(a_idx + 1, len(cluster_images)):
                image_a = cluster_images[a_idx]
                image_b = cluster_images[b_idx]
                pair_score = group_scores.get(
                    (image_a.id, image_b.id),
                    group_scores.get((image_b.id, image_a.id), 0),
                )

                match_info = _build_pair_match_info(
                    image_a, image_b, hash_type, static_dir
                )
                match_info["score"] = pair_score
                pair_matches.append(match_info)

        avg_score = sum(item["score"] for item in pair_matches) / max(
            len(pair_matches), 1
        )
        groups.append(
            {
                "group_id": group_id,
                "similarity_score": round(avg_score, 4),
                "images": [
                    {"id": image.id, "filename": image.filename}
                    for image in cluster_images
                ],
                "pair_matches": pair_matches,
            }
        )

    total_images = len(images)
    unique_count = total_images - sum(len(group["images"]) for group in groups)
    duplicate_rate = (
        round((total_images - unique_count) / total_images * 100, 1)
        if total_images > 0
        else 0
    )

    return {
        "project": {
            "id": project.id,
            "name": project.name,
            "description": project.description,
        },
        "generated_at": datetime.utcnow().isoformat(),
        "algorithm": hash_type,
        "threshold": threshold,
        "rotation_invariant": rotation_invariant,
        "images": image_meta,
        "matrix": {
            "names": matrix_payload["names"],
            "image_ids": matrix_payload["image_ids"],
            "values": matrix,
            "engine": matrix_payload.get("engine"),
        },
        "groups": groups,
        "summary": {
            "total_images": total_images,
            "similar_groups": len(groups),
            "unique_images": unique_count,
            "duplicate_rate": duplicate_rate,
        },
    }
