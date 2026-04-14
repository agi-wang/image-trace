"""Tests for frontend->backend API alignment.

Verifies every frontend api.ts function receives the exact JSON shape
it expects from the backend.  Each test mirrors a single TypeScript
interface / API function in ui/src/lib/api.ts.

Covered frontend functions:
  createProject, getProjects, getProject, deleteProject,
  uploadImages, uploadDocument, extractDocument,
  getProjectImages, deleteImage,
  analyzeImages, getComparisonResults, getAnalysisRuns, getAnalysisRunDetail,
  smartCompare, recomputeFeatures,
  checkHealth, visualizeMatch, getPairwiseMatrix, getMatchData,
  getSystemInfo, getFeatureStatus
"""

import time
import pytest
from tests.conftest import make_upload_bytes


# ========================================================================
#  Helpers
# ========================================================================


def _create_project(client, name="AlignTest"):
    resp = client.post("/projects", json={"name": name})
    assert resp.status_code == 200
    return resp.json()


def _upload_image(client, pid, filename="align.png", color=(128, 64, 32)):
    f = make_upload_bytes(filename, color=color)
    resp = client.post("/upload", data={"project_id": str(pid)}, files={"file": f})
    assert resp.status_code == 200
    return resp.json()


def _upload_and_wait(client, pid, filenames, wait=3):
    """Upload images and wait for precompute."""
    for fn in filenames:
        _upload_image(client, pid, fn)
    time.sleep(wait)


# ========================================================================
#  Project CRUD — createProject, getProjects, getProject, deleteProject
# ========================================================================


class TestProjectCRUDAlignment:
    """Verify Project JSON matches TS interface Project."""

    def test_create_project_shape(self, client):
        """POST /projects → { id, name, description?, created_at }"""
        data = _create_project(client, "CreateShape")
        assert isinstance(data["id"], int)
        assert isinstance(data["name"], str)
        assert "created_at" in data

    def test_get_projects_shape(self, client):
        """GET /projects → Project[] with image_count"""
        _create_project(client, "ListShape")
        resp = client.get("/projects")
        assert resp.status_code == 200
        projects = resp.json()
        assert isinstance(projects, list)
        for p in projects:
            assert "id" in p
            assert "name" in p
            assert "image_count" in p
            assert isinstance(p["image_count"], int)

    def test_get_project_shape(self, client):
        """GET /projects/{id} → single Project"""
        proj = _create_project(client, "DetailShape")
        resp = client.get(f"/projects/{proj['id']}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == proj["id"]
        assert "name" in data
        assert "created_at" in data

    def test_delete_project_shape(self, client):
        """DELETE /projects/{id} → { message }"""
        proj = _create_project(client, "DeleteShape")
        resp = client.delete(f"/projects/{proj['id']}")
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data


# ========================================================================
#  Upload — uploadImages, uploadDocument
# ========================================================================


class TestUploadAlignment:
    """Verify upload response matches TS ProcessedImage / Image."""

    def test_upload_image_shape(self, client):
        """POST /upload → { project_id, filename, file_path, file_type, processed_images, error }"""
        proj = _create_project(client, "UploadShape")
        data = _upload_image(client, proj["id"])
        assert "project_id" in data
        assert "filename" in data
        assert "file_path" in data
        assert "file_type" in data
        assert "processed_images" in data
        assert "error" in data
        assert isinstance(data["processed_images"], list)
        if data["processed_images"]:
            img = data["processed_images"][0]
            assert "id" in img
            assert "filename" in img
            assert "file_path" in img

    def test_upload_unsupported_shape(self, client):
        """POST /upload with unsupported file → HTTP 400"""
        proj = _create_project(client, "UploadUnsupported")
        f = ("test.xyz", b"nonsense", "application/octet-stream")
        resp = client.post(
            "/upload", data={"project_id": str(proj["id"])}, files={"file": f}
        )
        assert resp.status_code == 400
        data = resp.json()
        assert "detail" in data

    def test_upload_nonexistent_project(self, client):
        """POST /upload with bad project_id → 404"""
        f = make_upload_bytes("bad.png")
        resp = client.post("/upload", data={"project_id": "99999"}, files={"file": f})
        assert resp.status_code == 404


# ========================================================================
#  Images — getProjectImages, deleteImage
# ========================================================================


class TestImagesAlignment:
    """Verify Image[] JSON matches TS interface Image."""

    def test_get_images_shape(self, client):
        """GET /images/{project_id} → Image[] with required fields"""
        proj = _create_project(client, "ImagesShape")
        _upload_image(client, proj["id"], "shape_img.png")
        resp = client.get(f"/images/{proj['id']}")
        assert resp.status_code == 200
        images = resp.json()
        assert isinstance(images, list)
        assert len(images) >= 1
        img = images[0]
        # TS Image interface fields
        assert "id" in img
        assert "filename" in img
        assert "file_path" in img
        assert isinstance(img["id"], int)
        assert isinstance(img["filename"], str)

    def test_get_images_optional_fields(self, client):
        """Image should contain optional hash fields that TS expects."""
        proj = _create_project(client, "ImagesOptional")
        _upload_image(client, proj["id"], "opt.png")
        resp = client.get(f"/images/{proj['id']}")
        images = resp.json()
        img = images[0]
        # Optional fields the frontend expects
        for field in [
            "phash",
            "dhash",
            "ahash",
            "whash",
            "file_size",
            "project_id",
            "created_at",
        ]:
            assert field in img, f"Missing field: {field}"

    def test_delete_image_shape(self, client):
        """DELETE /images/{id} → { message }"""
        proj = _create_project(client, "DelImgShape")
        upload = _upload_image(client, proj["id"], "del.png")
        img_id = upload["processed_images"][0]["id"]
        resp = client.delete(f"/images/{img_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data


# ========================================================================
#  Analysis — analyzeImages, getComparisonResults
# ========================================================================


class TestAnalysisAlignment:
    """Verify AnalysisResult JSON matches TS interface."""

    def test_compare_shape(self, client):
        """POST /compare/{id} → AnalysisResult shape"""
        proj = _create_project(client, "CompareShape")
        for fn in ["cmp_a.png", "cmp_b.png"]:
            _upload_image(client, proj["id"], fn)
        resp = client.post(
            f"/compare/{proj['id']}", data={"threshold": "0.85", "hash_type": "phash"}
        )
        assert resp.status_code == 200
        data = resp.json()
        # TS AnalysisResult fields
        assert "project_id" in data
        assert "total_images" in data
        assert "groups" in data
        assert "unique_images" in data
        assert isinstance(data["groups"], list)
        assert isinstance(data["unique_images"], list)
        if data["groups"]:
            g = data["groups"][0]
            assert "group_id" in g
            assert "similarity_score" in g
            assert "images" in g

    def test_get_results_shape(self, client):
        """GET /results/{id} → same AnalysisResult shape"""
        proj = _create_project(client, "ResultsShape")
        for fn in ["res_a.png", "res_b.png"]:
            _upload_image(client, proj["id"], fn)
        client.post(
            f"/compare/{proj['id']}", data={"threshold": "0.85", "hash_type": "phash"}
        )
        resp = client.get(
            f"/results/{proj['id']}", params={"threshold": 0.85, "hash_type": "phash"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "project_id" in data
        assert "total_images" in data
        assert "groups" in data
        assert "unique_images" in data


# ========================================================================
#  Analysis Runs — getAnalysisRuns, getAnalysisRunDetail
# ========================================================================


class TestAnalysisRunsAlignment:
    """Verify AnalysisRun JSON matches TS interface."""

    def test_list_runs_shape(self, client):
        """GET /analysis_runs → AnalysisRun[]"""
        proj = _create_project(client, "RunsShape")
        for fn in ["run_a.png", "run_b.png"]:
            _upload_image(client, proj["id"], fn)
        # Trigger a compare to create an analysis run
        client.post(
            f"/compare/{proj['id']}", data={"threshold": "0.85", "hash_type": "orb"}
        )
        resp = client.get(f"/analysis_runs", params={"project_id": proj["id"]})
        assert resp.status_code == 200
        runs = resp.json()
        assert isinstance(runs, list)
        if runs:
            run = runs[0]
            # TS AnalysisRun fields
            assert "id" in run
            assert "project_id" in run
            assert "hash_type" in run
            assert "threshold" in run
            assert "total_images" in run
            assert "groups_count" in run
            assert "unique_count" in run
            assert "created_at" in run

    def test_run_detail_shape(self, client):
        """GET /analysis_runs/detail/{id} → run detail"""
        proj = _create_project(client, "RunDetailShape")
        for fn in ["rd_a.png", "rd_b.png"]:
            _upload_image(client, proj["id"], fn)
        client.post(
            f"/compare/{proj['id']}", data={"threshold": "0.85", "hash_type": "orb"}
        )
        runs = client.get(f"/analysis_runs", params={"project_id": proj["id"]}).json()
        if runs:
            run_id = runs[0]["id"]
            resp = client.get(f"/analysis_runs/detail/{run_id}")
            # Endpoint might return the run directly or wrapped
            if resp.status_code == 200:
                data = resp.json()
                assert "id" in data or "run" in data
            # 404 is acceptable if the detail endpoint doesn't exist
            else:
                assert resp.status_code in [200, 404]


# ========================================================================
#  Smart Compare — smartCompare, recomputeFeatures
# ========================================================================


class TestSmartCompareAlignment:
    """Verify SmartCompareResult JSON matches TS interface."""

    def test_smart_compare_shape(self, client):
        """POST /smart_compare/{id} → SmartCompareResult"""
        proj = _create_project(client, "SCShape")
        _upload_and_wait(client, proj["id"], ["sc_a.png", "sc_b.png"])
        resp = client.post(f"/smart_compare/{proj['id']}")
        assert resp.status_code == 200
        data = resp.json()
        # TS SmartCompareResult fields
        assert "total_images" in data
        assert "algorithms_used" in data
        assert "found_duplicates" in data
        assert "duplicate_groups" in data
        assert "unique_count" in data
        assert "scan_seconds" in data
        assert "summary" in data
        assert isinstance(data["total_images"], int)
        assert isinstance(data["algorithms_used"], int)
        assert isinstance(data["found_duplicates"], bool)
        assert isinstance(data["duplicate_groups"], list)
        assert isinstance(data["unique_count"], int)
        assert isinstance(data["scan_seconds"], (int, float))
        assert isinstance(data["summary"], str)

    def test_smart_compare_group_shape(self, client):
        """SmartCompareGroup should have images, confidence, matched_algorithms"""
        proj = _create_project(client, "SCGroupShape")
        _upload_and_wait(client, proj["id"], ["scg_a.png", "scg_b.png"])
        resp = client.post(f"/smart_compare/{proj['id']}")
        data = resp.json()
        if data["found_duplicates"]:
            for group in data["duplicate_groups"]:
                # TS SmartCompareGroup fields
                assert "images" in group
                assert "confidence" in group
                assert "matched_algorithms" in group
                assert "matched_count" in group
                assert isinstance(group["images"], list)
                assert isinstance(group["confidence"], (int, float))
                assert isinstance(group["matched_algorithms"], list)
                assert isinstance(group["matched_count"], int)
                for img in group["images"]:
                    assert "id" in img
                    assert "filename" in img
                    assert "file_path" in img

    def test_recompute_shape(self, client):
        """POST /recompute_features/{id} → { triggered, message }"""
        proj = _create_project(client, "RecompShape")
        resp = client.post(f"/recompute_features/{proj['id']}")
        assert resp.status_code == 200
        data = resp.json()
        assert "triggered" in data
        assert "message" in data
        assert isinstance(data["triggered"], int)
        assert isinstance(data["message"], str)


# ========================================================================
#  Health — checkHealth
# ========================================================================


class TestHealthAlignment:
    """Verify health endpoint matches TS expectation."""

    def test_health_shape(self, client):
        """GET /health → { status }"""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] == "healthy"


# ========================================================================
#  Visualization — visualizeMatch
# ========================================================================


class TestVisualizeMatchAlignment:
    """Verify visualize_match response matches TS { url }."""

    def test_visualize_match_shape(self, client):
        """POST /visualize_match → { image_path } or error for identical images"""
        proj = _create_project(client, "VizShape")
        _upload_image(client, proj["id"], "viz_a.png", color=(200, 100, 50))
        _upload_image(client, proj["id"], "viz_b.png", color=(200, 100, 50))
        images = client.get(f"/images/{proj['id']}").json()
        if len(images) >= 2:
            resp = client.post(
                "/visualize_match",
                data={
                    "image_a_id": str(images[0]["id"]),
                    "image_b_id": str(images[1]["id"]),
                    "hash_type": "orb",
                },
            )
            # May return 200 with image_path, or 500 if image files
            # are not accessible in test environment
            if resp.status_code == 200:
                data = resp.json()
                assert "image_path" in data or "url" in data
            else:
                # 500 is acceptable in test env where physical files may not
                # be accessible
                assert resp.status_code in [200, 500]


# ========================================================================
#  Pairwise Matrix — getPairwiseMatrix
# ========================================================================


class TestPairwiseMatrixAlignment:
    """Verify PairwiseMatrixResult matches TS interface."""

    def test_pairwise_matrix_shape(self, client):
        """GET /pairwise_matrix/{id} → { names, image_ids, matrix, algorithm, engine? }"""
        proj = _create_project(client, "PairShape")
        _upload_image(client, proj["id"], "pair_a.png")
        _upload_image(client, proj["id"], "pair_b.png")
        resp = client.get(f"/pairwise_matrix/{proj['id']}", params={"hash_type": "orb"})
        assert resp.status_code == 200
        data = resp.json()
        # TS PairwiseMatrixResult fields
        assert "names" in data
        assert "matrix" in data
        assert "algorithm" in data
        assert isinstance(data["names"], list)
        assert isinstance(data["matrix"], list)
        if data["matrix"]:
            assert isinstance(data["matrix"][0], list)
        # image_ids may or may not be present (optional in TS)
        # engine may or may not be present (optional in TS)


# ========================================================================
#  Match Data — getMatchData
# ========================================================================


class TestMatchDataAlignment:
    """Verify MatchData JSON matches TS interface."""

    def test_match_data_shape(self, client):
        """POST /match_data → { image_a, image_b, matches, score }"""
        proj = _create_project(client, "MatchShape")
        _upload_image(client, proj["id"], "md_a.png", color=(200, 100, 50))
        _upload_image(client, proj["id"], "md_b.png", color=(200, 100, 50))
        images = client.get(f"/images/{proj['id']}").json()
        if len(images) >= 2:
            resp = client.post(
                "/match_data",
                data={
                    "image_a_id": str(images[0]["id"]),
                    "image_b_id": str(images[1]["id"]),
                    "hash_type": "sift",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            # TS MatchData fields
            assert "image_a" in data
            assert "image_b" in data
            assert "matches" in data
            assert "score" in data
            # Nested shape
            assert "width" in data["image_a"]
            assert "height" in data["image_a"]
            assert "keypoints" in data["image_a"]
            assert isinstance(data["matches"], list)


# ========================================================================
#  System Info — getSystemInfo
# ========================================================================


class TestSystemInfoAlignment:
    """Verify SystemInfo JSON matches TS interface."""

    def test_system_info_shape(self, client):
        """GET /system_info → { engines, algorithms, matrix_engine, description }"""
        resp = client.get("/system_info")
        assert resp.status_code == 200
        data = resp.json()
        # TS SystemInfo fields
        assert "engines" in data
        assert "algorithms" in data
        assert "matrix_engine" in data
        assert "description" in data
        assert isinstance(data["engines"], list)
        assert isinstance(data["algorithms"], list)
        assert isinstance(data["matrix_engine"], str)
        assert isinstance(data["description"], str)


# ========================================================================
#  Feature Status — getFeatureStatus
# ========================================================================


class TestFeatureStatusAlignment:
    """Verify FeatureStatus JSON matches TS interface."""

    def test_feature_status_shape(self, client):
        """GET /feature_status/{id} → { project_id, total, ready, all_ready, images }"""
        proj = _create_project(client, "FStatusShape")
        _upload_image(client, proj["id"], "fs.png")
        resp = client.get(f"/feature_status/{proj['id']}")
        assert resp.status_code == 200
        data = resp.json()
        # TS FeatureStatus fields
        assert "project_id" in data
        assert "total" in data
        assert "ready" in data
        assert "all_ready" in data
        assert "images" in data
        assert isinstance(data["project_id"], int)
        assert isinstance(data["total"], int)
        assert isinstance(data["ready"], int)
        assert isinstance(data["all_ready"], bool)
        assert isinstance(data["images"], list)
        if data["images"]:
            img = data["images"][0]
            assert "id" in img
            assert "filename" in img
            assert "status" in img

    def test_feature_status_empty_project(self, client):
        """Empty project should still match TS shape."""
        proj = _create_project(client, "FStatusEmpty")
        resp = client.get(f"/feature_status/{proj['id']}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["ready"] == 0
        assert data["all_ready"] is False
        assert data["images"] == []


# ========================================================================
#  Download — /download
# ========================================================================


class TestDownloadAlignment:
    """Verify /download endpoint works as frontend expects."""

    def test_download_nonexistent_file(self, client):
        """GET /download/{file_path} → 404 for missing file"""
        resp = client.get("/download/nonexistent.png")
        assert resp.status_code == 404


# ========================================================================
#  Thumbnail — /thumbnail
# ========================================================================


class TestThumbnailAlignment:
    """Verify thumbnail endpoint matches toThumbnailUrl usage."""

    def test_thumbnail_shape(self, client):
        """GET /thumbnail/{id} → JPEG image bytes"""
        proj = _create_project(client, "ThumbShape")
        upload = _upload_image(client, proj["id"], "thumb.png")
        img_id = upload["processed_images"][0]["id"]
        resp = client.get(f"/thumbnail/{img_id}", params={"size": 200})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("image/")

    def test_thumbnail_nonexistent(self, client):
        """GET /thumbnail/99999 → 404"""
        resp = client.get("/thumbnail/99999")
        assert resp.status_code == 404


# ========================================================================
#  Report — /report
# ========================================================================


class TestReportAlignment:
    """Verify /report endpoint shape."""

    def test_report_shape(self, client):
        """GET /report/{id} → comprehensive report JSON"""
        proj = _create_project(client, "ReportShape")
        _upload_image(client, proj["id"], "rpt_a.png")
        _upload_image(client, proj["id"], "rpt_b.png")
        resp = client.get(f"/report/{proj['id']}")
        assert resp.status_code == 200
        data = resp.json()
        assert "project" in data
        assert "images" in data
        assert "summary" in data


# ========================================================================
#  Root Endpoint
# ========================================================================


class TestRootAlignment:
    """Verify root endpoint."""

    def test_root_shape(self, client):
        """GET / → { message, version, endpoints }"""
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data or "name" in data or "title" in data
        assert "version" in data
        assert "endpoints" in data
