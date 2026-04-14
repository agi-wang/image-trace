"""Tests for report endpoint, analysis runs, download, GET results, and orientation variants."""

import os
import pytest
from PIL import Image as PILImage
from tests.conftest import make_pattern_upload_bytes, make_upload_bytes


# ---------- /report/{project_id} -------------------------------------------


class TestReportEndpoint:
    """Tests for GET /report/{project_id}"""

    def _setup_project(self, client, count=3):
        resp = client.post("/projects", json={"name": "ReportTest"})
        pid = resp.json()["id"]
        colors = [(200, 50, 50), (200, 50, 50), (50, 200, 50)]
        for i in range(count):
            color = colors[i] if i < len(colors) else (i * 40, i * 30, i * 20)
            file_tuple = make_upload_bytes(f"rpt_{i}.png", size=(200, 200), color=color)
            client.post(
                "/upload", data={"project_id": str(pid)}, files={"file": file_tuple}
            )
        return pid

    def test_report_basic(self, client):
        pid = self._setup_project(client)
        resp = client.get(f"/report/{pid}?hash_type=orb&threshold=0.5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["project"]["id"] == pid
        assert data["algorithm"] == "orb"
        assert data["threshold"] == 0.5
        assert "summary" in data
        assert "matrix" in data
        assert "images" in data
        assert "groups" in data
        assert data["summary"]["total_images"] == 3

    def test_report_empty_project(self, client):
        resp = client.post("/projects", json={"name": "EmptyReport"})
        pid = resp.json()["id"]
        resp = client.get(f"/report/{pid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"]["total_images"] == 0
        assert data["groups"] == []

    def test_report_nonexistent_project(self, client):
        resp = client.get("/report/9999")
        assert resp.status_code == 404

    def test_report_with_descriptor_algo(self, client):
        """Report with descriptor algo should include match data in groups."""
        pid = self._setup_project(client, count=2)
        resp = client.get(f"/report/{pid}?hash_type=sift&threshold=0.01")
        assert resp.status_code == 200
        data = resp.json()
        # Should have matrix with values
        assert len(data["matrix"]["values"]) == 2
        assert len(data["matrix"]["values"][0]) == 2
        assert data["matrix"]["values"][0][0] == 1.0  # diagonal

    def test_report_descriptor_match_indexes_are_compact(self, client):
        resp = client.post("/projects", json={"name": "ReportPattern"})
        pid = resp.json()["id"]
        for name in ["rpt_pattern_a.png", "rpt_pattern_b.png"]:
            client.post(
                "/upload",
                data={"project_id": str(pid)},
                files={"file": make_pattern_upload_bytes(name)},
            )

        resp = client.get(f"/report/{pid}?hash_type=orb&threshold=0.01")
        assert resp.status_code == 200
        data = resp.json()
        pair_matches = data["groups"][0]["pair_matches"]
        assert pair_matches
        pair = pair_matches[0]
        assert pair["matches"]
        for match in pair["matches"]:
            assert 0 <= match["a_idx"] < len(pair["keypoints_a"])
            assert 0 <= match["b_idx"] < len(pair["keypoints_b"])

    def test_report_with_hash_algo(self, client):
        """Report with hash algo should work without keypoint data."""
        pid = self._setup_project(client)
        resp = client.get(f"/report/{pid}?hash_type=phash&threshold=0.5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["algorithm"] == "phash"

    def test_report_with_rotation_invariant(self, client):
        pid = self._setup_project(client, count=2)
        resp = client.get(f"/report/{pid}?hash_type=orb&rotation_invariant=true")
        assert resp.status_code == 200
        data = resp.json()
        assert data["rotation_invariant"] is True

    def test_report_matrix_dimensions(self, client):
        """Matrix should be N×N where N = number of images."""
        pid = self._setup_project(client, count=3)
        resp = client.get(f"/report/{pid}?hash_type=orb")
        assert resp.status_code == 200
        data = resp.json()
        n = len(data["images"])
        assert len(data["matrix"]["values"]) == n
        for row in data["matrix"]["values"]:
            assert len(row) == n


# ---------- /analysis_runs -------------------------------------------------


class TestAnalysisRuns:
    """Tests for GET /analysis_runs/{project_id} and /analysis_runs/detail/{id}"""

    def _run_compare(self, client, pid):
        # Use phash (works on solid-color test images; ORB would find 0 keypoints)
        client.post(f"/compare/{pid}", data={"threshold": "0.5", "hash_type": "phash"})

    def test_list_runs_empty(self, client):
        resp = client.post("/projects", json={"name": "RunsEmpty"})
        pid = resp.json()["id"]
        resp = client.get(f"/analysis_runs?project_id={pid}")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_runs_after_compare(self, client):
        resp = client.post("/projects", json={"name": "RunsAfter"})
        pid = resp.json()["id"]
        # Use slightly different colors: different file_hash but same phash
        file1 = make_upload_bytes("run_test1.png", color=(100, 100, 100))
        file2 = make_upload_bytes("run_test2.png", color=(101, 100, 100))
        client.post("/upload", data={"project_id": str(pid)}, files={"file": file1})
        client.post("/upload", data={"project_id": str(pid)}, files={"file": file2})
        self._run_compare(client, pid)
        resp = client.get(f"/analysis_runs?project_id={pid}")
        assert resp.status_code == 200
        runs = resp.json()
        assert len(runs) >= 1
        assert runs[0]["project_id"] == pid
        assert runs[0]["hash_type"] == "phash"

    def test_run_detail(self, client):
        resp = client.post("/projects", json={"name": "RunDetail"})
        pid = resp.json()["id"]
        file1 = make_upload_bytes("detail1.png", color=(50, 50, 50))
        file2 = make_upload_bytes("detail2.png", color=(51, 50, 50))
        client.post("/upload", data={"project_id": str(pid)}, files={"file": file1})
        client.post("/upload", data={"project_id": str(pid)}, files={"file": file2})
        self._run_compare(client, pid)
        runs = client.get(f"/analysis_runs?project_id={pid}").json()
        run_id = runs[0]["id"]
        resp = client.get(f"/analysis_runs/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run"]["id"] == run_id

    def test_run_detail_alias(self, client):
        resp = client.post("/projects", json={"name": "RunDetailAlias"})
        pid = resp.json()["id"]
        client.post(
            "/upload",
            data={"project_id": str(pid)},
            files={"file": make_upload_bytes("alias1.png", color=(50, 50, 50))},
        )
        client.post(
            "/upload",
            data={"project_id": str(pid)},
            files={"file": make_upload_bytes("alias2.png", color=(51, 50, 50))},
        )
        self._run_compare(client, pid)
        runs = client.get(f"/analysis_runs?project_id={pid}").json()
        run_id = runs[0]["id"]

        resp = client.get(f"/analysis_runs/detail/{run_id}")
        assert resp.status_code == 200
        assert resp.json()["run"]["id"] == run_id

    def test_run_detail_not_found(self, client):
        resp = client.get("/analysis_runs/9999")
        assert resp.status_code == 404


# ---------- GET /results/{project_id} --------------------------------------


class TestGetResults:
    """Tests for GET /results/{project_id} (read saved analysis only)"""

    def test_get_results(self, client):
        resp = client.post("/projects", json={"name": "GetResults"})
        pid = resp.json()["id"]
        file_tuple = make_upload_bytes("res.png", color=(80, 80, 80))
        client.post(
            "/upload", data={"project_id": str(pid)}, files={"file": file_tuple}
        )
        client.post(f"/compare/{pid}", data={"threshold": "0.85", "hash_type": "orb"})
        resp = client.get(f"/results/{pid}?threshold=0.85&hash_type=orb")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_images"] == 1
        assert "groups" in data

    def test_get_results_not_found_before_analysis(self, client):
        resp = client.post("/projects", json={"name": "GetResultsEmpty"})
        pid = resp.json()["id"]
        file_tuple = make_upload_bytes("res_empty.png", color=(80, 80, 80))
        client.post(
            "/upload", data={"project_id": str(pid)}, files={"file": file_tuple}
        )
        resp = client.get(f"/results/{pid}?threshold=0.85&hash_type=orb")
        assert resp.status_code == 404

    def test_get_results_does_not_create_new_run(self, client):
        resp = client.post("/projects", json={"name": "GetResultsReadOnly"})
        pid = resp.json()["id"]
        for name in ["res_ro_a.png", "res_ro_b.png"]:
            client.post(
                "/upload",
                data={"project_id": str(pid)},
                files={"file": make_upload_bytes(name)},
            )

        client.post(f"/compare/{pid}", data={"threshold": "0.85", "hash_type": "phash"})
        before = client.get(f"/analysis_runs?project_id={pid}").json()
        assert len(before) == 1

        resp = client.get(f"/results/{pid}?threshold=0.85&hash_type=phash")
        assert resp.status_code == 200

        after = client.get(f"/analysis_runs?project_id={pid}").json()
        assert len(after) == 1

    def test_get_results_not_found(self, client):
        resp = client.get("/results/9999")
        assert resp.status_code == 404


# ---------- /download ------------------------------------------------------


class TestDownload:
    """Tests for GET /download"""

    def test_download_uploaded_file(self, client):
        resp = client.post("/projects", json={"name": "DownloadTest"})
        pid = resp.json()["id"]
        file_tuple = make_upload_bytes("dl_test.png", color=(30, 30, 30))
        upload_resp = client.post(
            "/upload", data={"project_id": str(pid)}, files={"file": file_tuple}
        )
        img_data = upload_resp.json()["processed_images"][0]
        file_path = img_data.get("file_path", "")
        resp = client.get(f"/download/{file_path}")
        assert resp.status_code == 200

    def test_download_not_found(self, client):
        resp = client.get("/download/nonexistent/file.png")
        assert resp.status_code == 404


# ---------- image_processor: resize_image_if_needed -------------------------


class TestResizeImageIfNeeded:
    """Tests for resize_image_if_needed function."""

    def test_small_image_unchanged(self, tmp_path):
        from app.image_processor import resize_image_if_needed

        img_path = str(tmp_path / "small.png")
        PILImage.new("RGB", (100, 100), color="red").save(img_path)
        result = resize_image_if_needed(img_path, max_size=1024)
        assert result == img_path

    def test_large_image_resized(self, tmp_path):
        from app.image_processor import resize_image_if_needed

        img_path = str(tmp_path / "large.png")
        PILImage.new("RGB", (2048, 2048), color="blue").save(img_path)
        result = resize_image_if_needed(img_path, max_size=512)
        img = PILImage.open(result)
        assert max(img.size) <= 512

    def test_resize_to_output_path(self, tmp_path):
        from app.image_processor import resize_image_if_needed

        img_path = str(tmp_path / "orig.png")
        out_path = str(tmp_path / "resized.png")
        PILImage.new("RGB", (2000, 1000), color="green").save(img_path)
        result = resize_image_if_needed(img_path, max_size=500, output_path=out_path)
        assert os.path.exists(out_path)


# ---------- image_processor: orientation variants ---------------------------


class TestOrientationVariants:
    """Tests for rotation/flip invariance functions."""

    def test_generate_orientation_variants(self, tmp_path):
        from app.image_processor import _generate_orientation_variants

        img_path = str(tmp_path / "orient.png")
        PILImage.new("RGB", (100, 100), color="red").save(img_path)
        variants = _generate_orientation_variants(img_path)
        assert len(variants) == 8
        assert variants[0] == img_path  # first is original
        # Check all variant files exist
        for v in variants:
            assert os.path.exists(v)
        # Cleanup temp files
        for v in variants[1:]:
            os.unlink(v)

    def test_compute_features_for_variants(self, tmp_path):
        from app.image_processor import compute_features_for_variants

        img_path = str(tmp_path / "feat_var.png")
        PILImage.new("RGB", (100, 100), color="blue").save(img_path)
        variant_features = compute_features_for_variants(img_path)
        assert len(variant_features) == 8
        for feat in variant_features:
            assert "phash" in feat

    def test_compare_with_orientations(self, tmp_path):
        from app.image_processor import (
            compare_with_orientations,
            calculate_similarity,
            compute_image_features,
        )

        img_a_path = str(tmp_path / "ori_a.png")
        img_b_path = str(tmp_path / "ori_b.png")
        PILImage.new("RGB", (100, 100), color="red").save(img_a_path)
        # img_b is a 90° rotation of img_a
        PILImage.new("RGB", (100, 100), color="red").transpose(
            PILImage.Transpose.ROTATE_90
        ).save(img_b_path)

        feat_a = compute_image_features(img_a_path)

        def scorer(a, b):
            return calculate_similarity(feat_a.get("phash", ""), b.get("phash", ""))

        score = compare_with_orientations(
            img_a_path, img_b_path, scorer, features_a=feat_a
        )
        assert 0 <= score <= 1


# ---------- utils: delete_file_if_exists, cleanup_project_files -------------


class TestFileCleanup:
    """Tests for file cleanup utilities."""

    def test_delete_file_if_exists(self, tmp_path):
        from app.utils import delete_file_if_exists

        f = tmp_path / "to_delete.txt"
        f.write_text("content")
        assert f.exists()
        delete_file_if_exists(f)
        assert not f.exists()

    def test_delete_nonexistent_file_ok(self, tmp_path):
        from app.utils import delete_file_if_exists

        result = delete_file_if_exists(tmp_path / "nope.txt")
        # Should not raise

    def test_is_supported_image_format_webp(self):
        from app.utils import is_supported_image_format

        assert is_supported_image_format("photo.webp") is True

    def test_is_supported_image_format_heic(self):
        from app.utils import is_supported_image_format

        assert is_supported_image_format("photo.heic") is True
