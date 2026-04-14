"""API integration tests for Image Trace backend."""

import os
import time
from io import BytesIO

import pytest
from sqlmodel import Session, create_engine, select

from tests.conftest import make_upload_bytes


class TestHealthAndRoot:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "endpoints" in data


class TestProjects:
    def test_create_project(self, client):
        resp = client.post(
            "/projects", json={"name": "Test Project", "description": "desc"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Test Project"
        assert data["id"] is not None
        assert data["image_count"] == 0

    def test_list_projects(self, client):
        client.post("/projects", json={"name": "P1"})
        client.post("/projects", json={"name": "P2"})
        resp = client.get("/projects")
        assert resp.status_code == 200
        assert len(resp.json()) >= 2

    def test_get_project(self, client):
        create_resp = client.post("/projects", json={"name": "Detail"})
        pid = create_resp.json()["id"]
        resp = client.get(f"/projects/{pid}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "Detail"

    def test_project_image_counts_after_uploads(self, client):
        create_resp = client.post("/projects", json={"name": "Counted"})
        pid = create_resp.json()["id"]
        client.post(
            "/upload",
            data={"project_id": str(pid)},
            files={"file": make_upload_bytes("count_a.png")},
        )
        client.post(
            "/upload",
            data={"project_id": str(pid)},
            files={"file": make_upload_bytes("count_b.png")},
        )

        detail = client.get(f"/projects/{pid}")
        assert detail.status_code == 200
        assert detail.json()["image_count"] == 2

        listing = client.get("/projects")
        assert listing.status_code == 200
        project = next(item for item in listing.json() if item["id"] == pid)
        assert project["image_count"] == 2

    def test_get_project_not_found(self, client):
        resp = client.get("/projects/9999")
        assert resp.status_code == 404

    def test_delete_project(self, client):
        create_resp = client.post("/projects", json={"name": "ToDelete"})
        pid = create_resp.json()["id"]
        resp = client.delete(f"/projects/{pid}")
        assert resp.status_code == 200
        # 确认已删除
        resp2 = client.get(f"/projects/{pid}")
        assert resp2.status_code == 404

    def test_delete_project_cleans_artifacts(self, client, tmp_dir):
        from app.models import AnalysisRun, FeatureStore, Image, SimilarityCache

        create_resp = client.post("/projects", json={"name": "Cleanup"})
        pid = create_resp.json()["id"]

        up1 = client.post(
            "/upload",
            data={"project_id": str(pid)},
            files={"file": make_upload_bytes("cleanup_a.png")},
        )
        up2 = client.post(
            "/upload",
            data={"project_id": str(pid)},
            files={"file": make_upload_bytes("cleanup_b.png")},
        )
        assert up1.status_code == 200
        assert up2.status_code == 200

        images = client.get(f"/images/{pid}").json()
        assert len(images) == 2

        thumb_dir = tmp_dir / "thumbnails"
        thumb_dir.mkdir(exist_ok=True)
        thumb_path = thumb_dir / f"{images[0]['id']}_200.jpg"
        thumb_path.write_bytes(b"thumb")

        engine = create_engine(os.environ["DATABASE_URL"])
        with Session(engine) as session:
            img_rows = session.exec(select(Image).where(Image.project_id == pid)).all()
            assert len(img_rows) == 2

            for img in img_rows:
                session.add(
                    FeatureStore(
                        image_id=img.id,
                        variant_idx=0,
                        algorithm="phash_bits",
                        vector="Zm9v",
                        dimensions=3,
                    )
                )

            session.add(
                AnalysisRun(
                    project_id=pid,
                    hash_type="phash",
                    threshold=0.85,
                    total_images=2,
                    groups_count=1,
                    unique_count=0,
                    summary="{}",
                )
            )
            session.add(
                SimilarityCache(
                    hash_a=min(img_rows[0].file_hash, img_rows[1].file_hash),
                    hash_b=max(img_rows[0].file_hash, img_rows[1].file_hash),
                    algorithm="phash",
                    rotation_invariant=False,
                    score=1.0,
                )
            )
            session.commit()

            file_paths = [tmp_dir / img.file_path for img in img_rows]
            assert all(p.exists() for p in file_paths)

        resp = client.delete(f"/projects/{pid}")
        assert resp.status_code == 200
        assert thumb_path.exists() is False
        assert all(not p.exists() for p in file_paths)

        with Session(engine) as session:
            assert (
                session.exec(select(Image).where(Image.project_id == pid)).all() == []
            )
            assert session.exec(select(FeatureStore)).all() == []
            assert (
                session.exec(
                    select(AnalysisRun).where(AnalysisRun.project_id == pid)
                ).all()
                == []
            )
            assert session.exec(select(SimilarityCache)).all() == []


class TestUpload:
    def _create_project(self, client):
        resp = client.post("/projects", json={"name": "Upload Test"})
        return resp.json()["id"]

    def test_upload_png(self, client):
        pid = self._create_project(client)
        file_tuple = make_upload_bytes("test.png")
        resp = client.post(
            "/upload", data={"project_id": str(pid)}, files={"file": file_tuple}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["file_type"] == "image"
        assert data["error"] is None
        assert len(data["processed_images"]) == 1

    def test_upload_tif(self, client):
        """验证 .tif 格式上传可用。"""
        pid = self._create_project(client)
        file_tuple = make_upload_bytes("sample.tif")
        resp = client.post(
            "/upload", data={"project_id": str(pid)}, files={"file": file_tuple}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["file_type"] == "image"
        assert data["error"] is None

    def test_upload_jpg(self, client):
        pid = self._create_project(client)
        file_tuple = make_upload_bytes("photo.jpg")
        resp = client.post(
            "/upload", data={"project_id": str(pid)}, files={"file": file_tuple}
        )
        assert resp.status_code == 200
        assert resp.json()["file_type"] == "image"

    def test_upload_unsupported_format(self, client):
        pid = self._create_project(client)
        from io import BytesIO

        file_tuple = ("readme.txt", BytesIO(b"hello world"), "text/plain")
        resp = client.post(
            "/upload", data={"project_id": str(pid)}, files={"file": file_tuple}
        )
        assert resp.status_code == 400
        data = resp.json()
        assert "不支持的文件格式" in data["detail"]

    def test_upload_sanitizes_filename(self, client):
        pid = self._create_project(client)
        file_tuple = make_upload_bytes("../nested/test.png")
        resp = client.post(
            "/upload", data={"project_id": str(pid)}, files={"file": file_tuple}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "test.png"
        assert data["file_path"].endswith("test.png")

    def test_upload_to_nonexistent_project(self, client):
        file_tuple = make_upload_bytes("test.png")
        resp = client.post(
            "/upload", data={"project_id": "9999"}, files={"file": file_tuple}
        )
        assert resp.status_code == 404

    def test_document_upload_triggers_precompute(self, client, tmp_dir, monkeypatch):
        import app.image_processor as ip_mod
        import app.routers.uploads as uploads_mod
        from unittest.mock import MagicMock

        pid = self._create_project(client)
        extracted = tmp_dir / "extracted" / "doc_image.png"
        extracted.parent.mkdir(exist_ok=True)
        extracted.write_bytes(make_upload_bytes("doc_image.png")[1].read())

        features = ip_mod.compute_image_features(str(extracted))
        features.update(
            {
                "filename": "doc_image.png",
                "file_path": "extracted/doc_image.png",
                "extracted_from": "fake.pdf",
            }
        )

        mock_parser = MagicMock()
        mock_parser.process_document.return_value = {
            "status": "success",
            "images": [features],
        }
        monkeypatch.setattr(
            uploads_mod,
            "_get_doc_parser",
            lambda: mock_parser,
        )

        resp = client.post(
            "/upload",
            data={"project_id": str(pid)},
            files={"file": ("fake.pdf", BytesIO(b"%PDF-1.4"), "application/pdf")},
        )
        assert resp.status_code == 200
        time.sleep(3)

        status = client.get(f"/feature_status/{pid}")
        assert status.status_code == 200
        data = status.json()
        assert data["total"] == 1
        assert data["images"][0]["status"] in ("computing", "ready")


class TestImages:
    def _upload_image(self, client, pid, filename="test.png"):
        file_tuple = make_upload_bytes(filename)
        resp = client.post(
            "/upload", data={"project_id": str(pid)}, files={"file": file_tuple}
        )
        return resp.json()

    def test_list_project_images(self, client):
        resp = client.post("/projects", json={"name": "ImgList"})
        pid = resp.json()["id"]
        self._upload_image(client, pid, "a.png")
        self._upload_image(client, pid, "b.png")

        resp = client.get(f"/images/{pid}")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_list_images_nonexistent_project(self, client):
        resp = client.get("/images/9999")
        assert resp.status_code == 404

    def test_delete_image(self, client):
        resp = client.post("/projects", json={"name": "DelImg"})
        pid = resp.json()["id"]
        upload_data = self._upload_image(client, pid)
        img_id = upload_data["processed_images"][0]["id"]

        resp = client.delete(f"/images/{img_id}")
        assert resp.status_code == 200

        # 确认已删除
        resp2 = client.get(f"/images/{pid}")
        assert len(resp2.json()) == 0


class TestCompare:
    def _setup_project_with_images(self, client, count=3):
        resp = client.post("/projects", json={"name": "Compare Test"})
        pid = resp.json()["id"]
        colors = [(128, 64, 32), (128, 64, 32), (0, 255, 0)]
        for i in range(count):
            color = colors[i] if i < len(colors) else (i * 50, i * 30, i * 10)
            file_tuple = make_upload_bytes(f"img_{i}.png", color=color)
            client.post(
                "/upload", data={"project_id": str(pid)}, files={"file": file_tuple}
            )
        return pid

    def test_compare_empty_project(self, client):
        resp = client.post("/projects", json={"name": "Empty"})
        pid = resp.json()["id"]
        resp = client.post(
            f"/compare/{pid}", data={"threshold": "0.85", "hash_type": "orb"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_images"] == 0

    def test_compare_with_images(self, client):
        pid = self._setup_project_with_images(client)
        resp = client.post(
            f"/compare/{pid}", data={"threshold": "0.85", "hash_type": "orb"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_images"] == 3
        assert "groups" in data
        assert "unique_images" in data

    def test_compare_nonexistent_project(self, client):
        resp = client.post(
            "/compare/9999", data={"threshold": "0.85", "hash_type": "orb"}
        )
        assert resp.status_code == 404

    def test_compare_invalid_threshold(self, client):
        resp = client.post("/projects", json={"name": "Invalid"})
        pid = resp.json()["id"]
        resp = client.post(
            f"/compare/{pid}", data={"threshold": "2.0", "hash_type": "orb"}
        )
        assert resp.status_code == 400

    def test_compare_invalid_hash_type(self, client):
        resp = client.post("/projects", json={"name": "Invalid2"})
        pid = resp.json()["id"]
        resp = client.post(
            f"/compare/{pid}", data={"threshold": "0.85", "hash_type": "invalid"}
        )
        assert resp.status_code == 400
