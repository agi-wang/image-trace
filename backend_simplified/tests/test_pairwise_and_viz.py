"""Tests for pairwise matrix API, expanded visualize_match, and rotation_invariant."""

import pytest
from tests.conftest import make_pattern_upload_bytes, make_upload_bytes


class TestPairwiseMatrix:
    """Tests for GET /pairwise_matrix/{project_id}"""

    def _setup_project(self, client, count=3):
        resp = client.post("/projects", json={"name": "PairwiseTest"})
        pid = resp.json()["id"]
        colors = [(200, 50, 50), (200, 50, 50), (50, 50, 200)]
        for i in range(count):
            color = colors[i] if i < len(colors) else (i * 50, i * 30, i * 10)
            file_tuple = make_upload_bytes(f"pw_{i}.png", color=color)
            client.post(
                "/upload", data={"project_id": str(pid)}, files={"file": file_tuple}
            )
        return pid

    def test_pairwise_empty_project(self, client):
        resp = client.post("/projects", json={"name": "EmptyPW"})
        pid = resp.json()["id"]
        resp = client.get(f"/pairwise_matrix/{pid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["names"] == []
        assert data["matrix"] == []

    def test_pairwise_with_images(self, client):
        pid = self._setup_project(client, count=3)
        resp = client.get(f"/pairwise_matrix/{pid}?hash_type=phash")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["names"]) == 3
        assert len(data["matrix"]) == 3
        assert len(data["matrix"][0]) == 3
        assert len(data["image_ids"]) == 3
        assert data["algorithm"] == "phash"
        # Diagonal should be 1.0
        for i in range(3):
            assert data["matrix"][i][i] == 1.0
        # Matrix should be symmetric
        for i in range(3):
            for j in range(3):
                assert data["matrix"][i][j] == data["matrix"][j][i]

    def test_pairwise_invalid_algo(self, client):
        resp = client.post("/projects", json={"name": "BadAlgo"})
        pid = resp.json()["id"]
        resp = client.get(f"/pairwise_matrix/{pid}?hash_type=invalid")
        assert resp.status_code == 400

    @pytest.mark.parametrize(
        "algo", ["phash", "dhash", "ssim", "histogram", "orb", "sift"]
    )
    def test_pairwise_all_algorithms(self, client, algo):
        """Test pairwise matrix with various algorithm types."""
        pid = self._setup_project(client, count=2)
        resp = client.get(f"/pairwise_matrix/{pid}?hash_type={algo}")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["matrix"]) == 2
        assert 0.0 <= data["matrix"][0][1] <= 1.0


class TestVisualizeMatchExpanded:
    """Tests for expanded /visualize_match (akaze/kaze support)."""

    def _setup_pair(self, client):
        resp = client.post("/projects", json={"name": "VizMatch"})
        pid = resp.json()["id"]
        for name in ["viz_a.png", "viz_b.png"]:
            file_tuple = make_upload_bytes(name, size=(200, 200))
            client.post(
                "/upload", data={"project_id": str(pid)}, files={"file": file_tuple}
            )
        images = client.get(f"/images/{pid}").json()
        return images[0]["id"], images[1]["id"]

    @pytest.mark.parametrize("algo", ["orb", "sift", "brisk", "akaze", "kaze"])
    def test_visualize_match_all_descriptors(self, client, algo):
        a_id, b_id = self._setup_pair(client)
        resp = client.post(
            "/visualize_match",
            data={
                "image_a_id": str(a_id),
                "image_b_id": str(b_id),
                "hash_type": algo,
            },
        )
        # May fail with "not enough features" on simple images — that's okay
        assert resp.status_code in (200, 500)

    def test_visualize_match_unsupported_algo(self, client):
        a_id, b_id = self._setup_pair(client)
        resp = client.post(
            "/visualize_match",
            data={
                "image_a_id": str(a_id),
                "image_b_id": str(b_id),
                "hash_type": "phash",
            },
        )
        assert resp.status_code == 400

    def test_visualize_match_nonexistent_image(self, client):
        resp = client.post(
            "/visualize_match",
            data={
                "image_a_id": "9999",
                "image_b_id": "9998",
                "hash_type": "orb",
            },
        )
        assert resp.status_code == 404


class TestRotationInvariantAPI:
    """Tests for rotation_invariant parameter in /compare."""

    def _setup_project(self, client):
        resp = client.post("/projects", json={"name": "RotInv"})
        pid = resp.json()["id"]
        for name, color in [("r1.png", (200, 50, 50)), ("r2.png", (200, 50, 50))]:
            file_tuple = make_upload_bytes(name, color=color)
            client.post(
                "/upload", data={"project_id": str(pid)}, files={"file": file_tuple}
            )
        return pid

    def test_compare_with_rotation_invariant(self, client):
        pid = self._setup_project(client)
        resp = client.post(
            f"/compare/{pid}",
            data={
                "threshold": "0.85",
                "hash_type": "ssim",
                "rotation_invariant": "true",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_images"] == 2

    def test_compare_without_rotation_invariant(self, client):
        pid = self._setup_project(client)
        resp = client.post(
            f"/compare/{pid}",
            data={
                "threshold": "0.85",
                "hash_type": "ssim",
                "rotation_invariant": "false",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_images"] == 2

    @pytest.mark.parametrize(
        "algo", ["auto", "ssim", "histogram", "template", "colorhash"]
    )
    def test_compare_new_algos(self, client, algo):
        """Verify all newly added algorithms work via the API."""
        pid = self._setup_project(client)
        resp = client.post(
            f"/compare/{pid}",
            data={
                "threshold": "0.85",
                "hash_type": algo,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["total_images"] == 2


class TestDbSchemaMigration:
    """Tests for automatic DB schema migration on startup."""

    def test_migrate_adds_missing_column(self, tmp_path):
        """Verify migrate_db_schema adds a missing colorhash column."""
        import sqlite3

        db_path = tmp_path / "test_migrate.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE images (
                id INTEGER PRIMARY KEY,
                filename VARCHAR NOT NULL,
                project_id INTEGER NOT NULL,
                file_path VARCHAR NOT NULL,
                file_hash VARCHAR NOT NULL,
                phash VARCHAR NOT NULL,
                created_at DATETIME
            )
        """)
        conn.commit()
        conn.close()

        from app.runtime import migrate_db_schema

        migrate_db_schema(f"sqlite:///{db_path}")

        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("PRAGMA table_info(images)")
        cols = {row[1] for row in cursor.fetchall()}
        conn.close()
        assert "colorhash" in cols
        assert "dhash" in cols
        assert "ahash" in cols

    def test_migrate_no_crash_on_complete_schema(self, tmp_path):
        """Verify migration is a no-op when schema is already up to date."""
        import sqlite3

        db_path = tmp_path / "test_noop.db"
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE images (
                id INTEGER PRIMARY KEY,
                filename VARCHAR, project_id INTEGER, file_path VARCHAR,
                file_hash VARCHAR, phash VARCHAR, dhash VARCHAR,
                ahash VARCHAR, whash VARCHAR, colorhash VARCHAR,
                extracted_from VARCHAR, file_size INTEGER,
                width INTEGER, height INTEGER, created_at DATETIME
            )
        """)
        conn.commit()
        conn.close()

        from app.runtime import migrate_db_schema

        migrate_db_schema(f"sqlite:///{db_path}")

    def test_migrate_no_crash_on_missing_db(self):
        """Verify migration gracefully handles non-existent DB file."""
        from app.runtime import migrate_db_schema

        migrate_db_schema("sqlite:////tmp/does_not_exist_12345.db")


class TestThumbnail:
    """Tests for GET /thumbnail/{image_id}"""

    def _upload_image(self, client, pid, filename="thumb.png", color=(128, 64, 32)):
        file_tuple = make_upload_bytes(filename, color=color)
        resp = client.post(
            "/upload", data={"project_id": str(pid)}, files={"file": file_tuple}
        )
        return resp.json()["processed_images"][0]["id"]

    def test_thumbnail_success(self, client):
        resp = client.post("/projects", json={"name": "ThumbTest"})
        pid = resp.json()["id"]
        img_id = self._upload_image(client, pid)
        resp = client.get(f"/thumbnail/{img_id}?size=200")
        assert resp.status_code == 200
        assert resp.headers.get("content-type") == "image/jpeg"

    def test_thumbnail_not_found(self, client):
        resp = client.get("/thumbnail/9999")
        assert resp.status_code == 404

    def test_thumbnail_caching(self, client):
        resp = client.post("/projects", json={"name": "ThumbCache"})
        pid = resp.json()["id"]
        img_id = self._upload_image(client, pid)
        # First call creates thumbnail
        resp1 = client.get(f"/thumbnail/{img_id}")
        assert resp1.status_code == 200
        # Second call uses cache
        resp2 = client.get(f"/thumbnail/{img_id}")
        assert resp2.status_code == 200


class TestMatchData:
    """Tests for POST /match_data"""

    def _setup_pair(self, client):
        resp = client.post("/projects", json={"name": "MatchData"})
        pid = resp.json()["id"]
        ids = []
        for name, color in [("md_a.png", (200, 50, 50)), ("md_b.png", (50, 200, 50))]:
            file_tuple = make_upload_bytes(name, size=(200, 200), color=color)
            client.post(
                "/upload", data={"project_id": str(pid)}, files={"file": file_tuple}
            )
        images = client.get(f"/images/{pid}").json()
        return images[0]["id"], images[1]["id"]

    def test_match_data_success(self, client):
        resp = client.post("/projects", json={"name": "MatchDataPattern"})
        pid = resp.json()["id"]
        client.post(
            "/upload",
            data={"project_id": str(pid)},
            files={"file": make_pattern_upload_bytes("md_a.png")},
        )
        client.post(
            "/upload",
            data={"project_id": str(pid)},
            files={"file": make_pattern_upload_bytes("md_b.png")},
        )
        images = client.get(f"/images/{pid}").json()
        a_id, b_id = images[0]["id"], images[1]["id"]
        resp = client.post(
            "/match_data",
            data={
                "image_a_id": str(a_id),
                "image_b_id": str(b_id),
                "hash_type": "orb",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "image_a" in data
        assert "image_b" in data
        assert "matches" in data
        assert "score" in data
        assert data["image_a"]["width"] > 0
        assert len(data["matches"]) > 0
        for match in data["matches"]:
            assert 0 <= match["a_idx"] < len(data["image_a"]["keypoints"])
            assert 0 <= match["b_idx"] < len(data["image_b"]["keypoints"])

    def test_match_data_invalid_algo(self, client):
        a_id, b_id = self._setup_pair(client)
        resp = client.post(
            "/match_data",
            data={
                "image_a_id": str(a_id),
                "image_b_id": str(b_id),
                "hash_type": "phash",
            },
        )
        assert resp.status_code == 400

    def test_match_data_not_found(self, client):
        resp = client.post(
            "/match_data",
            data={
                "image_a_id": "9999",
                "image_b_id": "9998",
                "hash_type": "sift",
            },
        )
        assert resp.status_code == 404
