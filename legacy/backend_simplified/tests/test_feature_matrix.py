"""Tests for feature_matrix.py — precompute, serialize, matrix comparison."""

import os
import numpy as np
import pytest
from PIL import Image as PILImage
from tests.conftest import make_upload_bytes


# ---------- Vector serialization -------------------------------------------


class TestVectorSerialization:
    def test_b64_roundtrip_float32(self):
        from app.feature_matrix import vector_to_b64, b64_to_vector

        arr = np.random.rand(128).astype(np.float32)
        b64 = vector_to_b64(arr)
        recovered = b64_to_vector(b64, dtype=np.float32)
        np.testing.assert_array_almost_equal(arr, recovered)

    def test_b64_roundtrip_uint8(self):
        from app.feature_matrix import vector_to_b64, b64_to_vector

        arr = np.array([0, 1, 1, 0, 1] * 12 + [0, 1, 1, 0], dtype=np.uint8)
        b64 = vector_to_b64(arr)
        recovered = b64_to_vector(b64, dtype=np.uint8)
        np.testing.assert_array_equal(arr, recovered)


# ---------- Hash bit conversion -------------------------------------------


class TestHashBits:
    def test_hash_str_to_bits(self):
        from app.feature_matrix import hash_str_to_bits

        bits = hash_str_to_bits("ff00")
        assert bits.shape == (64,)
        # 'f' = 1111, 'f' = 1111, '0' = 0000, '0' = 0000
        assert bits[0] == 1  # first nibble of 'f'
        assert bits[8] == 0  # first nibble of '0'

    def test_empty_hash(self):
        from app.feature_matrix import hash_str_to_bits

        bits = hash_str_to_bits("")
        assert bits.shape == (64,)
        assert bits.sum() == 0


# ---------- Single variant feature computation ----------------------------


class TestSingleVariantFeatures:
    def test_compute_features(self, tmp_path):
        from app.feature_matrix import _compute_single_variant_features

        img_path = str(tmp_path / "test.png")
        PILImage.new("RGB", (200, 200), color="red").save(img_path)
        features = _compute_single_variant_features(img_path)

        # Should have hash features
        assert "phash_bits" in features
        assert "dhash_bits" in features
        assert features["phash_bits"][1] == 64  # dimensions

        # Should have histogram
        assert "histogram_hsv" in features
        assert features["histogram_hsv"][1] == 3000

        # Should have gray flat
        assert "gray_flat" in features
        assert features["gray_flat"][1] == 128 * 128

        # Should have descriptor pooled
        assert "sift_pooled" in features
        assert "orb_pooled" in features


# ---------- Variant generation -------------------------------------------


class TestVariantGeneration:
    def test_generate_8_variants(self, tmp_path):
        from app.feature_matrix import _generate_variant_paths

        img_path = str(tmp_path / "test.png")
        PILImage.new("RGB", (100, 100), color="blue").save(img_path)
        paths = _generate_variant_paths(img_path)
        assert len(paths) == 8
        assert paths[0] == img_path  # original
        for p in paths:
            assert os.path.exists(p)
        # Cleanup
        for p in paths[1:]:
            os.unlink(p)


# ---------- Hash matrix comparison ----------------------------------------


class TestHashMatrix:
    def test_identical_images_score_1(self):
        from app.feature_matrix import compute_hash_similarity_matrix

        bits = np.array([1, 0, 1, 0] * 16, dtype=np.uint8)
        vectors = {
            1: {0: bits.copy()},
            2: {0: bits.copy()},
        }
        matrix = compute_hash_similarity_matrix(vectors, [1, 2])
        assert matrix[0, 1] == 1.0
        assert matrix[1, 0] == 1.0

    def test_different_images_score_low(self):
        from app.feature_matrix import compute_hash_similarity_matrix

        vectors = {
            1: {0: np.zeros(64, dtype=np.uint8)},
            2: {0: np.ones(64, dtype=np.uint8)},
        }
        matrix = compute_hash_similarity_matrix(vectors, [1, 2])
        assert matrix[0, 1] == 0.0  # all bits different


# ---------- Cosine matrix comparison ---------------------------------------


class TestCosineMatrix:
    def test_identical_vectors_score_1(self):
        from app.feature_matrix import compute_cosine_similarity_matrix

        vec = np.random.rand(128).astype(np.float32)
        vectors = {
            1: {0: vec.copy()},
            2: {0: vec.copy()},
        }
        matrix = compute_cosine_similarity_matrix(vectors, [1, 2])
        assert abs(matrix[0, 1] - 1.0) < 0.001

    def test_orthogonal_vectors_score_0(self):
        from app.feature_matrix import compute_cosine_similarity_matrix

        vec_a = np.array([1, 0, 0, 0], dtype=np.float32)
        vec_b = np.array([0, 1, 0, 0], dtype=np.float32)
        vectors = {
            1: {0: vec_a},
            2: {0: vec_b},
        }
        matrix = compute_cosine_similarity_matrix(vectors, [1, 2])
        assert abs(matrix[0, 1]) < 0.001

    def test_rotation_invariant_best_variant(self):
        from app.feature_matrix import compute_cosine_similarity_matrix

        vec_match = np.array([1, 0, 0, 0], dtype=np.float32)
        vec_bad = np.array([0, 1, 0, 0], dtype=np.float32)
        # Batch approach: compares same-variant across images, takes max.
        # variant 0: bad match, variant 1: perfect match
        vectors = {
            1: {0: vec_bad, 1: vec_match},
            2: {0: vec_bad, 1: vec_match},
        }
        matrix = compute_cosine_similarity_matrix(
            vectors, [1, 2], rotation_invariant=True
        )
        assert abs(matrix[0, 1] - 1.0) < 0.001  # variant 1 gives perfect match


# ---------- Feature status endpoint ----------------------------------------


class TestFeatureStatus:
    def test_feature_status_empty(self, client):
        resp = client.post("/projects", json={"name": "StatusTest"})
        pid = resp.json()["id"]
        resp = client.get(f"/feature_status/{pid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["all_ready"] is False

    def test_feature_status_after_upload(self, client):
        resp = client.post("/projects", json={"name": "StatusUpload"})
        pid = resp.json()["id"]
        file_tuple = make_upload_bytes("status_test.png", color=(100, 100, 100))
        client.post(
            "/upload", data={"project_id": str(pid)}, files={"file": file_tuple}
        )
        resp = client.get(f"/feature_status/{pid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert len(data["images"]) == 1


# ---------- Precompute feature matrix pipeline ----------------------------


class TestPrecomputePipeline:
    """Test the full precompute_feature_matrix pipeline."""

    def test_precompute_stores_features(self, tmp_path):
        """precompute_feature_matrix should store features in FeatureStore table."""
        from sqlmodel import Session as _Session, SQLModel, create_engine, select
        from sqlalchemy.pool import StaticPool
        from app.feature_matrix import precompute_feature_matrix
        from app.models import FeatureStore, Image, Project

        engine = create_engine(
            "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
        )
        SQLModel.metadata.create_all(engine)

        img_path = str(tmp_path / "test_precomp.png")
        PILImage.new("RGB", (200, 200), color="green").save(img_path)

        with _Session(engine) as session:
            project = Project(name="PrecompTest")
            session.add(project)
            session.commit()
            session.refresh(project)

            image = Image(
                filename="test_precomp.png",
                project_id=project.id,
                file_path="uploads/test_precomp.png",
                file_hash="abc123",
                phash="ffff0000",
            )
            session.add(image)
            session.commit()
            session.refresh(image)

            precompute_feature_matrix(image.id, img_path, session)

            # Should have stored features
            features = session.exec(
                select(FeatureStore).where(FeatureStore.image_id == image.id)
            ).all()
            assert len(features) > 0

            # Should have features for multiple algorithms
            algos = set(f.algorithm for f in features)
            assert len(algos) >= 5  # at least hash + some descriptors

            # Image status should be 'ready'
            session.refresh(image)
            assert image.feature_status == "ready"

    def test_precompute_stores_8_variants(self, tmp_path):
        """Each algorithm should have up to 8 variant vectors stored."""
        from sqlmodel import Session as _Session, SQLModel, create_engine, select
        from sqlalchemy.pool import StaticPool
        from app.feature_matrix import precompute_feature_matrix
        from app.models import FeatureStore, Image, Project

        engine = create_engine(
            "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
        )
        SQLModel.metadata.create_all(engine)

        img_path = str(tmp_path / "variant_test.png")
        PILImage.new("RGB", (200, 200), color="blue").save(img_path)

        with _Session(engine) as session:
            project = Project(name="VariantTest")
            session.add(project)
            session.commit()
            session.refresh(project)

            image = Image(
                filename="variant_test.png",
                project_id=project.id,
                file_path="uploads/variant_test.png",
                file_hash="def456",
                phash="00ff00ff",
            )
            session.add(image)
            session.commit()
            session.refresh(image)

            precompute_feature_matrix(image.id, img_path, session)

            # Check variant count for a hash feature (should have 8)
            phash_features = session.exec(
                select(FeatureStore).where(
                    FeatureStore.image_id == image.id,
                    FeatureStore.algorithm == "phash_bits",
                )
            ).all()
            assert len(phash_features) == 8


# ---------- are_features_ready -----------------------------------------


class TestAreFeaturesReady:
    """Test the are_features_ready function."""

    def test_no_images_returns_true(self):
        """Empty list should return True (vacuously true)."""
        from sqlmodel import Session as _Session, SQLModel, create_engine
        from sqlalchemy.pool import StaticPool
        from app.feature_matrix import are_features_ready

        engine = create_engine(
            "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
        )
        SQLModel.metadata.create_all(engine)
        with _Session(engine) as session:
            assert are_features_ready(session, []) is True

    def test_pending_image_returns_false(self):
        """Image with status 'pending' should return False."""
        from sqlmodel import Session as _Session, SQLModel, create_engine
        from sqlalchemy.pool import StaticPool
        from app.feature_matrix import are_features_ready
        from app.models import Image, Project

        engine = create_engine(
            "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
        )
        SQLModel.metadata.create_all(engine)
        with _Session(engine) as session:
            project = Project(name="PendingCheck")
            session.add(project)
            session.commit()
            session.refresh(project)
            image = Image(
                filename="pending.png",
                project_id=project.id,
                file_path="uploads/pending.png",
                file_hash="pend1",
                phash="0000",
                feature_status="pending",
            )
            session.add(image)
            session.commit()
            session.refresh(image)
            assert are_features_ready(session, [image.id]) is False

    def test_ready_image_returns_true(self):
        """Image with status 'ready' should return True."""
        from sqlmodel import Session as _Session, SQLModel, create_engine
        from sqlalchemy.pool import StaticPool
        from app.feature_matrix import are_features_ready
        from app.models import Image, Project

        engine = create_engine(
            "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
        )
        SQLModel.metadata.create_all(engine)
        with _Session(engine) as session:
            project = Project(name="ReadyCheck")
            session.add(project)
            session.commit()
            session.refresh(project)
            image = Image(
                filename="ready.png",
                project_id=project.id,
                file_path="uploads/ready.png",
                file_hash="rdy1",
                phash="ffff",
                feature_status="ready",
            )
            session.add(image)
            session.commit()
            session.refresh(image)
            assert are_features_ready(session, [image.id]) is True

    def test_mixed_status_returns_false(self):
        from sqlmodel import Session as _Session, SQLModel, create_engine
        from sqlalchemy.pool import StaticPool
        from app.feature_matrix import are_features_ready
        from app.models import Image, Project

        engine = create_engine(
            "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
        )
        SQLModel.metadata.create_all(engine)
        with _Session(engine) as session:
            project = Project(name="MixedReadyCheck")
            session.add(project)
            session.commit()
            session.refresh(project)

            ready_image = Image(
                filename="ready.png",
                project_id=project.id,
                file_path="uploads/ready.png",
                file_hash="rdy2",
                phash="ffff",
                feature_status="ready",
            )
            pending_image = Image(
                filename="pending.png",
                project_id=project.id,
                file_path="uploads/pending.png",
                file_hash="pend2",
                phash="0000",
                feature_status="pending",
            )
            session.add(ready_image)
            session.add(pending_image)
            session.commit()
            session.refresh(ready_image)
            session.refresh(pending_image)

            assert (
                are_features_ready(session, [ready_image.id, pending_image.id]) is False
            )


# ---------- _build_matrix_from_vectors -----------------------------------


class TestBuildMatrix:
    """Test _build_matrix_from_vectors edge cases."""

    def test_empty_vectors(self):
        """Empty vectors dict should return zeros matrix."""
        from app.feature_matrix import _build_matrix_from_vectors

        matrix = _build_matrix_from_vectors({}, [1, 2], variant=0)
        assert matrix.shape[0] == 2

    def test_missing_image_fills_zeros(self):
        """Missing image should get zero row."""
        from app.feature_matrix import _build_matrix_from_vectors

        vec = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        vectors = {1: {0: vec}}  # image 2 is missing
        matrix = _build_matrix_from_vectors(vectors, [1, 2], variant=0)
        assert matrix.shape == (2, 3)
        np.testing.assert_array_equal(matrix[0], vec)
        np.testing.assert_array_equal(matrix[1], np.zeros(3))


# ---------- ALGO_TO_FEATURE mapping consistency ---------------------------


class TestAlgoFeatureMapping:
    """Verify ALGO_TO_FEATURE and FEATURE_TO_ALGO mappings are consistent."""

    def test_algo_to_feature_all_have_inverses(self):
        """Every ALGO_TO_FEATURE key should have an inverse in FEATURE_TO_ALGO."""
        from app.feature_matrix import ALGO_TO_FEATURE, FEATURE_TO_ALGO

        for algo, feature in ALGO_TO_FEATURE.items():
            assert feature in FEATURE_TO_ALGO, (
                f"ALGO_TO_FEATURE['{algo}'] = '{feature}' not in FEATURE_TO_ALGO"
            )
            assert FEATURE_TO_ALGO[feature] == algo, (
                f"FEATURE_TO_ALGO['{feature}'] = '{FEATURE_TO_ALGO[feature]}' != '{algo}'"
            )

    def test_feature_to_algo_all_have_inverses(self):
        """Every FEATURE_TO_ALGO key should have an inverse in ALGO_TO_FEATURE."""
        from app.feature_matrix import ALGO_TO_FEATURE, FEATURE_TO_ALGO

        for feature, algo in FEATURE_TO_ALGO.items():
            assert algo in ALGO_TO_FEATURE, (
                f"FEATURE_TO_ALGO['{feature}'] = '{algo}' not in ALGO_TO_FEATURE"
            )

    def test_all_features_have_known_types(self):
        """All features should be in HASH, DESCRIPTOR, or PIXEL category."""
        from app.feature_matrix import (
            ALL_FEATURES,
            HASH_FEATURES,
            DESCRIPTOR_FEATURES,
            PIXEL_FEATURES,
        )

        expected = set(HASH_FEATURES + DESCRIPTOR_FEATURES + PIXEL_FEATURES)
        assert set(ALL_FEATURES) == expected


# ---------- FeatureStore model -------------------------------------------


class TestFeatureStoreModel:
    """Test the FeatureStore SQLModel."""

    def test_create_and_read(self):
        """Should be able to create and read a FeatureStore record."""
        from sqlmodel import Session as _Session, SQLModel, create_engine, select
        from sqlalchemy.pool import StaticPool
        from app.models import FeatureStore, Image, Project
        from app.feature_matrix import vector_to_b64

        engine = create_engine(
            "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
        )
        SQLModel.metadata.create_all(engine)

        with _Session(engine) as session:
            project = Project(name="FSTest")
            session.add(project)
            session.commit()
            session.refresh(project)

            image = Image(
                filename="fs_test.png",
                project_id=project.id,
                file_path="uploads/fs_test.png",
                file_hash="fs1",
                phash="aaaa",
            )
            session.add(image)
            session.commit()
            session.refresh(image)

            vec = np.random.rand(64).astype(np.float32)
            fs = FeatureStore(
                image_id=image.id,
                variant_idx=0,
                algorithm="phash_bits",
                vector=vector_to_b64(vec),
                dimensions=64,
            )
            session.add(fs)
            session.commit()
            session.refresh(fs)

            assert fs.id is not None
            assert fs.image_id == image.id
            assert fs.algorithm == "phash_bits"
            assert fs.dimensions == 64

    def test_multiple_variants_per_image(self):
        """Should be able to store multiple variants for same image."""
        from sqlmodel import Session as _Session, SQLModel, create_engine, select
        from sqlalchemy.pool import StaticPool
        from app.models import FeatureStore, Image, Project
        from app.feature_matrix import vector_to_b64

        engine = create_engine(
            "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
        )
        SQLModel.metadata.create_all(engine)

        with _Session(engine) as session:
            project = Project(name="MultiVariant")
            session.add(project)
            session.commit()
            session.refresh(project)

            image = Image(
                filename="mv_test.png",
                project_id=project.id,
                file_path="uploads/mv_test.png",
                file_hash="mv1",
                phash="bbbb",
            )
            session.add(image)
            session.commit()
            session.refresh(image)

            for variant_idx in range(8):
                vec = np.random.rand(64).astype(np.float32)
                fs = FeatureStore(
                    image_id=image.id,
                    variant_idx=variant_idx,
                    algorithm="phash_bits",
                    vector=vector_to_b64(vec),
                    dimensions=64,
                )
                session.add(fs)
            session.commit()

            found = session.exec(
                select(FeatureStore).where(
                    FeatureStore.image_id == image.id,
                    FeatureStore.algorithm == "phash_bits",
                )
            ).all()
            assert len(found) == 8
            variants = sorted(f.variant_idx for f in found)
            assert variants == list(range(8))
