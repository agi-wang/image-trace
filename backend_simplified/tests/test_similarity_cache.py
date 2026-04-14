"""Tests for Similarity Cache Layer in image_processor.py."""

import pytest
import numpy as np
from collections import OrderedDict
from pathlib import Path
from PIL import Image as PILImage
from sqlmodel import Session, SQLModel, create_engine, select
from sqlmodel.pool import StaticPool

from app.image_processor import (
    _cache_key,
    _raw_similarity,
    get_or_compute_similarity,
    invalidate_similarity_cache,
    invalidate_feature_cache,
    compute_image_features,
    _GRAY_CACHE,
    _COLOR_CACHE,
    _HIST_CACHE,
    _T2_LOCK,
    _evict_if_needed,
    HASH_ALGOS,
    DESCRIPTOR_ALGOS,
)
from app.models import SimilarityCache


# ---------- fixtures -------------------------------------------------------

@pytest.fixture
def cache_session():
    """In-memory DB session with SimilarityCache table."""
    engine = create_engine(
        "sqlite://", poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture
def textured_pair(tmp_path):
    """Two textured images for similarity computation."""
    np.random.seed(42)
    arr_a = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    path_a = tmp_path / "cache_a.png"
    PILImage.fromarray(arr_a).save(str(path_a))

    arr_b = arr_a.copy()
    arr_b[50:100, 50:100] = 0  # blacken a patch → different
    path_b = tmp_path / "cache_b.png"
    PILImage.fromarray(arr_b).save(str(path_b))

    feat_a = compute_image_features(str(path_a))
    feat_b = compute_image_features(str(path_b))
    return str(path_a), str(path_b), feat_a, feat_b


@pytest.fixture
def identical_pair(tmp_path):
    """Two identical images."""
    img = PILImage.new("RGB", (100, 100), color=(128, 64, 32))
    path_a = tmp_path / "id_a.png"
    path_b = tmp_path / "id_b.png"
    img.save(str(path_a))
    img.save(str(path_b))
    feat_a = compute_image_features(str(path_a))
    feat_b = compute_image_features(str(path_b))
    return str(path_a), str(path_b), feat_a, feat_b


# ============================================================================
#  _cache_key
# ============================================================================

class TestCacheKey:
    def test_canonical_ordering(self):
        """_cache_key always returns (min, max) regardless of input order."""
        assert _cache_key("abc", "xyz") == ("abc", "xyz")
        assert _cache_key("xyz", "abc") == ("abc", "xyz")

    def test_same_key(self):
        """Same hash → same pair."""
        assert _cache_key("aaa", "aaa") == ("aaa", "aaa")

    def test_empty_strings(self):
        """Empty strings don't crash."""
        assert _cache_key("", "") == ("", "")
        assert _cache_key("", "z") == ("", "z")


# ============================================================================
#  _raw_similarity
# ============================================================================

class TestRawSimilarity:
    def test_hash_algo_identical(self, identical_pair):
        """Hash algorithm returns 1.0 for identical images."""
        _, _, feat_a, feat_b = identical_pair
        score = _raw_similarity("", "", "phash", feat_a, feat_b)
        assert score == 1.0

    def test_hash_algo_different(self, textured_pair):
        """Hash algorithm returns valid score for different images."""
        pa, pb, fa, fb = textured_pair
        score = _raw_similarity(pa, pb, "phash", fa, fb)
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("algo", ["ssim", "histogram", "template"])
    def test_pixel_algos(self, textured_pair, algo):
        """Pixel algorithms return valid 0-1 scores."""
        pa, pb, fa, fb = textured_pair
        score = _raw_similarity(pa, pb, algo, fa, fb)
        assert 0.0 <= score <= 1.0

    def test_descriptor_algo(self, textured_pair):
        """Descriptor algorithm returns valid score."""
        pa, pb, fa, fb = textured_pair
        score = _raw_similarity(pa, pb, "orb", fa, fb)
        assert 0.0 <= score <= 1.0

    def test_auto_fusion(self, textured_pair):
        """Auto (hybrid) fusion returns valid score."""
        pa, pb, fa, fb = textured_pair
        score = _raw_similarity(pa, pb, "auto", fa, fb)
        assert 0.0 <= score <= 1.0

    def test_unknown_algo_returns_zero(self, textured_pair):
        """Unknown algorithm name returns 0.0."""
        pa, pb, fa, fb = textured_pair
        score = _raw_similarity(pa, pb, "nonexistent_algo", fa, fb)
        assert score == 0.0


# ============================================================================
#  get_or_compute_similarity
# ============================================================================

class TestGetOrComputeSimilarity:
    def test_cache_miss_computes_and_stores(self, cache_session, textured_pair):
        """First call computes, stores in DB, and returns a valid score."""
        pa, pb, fa, fb = textured_pair
        score = get_or_compute_similarity(
            path_a=pa, path_b=pb,
            file_hash_a=fa["file_hash"], file_hash_b=fb["file_hash"],
            algorithm="phash", features_a=fa, features_b=fb,
            session=cache_session,
        )
        assert 0.0 <= score <= 1.0

        # Verify the cache entry was written
        entries = cache_session.exec(select(SimilarityCache)).all()
        assert len(entries) == 1
        assert entries[0].score == score

    def test_cache_hit_skips_recompute(self, cache_session, textured_pair):
        """Second call returns cached value without recomputing."""
        pa, pb, fa, fb = textured_pair
        score1 = get_or_compute_similarity(
            path_a=pa, path_b=pb,
            file_hash_a=fa["file_hash"], file_hash_b=fb["file_hash"],
            algorithm="phash", features_a=fa, features_b=fb,
            session=cache_session,
        )
        score2 = get_or_compute_similarity(
            path_a=pa, path_b=pb,
            file_hash_a=fa["file_hash"], file_hash_b=fb["file_hash"],
            algorithm="phash", features_a=fa, features_b=fb,
            session=cache_session,
        )
        assert score1 == score2

        # Should still have only 1 entry (cache hit, no new write)
        entries = cache_session.exec(select(SimilarityCache)).all()
        assert len(entries) == 1

    def test_no_session_mode(self, textured_pair):
        """Works without a session (no caching)."""
        pa, pb, fa, fb = textured_pair
        score = get_or_compute_similarity(
            path_a=pa, path_b=pb,
            file_hash_a=fa["file_hash"], file_hash_b=fb["file_hash"],
            algorithm="phash", features_a=fa, features_b=fb,
            session=None,
        )
        assert 0.0 <= score <= 1.0

    def test_rotation_invariant_mode(self, cache_session, textured_pair):
        """rotation_invariant=True uses orientation variants."""
        pa, pb, fa, fb = textured_pair
        score = get_or_compute_similarity(
            path_a=pa, path_b=pb,
            file_hash_a=fa["file_hash"], file_hash_b=fb["file_hash"],
            algorithm="phash", features_a=fa, features_b=fb,
            rotation_invariant=True,
            session=cache_session,
        )
        assert 0.0 <= score <= 1.0

    def test_different_algo_different_cache_entry(self, cache_session, textured_pair):
        """Different algorithms create separate cache entries."""
        pa, pb, fa, fb = textured_pair
        get_or_compute_similarity(
            path_a=pa, path_b=pb,
            file_hash_a=fa["file_hash"], file_hash_b=fb["file_hash"],
            algorithm="phash", features_a=fa, features_b=fb,
            session=cache_session,
        )
        get_or_compute_similarity(
            path_a=pa, path_b=pb,
            file_hash_a=fa["file_hash"], file_hash_b=fb["file_hash"],
            algorithm="dhash", features_a=fa, features_b=fb,
            session=cache_session,
        )
        entries = cache_session.exec(select(SimilarityCache)).all()
        assert len(entries) == 2

    def test_symmetric_hash_order(self, cache_session, textured_pair):
        """A↔B and B↔A produce the same cache entry."""
        pa, pb, fa, fb = textured_pair
        score1 = get_or_compute_similarity(
            path_a=pa, path_b=pb,
            file_hash_a=fa["file_hash"], file_hash_b=fb["file_hash"],
            algorithm="phash", features_a=fa, features_b=fb,
            session=cache_session,
        )
        score2 = get_or_compute_similarity(
            path_a=pb, path_b=pa,
            file_hash_a=fb["file_hash"], file_hash_b=fa["file_hash"],
            algorithm="phash", features_a=fb, features_b=fa,
            session=cache_session,
        )
        assert score1 == score2
        # Should still be 1 entry (canonical ordering)
        entries = cache_session.exec(select(SimilarityCache)).all()
        assert len(entries) == 1


# ============================================================================
#  invalidate_similarity_cache
# ============================================================================

class TestInvalidateSimilarityCache:
    def test_removes_matching_entries(self, cache_session):
        """Removes all entries involving a given hash."""
        cache_session.add(SimilarityCache(
            hash_a="aaa", hash_b="bbb", algorithm="phash", score=0.9
        ))
        cache_session.add(SimilarityCache(
            hash_a="bbb", hash_b="ccc", algorithm="phash", score=0.8
        ))
        cache_session.add(SimilarityCache(
            hash_a="ddd", hash_b="eee", algorithm="phash", score=0.7
        ))
        cache_session.commit()

        invalidate_similarity_cache(cache_session, "bbb")

        remaining = cache_session.exec(select(SimilarityCache)).all()
        assert len(remaining) == 1
        assert remaining[0].hash_a == "ddd"

    def test_no_op_on_unknown_hash(self, cache_session):
        """No error when hash doesn't exist in cache."""
        cache_session.add(SimilarityCache(
            hash_a="aaa", hash_b="bbb", algorithm="phash", score=0.9
        ))
        cache_session.commit()

        invalidate_similarity_cache(cache_session, "zzz")

        remaining = cache_session.exec(select(SimilarityCache)).all()
        assert len(remaining) == 1

    def test_empty_table(self, cache_session):
        """No crash on empty table."""
        invalidate_similarity_cache(cache_session, "anything")
        remaining = cache_session.exec(select(SimilarityCache)).all()
        assert len(remaining) == 0


# ============================================================================
#  invalidate_feature_cache (T2 in-memory caches)
# ============================================================================

class TestInvalidateFeatureCache:
    """Monkeypatch global caches for full isolation."""

    @pytest.fixture(autouse=True)
    def _isolated_caches(self, monkeypatch):
        """Replace global caches with fresh OrderedDicts for each test."""
        import app.image_processor as ip
        monkeypatch.setattr(ip, "_GRAY_CACHE", OrderedDict())
        monkeypatch.setattr(ip, "_COLOR_CACHE", OrderedDict())
        monkeypatch.setattr(ip, "_HIST_CACHE", OrderedDict())
        monkeypatch.setattr(ip, "_t2_bytes", 0)

    def test_clears_gray_and_color_caches(self):
        """Clears gray, color, and histogram caches for a given path."""
        import app.image_processor as ip
        test_path = "/test/invalidate.png"
        fake_array = np.zeros((10, 10), dtype=np.uint8)

        ip._GRAY_CACHE[(test_path, 512)] = fake_array
        ip._COLOR_CACHE[(test_path, 512)] = fake_array
        ip._HIST_CACHE[test_path] = fake_array

        assert (test_path, 512) in ip._GRAY_CACHE

        invalidate_feature_cache(test_path)

        assert (test_path, 512) not in ip._GRAY_CACHE
        assert (test_path, 512) not in ip._COLOR_CACHE
        assert test_path not in ip._HIST_CACHE

    def test_no_op_for_unknown_path(self):
        """No crash when path doesn't exist in any cache."""
        invalidate_feature_cache("/nonexistent/path.png")

    def test_preserves_other_entries(self):
        """Only removes entries for the specified path."""
        import app.image_processor as ip
        fake_array = np.zeros((10, 10), dtype=np.uint8)
        keep_path = "/keep/this.png"
        remove_path = "/remove/that.png"

        ip._GRAY_CACHE[(keep_path, 512)] = fake_array
        ip._GRAY_CACHE[(remove_path, 512)] = fake_array

        invalidate_feature_cache(remove_path)

        assert (keep_path, 512) in ip._GRAY_CACHE
        assert (remove_path, 512) not in ip._GRAY_CACHE


# ============================================================================
#  _evict_if_needed
# ============================================================================

class TestEvictIfNeeded:
    def test_no_op_below_threshold(self):
        """Cache stays intact when below max_size."""
        cache = OrderedDict((i, i) for i in range(5))
        _evict_if_needed(cache, max_size=10)
        assert len(cache) == 5

    def test_evicts_oldest_when_over(self):
        """Evicts oldest entries when cache exceeds max_size (LRU)."""
        cache = OrderedDict((i, i) for i in range(20))
        _evict_if_needed(cache, max_size=10)
        assert len(cache) == 10
        # Oldest 10 keys should be removed (LRU: first-in = oldest)
        for i in range(10):
            assert i not in cache
        for i in range(10, 20):
            assert i in cache

    def test_empty_dict_safe(self):
        """No crash on empty OrderedDict."""
        cache = OrderedDict()
        _evict_if_needed(cache, max_size=10)
        assert len(cache) == 0

    def test_exact_threshold(self):
        """At exactly max_size, no eviction (only > triggers)."""
        cache = OrderedDict((i, i) for i in range(10))
        _evict_if_needed(cache, max_size=10)
        assert len(cache) == 10
