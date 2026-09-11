"""Tests for new comparison algorithms (Tier 2 + fusion)."""

import pytest
import numpy as np
from PIL import Image as PILImage

from app.image_processor import (
    calculate_ssim_similarity,
    calculate_histogram_similarity,
    calculate_template_similarity,
    calculate_hybrid_similarity,
    compute_image_features,
    compute_descriptor,
    calculate_descriptor_similarity,
    HASH_ALGOS, PIXEL_ALGOS, DESCRIPTOR_ALGOS, FUSION_ALGOS, ALL_ALGOS,
)


# ---------- 辅助 fixtures ---------------------------------------------------

@pytest.fixture
def identical_pair(tmp_path):
    """两张内容相同的有纹理图片（确保 AKAZE/KAZE 能提取到特征点）。"""
    np.random.seed(42)
    # 棋盘格 + 随机噪声 → 丰富的特征点
    arr = np.zeros((200, 200, 3), dtype=np.uint8)
    for y in range(200):
        for x in range(200):
            if (x // 20 + y // 20) % 2 == 0:
                arr[y, x] = [200, 100, 50]
            else:
                arr[y, x] = [50, 100, 200]
    # 添加少量噪声
    noise = np.random.randint(-10, 10, arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    pa = tmp_path / "id_a.png"
    pb = tmp_path / "id_b.png"
    PILImage.fromarray(arr).save(str(pa))
    PILImage.fromarray(arr).save(str(pb))
    return str(pa), str(pb)


@pytest.fixture
def different_pair(tmp_path):
    """两张完全不同的图片。"""
    arr_a = np.zeros((100, 100, 3), dtype=np.uint8)
    for x in range(100):
        arr_a[:, x] = [int(x * 2.55), 0, 0]
    pa = tmp_path / "diff_a.png"
    PILImage.fromarray(arr_a).save(str(pa))

    arr_b = np.zeros((100, 100, 3), dtype=np.uint8)
    for y in range(100):
        arr_b[y, :] = [0, 0, 255] if y % 20 < 10 else [255, 255, 0]
    pb = tmp_path / "diff_b.png"
    PILImage.fromarray(arr_b).save(str(pb))
    return str(pa), str(pb)


# ---------- Algorithm Registry -----------------------------------------------

class TestAlgoRegistry:
    def test_all_algos_contains_all(self):
        assert 'phash' in ALL_ALGOS
        assert 'ssim' in ALL_ALGOS
        assert 'orb' in ALL_ALGOS
        assert 'auto' in ALL_ALGOS
        assert 'akaze' in ALL_ALGOS
        assert 'kaze' in ALL_ALGOS
        assert 'colorhash' in ALL_ALGOS

    def test_tier_separation(self):
        assert len(HASH_ALGOS & PIXEL_ALGOS) == 0
        assert len(PIXEL_ALGOS & DESCRIPTOR_ALGOS) == 0
        assert ALL_ALGOS == HASH_ALGOS | PIXEL_ALGOS | DESCRIPTOR_ALGOS | FUSION_ALGOS


# ---------- SSIM ------------------------------------------------------------

class TestSSIM:
    def test_identical_high(self, identical_pair):
        a, b = identical_pair
        score = calculate_ssim_similarity(a, b)
        assert score >= 0.99

    def test_different_low(self, different_pair):
        a, b = different_pair
        score = calculate_ssim_similarity(a, b)
        assert score < 0.8

    def test_range(self, different_pair):
        a, b = different_pair
        score = calculate_ssim_similarity(a, b)
        assert 0.0 <= score <= 1.0


# ---------- Histogram -------------------------------------------------------

class TestHistogram:
    def test_identical_high(self, identical_pair):
        a, b = identical_pair
        score = calculate_histogram_similarity(a, b)
        assert score >= 0.95

    def test_different_lower(self, different_pair):
        a, b = different_pair
        score = calculate_histogram_similarity(a, b)
        assert score < 0.95

    def test_range(self, different_pair):
        a, b = different_pair
        score = calculate_histogram_similarity(a, b)
        assert 0.0 <= score <= 1.0


# ---------- Template Matching -----------------------------------------------

class TestTemplateMatching:
    def test_identical_high(self, identical_pair):
        a, b = identical_pair
        score = calculate_template_similarity(a, b)
        assert score >= 0.99

    def test_different_lower(self, different_pair):
        a, b = different_pair
        score = calculate_template_similarity(a, b)
        assert score < 0.9

    def test_range(self, different_pair):
        a, b = different_pair
        score = calculate_template_similarity(a, b)
        assert 0.0 <= score <= 1.0


# ---------- New Descriptors (AKAZE / KAZE) ----------------------------------

class TestNewDescriptors:
    @pytest.fixture
    def textured_pair(self, tmp_path):
        """专门为特征描述子测试创建有丰富纹理的图片对。"""
        np.random.seed(123)
        arr_a = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        pa = tmp_path / "tex_a.png"
        PILImage.fromarray(arr_a).save(str(pa))

        # B: 相同内容 + 微小扰动 → 不同文件但描述子相似
        arr_b = arr_a.copy()
        arr_b[0, 0] = [0, 0, 0]
        arr_b[199, 199] = [255, 255, 255]
        pb = tmp_path / "tex_b.png"
        PILImage.fromarray(arr_b).save(str(pb))
        return str(pa), str(pb)

    @pytest.mark.parametrize("algo", ["akaze", "kaze"])
    def test_compute_returns_desc(self, textured_pair, algo):
        a, _ = textured_pair
        desc, norm = compute_descriptor(a, algo)
        assert desc is not None
        assert desc.shape[0] > 0

    @pytest.mark.parametrize("algo", ["akaze", "kaze"])
    def test_similar_descriptors(self, textured_pair, algo):
        a, b = textured_pair
        desc_a, norm = compute_descriptor(a, algo)
        desc_b, _ = compute_descriptor(b, algo)
        sim = calculate_descriptor_similarity(desc_a, desc_b, norm)
        assert sim >= 0.5  # 近似相同图片


# ---------- Colorhash -------------------------------------------------------

class TestColorhash:
    def test_compute_features_has_colorhash(self, identical_pair):
        a, _ = identical_pair
        features = compute_image_features(a)
        assert 'colorhash' in features
        assert len(features['colorhash']) > 0


# ---------- Hybrid / Auto Fusion --------------------------------------------

class TestHybridFusion:
    def test_identical_high(self, identical_pair):
        a, b = identical_pair
        fa = compute_image_features(a)
        fb = compute_image_features(b)
        score = calculate_hybrid_similarity(a, b, fa, fb)
        # ORB 对相同文件返回 0（file_hash 相同被 BFMatcher 跳过），所以融合分会偏低
        assert score >= 0.5

    def test_different_lower(self, different_pair):
        a, b = different_pair
        fa = compute_image_features(a)
        fb = compute_image_features(b)
        score = calculate_hybrid_similarity(a, b, fa, fb)
        assert score < 0.9

    def test_custom_weights(self, identical_pair):
        a, b = identical_pair
        fa = compute_image_features(a)
        fb = compute_image_features(b)
        score = calculate_hybrid_similarity(a, b, fa, fb, weights={'ssim': 1.0})
        assert score >= 0.99

    def test_range(self, different_pair):
        a, b = different_pair
        fa = compute_image_features(a)
        fb = compute_image_features(b)
        score = calculate_hybrid_similarity(a, b, fa, fb)
        assert 0.0 <= score <= 1.0


# ---------- Rotation / Flip Invariance ------------------------------------

class TestRotationInvariance:
    """验证 compare_with_orientations 增强后旋转/翻转鲁棒性。"""

    @pytest.fixture
    def rotated_pair(self, tmp_path):
        """原图 + 旋转90° 版本。"""
        from app.image_processor import compare_with_orientations
        np.random.seed(42)
        arr = np.random.randint(30, 225, (200, 200, 3), dtype=np.uint8)
        arr[:40, :40] = [220, 30, 30]
        pa = tmp_path / "rot_orig.png"
        PILImage.fromarray(arr).save(str(pa))

        rotated = PILImage.fromarray(arr).rotate(90, expand=True)
        pb = tmp_path / "rot_90.png"
        rotated.save(str(pb))
        return str(pa), str(pb)

    @pytest.fixture
    def flipped_pair(self, tmp_path):
        """原图 + 水平翻转 版本。"""
        np.random.seed(42)
        arr = np.random.randint(30, 225, (200, 200, 3), dtype=np.uint8)
        arr[:40, :40] = [220, 30, 30]
        pa = tmp_path / "flip_orig.png"
        PILImage.fromarray(arr).save(str(pa))

        flipped = PILImage.fromarray(arr).transpose(PILImage.FLIP_LEFT_RIGHT)
        pb = tmp_path / "flip_h.png"
        flipped.save(str(pb))
        return str(pa), str(pb)

    def test_ssim_rotation_enhanced(self, rotated_pair):
        from app.image_processor import compare_with_orientations
        a, b = rotated_pair
        # 原始 SSIM 对旋转失效
        raw = calculate_ssim_similarity(a, b)
        assert raw < 0.5
        # 增强后应接近 1.0
        enhanced = compare_with_orientations(a, b, calculate_ssim_similarity)
        assert enhanced >= 0.90

    def test_ssim_flip_enhanced(self, flipped_pair):
        from app.image_processor import compare_with_orientations
        a, b = flipped_pair
        enhanced = compare_with_orientations(a, b, calculate_ssim_similarity)
        assert enhanced >= 0.90

    def test_orb_flip_enhanced(self, flipped_pair):
        from app.image_processor import compare_with_orientations, compute_descriptor
        a, b = flipped_pair
        def orb_scorer(pa, pb):
            da, na = compute_descriptor(pa, 'orb')
            db, _ = compute_descriptor(pb, 'orb')
            return calculate_descriptor_similarity(da, db, na)
        enhanced = compare_with_orientations(a, b, orb_scorer)
        assert enhanced >= 0.90

    def test_histogram_rotation_invariant(self, rotated_pair):
        """直方图本身对旋转不敏感，增强后保持高分。"""
        from app.image_processor import compare_with_orientations
        a, b = rotated_pair
        enhanced = compare_with_orientations(a, b, calculate_histogram_similarity)
        assert enhanced >= 0.90

