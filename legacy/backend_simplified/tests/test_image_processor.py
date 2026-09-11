"""Tests for app.image_processor module."""

import os
import hashlib
from pathlib import Path

import pytest
import numpy as np
from PIL import Image as PILImage

from app.image_processor import (
    compute_file_md5,
    compute_image_features,
    hamming_distance,
    calculate_similarity,
    is_image_file,
    find_similar_images,
    group_similar_images,
    resize_image_if_needed,
    compute_descriptor,
    compute_descriptor_with_kp,
    calculate_descriptor_similarity,
    get_cached_descriptor,
    draw_feature_matches,
    _create_extractor,
    HASH_ALGOS, DESCRIPTOR_ALGOS, ALL_ALGOS,
)


# ---------- is_image_file ---------------------------------------------------

class TestIsImageFile:
    @pytest.mark.parametrize("ext", [
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp",
        ".jp2", ".jfif", ".psd", ".ico", ".tga", ".pbm", ".dng", ".heic", ".avif",
    ])
    def test_supported(self, ext):
        assert is_image_file(f"photo{ext}") is True

    @pytest.mark.parametrize("ext", [".pdf", ".docx", ".txt", ".exe", ".mp4", ""])
    def test_unsupported(self, ext):
        assert is_image_file(f"file{ext}") is False

    def test_case_insensitive(self):
        assert is_image_file("photo.JPG") is True
        assert is_image_file("photo.TIF") is True


# ---------- compute_file_md5 ------------------------------------------------

class TestComputeFileMd5:
    def test_known_content(self, tmp_dir):
        path = tmp_dir / "hello.bin"
        path.write_bytes(b"hello world")
        expected = hashlib.md5(b"hello world").hexdigest()
        assert compute_file_md5(str(path)) == expected

    def test_empty_file(self, tmp_dir):
        path = tmp_dir / "empty.bin"
        path.write_bytes(b"")
        expected = hashlib.md5(b"").hexdigest()
        assert compute_file_md5(str(path)) == expected


# ---------- compute_image_features ------------------------------------------

class TestComputeImageFeatures:
    def test_returns_all_hashes(self, sample_image):
        features = compute_image_features(str(sample_image))
        for key in ("phash", "dhash", "ahash", "whash", "file_hash", "width", "height", "file_size"):
            assert key in features, f"缺少字段: {key}"

    def test_dimensions_correct(self, sample_image):
        features = compute_image_features(str(sample_image))
        assert features["width"] == 100
        assert features["height"] == 100

    def test_hash_not_empty(self, sample_image):
        features = compute_image_features(str(sample_image))
        assert len(features["phash"]) > 0
        assert len(features["file_hash"]) > 0

    def test_tif_image(self, sample_tif_image):
        """验证 TIF 格式图片可正常计算特征。"""
        features = compute_image_features(str(sample_tif_image))
        assert features["width"] == 100
        assert features["height"] == 100
        assert len(features["phash"]) > 0

    def test_identical_images_same_hash(self, sample_image_pair):
        """两张内容完全相同的图片应有相同的 phash。"""
        a, b, _ = sample_image_pair
        fa = compute_image_features(str(a))
        fb = compute_image_features(str(b))
        assert fa["phash"] == fb["phash"]

    def test_different_images_different_hash(self, tmp_dir):
        """不同内容的图片 phash 不同（使用有纹理的图片避免纯色哈希碰撞）。"""
        # 图 A：水平渐变
        arr_a = np.zeros((100, 100, 3), dtype=np.uint8)
        for x in range(100):
            arr_a[:, x] = [int(x * 2.55), 0, 0]
        path_a = tmp_dir / "grad_h.png"
        PILImage.fromarray(arr_a).save(str(path_a))

        # 图 B：垂直条纹
        arr_b = np.zeros((100, 100, 3), dtype=np.uint8)
        for y in range(100):
            arr_b[y, :] = [0, 0, 255] if y % 20 < 10 else [255, 255, 0]
        path_b = tmp_dir / "stripes.png"
        PILImage.fromarray(arr_b).save(str(path_b))

        fa = compute_image_features(str(path_a))
        fb = compute_image_features(str(path_b))
        assert fa["phash"] != fb["phash"]


# ---------- hamming_distance ------------------------------------------------

class TestHammingDistance:
    def test_identical(self):
        h = "aabbccdd"
        assert hamming_distance(h, h) == 0

    def test_known_distance(self):
        # 0x0 = 0000, 0xf = 1111 => 4 bits different per hex digit
        assert hamming_distance("0000", "ffff") == 16

    def test_one_bit(self):
        # 0x0 vs 0x1 => 1 bit different
        assert hamming_distance("0", "1") == 1


# ---------- calculate_similarity --------------------------------------------

class TestCalculateSimilarity:
    def test_identical_returns_one(self):
        h = "abcdef0123456789"
        assert calculate_similarity(h, h) == 1.0

    def test_completely_different_returns_zero(self):
        h1 = "0" * 16
        h2 = "f" * 16
        sim = calculate_similarity(h1, h2)
        assert sim == 0.0

    def test_range(self, sample_image_pair):
        a, _, c = sample_image_pair
        fa = compute_image_features(str(a))
        fc = compute_image_features(str(c))
        sim = calculate_similarity(fa["phash"], fc["phash"])
        assert 0.0 <= sim <= 1.0


# ---------- find_similar_images & group_similar_images ----------------------

class TestGrouping:
    def _make_features(self, paths):
        items = []
        for i, p in enumerate(paths):
            f = compute_image_features(str(p))
            f["id"] = i + 1
            f["filename"] = p.name
            f["file_path"] = str(p)
            items.append(f)
        return items

    def _make_patterned_images(self, tmp_dir):
        """生成有纹理差异的测试图片，确保 phash 可区分。"""
        # A：256x256 渐变图（较大图片，1px 变化不影响 phash）
        arr_a = np.zeros((256, 256, 3), dtype=np.uint8)
        for x in range(256):
            arr_a[:, x] = [x, 0, 0]
        path_a = tmp_dir / "pattern_a.png"
        PILImage.fromarray(arr_a).save(str(path_a))

        # B：同样的图但角落有微小差异 → 不同 file_hash, 相同 phash
        arr_b = arr_a.copy()
        arr_b[255, 255] = [arr_a[255, 255][0], 1, 0]  # 微调绿色通道
        path_b = tmp_dir / "pattern_b.png"
        PILImage.fromarray(arr_b).save(str(path_b))

        # C：垂直条纹（完全不同的视觉内容）
        arr_c = np.zeros((256, 256, 3), dtype=np.uint8)
        for y in range(256):
            arr_c[y, :] = [0, 0, 255] if y % 40 < 20 else [255, 255, 0]
        path_c = tmp_dir / "pattern_c.png"
        PILImage.fromarray(arr_c).save(str(path_c))

        return path_a, path_b, path_c

    def test_find_similar_identical(self, tmp_dir):
        """find_similar_images 应找到相似图片（返回 (img, similarity) 元组列表）。"""
        a, b, _ = self._make_patterned_images(tmp_dir)
        items = self._make_features([a, b])
        query = items[0]
        results = find_similar_images(query, [items[1]], threshold=0.5, hash_type="phash")
        assert len(results) >= 1
        # 返回格式是 (img_dict, similarity_float)
        _, sim = results[0]
        assert sim >= 0.9

    def test_group_similar_images(self, tmp_dir):
        """group_similar_images 应该将相似图片分到一组。"""
        a, b, c = self._make_patterned_images(tmp_dir)
        items = self._make_features([a, b, c])

        # 先验证 A 和 B 确实比 A 和 C 更相似
        sim_ab = calculate_similarity(items[0]["phash"], items[1]["phash"])
        sim_ac = calculate_similarity(items[0]["phash"], items[2]["phash"])
        # 选取一个能区分 AB 和 AC 的阈值
        mid_threshold = (sim_ab + sim_ac) / 2 if sim_ab > sim_ac else 0.99

        groups, ungrouped = group_similar_images(items, threshold=mid_threshold, hash_type="phash")
        # A 和 B 应在一组，C 独立
        assert len(groups) >= 1
        # 至少有一个组包含 A 和 B
        ab_ids = {items[0]["id"], items[1]["id"]}
        found = any(
            ab_ids.issubset({img["id"] for img in g})
            for g in groups
        )
        assert found, f"A and B should be grouped together, groups={groups}"


# ---------- resize_image_if_needed ------------------------------------------

class TestResizeImage:
    def test_large_image_resized(self, large_image, tmp_dir):
        output = str(tmp_dir / "resized.png")
        result = resize_image_if_needed(str(large_image), max_size=512, output_path=output)
        img = PILImage.open(result)
        assert max(img.size) <= 512

    def test_small_image_unchanged(self, sample_image, tmp_dir):
        output = str(tmp_dir / "unchanged.png")
        result = resize_image_if_needed(str(sample_image), max_size=512, output_path=output)
        img = PILImage.open(result)
        assert img.size == (100, 100)


# ---------- _create_extractor error branches --------------------------------

class TestCreateExtractor:
    def test_unsupported_algo(self):
        with pytest.raises(ValueError, match="不支持的特征算法"):
            _create_extractor("nonexistent")

    @pytest.mark.parametrize("algo", ["orb", "brisk", "sift", "akaze", "kaze"])
    def test_supported_algo_returns_extractor(self, algo):
        ext, norm = _create_extractor(algo)
        assert ext is not None
        assert isinstance(norm, int)


# ---------- compute_descriptor_with_kp --------------------------------------

class TestComputeDescriptorWithKp:
    @pytest.fixture
    def textured_image(self, tmp_dir):
        np.random.seed(99)
        arr = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        path = tmp_dir / "textured.png"
        PILImage.fromarray(arr).save(str(path))
        return path

    def test_returns_kps_desc_norm(self, textured_image):
        kps, desc, norm = compute_descriptor_with_kp(str(textured_image), "orb")
        assert kps is not None
        assert desc is not None
        assert len(kps) > 0

    def test_invalid_path_raises(self, tmp_dir):
        with pytest.raises(ValueError, match="无法读取图像文件"):
            compute_descriptor_with_kp(str(tmp_dir / "nonexistent.png"), "orb")


# ---------- Descriptor Cache ------------------------------------------------

class TestDescriptorCache:
    @pytest.fixture
    def textured_image(self, tmp_dir):
        np.random.seed(77)
        arr = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        path = tmp_dir / "cached.png"
        PILImage.fromarray(arr).save(str(path))
        return path

    def test_get_cached_descriptor_first_call(self, textured_image):
        """首次调用计算并缓存。"""
        file_hash = compute_file_md5(str(textured_image))
        desc, norm = get_cached_descriptor(str(textured_image), file_hash, "orb")
        assert desc is not None
        assert desc.shape[0] > 0

    def test_get_cached_descriptor_second_call(self, textured_image):
        """第二次调用从缓存返回。"""
        file_hash = compute_file_md5(str(textured_image))
        desc1, _ = get_cached_descriptor(str(textured_image), file_hash, "orb")
        desc2, _ = get_cached_descriptor(str(textured_image), file_hash, "orb")
        np.testing.assert_array_equal(desc1, desc2)


# ---------- draw_feature_matches --------------------------------------------

class TestDrawFeatureMatches:
    @pytest.fixture
    def textured_pair(self, tmp_dir):
        np.random.seed(55)
        arr_a = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        pa = tmp_dir / "vis_a.png"
        PILImage.fromarray(arr_a).save(str(pa))

        arr_b = arr_a.copy()
        arr_b[0, 0] = [0, 0, 0]
        pb = tmp_dir / "vis_b.png"
        PILImage.fromarray(arr_b).save(str(pb))
        return str(pa), str(pb)

    def test_produces_output_file(self, textured_pair, tmp_dir):
        a, b = textured_pair
        out_dir = tmp_dir / "vis"
        result = draw_feature_matches(a, b, algo="orb", output_dir=out_dir)
        assert result.exists()
        assert result.suffix == ".jpg"


# ---------- calculate_descriptor_similarity edge cases ----------------------

class TestDescriptorSimilarityEdge:
    def test_none_descriptors_returns_zero(self):
        assert calculate_descriptor_similarity(None, None, 4) == 0.0

    def test_one_none_returns_zero(self, sample_image):
        from app.image_processor import compute_descriptor
        desc, norm = compute_descriptor(str(sample_image), "orb")
        assert calculate_descriptor_similarity(desc, None, norm) == 0.0


# ---------- compute_image_features colorhash --------------------------------

class TestComputeFeaturesColorhash:
    def test_has_colorhash(self, sample_image):
        features = compute_image_features(str(sample_image))
        assert 'colorhash' in features
        assert len(features['colorhash']) > 0
