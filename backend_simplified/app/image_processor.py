import hashlib
import logging
import os
import threading
import uuid
from collections import OrderedDict
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import numpy as np
from PIL import Image as PILImage
import imagehash

logger = logging.getLogger(__name__)

# ---------- 自动注册 HEIF/AVIF 插件 ----------
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass  # pillow-heif 未安装时静默跳过

try:
    import pillow_avif  # noqa: F401  注册 AVIF 解码器
except ImportError:
    pass

# RAW 格式支持
try:
    import rawpy as _rawpy
except ImportError:
    _rawpy = None

RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf"}

try:
    import cv2
except Exception:
    cv2 = None  # 延迟到使用特征匹配类算法时再报错

try:
    from skimage.metrics import structural_similarity as _ssim_fn
except Exception:
    _ssim_fn = None


# ---------- 所有支持的比对算法 ----------
# Tier 1: 感知哈希（毫秒级）
HASH_ALGOS = {"phash", "dhash", "ahash", "whash", "colorhash"}
# Tier 2: 像素/结构级（百毫秒级）
PIXEL_ALGOS = {"ssim", "histogram", "template"}
# Tier 3: 特征描述子（秒级）
DESCRIPTOR_ALGOS = {"orb", "brisk", "sift", "akaze", "kaze"}
# 融合模式
FUSION_ALGOS = {"auto"}
ALL_ALGOS = HASH_ALGOS | PIXEL_ALGOS | DESCRIPTOR_ALGOS | FUSION_ALGOS

# ---------- 支持的图像格式（全品种） ----------
SUPPORTED_IMAGE_EXTENSIONS = {
    # 常用格式
    ".jpg",
    ".jpeg",
    ".jfif",
    ".jpe",
    ".png",
    ".apng",
    ".gif",
    ".bmp",
    ".dib",
    ".tif",
    ".tiff",
    ".webp",
    # JPEG 2000 系列
    ".jp2",
    ".j2k",
    ".j2c",
    ".jpf",
    ".jpx",
    ".jpc",
    # 专业/设计格式
    ".psd",  # Adobe Photoshop
    ".eps",
    ".ps",  # Encapsulated PostScript
    ".svg",  # SVG (矢量，仅记录支持，实际需 cairosvg 转换)
    # 图标格式
    ".ico",
    ".cur",
    ".icns",
    # Targa 系列
    ".tga",
    ".vda",
    ".icb",
    ".vst",
    # 科学图像
    ".fits",
    ".fit",
    ".fts",
    # 传统格式
    ".pcx",
    ".dcx",
    ".dds",  # DirectDraw Surface
    ".qoi",  # Quite Ok Image
    ".mpo",  # Multi-Picture Object
    # NetPBM 系列
    ".pbm",
    ".pgm",
    ".ppm",
    ".pnm",
    # SGI 系列
    ".sgi",
    ".rgb",
    ".rgba",
    ".bw",
    # Sun Raster
    ".ras",
    # 动画帧
    ".flc",
    ".fli",
    # X Window
    ".xbm",
    ".xpm",
    # RAW 相机格式（需要 rawpy/libraw，记录支持但降级处理）
    ".cr2",
    ".cr3",
    ".nef",
    ".arw",
    ".dng",
    ".orf",
    ".rw2",
    ".raf",
    # HEIF/AVIF（需要 pillow-heif/pillow-avif 插件）
    ".heic",
    ".heif",
    ".avif",
}


def compute_file_md5(file_path: str) -> str:
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def compute_image_features(image_path: str) -> Dict[str, Any]:
    """
    计算图像的多种哈希值和基本信息

    Args:
        image_path: 图像文件路径

    Returns:
        包含各种哈希值和图像信息的字典
    """
    features = {}

    # 文件哈希
    features["file_hash"] = compute_file_md5(image_path)
    features["file_size"] = os.path.getsize(image_path)

    _, ext = os.path.splitext(image_path.lower())

    try:
        # RAW 格式使用 rawpy 解码
        if ext in RAW_EXTENSIONS and _rawpy is not None:
            with _rawpy.imread(image_path) as raw:
                rgb = raw.postprocess()
            img = PILImage.fromarray(rgb)
        else:
            img = PILImage.open(image_path)

        with img:
            # 获取图像尺寸
            features["width"], features["height"] = img.size

            # 转换为RGB模式（如果需要）
            if img.mode != "RGB":
                img = img.convert("RGB")

            # 计算各种感知哈希
            features["phash"] = str(imagehash.phash(img, hash_size=8))
            features["dhash"] = str(imagehash.dhash(img, hash_size=8))
            features["ahash"] = str(imagehash.average_hash(img, hash_size=8))
            features["whash"] = str(imagehash.whash(img, hash_size=8))
            features["colorhash"] = str(imagehash.colorhash(img))

    except Exception as e:
        raise ValueError(f"无法处理图像文件 {image_path}: {str(e)}")

    return features


def hamming_distance(hash1: str, hash2: str) -> int:
    """
    计算两个哈希值的汉明距离

    Args:
        hash1, hash2: 十六进制哈希字符串

    Returns:
        汉明距离（不同位的数量）
    """
    # 将十六进制转换为二进制
    bin1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
    bin2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)

    # 计算不同位的数量
    return sum(c1 != c2 for c1, c2 in zip(bin1, bin2))


def calculate_similarity(hash1: str, hash2: str, max_bits: int = 64) -> float:
    """
    计算两个哈希值的相似度

    Args:
        hash1, hash2: 十六进制哈希字符串
        max_bits: 哈希的最大位数（默认64位）

    Returns:
        相似度分数 (0-1，1表示完全相同)
    """
    distance = hamming_distance(hash1, hash2)
    similarity = (max_bits - distance) / max_bits
    return max(0.0, min(1.0, similarity))


def _ensure_cv2():
    if cv2 is None:
        raise RuntimeError(
            "未安装 OpenCV（cv2），无法使用特征描述子算法。请安装 opencv-contrib-python-headless。"
        )


def _create_extractor(algo: str, max_features: int = 512):
    """
    创建特征提取器，返回 (extractor, norm_type)
    支持: orb / brisk / sift / akaze / kaze
    """
    _ensure_cv2()
    algo = algo.lower()
    if algo == "orb":
        return cv2.ORB_create(
            nfeatures=max_features, scaleFactor=1.2, nlevels=8
        ), cv2.NORM_HAMMING
    elif algo == "brisk":
        return cv2.BRISK_create(), cv2.NORM_HAMMING
    elif algo == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("当前 OpenCV 未包含 SIFT。")
        return cv2.SIFT_create(nfeatures=max_features), cv2.NORM_L2
    elif algo == "akaze":
        return cv2.AKAZE_create(), cv2.NORM_HAMMING
    elif algo == "kaze":
        return cv2.KAZE_create(), cv2.NORM_L2
    elif algo == "surf":
        if not hasattr(cv2, "xfeatures2d") or not hasattr(
            cv2.xfeatures2d, "SURF_create"
        ):
            raise RuntimeError("当前 OpenCV 未包含 SURF。")
        return cv2.xfeatures2d.SURF_create(), cv2.NORM_L2
    else:
        raise ValueError(f"不支持的特征算法: {algo}")


def compute_descriptor(image_path: str, algo: str, max_features: int = 512):
    """
    计算局部特征描述子，支持 orb/sift/brisk/akaze/kaze
    返回 (descriptor, norm_type)
    """
    _ensure_cv2()
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像文件: {image_path}")

    extractor, norm = _create_extractor(algo, max_features)
    _, desc = extractor.detectAndCompute(img, None)
    return desc, norm


def compute_descriptor_with_kp(image_path: str, algo: str, max_features: int = 512):
    """
    返回 (keypoints, descriptor, norm)
    """
    _ensure_cv2()
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像文件: {image_path}")

    extractor, norm = _create_extractor(algo, max_features)
    kps, desc = extractor.detectAndCompute(img, None)
    return kps, desc, norm


# 内存缓存：键 (algo, file_hash) -> (descriptor, norm)
_DESC_CACHE: Dict[Tuple[str, str], Tuple[Any, int]] = {}
_DESC_LOCK = threading.Lock()


def _disk_path(algo: str, file_hash: str) -> Path:
    from app.runtime import get_desc_dir

    return get_desc_dir() / algo / f"{file_hash}.npz"


def _load_descriptor_from_disk(algo: str, file_hash: str) -> Optional[Tuple[Any, int]]:
    path = _disk_path(algo, file_hash)
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        desc = data["desc"]
        norm = int(data["norm"])
        return desc, norm
    except Exception as exc:
        logger.debug("Descriptor disk load failed for %s/%s: %s", algo, file_hash, exc)
        return None


def _save_descriptor_to_disk(algo: str, file_hash: str, desc, norm: int):
    path = _disk_path(algo, file_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.savez_compressed(path, desc=desc, norm=norm)
    except Exception as exc:
        logger.warning(
            "Descriptor disk save failed for %s/%s: %s", algo, file_hash, exc
        )


def get_cached_descriptor(
    image_path: str, file_hash: str, algo: str, max_features: int = 512
):
    key = (algo, file_hash)
    with _DESC_LOCK:
        if key in _DESC_CACHE:
            return _DESC_CACHE[key]

    # 尝试磁盘缓存
    disk = _load_descriptor_from_disk(algo, file_hash)
    if disk:
        with _DESC_LOCK:
            _DESC_CACHE[key] = disk
        return disk

    # 计算并写入缓存
    desc, norm = compute_descriptor(image_path, algo, max_features=max_features)
    with _DESC_LOCK:
        _DESC_CACHE[key] = (desc, norm)
    _save_descriptor_to_disk(algo, file_hash, desc, norm)
    return desc, norm


def calculate_descriptor_similarity(desc1, desc2, norm: int, top_k: int = 64) -> float:
    """
    通用局部特征相似度（BFMatcher）
    返回 0-1，越大越相似。
    """
    _ensure_cv2()
    if desc1 is None or desc2 is None:
        return 0.0
    matcher = cv2.BFMatcher(norm, crossCheck=True)
    matches = matcher.match(desc1, desc2)
    if not matches:
        return 0.0
    matches = sorted(matches, key=lambda m: m.distance)[:top_k]
    avg_dist = sum(m.distance for m in matches) / len(matches)
    # 距离转相似度：对 Hamming 用 256 归一，对 L2 用 512 近似上界
    if norm == cv2.NORM_HAMMING:
        similarity = 1.0 - min(max(avg_dist, 0.0), 256.0) / 256.0
    else:
        similarity = 1.0 - min(max(avg_dist, 0.0), 512.0) / 512.0
    return max(0.0, min(1.0, similarity))


def draw_feature_matches(
    image_path_a: str,
    image_path_b: str,
    algo: str = "orb",
    output_dir: Optional[Path] = None,
    max_features: int = 512,
    max_matches: int = 40,
) -> Path:
    _ensure_cv2()
    if output_dir is None:
        from app.runtime import get_static_dir

        output_dir = get_static_dir() / "visualizations"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kps1, desc1, norm = compute_descriptor_with_kp(
        image_path_a, algo, max_features=max_features
    )
    kps2, desc2, _ = compute_descriptor_with_kp(
        image_path_b, algo, max_features=max_features
    )

    if desc1 is None or desc2 is None:
        raise ValueError("未能提取到足够的特征点")

    matcher = cv2.BFMatcher(norm, crossCheck=True)
    matches = matcher.match(desc1, desc2)
    matches = sorted(matches, key=lambda m: m.distance)[:max_matches]

    img1 = cv2.imread(str(image_path_a))
    img2 = cv2.imread(str(image_path_b))
    match_vis = cv2.drawMatches(
        img1,
        kps1,
        img2,
        kps2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    fname = f"match_{algo}_{uuid.uuid4().hex[:8]}.jpg"
    out_path = output_dir / fname
    cv2.imwrite(str(out_path), match_vis, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return out_path


def find_similar_images(
    query_image: Dict[str, Any],
    image_list: List[Dict[str, Any]],
    threshold: float = 0.85,
    hash_type: str = "phash",
) -> List[Tuple[Dict[str, Any], float]]:
    """
    查找与查询图像相似的图像

    Args:
        query_image: 查询图像的特征字典
        image_list: 待比对的图像列表
        threshold: 相似度阈值（0-1）
        hash_type: 使用的哈希类型（'phash', 'dhash', 'ahash', 'whash'）

    Returns:
        包含相似图像和相似度分数的列表，按相似度降序排列
    """
    query_hash = query_image.get(hash_type)
    if not query_hash:
        raise ValueError(f"查询图像缺少 {hash_type} 哈希值")

    similar_images = []

    for img in image_list:
        img_hash = img.get(hash_type)
        if not img_hash:
            continue

        # 跳过与自身比较
        if img["file_hash"] == query_image["file_hash"]:
            continue

        similarity = calculate_similarity(query_hash, img_hash)

        if similarity >= threshold:
            similar_images.append((img, similarity))

    # 按相似度降序排列
    similar_images.sort(key=lambda x: x[1], reverse=True)

    return similar_images


def group_similar_images(
    images: List[Dict[str, Any]], threshold: float = 0.85, hash_type: str = "phash"
) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    将相似的图像分组

    Args:
        images: 图像列表
        threshold: 相似度阈值
        hash_type: 使用的哈希类型

    Returns:
        (相似组列表, 未分组图像列表)
    """
    if not images:
        return [], []

    # 复制列表，避免修改原数据
    remaining_images = images.copy()
    groups = []
    processed_hashes = set()

    for i, query_img in enumerate(images):
        # 跳过已处理的图像
        if query_img["file_hash"] in processed_hashes:
            continue

        # 查找相似图像
        similar = find_similar_images(query_img, remaining_images, threshold, hash_type)

        if similar:
            # 创建新组，包含查询图像和所有相似图像
            group = [query_img] + [img for img, _ in similar]
            groups.append(group)

            # 标记已处理的图像
            processed_hashes.add(query_img["file_hash"])
            for img, _ in similar:
                processed_hashes.add(img["file_hash"])

    # 找出未分组的图像
    ungrouped = [img for img in images if img["file_hash"] not in processed_hashes]

    return groups, ungrouped


def is_image_file(file_path: str) -> bool:
    """检查文件是否为支持的图像格式（全品种）"""
    _, ext = os.path.splitext(file_path.lower())
    return ext in SUPPORTED_IMAGE_EXTENSIONS


def resize_image_if_needed(
    image_path: str, max_size: int = 1024, output_path: str = None
) -> str:
    """
    如果图像尺寸过大，调整图像大小

    Args:
        image_path: 输入图像路径
        max_size: 最大尺寸（宽度或高度）
        output_path: 输出路径（如果为None，则覆盖原文件）

    Returns:
        处理后的图像路径
    """
    with PILImage.open(image_path) as img:
        width, height = img.size

        # 如果图像尺寸在限制内，直接返回
        if max(width, height) <= max_size:
            return image_path

        # 计算新尺寸
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)

        # 调整大小
        resized_img = img.resize((new_width, new_height), PILImage.Resampling.LANCZOS)

        # 保存图像
        output = output_path or image_path
        resized_img.save(output, quality=85, optimize=True)

        return output


# ============================================================================
#  Tier 2: 像素/结构级比对 — 含 per-file 特征缓存
# ============================================================================

# Per-file feature caches: keyed by (file_path, max_side) → numpy array
# LRU eviction when > MAX_T2_CACHE entries or _t2_bytes > MAX_T2_BYTES to bound memory
MAX_T2_CACHE = 200
MAX_T2_BYTES = 512 * 1024 * 1024  # 512 MB total memory cap for T2 caches
_GRAY_CACHE: OrderedDict = OrderedDict()
_COLOR_CACHE: OrderedDict = OrderedDict()
_HIST_CACHE: OrderedDict = OrderedDict()
_t2_bytes: int = 0  # cumulative nbytes across all T2 caches
_T2_LOCK = threading.Lock()


def _evict_if_needed(cache: OrderedDict, max_size: int = MAX_T2_CACHE):
    """LRU eviction: pop oldest entries when cache exceeds count or byte limit."""
    global _t2_bytes
    while len(cache) > max_size and cache:
        _, v = cache.popitem(last=False)
        _t2_bytes -= getattr(v, "nbytes", 0)
    while _t2_bytes > MAX_T2_BYTES and cache:
        _, v = cache.popitem(last=False)
        _t2_bytes -= getattr(v, "nbytes", 0)
    if _t2_bytes < 0:
        _t2_bytes = 0


def _load_gray_resized(path: str, max_side: int = 512) -> "np.ndarray":
    """加载灰度图并限制最大尺寸（带 LRU 内存缓存）。"""
    global _t2_bytes
    key = (path, max_side)
    with _T2_LOCK:
        if key in _GRAY_CACHE:
            _GRAY_CACHE.move_to_end(key)
            return _GRAY_CACHE[key]

    _ensure_cv2()
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {path}")
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    with _T2_LOCK:
        _evict_if_needed(_GRAY_CACHE)
        _GRAY_CACHE[key] = img
        _t2_bytes += getattr(img, "nbytes", 0)
    return img


def _load_color_resized(path: str, max_side: int = 512) -> "np.ndarray":
    """加载彩色图并限制最大尺寸（带 LRU 内存缓存）。"""
    global _t2_bytes
    key = (path, max_side)
    with _T2_LOCK:
        if key in _COLOR_CACHE:
            _COLOR_CACHE.move_to_end(key)
            return _COLOR_CACHE[key]

    _ensure_cv2()
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"无法读取图像: {path}")
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    with _T2_LOCK:
        _evict_if_needed(_COLOR_CACHE)
        _COLOR_CACHE[key] = img
        _t2_bytes += getattr(img, "nbytes", 0)
    return img


def _get_cached_histogram(path: str) -> "np.ndarray":
    """计算并缓存 HSV H+S 二维直方图（per-file，避免重复计算）。"""
    global _t2_bytes
    with _T2_LOCK:
        if path in _HIST_CACHE:
            _HIST_CACHE.move_to_end(path)
            return _HIST_CACHE[path]

    _ensure_cv2()
    img = _load_color_resized(path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)

    with _T2_LOCK:
        _evict_if_needed(_HIST_CACHE)
        _HIST_CACHE[path] = hist
        _t2_bytes += getattr(hist, "nbytes", 0)
    return hist


def invalidate_feature_cache(path: str):
    """清除指定文件的所有 Tier 2 特征缓存（用于文件删除/更新时）。"""
    global _t2_bytes
    with _T2_LOCK:
        keys_to_remove = [k for k in _GRAY_CACHE if k[0] == path]
        for k in keys_to_remove:
            _t2_bytes -= getattr(_GRAY_CACHE[k], "nbytes", 0)
            del _GRAY_CACHE[k]
        keys_to_remove = [k for k in _COLOR_CACHE if k[0] == path]
        for k in keys_to_remove:
            _t2_bytes -= getattr(_COLOR_CACHE[k], "nbytes", 0)
            del _COLOR_CACHE[k]
        removed = _HIST_CACHE.pop(path, None)
        if removed is not None:
            _t2_bytes -= getattr(removed, "nbytes", 0)
        if _t2_bytes < 0:
            _t2_bytes = 0


def calculate_ssim_similarity(path_a: str, path_b: str) -> float:
    """
    SSIM 结构相似性指数（与人眼感知高度一致）。
    返回 0-1，1 表示完全相同。
    """
    if _ssim_fn is None:
        raise RuntimeError(
            "未安装 scikit-image，无法使用 SSIM。请: pip install scikit-image"
        )

    img_a = _load_gray_resized(path_a)
    img_b = _load_gray_resized(path_b)

    # 统一尺寸（取较小者）
    h = min(img_a.shape[0], img_b.shape[0])
    w = min(img_a.shape[1], img_b.shape[1])
    img_a = cv2.resize(img_a, (w, h))
    img_b = cv2.resize(img_b, (w, h))

    score = _ssim_fn(img_a, img_b, data_range=255)
    return float(max(0.0, min(1.0, score)))


def calculate_histogram_similarity(path_a: str, path_b: str) -> float:
    """
    颜色直方图相关性（对色彩分布变化敏感）。
    使用 HSV 色彩空间的 H+S 通道，更符合人眼感知。
    返回 0-1。使用 per-file 缓存的直方图特征。
    """
    _ensure_cv2()
    hist_a = _get_cached_histogram(path_a)
    hist_b = _get_cached_histogram(path_b)

    # 相关性 [-1, 1] → 映射到 [0, 1]
    corr = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
    return float(max(0.0, min(1.0, (corr + 1.0) / 2.0)))


def calculate_template_similarity(path_a: str, path_b: str) -> float:
    """
    模板匹配 — 归一化互相关（NCC）。
    适合检测子图嵌入、水印、局部复制。
    返回 0-1。
    """
    _ensure_cv2()
    img_a = _load_gray_resized(path_a, max_side=256)
    img_b = _load_gray_resized(path_b, max_side=256)

    # 统一尺寸
    h = min(img_a.shape[0], img_b.shape[0])
    w = min(img_a.shape[1], img_b.shape[1])
    img_a = cv2.resize(img_a, (w, h))
    img_b = cv2.resize(img_b, (w, h))

    result = cv2.matchTemplate(img_a, img_b, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return float(max(0.0, min(1.0, max_val)))


# ============================================================================
#  融合模式
# ============================================================================


def calculate_hybrid_similarity(
    path_a: str,
    path_b: str,
    features_a: Dict[str, Any],
    features_b: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    多算法加权融合评分（auto 模式）。

    默认权重：pHash(0.3) + SSIM(0.3) + ORB(0.4)
    返回 0-1。
    """
    if weights is None:
        weights = {"phash": 0.3, "ssim": 0.3, "orb": 0.4}

    total_weight = 0.0
    total_score = 0.0

    for algo, w in weights.items():
        try:
            if algo in HASH_ALGOS:
                ha = features_a.get(algo)
                hb = features_b.get(algo)
                if ha and hb:
                    score = calculate_similarity(ha, hb)
                    total_score += score * w
                    total_weight += w

            elif algo == "ssim":
                score = calculate_ssim_similarity(path_a, path_b)
                total_score += score * w
                total_weight += w

            elif algo == "histogram":
                score = calculate_histogram_similarity(path_a, path_b)
                total_score += score * w
                total_weight += w

            elif algo in DESCRIPTOR_ALGOS:
                desc_a, norm_a = compute_descriptor(path_a, algo)
                desc_b, norm_b = compute_descriptor(path_b, algo)
                score = calculate_descriptor_similarity(desc_a, desc_b, norm_a)
                total_score += score * w
                total_weight += w
        except Exception as exc:
            logger.debug(
                "Algorithm %s failed during hybrid comparison, skipping: %s", algo, exc
            )

    if total_weight == 0:
        return 0.0
    return total_score / total_weight


# ============================================================================
#  旋转/翻转不变性增强
# ============================================================================

# 8 种方向变体: 4 旋转 × (原图 + 水平翻转)
_ORIENTATIONS = [
    None,  # 原图
    PILImage.Transpose.ROTATE_90,  # 旋转 90°
    PILImage.Transpose.ROTATE_180,  # 旋转 180°
    PILImage.Transpose.ROTATE_270,  # 旋转 270°
    PILImage.Transpose.FLIP_LEFT_RIGHT,  # 水平翻转
    (PILImage.Transpose.FLIP_LEFT_RIGHT, PILImage.Transpose.ROTATE_90),  # 翻转 + 90°
    (PILImage.Transpose.FLIP_LEFT_RIGHT, PILImage.Transpose.ROTATE_180),  # 翻转 + 180°
    (PILImage.Transpose.FLIP_LEFT_RIGHT, PILImage.Transpose.ROTATE_270),  # 翻转 + 270°
]

import tempfile as _tempfile


def _generate_orientation_variants(image_path: str) -> List[str]:
    """
    生成图像的 8 种方向变体（4 旋转 × 2 翻转态）。
    返回 [原图路径, 变体1路径, ..., 变体7路径]。
    调用方负责清理临时文件。
    """
    paths = [image_path]  # 第一个始终是原图
    img = PILImage.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    for orient in _ORIENTATIONS[1:]:  # 跳过 None（原图）
        variant = img.copy()
        if isinstance(orient, tuple):
            for op in orient:
                variant = variant.transpose(op)
        else:
            variant = variant.transpose(orient)

        fd, tmp_path = _tempfile.mkstemp(suffix=".png")
        os.close(fd)
        variant.save(tmp_path)
        paths.append(tmp_path)

    img.close()
    return paths


def compare_with_orientations(
    path_a: str,
    path_b: str,
    scorer,
    features_a: Optional[Dict[str, Any]] = None,
    features_b_list: Optional[List[Dict[str, Any]]] = None,
) -> float:
    """
    对 image_b 的 8 种方向变体逐一与 image_a 比较，返回最高分。

    Args:
        path_a: 基准图像路径
        path_b: 待比较图像路径
        scorer: 评分函数 scorer(path_a, path_b_variant, feat_a?, feat_b?) -> float
        features_a: 基准图特征（hash 类算法需要）
        features_b_list: 各变体预算特征（hash 类算法需要，长度=8）

    Returns:
        所有方向变体中的最高相似度 (0-1)
    """
    variants = _generate_orientation_variants(path_b)
    best_score = 0.0

    try:
        for i, variant_path in enumerate(variants):
            try:
                if features_a is not None and features_b_list is not None:
                    score = scorer(path_a, variant_path, features_a, features_b_list[i])
                else:
                    score = scorer(path_a, variant_path)
                best_score = max(best_score, score)
                # 如果已达到 0.95+，提前退出（完美匹配不需要继续）
                if best_score >= 0.95:
                    break
            except Exception as exc:
                logger.debug("Variant comparison failed, continuing: %s", exc)
                continue
    finally:
        # 清理临时文件（跳过第一个，那是原图）
        for p in variants[1:]:
            try:
                os.unlink(p)
            except OSError:
                pass

    return best_score


def compute_features_for_variants(image_path: str) -> List[Dict[str, Any]]:
    """
    计算一张图片 8 种方向变体的哈希特征。
    用于 hash 类算法的方向不变比对。
    """
    variants = _generate_orientation_variants(image_path)
    features_list = []

    try:
        for v_path in variants:
            features_list.append(compute_image_features(v_path))
    finally:
        for p in variants[1:]:
            try:
                os.unlink(p)
            except OSError:
                pass

    return features_list


# ============================================================================
#  Pairwise Similarity Cache Layer
# ============================================================================


def _cache_key(file_hash_a: str, file_hash_b: str):
    """Return ordered (min, max) hash pair to canonicalize A↔B."""
    return (min(file_hash_a, file_hash_b), max(file_hash_a, file_hash_b))


def get_or_compute_similarity(
    *,
    path_a: str,
    path_b: str,
    file_hash_a: str,
    file_hash_b: str,
    algorithm: str,
    features_a: Dict[str, Any],
    features_b: Dict[str, Any],
    rotation_invariant: bool = False,
    session=None,
) -> float:
    """
    Central similarity computation with DB cache.

    Cache key: (ordered_hash_a, ordered_hash_b, algorithm, rotation_invariant)
    On cache hit  → returns stored score immediately.
    On cache miss → computes, writes to similarity_cache, returns score.

    Parameters
    ----------
    path_a, path_b       : full paths to image files
    file_hash_a, file_hash_b : MD5 hashes from the Image table
    algorithm            : one of ALL_ALGOS
    features_a, features_b   : dict with phash/dhash/ahash/whash/colorhash
    rotation_invariant   : whether to try 8 orientations
    session              : SQLModel session for cache read/write (optional)
    """
    ha, hb = _cache_key(file_hash_a, file_hash_b)

    # ---------- 1. Cache lookup ----------
    if session is not None:
        try:
            from sqlmodel import select as _sel
            from app.models import SimilarityCache

            stmt = _sel(SimilarityCache).where(
                SimilarityCache.hash_a == ha,
                SimilarityCache.hash_b == hb,
                SimilarityCache.algorithm == algorithm,
                SimilarityCache.rotation_invariant == rotation_invariant,
            )
            cached = session.exec(stmt).first()
            if cached is not None:
                return cached.score
        except Exception as exc:
            logger.debug("Similarity cache read failed: %s", exc)

    # ---------- 2. Compute ----------
    score = _raw_similarity(path_a, path_b, algorithm, features_a, features_b)

    if rotation_invariant:
        try:
            ri_score = calculate_rotation_invariant_similarity(
                path_a, path_b, algorithm, features_a, features_b
            )
            score = max(score, ri_score)
        except Exception as exc:
            logger.debug(
                "Rotation invariant comparison failed for %s vs %s [%s]: %s",
                path_a,
                path_b,
                algorithm,
                exc,
            )

    score = round(score, 4)

    # ---------- 3. Cache write ----------
    if session is not None:
        try:
            from app.models import SimilarityCache

            entry = SimilarityCache(
                hash_a=ha,
                hash_b=hb,
                algorithm=algorithm,
                rotation_invariant=rotation_invariant,
                score=score,
            )
            session.add(entry)
            session.commit()
        except Exception as e:
            logger.warning(
                "Similarity cache write failed for %s↔%s [%s]: %s", ha, hb, algorithm, e
            )
            try:
                session.rollback()
            except Exception as rollback_exc:
                logger.debug("Rollback failed: %s", rollback_exc)

    return score


def _raw_similarity(
    path_a: str,
    path_b: str,
    algorithm: str,
    features_a: Dict[str, Any],
    features_b: Dict[str, Any],
) -> float:
    """Compute similarity score without caching (pure computation)."""
    if algorithm in HASH_ALGOS:
        return calculate_similarity(
            features_a.get(algorithm, ""), features_b.get(algorithm, "")
        )
    elif algorithm in DESCRIPTOR_ALGOS:
        desc_a, norm_a = get_cached_descriptor(
            path_a, features_a.get("file_hash", ""), algorithm
        )
        desc_b, _ = get_cached_descriptor(
            path_b, features_b.get("file_hash", ""), algorithm
        )
        return calculate_descriptor_similarity(desc_a, desc_b, norm_a)
    elif algorithm == "ssim":
        return calculate_ssim_similarity(path_a, path_b)
    elif algorithm == "histogram":
        return calculate_histogram_similarity(path_a, path_b)
    elif algorithm == "template":
        return calculate_template_similarity(path_a, path_b)
    elif algorithm == "auto":
        return calculate_hybrid_similarity(path_a, path_b, features_a, features_b)
    return 0.0


def calculate_rotation_invariant_similarity(
    path_a: str,
    path_b: str,
    algorithm: str,
    features_a: Dict[str, Any],
    features_b: Dict[str, Any],
) -> float:
    """Compare using 8 orientation variants, return the max score."""
    if algorithm in HASH_ALGOS:
        b_features_list = compute_features_for_variants(path_b)
        best = 0.0
        for bf in b_features_list:
            s = calculate_similarity(
                features_a.get(algorithm, ""), bf.get(algorithm, "")
            )
            best = max(best, s)
        return best
    else:

        def scorer(fa, fb):
            return _raw_similarity(path_a, path_b, algorithm, fa, fb)

        return compare_with_orientations(path_a, path_b, scorer, features_a=features_a)


def invalidate_similarity_cache(session, file_hash: str):
    """Remove all cached scores involving a specific image (e.g. on delete)."""
    try:
        from sqlmodel import select as _sel
        from app.models import SimilarityCache

        stmt = _sel(SimilarityCache).where(
            (SimilarityCache.hash_a == file_hash)
            | (SimilarityCache.hash_b == file_hash)
        )
        for entry in session.exec(stmt).all():
            session.delete(entry)
        session.commit()
    except Exception as e:
        logger.warning(
            "Similarity cache invalidation failed for hash %s: %s", file_hash, e
        )
        try:
            session.rollback()
        except Exception as rollback_exc:
            logger.debug("Rollback failed during cache invalidation: %s", rollback_exc)
