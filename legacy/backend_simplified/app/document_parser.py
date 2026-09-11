import logging
import os
import uuid
import zipfile
from typing import List, Dict, Any, Tuple
from pathlib import Path
import io

logger = logging.getLogger(__name__)

# PDF处理
import fitz  # PyMuPDF

# 图像处理
from PIL import Image as PILImage

from .image_processor import compute_image_features, is_image_file


class DocumentParser:
    """文档解析器，支持从PDF、DOCX、PPTX中提取图片"""

    def __init__(
        self, upload_dir: str = "data/uploads", extract_dir: str = "data/extracted"
    ):
        self.upload_dir = Path(upload_dir)
        self.extract_dir = Path(extract_dir)
        # data 目录（用于生成相对路径：uploads/...、extracted/...）
        self.base_dir = self.upload_dir.parent
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.extract_dir.mkdir(parents=True, exist_ok=True)

    def _rel_to_base(self, path: Path) -> str:
        """将路径转换为相对于 data 目录的路径字符串"""
        try:
            return str(path.resolve().relative_to(self.base_dir.resolve()))
        except Exception as exc:
            logger.debug("Path relativization failed for %s: %s", path, exc)
            return str(path)

    def extract_images_from_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        从文档中提取所有图片

        Args:
            file_path: 文档文件路径

        Returns:
            提取的图片信息列表
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            return self._extract_from_pdf(file_path)
        elif ext in [".docx", ".pptx"]:
            return self._extract_from_office(file_path)
        else:
            raise ValueError(f"不支持的文档格式: {ext}")

    def _extract_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """从PDF文件提取图片"""
        extracted_images = []
        pdf_name = pdf_path.stem

        try:
            # 方法1: 使用PyMuPDF提取嵌入图片
            doc = fitz.open(str(pdf_path))
            img_index = 0

            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)

                for img_info in image_list:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)

                    if base_image:
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        # 生成文件名
                        filename = f"{pdf_name}_page{page_num + 1}_img{img_index + 1}.{image_ext}"
                        save_path = self.extract_dir / filename

                        # 保存图片
                        with open(save_path, "wb") as f:
                            f.write(image_bytes)

                        # 如果是PDF格式，转换为JPEG
                        if image_ext.lower() == "pdf":
                            save_path = self._convert_to_jpeg(save_path)

                        # 计算特征
                        features = compute_image_features(str(save_path))
                        features.update(
                            {
                                "filename": filename,
                                "file_path": self._rel_to_base(save_path),
                                "extracted_from": self._rel_to_base(pdf_path),
                                "extraction_method": "embedded",
                                "page_number": page_num + 1,
                                "image_index": img_index,
                            }
                        )

                        extracted_images.append(features)
                        img_index += 1

            # 方法2/3: 如果没有找到图片，或需要更多图片，将页面渲染为图片
            if not extracted_images:
                rendered_images = self._render_pdf_pages(pdf_path, pdf_name)
                extracted_images.extend(rendered_images)

            doc.close()

        except Exception as e:
            # 如果 PyMuPDF 解析失败，尝试退化为页面渲染
            try:
                rendered_images = self._render_pdf_pages(pdf_path, pdf_name)
                extracted_images.extend(rendered_images)
            except Exception as e2:
                raise Exception(f"PDF解析失败: {str(e)}, 备用方法也失败: {str(e2)}")

        return extracted_images

    def _render_pdf_pages(self, pdf_path: Path, pdf_name: str) -> List[Dict[str, Any]]:
        """将PDF页面渲染为图片"""
        rendered_images = []

        try:
            # 使用 PyMuPDF 直接渲染页面（不依赖 poppler/pdf2image）
            doc = fitz.open(str(pdf_path))
            try:
                for page_num in range(len(doc)):
                    page = doc[page_num]

                    # 约等于 144dpi（默认 72dpi * 2）
                    mat = fitz.Matrix(2, 2)
                    pix = page.get_pixmap(matrix=mat, alpha=False)

                    filename = f"{pdf_name}_rendered_page{page_num + 1}.jpg"
                    save_path = self.extract_dir / filename

                    # 转成 JPEG 保存
                    img_bytes = pix.tobytes("png")
                    pil_img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                    pil_img.save(save_path, "JPEG", quality=85, optimize=True)

                    # 计算特征
                    features = compute_image_features(str(save_path))
                    features.update(
                        {
                            "filename": filename,
                            "file_path": self._rel_to_base(save_path),
                            "extracted_from": self._rel_to_base(pdf_path),
                            "extraction_method": "rendered",
                            "page_number": page_num + 1,
                            "image_index": 0,
                        }
                    )

                    rendered_images.append(features)
            finally:
                doc.close()

        except Exception as e:
            print(f"渲染PDF页面失败: {str(e)}")

        return rendered_images

    def _extract_from_office(self, file_path: Path) -> List[Dict[str, Any]]:
        """从Office文档（DOCX/PPTX）提取图片"""
        extracted_images = []
        file_name = file_path.stem
        file_type = file_path.suffix[1:].upper()  # 去掉点号，转为大写

        try:
            # Office文档实际上是ZIP文件
            with zipfile.ZipFile(file_path, "r") as zip_file:
                # 查找媒体文件
                media_files = []
                for file_info in zip_file.filelist:
                    if file_info.filename.startswith(
                        "word/media/"
                    ) or file_info.filename.startswith("ppt/media/"):
                        media_files.append(file_info)

                # 提取媒体文件
                for idx, file_info in enumerate(media_files):
                    # 读取文件数据
                    with zip_file.open(file_info) as source:
                        file_data = source.read()

                    # 获取文件扩展名
                    original_name = os.path.basename(file_info.filename)
                    _, ext = os.path.splitext(original_name)
                    if not ext:
                        # 根据文件头判断类型
                        ext = self._guess_image_extension(file_data)

                    if ext:
                        # 生成新文件名
                        filename = f"{file_name}_media{idx + 1}{ext}"
                        save_path = self.extract_dir / filename

                        # 保存文件
                        with open(save_path, "wb") as f:
                            f.write(file_data)

                        # 如果是PDF格式，转换为JPEG
                        if ext.lower() == ".pdf":
                            save_path = self._convert_to_jpeg(save_path)
                            ext = ".jpg"
                            filename = save_path.name

                        # 计算特征
                        features = compute_image_features(str(save_path))
                        features.update(
                            {
                                "filename": filename,
                                "file_path": self._rel_to_base(save_path),
                                "extracted_from": self._rel_to_base(file_path),
                                "extraction_method": f"{file_type}_media",
                                "page_number": None,
                                "image_index": idx,
                                "media_path": file_info.filename,
                            }
                        )

                        extracted_images.append(features)

        except Exception as e:
            raise Exception(f"从{file_type}文档提取图片失败: {str(e)}")

        return extracted_images

    def _guess_image_extension(self, file_data: bytes) -> str:
        """根据文件头猜测图像格式"""
        if file_data.startswith(b"\xff\xd8\xff"):
            return ".jpg"
        elif file_data.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        elif file_data.startswith(b"GIF87a") or file_data.startswith(b"GIF89a"):
            return ".gif"
        elif file_data.startswith(b"BM"):
            return ".bmp"
        elif file_data.startswith(b"II*\x00") or file_data.startswith(b"MM\x00*"):
            return ".tiff"
        elif file_data.startswith(b"%PDF"):
            return ".pdf"
        else:
            return ".jpg"  # 默认为JPEG

    def _convert_to_jpeg(self, image_path: Path) -> Path:
        """将图片转换为JPEG格式"""
        try:
            with PILImage.open(image_path) as img:
                # 转换为RGB模式（如果需要）
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # 生成新文件名
                new_path = image_path.with_suffix(".jpg")

                # 保存为JPEG
                img.save(new_path, "JPEG", quality=85, optimize=True)

                # 删除原文件
                if new_path != image_path:
                    image_path.unlink()

                return new_path
        except Exception as e:
            print(f"转换图片为JPEG失败: {str(e)}")
            return image_path

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        处理文档文件，返回提取结果

        Args:
            file_path: 文档文件路径

        Returns:
            处理结果字典
        """
        file_path = Path(file_path)

        # 检查文件是否存在
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 检查文件类型
        ext = file_path.suffix.lower()
        if ext not in [".pdf", ".docx", ".pptx"]:
            raise ValueError(f"不支持的文档格式: {ext}")

        # 提取图片
        try:
            extracted_images = self.extract_images_from_document(file_path)

            return {
                "status": "success",
                "document_path": self._rel_to_base(file_path),
                "document_name": file_path.name,
                "document_type": ext[1:].upper(),
                "extracted_count": len(extracted_images),
                "images": extracted_images,
            }

        except Exception as e:
            return {
                "status": "error",
                "document_path": self._rel_to_base(file_path),
                "error": str(e),
                "extracted_count": 0,
                "images": [],
            }
