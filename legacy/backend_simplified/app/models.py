from datetime import datetime
from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, TEXT


class ProjectBase(SQLModel):
    name: str = Field(max_length=255)
    description: Optional[str] = Field(default=None, max_length=1000)


class Project(ProjectBase, table=True):
    __tablename__ = "projects"

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)

    # 关联关系
    images: List["Image"] = Relationship(back_populates="project")


class ProjectCreate(ProjectBase):
    pass


class ProjectRead(ProjectBase):
    id: int
    created_at: datetime
    image_count: Optional[int] = 0


class ImageBase(SQLModel):
    filename: str = Field(max_length=255)
    project_id: int = Field(foreign_key="projects.id")
    file_path: str = Field(max_length=500)
    file_hash: str = Field(index=True, max_length=32)  # MD5 hash
    phash: str = Field(index=True, max_length=64)     # Perceptual hash
    dhash: Optional[str] = Field(default=None, max_length=64)  # Difference hash
    ahash: Optional[str] = Field(default=None, max_length=64)  # Average hash
    whash: Optional[str] = Field(default=None, max_length=64)  # Wavelet hash
    colorhash: Optional[str] = Field(default=None, max_length=64)  # Color hash
    extracted_from: Optional[str] = Field(default=None, max_length=255)  # 来源文档名
    file_size: Optional[int] = None  # 文件大小（字节）
    width: Optional[int] = None      # 图片宽度
    height: Optional[int] = None     # 图片高度
    feature_status: Optional[str] = Field(default="pending", max_length=16)  # pending/computing/ready


class Image(ImageBase, table=True):
    __tablename__ = "images"

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)

    # 关联关系
    project: Project = Relationship(back_populates="images")


class ImageCreate(ImageBase):
    pass


class ImageRead(ImageBase):
    id: int
    created_at: datetime


class SimilarGroup(SQLModel):
    """相似图片组"""
    group_id: int
    similarity_score: float  # 相似度分数 (0-1)
    images: List[ImageRead]


class ComparisonResult(SQLModel):
    """比对结果"""
    project_id: int
    total_images: int
    groups: List[SimilarGroup]
    unique_images: List[ImageRead]  # 未找到相似图的独立图片


class AnalysisRun(SQLModel, table=True):
    """分析运行记录"""
    __tablename__ = "analysis_runs"

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="projects.id")
    hash_type: str = Field(max_length=32)
    threshold: float = Field(default=0.85)
    total_images: int
    groups_count: int
    unique_count: int
    summary: Optional[str] = Field(default=None, sa_column=Column(TEXT))
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)


class SimilarityCache(SQLModel, table=True):
    """
    缓存配对相似度分数，避免重复计算。
    缓存键: (hash_a, hash_b, algorithm, rotation_invariant)
    注意: hash_a/hash_b 始终按字典序存储 (min, max)，确保 A↔B 只存一份。
    """
    __tablename__ = "similarity_cache"

    id: Optional[int] = Field(default=None, primary_key=True)
    hash_a: str = Field(index=True, max_length=32)
    hash_b: str = Field(index=True, max_length=32)
    algorithm: str = Field(index=True, max_length=32)
    rotation_invariant: bool = Field(default=False)
    score: float
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)


class FeatureStore(SQLModel, table=True):
    """
    预计算特征矩阵存储。
    每张图像 × 8变体 × N算法 → 固定维度特征向量(BLOB)。
    比对时直接加载向量做矩阵运算，毫秒级出结果。
    """
    __tablename__ = "feature_store"

    id: Optional[int] = Field(default=None, primary_key=True)
    image_id: int = Field(index=True, foreign_key="images.id")
    variant_idx: int = Field(default=0)        # 0=原图, 1-7=旋转翻转变体
    algorithm: str = Field(max_length=32)       # phash_bits/sift_pooled/histogram_hsv/...
    vector: bytes = Field(sa_column=Column("vector", TEXT))  # numpy→base64 bytes
    dimensions: int                             # 向量维度
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)