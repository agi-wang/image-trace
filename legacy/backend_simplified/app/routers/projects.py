from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from app.models import Image, Project, ProjectCreate, ProjectRead
from app.runtime import get_db, get_static_dir
from app.services.project_service import (
    delete_image_artifacts,
    delete_project_with_artifacts,
    project_image_counts,
    serialize_project,
)

router = APIRouter()


@router.post("/projects", response_model=ProjectRead)
async def create_project(project: ProjectCreate, session: Session = Depends(get_db)):
    db_project = Project.model_validate(project)
    session.add(db_project)
    session.commit()
    session.refresh(db_project)
    return serialize_project(db_project, image_count=0)


@router.get("/projects", response_model=List[ProjectRead])
async def list_projects(
    skip: int = 0,
    limit: int = 100,
    session: Session = Depends(get_db),
):
    projects = session.exec(select(Project).offset(skip).limit(limit)).all()
    counts = project_image_counts(
        session, [project.id for project in projects if project.id]
    )
    return [
        serialize_project(project, counts.get(project.id, 0)) for project in projects
    ]


@router.get("/projects/{project_id}", response_model=ProjectRead)
async def get_project(project_id: int, session: Session = Depends(get_db)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")

    count = project_image_counts(session, [project_id]).get(project_id, 0)
    return serialize_project(project, count)


@router.delete("/projects/{project_id}")
async def delete_project(project_id: int, session: Session = Depends(get_db)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")

    delete_project_with_artifacts(session, project, get_static_dir())
    return {"message": "项目已删除"}


@router.get("/images/{project_id}")
async def list_project_images(
    project_id: int,
    skip: int = 0,
    limit: int = 100,
    session: Session = Depends(get_db),
):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")

    statement = (
        select(Image).where(Image.project_id == project_id).offset(skip).limit(limit)
    )
    return session.exec(statement).all()


@router.delete("/images/{image_id}")
async def delete_image(image_id: int, session: Session = Depends(get_db)):
    image = session.get(Image, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="图像不存在")

    delete_image_artifacts(session, image, get_static_dir())
    session.delete(image)
    session.commit()
    return {"message": "图像已删除"}
