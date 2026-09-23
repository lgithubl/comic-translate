import io
import json
import os
import secrets
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("CT_WEB_DATA_DIR", BASE_DIR.parent / ".ct-web-data")).resolve()
PROJECTS_DIR = DATA_DIR / "projects"
STATIC_DIR = BASE_DIR / "static"

PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Comic Translate Web")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class Box(BaseModel):
    id: str = Field(default_factory=lambda: secrets.token_hex(8))
    x: float
    y: float
    width: float
    height: float
    rotation: float = 0
    text: str = ""
    translation: str = ""


class ProjectState(BaseModel):
    boxes: list[Box] = Field(default_factory=list)


class PageInfo(BaseModel):
    id: str
    filename: str
    width: int
    height: int


def project_dir(project_id: str) -> Path:
    if not project_id or any(ch in project_id for ch in "/\\:."):
        raise HTTPException(400, detail="Invalid project id")
    path = PROJECTS_DIR / project_id
    if not path.exists():
        raise HTTPException(404, detail="Project not found")
    return path


def page_dir(path: Path, page_id: str) -> Path:
    if not page_id or any(ch in page_id for ch in "/\\:."):
        raise HTTPException(400, detail="Invalid page id")
    target = path / "pages" / page_id
    if not target.exists():
        raise HTTPException(404, detail="Page not found")
    return target


def manifest_path(path: Path) -> Path:
    return path / "project.json"


def read_manifest(path: Path) -> dict[str, Any]:
    target = manifest_path(path)
    if not target.exists():
        return {"pages": []}
    return json.loads(target.read_text(encoding="utf-8"))


def write_manifest(path: Path, pages: list[PageInfo]) -> dict[str, Any]:
    payload = {"pages": [page.dict() for page in pages]}
    manifest_path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def state_path(path: Path) -> Path:
    return path / "state.json"


def read_state(path: Path) -> dict[str, Any]:
    target = state_path(path)
    if not target.exists():
        return {"boxes": []}
    return json.loads(target.read_text(encoding="utf-8"))


def write_state(path: Path, state: ProjectState) -> dict[str, Any]:
    payload = state.dict()
    state_path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def image_path(path: Path) -> Path:
    for name in ("source.png", "source.jpg", "source.jpeg", "source.webp"):
        candidate = path / name
        if candidate.exists():
            return candidate
    raise HTTPException(404, detail="Source image not found")


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: float) -> list[str]:
    if not text:
        return []
    lines: list[str] = []
    for paragraph in text.splitlines() or [text]:
        current = ""
        for char in paragraph:
            candidate = current + char
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if current and bbox[2] - bbox[0] > max_width:
                lines.append(current)
                current = char
            else:
                current = candidate
        if current:
            lines.append(current)
    return lines


def render_page(path: Path) -> bytes:
    source = Image.open(image_path(path)).convert("RGB")
    state = ProjectState(**read_state(path))
    output = source.copy()
    draw = ImageDraw.Draw(output)

    for box in state.boxes:
        if box.width <= 1 or box.height <= 1:
            continue
        text = (box.translation or box.text or "").strip()
        if not text:
            continue

        rect = [box.x, box.y, box.x + box.width, box.y + box.height]
        draw.rectangle(rect, fill="white")
        font_size = max(10, min(42, int(box.height / 3)))
        font = load_font(font_size)
        lines = wrap_text(draw, text, font, max(4, box.width - 12))
        line_height = font_size + 4
        total_height = len(lines) * line_height
        y = box.y + max(4, (box.height - total_height) / 2)
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            x = box.x + max(4, (box.width - line_width) / 2)
            draw.text((x, y), line, fill="black", font=font)
            y += line_height

    buf = io.BytesIO()
    output.save(buf, format="PNG")
    return buf.getvalue()


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/projects")
async def create_project(
    images: list[UploadFile] | None = File(default=None),
    image: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    uploads = list(images or [])
    if image is not None:
        uploads.append(image)
    if not uploads:
        raise HTTPException(400, detail="At least one image is required")

    project_id = secrets.token_hex(8)
    path = PROJECTS_DIR / project_id
    path.mkdir(parents=True, exist_ok=False)
    pages_dir = path / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    pages: list[PageInfo] = []
    for index, upload in enumerate(uploads, start=1):
        raw = await upload.read()
        try:
            pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            raise HTTPException(400, detail=f"Invalid image {upload.filename or index}: {exc}") from exc

        page_id = f"{index:04d}-{secrets.token_hex(4)}"
        target = pages_dir / page_id
        target.mkdir(parents=True, exist_ok=False)
        pil_image.save(target / "source.png", format="PNG")
        write_state(target, ProjectState())
        pages.append(
            PageInfo(
                id=page_id,
                filename=upload.filename or f"page-{index:04d}.png",
                width=pil_image.width,
                height=pil_image.height,
            )
        )

    manifest = write_manifest(path, pages)

    return {
        "project_id": project_id,
        "pages": [
            {
                **page,
                "image_url": f"/api/projects/{project_id}/pages/{page['id']}/image",
                "state": read_state(page_dir(path, page["id"])),
            }
            for page in manifest["pages"]
        ],
    }


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str) -> dict[str, Any]:
    path = project_dir(project_id)
    return {
        "project_id": project_id,
        "pages": [
            {
                **page,
                "image_url": f"/api/projects/{project_id}/pages/{page['id']}/image",
                "state": read_state(page_dir(path, page["id"])),
            }
            for page in read_manifest(path)["pages"]
        ],
    }


@app.get("/api/projects/{project_id}/pages/{page_id}/image")
async def get_image(project_id: str, page_id: str) -> Response:
    path = project_dir(project_id)
    return Response(image_path(page_dir(path, page_id)).read_bytes(), media_type="image/png")


@app.put("/api/projects/{project_id}/pages/{page_id}/state")
async def save_project_state(project_id: str, page_id: str, state: ProjectState) -> dict[str, Any]:
    path = project_dir(project_id)
    return {"state": write_state(page_dir(path, page_id), state)}


@app.post("/api/projects/{project_id}/pages/{page_id}/boxes")
async def detect_boxes(project_id: str, page_id: str) -> dict[str, Any]:
    path = project_dir(project_id)
    target = page_dir(path, page_id)
    with Image.open(image_path(target)) as source:
        width, height = source.size
    box = Box(
        x=width * 0.25,
        y=height * 0.25,
        width=width * 0.5,
        height=max(80, height * 0.16),
        text="",
        translation="",
    )
    state = ProjectState(**read_state(target))
    state.boxes.append(box)
    return {"state": write_state(target, state)}


@app.get("/api/projects/{project_id}/pages/{page_id}/render.png")
async def render_png(project_id: str, page_id: str) -> Response:
    return Response(render_page(page_dir(project_dir(project_id), page_id)), media_type="image/png")


@app.get("/api/projects/{project_id}/download.zip")
async def download_zip(project_id: str) -> StreamingResponse:
    path = project_dir(project_id)
    manifest = read_manifest(path)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("project.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        for index, page in enumerate(manifest["pages"], start=1):
            target = page_dir(path, page["id"])
            stem = f"{index:04d}-{Path(page['filename']).stem or page['id']}"
            archive.writestr(f"final/{stem}.png", render_page(target))
            archive.writestr(f"state/{stem}.json", json.dumps(read_state(target), ensure_ascii=False, indent=2))
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=comic-translate-{project_id}.zip"},
    )
