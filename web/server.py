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


def project_dir(project_id: str) -> Path:
    if not project_id or any(ch in project_id for ch in "/\\:."):
        raise HTTPException(400, detail="Invalid project id")
    path = PROJECTS_DIR / project_id
    if not path.exists():
        raise HTTPException(404, detail="Project not found")
    return path


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


def render_project(path: Path) -> bytes:
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
async def create_project(image: UploadFile = File(...)) -> dict[str, Any]:
    project_id = secrets.token_hex(8)
    path = PROJECTS_DIR / project_id
    path.mkdir(parents=True, exist_ok=False)

    raw = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, detail=f"Invalid image: {exc}") from exc

    source_path = path / "source.png"
    pil_image.save(source_path, format="PNG")
    write_state(path, ProjectState())

    return {
        "project_id": project_id,
        "width": pil_image.width,
        "height": pil_image.height,
        "image_url": f"/api/projects/{project_id}/image",
        "state": {"boxes": []},
    }


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str) -> dict[str, Any]:
    path = project_dir(project_id)
    with Image.open(image_path(path)) as source:
        width, height = source.size
    return {
        "project_id": project_id,
        "width": width,
        "height": height,
        "image_url": f"/api/projects/{project_id}/image",
        "state": read_state(path),
    }


@app.get("/api/projects/{project_id}/image")
async def get_image(project_id: str) -> Response:
    path = project_dir(project_id)
    return Response(image_path(path).read_bytes(), media_type="image/png")


@app.put("/api/projects/{project_id}/state")
async def save_project_state(project_id: str, state: ProjectState) -> dict[str, Any]:
    path = project_dir(project_id)
    return {"state": write_state(path, state)}


@app.post("/api/projects/{project_id}/boxes")
async def detect_boxes(project_id: str) -> dict[str, Any]:
    path = project_dir(project_id)
    with Image.open(image_path(path)) as source:
        width, height = source.size
    box = Box(
        x=width * 0.25,
        y=height * 0.25,
        width=width * 0.5,
        height=max(80, height * 0.16),
        text="",
        translation="",
    )
    state = ProjectState(**read_state(path))
    state.boxes.append(box)
    return {"state": write_state(path, state)}


@app.get("/api/projects/{project_id}/render.png")
async def render_png(project_id: str) -> Response:
    return Response(render_project(project_dir(project_id)), media_type="image/png")


@app.get("/api/projects/{project_id}/download.zip")
async def download_zip(project_id: str) -> StreamingResponse:
    rendered = render_project(project_dir(project_id))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("final.png", rendered)
        archive.writestr("state.json", json.dumps(read_state(project_dir(project_id)), ensure_ascii=False, indent=2))
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=comic-translate-{project_id}.zip"},
    )
