import io
import json
import os
import re
import secrets
import zipfile
import base64
from pathlib import Path
from typing import Any

import httpx
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


class AutoTranslateRequest(BaseModel):
    api_url: str = ""
    api_key: str = ""
    model: str = ""
    target_language: str = "Simplified Chinese"
    source_language: str = "auto"
    extra_context: str = ""


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


def chat_completions_url(api_url: str) -> str:
    base = (api_url or "").strip().rstrip("/")
    if not base:
        raise HTTPException(400, detail="API URL is required")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def extract_json_object(text: str) -> dict[str, Any]:
    value = (text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, flags=re.S)
        if not match:
            raise HTTPException(502, detail=f"Model did not return JSON: {text[:500]}")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise HTTPException(502, detail=f"Could not parse model JSON: {exc}") from exc


def normalize_model_boxes(payload: dict[str, Any], width: int, height: int) -> list[Box]:
    boxes: list[Box] = []
    raw_boxes = payload.get("boxes")
    if not isinstance(raw_boxes, list):
        raise HTTPException(502, detail="Model JSON must contain a boxes array")
    for raw in raw_boxes:
        if not isinstance(raw, dict):
            continue
        try:
            x = float(raw.get("x", 0))
            y = float(raw.get("y", 0))
            box_width = float(raw.get("width", raw.get("w", 0)))
            box_height = float(raw.get("height", raw.get("h", 0)))
        except (TypeError, ValueError):
            continue
        if box_width <= 1 or box_height <= 1:
            continue
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        box_width = max(2, min(box_width, width - x))
        box_height = max(2, min(box_height, height - y))
        boxes.append(
            Box(
                x=x,
                y=y,
                width=box_width,
                height=box_height,
                text=str(raw.get("text", "") or ""),
                translation=str(raw.get("translation", "") or ""),
            )
        )
    return boxes


async def call_chat_completions(
    request: AutoTranslateRequest,
    image_bytes: bytes,
    image_width: int,
    image_height: int,
    *,
    stream: bool = False,
) -> str:
    if not request.model.strip():
        raise HTTPException(400, detail="Model is required")
    image_data = base64.b64encode(image_bytes).decode("ascii")
    system_prompt = (
        "You are a comic translation assistant. Detect every visible speech bubble, caption, sign, "
        "or sound-effect text region that should be translated. OCR the source text and translate it. "
        "Return only valid JSON."
    )
    user_prompt = (
        f"Image size is {image_width}x{image_height}. Source language: {request.source_language or 'auto'}. "
        f"Target language: {request.target_language or 'Simplified Chinese'}.\n"
        "Return this exact JSON shape: "
        '{"boxes":[{"x":0,"y":0,"width":100,"height":40,"text":"source text","translation":"translated text"}]}.\n'
        "Coordinates must be pixel coordinates in the original image. Merge text lines that belong to the same bubble. "
        "Skip decorative art without readable text. Do not include markdown."
    )
    if request.extra_context.strip():
        user_prompt += f"\nExtra translation context: {request.extra_context.strip()}"

    headers = {"Content-Type": "application/json"}
    if request.api_key.strip():
        headers["Authorization"] = f"Bearer {request.api_key.strip()}"
    body: dict[str, Any] = {
        "model": request.model.strip(),
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
                ],
            },
        ],
        "temperature": 0,
        "stream": stream,
    }
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(chat_completions_url(request.api_url), headers=headers, json=body)
    if response.status_code >= 400:
        raise HTTPException(response.status_code, detail=response.text)
    if not stream:
        data = response.json()
        return data["choices"][0]["message"]["content"]

    chunks: list[str] = []
    for line in response.text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        if not data:
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        delta = event.get("choices", [{}])[0].get("delta", {})
        if "content" in delta:
            chunks.append(delta["content"])
    return "".join(chunks)


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


@app.post("/api/projects/{project_id}/pages/{page_id}/auto-translate")
async def auto_translate_page(project_id: str, page_id: str, request: AutoTranslateRequest) -> dict[str, Any]:
    path = project_dir(project_id)
    target = page_dir(path, page_id)
    source_path = image_path(target)
    image_bytes = source_path.read_bytes()
    with Image.open(source_path) as source:
        width, height = source.size

    try:
        content = await call_chat_completions(request, image_bytes, width, height, stream=False)
    except HTTPException as exc:
        if exc.status_code != 400 or "Stream must be set to true" not in str(exc.detail):
            raise
        content = await call_chat_completions(request, image_bytes, width, height, stream=True)

    boxes = normalize_model_boxes(extract_json_object(content), width, height)
    state = ProjectState(boxes=boxes)
    return {"state": write_state(target, state), "raw": content}


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
