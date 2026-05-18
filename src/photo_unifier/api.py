from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Optional
import json
import mimetypes
import threading
import time

from PIL import Image

try:  # optional at import time so core tests can run without FastAPI installed
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
except Exception:  # pragma: no cover - optional dependency fallback
    FastAPI = None  # type: ignore[assignment]
    HTTPException = Query = None  # type: ignore[assignment]
    FileResponse = HTMLResponse = RedirectResponse = Response = None  # type: ignore[assignment]

from .config import DEFAULT_CONFIG_PATH, AppConfig, load_config, save_config
from .derivatives import _build_image_thumbnail
from .metadata import manifest
from .pipeline import run_face_detection, run_full_pipeline


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{escape(title)}</title>
    <style>
      :root {{
        --bg: #f3f4f6;
        --panel: rgba(255,255,255,.88);
        --ink: #111111;
        --muted: #6e6e73;
        --line: rgba(17,17,17,.08);
        --accent: #0a84ff;
        --accent-soft: rgba(10,132,255,.12);
        --shadow: 0 16px 34px rgba(17,17,17,.08);
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        color: var(--ink);
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", sans-serif;
        background:
          radial-gradient(circle at top left, rgba(10,132,255,.08), transparent 22rem),
          linear-gradient(180deg, #fbfbfd 0%, #f2f2f7 100%);
      }}
      a {{ color: inherit; text-decoration: none; }}
      .app {{
        display: grid;
        grid-template-columns: 220px minmax(0, 1fr);
        min-height: 100vh;
      }}
      .sidebar {{
        border-right: 1px solid var(--line);
        background: rgba(255,255,255,.58);
        backdrop-filter: blur(18px);
        padding: 26px 16px;
        position: sticky;
        top: 0;
        height: 100vh;
      }}
      .brand {{
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 24px;
      }}
      .nav-group {{
        display: grid;
        gap: 6px;
        margin-bottom: 22px;
      }}
      .nav-label {{
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .06em;
        color: var(--muted);
        text-transform: uppercase;
        margin: 0 10px 6px;
      }}
      .nav-item {{
        padding: 10px 12px;
        border-radius: 12px;
        font-weight: 520;
        color: #243041;
      }}
      .nav-item:hover {{
        background: var(--accent-soft);
        color: #0b61bb;
      }}
      .main {{
        padding: 28px 30px 42px;
      }}
      .toolbar {{
        display: flex;
        justify-content: space-between;
        gap: 18px;
        align-items: center;
        margin-bottom: 20px;
      }}
      .title h1 {{
        margin: 0;
        font-size: 2rem;
        line-height: 1.1;
      }}
      .title p {{
        margin: 6px 0 0;
        color: var(--muted);
      }}
      .button-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }}
      .hero-card {{
        display: grid;
        grid-template-columns: minmax(0, 1.2fr) minmax(320px, .8fr);
        gap: 18px;
        margin-bottom: 24px;
      }}
      .btn {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border: 0;
        border-radius: 999px;
        padding: 11px 16px;
        background: #111827;
        color: #fff;
        font: inherit;
        cursor: pointer;
        box-shadow: var(--shadow);
      }}
      .btn.secondary {{
        background: rgba(255,255,255,.92);
        color: #0f172a;
        border: 1px solid var(--line);
        box-shadow: none;
      }}
      .flash {{
        margin-bottom: 16px;
        padding: 14px 16px;
        border-radius: 16px;
        background: #eaf3ff;
        color: #0b61bb;
        border: 1px solid rgba(10,132,255,.18);
      }}
      .metrics {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 14px;
        margin-bottom: 20px;
      }}
      .card {{
        background: var(--panel);
        backdrop-filter: blur(18px);
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 18px;
        box-shadow: var(--shadow);
      }}
      .metric-label {{
        color: var(--muted);
        font-size: .88rem;
      }}
      .metric-value {{
        margin-top: 8px;
        font-size: 2rem;
        font-weight: 700;
      }}
      .table {{
        width: 100%;
        border-collapse: collapse;
      }}
      .table th, .table td {{
        text-align: left;
        padding: 12px 10px;
        border-bottom: 1px solid var(--line);
        vertical-align: top;
      }}
      .table th {{
        color: var(--muted);
        font-size: .88rem;
        font-weight: 600;
      }}
      .badge {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: .82rem;
        background: var(--accent-soft);
        color: #0b61bb;
      }}
      .asset-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
        gap: 12px;
      }}
      .asset-card {{
        display: grid;
        gap: 8px;
      }}
      .asset-link {{
        display: block;
      }}
      .asset-thumb {{
        width: 100%;
        aspect-ratio: 1 / 1;
        object-fit: cover;
        border-radius: 20px;
        background: linear-gradient(135deg, #eceef3, #d8dde7);
        border: 1px solid rgba(255,255,255,.65);
      }}
      .asset-frame {{
        position: relative;
      }}
      .asset-frame::after {{
        content: "";
        position: absolute;
        inset: 0;
        border-radius: 18px;
        box-shadow: inset 0 0 0 1px rgba(15,23,42,.06);
        pointer-events: none;
      }}
      .media-badge {{
        position: absolute;
        left: 10px;
        bottom: 10px;
        padding: 5px 9px;
        border-radius: 999px;
        background: rgba(15,23,42,.72);
        color: #fff;
        font-size: .74rem;
        backdrop-filter: blur(10px);
      }}
      .asset-name {{
        font-size: .9rem;
        font-weight: 600;
        line-height: 1.25;
      }}
      .asset-meta {{
        font-size: .82rem;
        color: var(--muted);
      }}
      .searchbar {{
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 18px;
      }}
      input[type="search"], input[type="text"], input[type="number"], textarea {{
        width: 100%;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: rgba(255,255,255,.92);
        padding: 12px 14px;
        font: inherit;
        color: var(--ink);
      }}
      input[type="search"] {{
        flex: 1;
        min-width: 240px;
      }}
      textarea {{
        min-height: 120px;
        resize: vertical;
      }}
      .form-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 16px;
      }}
      .field {{
        display: grid;
        gap: 8px;
      }}
      .field span {{
        color: var(--muted);
        font-size: .9rem;
      }}
      .section {{
        margin-bottom: 24px;
      }}
      .section-header {{
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 14px;
        margin: 0 0 14px;
      }}
      .section-header h2 {{
        margin: 0;
        font-size: 1.1rem;
      }}
      .section-header p {{
        margin: 0;
        color: var(--muted);
        font-size: .9rem;
      }}
      .moment {{
        margin-bottom: 28px;
      }}
      .moment h3 {{
        margin: 0 0 12px;
        font-size: 1.1rem;
      }}
      .summary-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 14px;
        margin-bottom: 24px;
      }}
      .summary-card {{
        background: linear-gradient(180deg, rgba(255,255,255,.94), rgba(255,255,255,.82));
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 22px;
        box-shadow: var(--shadow);
      }}
      .summary-card h2 {{
        margin: 0 0 8px;
        font-size: 1.05rem;
      }}
      .summary-big {{
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -.03em;
      }}
      .summary-copy {{
        color: var(--muted);
        font-size: .95rem;
        line-height: 1.45;
      }}
      .top-rail {{
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: center;
        margin-bottom: 22px;
      }}
      .top-rail h1 {{
        margin: 0;
        font-size: 2.35rem;
        letter-spacing: -.04em;
      }}
      .top-rail p {{
        margin: 6px 0 0;
        color: var(--muted);
      }}
      .help-list {{
        display: grid;
        gap: 10px;
        margin: 14px 0 0;
      }}
      .help-item {{
        padding: 12px 14px;
        border-radius: 16px;
        background: rgba(10,132,255,.06);
        color: #1d1d1f;
      }}
      .empty-thumb {{
        width: 100%;
        aspect-ratio: 1 / 1;
        border-radius: 20px;
        display: grid;
        place-items: center;
        background: linear-gradient(135deg, #eceef3, #d8dde7);
        color: var(--muted);
        font-size: .85rem;
      }}
      .muted {{
        color: var(--muted);
      }}
      .dupe-actions {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 10px 0 14px;
      }}
      .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,.92);
        border: 1px solid var(--line);
        color: var(--muted);
      }}
      .viewer-shell {{
        display: grid;
        gap: 18px;
      }}
      .viewer-card {{
        overflow: hidden;
        padding: 0;
      }}
      .viewer-stage {{
        min-height: 420px;
        display: grid;
        place-items: center;
        background:
          radial-gradient(circle at top left, rgba(10,132,255,.12), transparent 18rem),
          linear-gradient(180deg, #f7f9fc 0%, #e9eef6 100%);
      }}
      .viewer-stage img,
      .viewer-stage video,
      .viewer-stage audio,
      .viewer-stage iframe {{
        width: 100%;
        height: 100%;
        border: 0;
      }}
      .viewer-stage img {{
        object-fit: contain;
        max-height: 78vh;
      }}
      .viewer-stage video {{
        max-height: 78vh;
        background: #000;
      }}
      .viewer-stage audio {{
        width: min(560px, calc(100% - 48px));
        height: auto;
      }}
      .viewer-fallback {{
        padding: 32px;
        text-align: center;
        display: grid;
        gap: 12px;
      }}
      .file-pill {{
        display: inline-flex;
        justify-self: center;
        padding: 8px 14px;
        border-radius: 999px;
        background: rgba(10,132,255,.1);
        color: #0b61bb;
        font-weight: 600;
      }}
      @media (max-width: 960px) {{
        .app {{
          grid-template-columns: 1fr;
        }}
        .sidebar {{
          position: static;
          height: auto;
          border-right: 0;
          border-bottom: 1px solid var(--line);
        }}
        .hero-card {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>
  </head>
  <body>
    <div class="app">
      <aside class="sidebar">
        <div class="brand">Photo Unifier</div>
        <div class="nav-group">
          <div class="nav-label">Library</div>
          <a class="nav-item" href="/">Home</a>
          <a class="nav-item" href="/app/assets">Library</a>
          <a class="nav-item" href="/app/faces">People</a>
          <a class="nav-item" href="/app/duplicates">Duplicates</a>
          <a class="nav-item" href="/app/cleanup">Cleanup</a>
          <a class="nav-item" href="/app/metadata">Metadata</a>
          <a class="nav-item" href="/app/jobs">Imports</a>
        </div>
        <div class="nav-group">
          <div class="nav-label">System</div>
          <a class="nav-item" href="/app/settings">Settings</a>
          <a class="nav-item" href="/docs">API Docs</a>
        </div>
      </aside>
      <main class="main">
        {body}
      </main>
    </div>
  </body>
</html>"""


def _format_moment_label(value: Optional[str]) -> str:
    if not value:
        return "Unknown Date"
    try:
        dt = datetime.strptime(value[:19], "%Y-%m-%dT%H:%M:%S")
        return dt.strftime("%B %Y")
    except Exception:
        return "Unknown Date"


def _asset_thumb_html(asset_id: str, filename: str, managed_path: Optional[str], dt_original: Optional[str], media_type: str, status: str) -> str:
    thumb_src = f"/poster/{asset_id}"
    managed_link = f'<a href="/managed/{asset_id}">open</a>' if managed_path else '<span class="muted">pending</span>'
    return f"""
    <article class="asset-card">
      <a class="asset-link" href="/app/assets/{asset_id}">
        <div class="asset-frame">
          <img class="asset-thumb" src="{thumb_src}" alt="{escape(filename)}" loading="lazy" decoding="async">
          <span class="media-badge">{escape(media_type)}</span>
        </div>
      </a>
      <div class="asset-name"><a href="/app/assets/{asset_id}">{escape(filename)}</a></div>
      <div class="asset-meta">{escape(media_type)} · {escape(status)}</div>
      <div class="asset-meta">{escape(dt_original or "unknown time")} · {managed_link}</div>
    </article>
    """


def _group_assets(items: list[dict]) -> list[tuple[str, list[dict]]]:
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for item in items:
        key = _format_moment_label(item.get("dt_original"))
        groups.setdefault(key, []).append(item)
    return list(groups.items())


def _render_metadata_value(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, (dict, list)):
        return escape(json.dumps(value, indent=2, sort_keys=True))
    return escape(str(value))


def _guess_media_kind(path: Optional[str], media_type: Optional[str]) -> str:
    if media_type == "video":
        return "video"
    if media_type in {"image", "raw"}:
        return "image"
    guessed, _ = mimetypes.guess_type(path or "")
    if guessed:
        if guessed.startswith("image/"):
            return "image"
        if guessed.startswith("video/"):
            return "video"
        if guessed.startswith("audio/"):
            return "audio"
        if guessed in {"application/pdf", "text/plain", "text/html"}:
            return "document"
    return "fallback"


def _viewer_html(asset_id: str, item: dict) -> str:
    managed_path = item.get("managed_path")
    filename = item.get("orig_filename") or "asset"
    media_kind = _guess_media_kind(managed_path, item.get("media_type"))
    inline_url = f"/inline/{asset_id}"
    thumb_url = f"/thumbnail/{asset_id}"
    actions = f"""
    <div class="button-row" style="padding: 18px;">
      <a href="/managed/{asset_id}"><button class="btn" type="button">Open File</button></a>
      <a href="/download/{asset_id}"><button class="btn secondary" type="button">Download</button></a>
    </div>
    """
    if not managed_path:
        return f"""
        <section class="card viewer-card">
          <div class="viewer-fallback">
            <div class="file-pill">Managed file pending</div>
            <div class="muted">This item has been indexed, but the managed copy is not ready yet.</div>
          </div>
        </section>
        """
    if media_kind == "image":
        return f"""
        <section class="card viewer-card">
          <div class="viewer-stage">
            <img src="{inline_url}" alt="{escape(filename)}" loading="lazy" decoding="async">
          </div>
          {actions}
        </section>
        """
    if media_kind == "video":
        return f"""
        <section class="card viewer-card">
          <div class="viewer-stage">
            <video controls preload="metadata" poster="{thumb_url}">
              <source src="{inline_url}">
            </video>
          </div>
          {actions}
        </section>
        """
    if media_kind == "audio":
        return f"""
        <section class="card viewer-card">
          <div class="viewer-stage">
            <audio controls preload="metadata">
              <source src="{inline_url}">
            </audio>
          </div>
          {actions}
        </section>
        """
    if media_kind == "document":
        return f"""
        <section class="card viewer-card">
          <div class="viewer-stage">
            <iframe src="{inline_url}" title="{escape(filename)}"></iframe>
          </div>
          {actions}
        </section>
        """
    ext = Path(filename).suffix.lower() or "file"
    return f"""
    <section class="card viewer-card">
      <div class="viewer-fallback">
        <div class="file-pill">{escape(ext)} preview</div>
        <div>This file type does not have an inline viewer yet, but it is still available in your library.</div>
        <div class="muted">You can open the managed file directly, download it, or use the generated preview if one exists.</div>
        <div style="justify-self:center; width:min(420px, 100%);">
          <img class="asset-thumb" style="aspect-ratio:4/3;" src="/poster/{asset_id}" alt="{escape(filename)}" loading="lazy" decoding="async">
        </div>
      </div>
      {actions}
    </section>
    """


def _safe_thumbnail_path(derivatives_dir: Path, asset_id: str, kind: str) -> Path:
    return derivatives_dir / kind / f"{asset_id}.jpg"


def _resolve_poster_path(
    db_path: Path,
    derivatives_dir: Path,
    managed_library_dir: Path,
    asset_id: str,
) -> tuple[Optional[Path], Optional[dict]]:
    item = manifest.get_asset(db_path, asset_id)
    if not item:
        return None, None
    stored = item.get("thumbnail_path")
    if stored and Path(stored).exists():
        return Path(stored), item
    if item.get("media_type") == "video":
        candidate = _safe_thumbnail_path(derivatives_dir, asset_id, "video_preview")
        if candidate.exists():
            return candidate, item
        return None, item
    candidate = _safe_thumbnail_path(derivatives_dir, asset_id, "primary")
    if candidate.exists():
        return candidate, item
    managed_rel = item.get("managed_path")
    if managed_rel:
        managed_file = managed_library_dir / str(managed_rel)
        if managed_file.exists():
            try:
                width, height = _build_image_thumbnail(managed_file, candidate, 640)
                manifest.record_thumbnail(db_path, asset_id, "primary", str(candidate), "READY", width=width, height=height)
                return candidate, item
            except Exception:
                return managed_file, item
    return None, item


def create_app(config_path: Path | str = DEFAULT_CONFIG_PATH) -> FastAPI:
    if FastAPI is None:
        raise RuntimeError("FastAPI is required to run the web UI. Install dependencies first.")
    config_path = Path(config_path).resolve()
    auto_sync_lock = threading.Lock()
    auto_sync_state = {
        "running": False,
        "last_result": None,
        "last_error": None,
        "last_run_at": None,
    }

    def _current_config() -> tuple[AppConfig, Path, Path, Path]:
        current_config, resolved = load_config(config_path)
        current_config.ensure_workspace_dirs(resolved)
        current_db_path = current_config.db_path(resolved)
        current_managed = current_config.managed_library_dir(resolved)
        manifest.init_db(current_db_path)
        return current_config, resolved, current_db_path, current_managed

    def _run_full_pipeline_background(batch_limit: Optional[int] = None):
        if not auto_sync_lock.acquire(blocking=False):
            return
        auto_sync_state["running"] = True
        auto_sync_state["last_error"] = None
        try:
            result = run_full_pipeline(config_path, batch_limit=batch_limit)
            auto_sync_state["last_result"] = result
            auto_sync_state["last_run_at"] = datetime.utcnow().isoformat(timespec="seconds")
        except Exception as exc:  # pragma: no cover
            auto_sync_state["last_error"] = str(exc)
            auto_sync_state["last_run_at"] = datetime.utcnow().isoformat(timespec="seconds")
        finally:
            auto_sync_state["running"] = False
            auto_sync_lock.release()

    def _start_background(action_name: str):
        current_config, _resolved, _db_path, _managed = _current_config()
        safe_batch_limit = max(25, min(int(current_config.pipeline.batch_size or 500), 1000))
        action_map = {
            "run-now": lambda: _run_full_pipeline_background(batch_limit=safe_batch_limit),
            "detect-faces": lambda: run_face_detection(config_path, limit=safe_batch_limit, force=True),
            "toggle-auto": None,
        }
        if action_name not in action_map:
            raise HTTPException(status_code=404, detail="Unknown action")
        if action_name == "toggle-auto":
            return
        worker = threading.Thread(target=action_map[action_name], daemon=True)
        worker.start()

    def _auto_sync_loop():
        while True:
            try:
                current_config, _resolved, _db_path, _managed = _current_config()
                if current_config.pipeline.auto_sync_enabled and not auto_sync_state["running"]:
                    safe_batch_limit = max(25, min(int(current_config.pipeline.batch_size or 500), 1000))
                    _run_full_pipeline_background(batch_limit=safe_batch_limit)
                interval = max(30, int(current_config.pipeline.auto_sync_interval_seconds))
            except Exception as exc:  # pragma: no cover
                auto_sync_state["last_error"] = str(exc)
                interval = 60
            time.sleep(interval)

    initial_config, _resolved, _db_path, _managed = _current_config()
    if initial_config.pipeline.auto_sync_enabled:
        threading.Thread(target=_auto_sync_loop, daemon=True).start()
    app = FastAPI(title="Photo Unifier", version="0.4.0")

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/status")
    def status():
        _config, _resolved, db_path, _managed = _current_config()
        return manifest.get_overview(db_path)

    @app.get("/jobs")
    def jobs(limit: int = Query(default=20, ge=1, le=200)):
        _config, _resolved, db_path, _managed = _current_config()
        return {"items": manifest.list_jobs(db_path, limit=limit)}

    @app.get("/assets")
    def assets(
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        q: Optional[str] = None,
    ):
        _config, _resolved, db_path, _managed = _current_config()
        return {"items": manifest.list_assets(db_path, limit=limit, offset=offset, query=q)}

    @app.get("/search")
    def search(q: str, limit: int = Query(default=100, ge=1, le=500), offset: int = Query(default=0, ge=0)):
        _config, _resolved, db_path, _managed = _current_config()
        return {"query": q, "items": manifest.list_assets(db_path, limit=limit, offset=offset, query=q)}

    @app.get("/duplicates")
    def duplicates(limit: int = Query(default=50, ge=1, le=200)):
        _config, _resolved, db_path, _managed = _current_config()
        return {"items": manifest.list_duplicate_groups(db_path, limit=limit)}

    @app.get("/duplicates/{group_id}")
    def duplicate_group(group_id: str):
        _config, _resolved, db_path, _managed = _current_config()
        items = manifest.list_duplicate_group_items(db_path, group_id)
        if not items:
            raise HTTPException(status_code=404, detail="Duplicate group not found")
        return {"items": items}

    @app.get("/faces")
    def faces(limit: int = Query(default=50, ge=1, le=200)):
        _config, _resolved, db_path, _managed = _current_config()
        return {"items": manifest.list_faces(db_path, limit=limit)}

    @app.get("/app/faces/assign")
    def assign_face(identity_label: str, face_id: str):
        _config, _resolved, db_path, _managed = _current_config()
        identities = {item["label"]: item["id"] for item in manifest.list_face_identities(db_path, limit=500)}
        identity_id = identities.get(identity_label)
        if not identity_id:
            identity_id = manifest.create_face_identity(db_path, identity_label)
        manifest.assign_face_identity(db_path, face_id, identity_id)
        return RedirectResponse(url="/app/faces?message=Face+labeled", status_code=303)

    @app.get("/app/faces/reject")
    def reject_face(face_id: str):
        _config, _resolved, db_path, _managed = _current_config()
        manifest.reject_face(db_path, face_id)
        return RedirectResponse(url="/app/faces?message=Detection+hidden", status_code=303)

    @app.get("/thumbnail/{asset_id}")
    def thumbnail(asset_id: str):
        current_config, resolved, db_path, managed_library_dir = _current_config()
        path, item = _resolve_poster_path(
            db_path,
            current_config.derivatives_dir(resolved),
            managed_library_dir,
            asset_id,
        )
        if not path or not path.exists():
            raise HTTPException(status_code=404, detail="Thumbnail not found")
        media_type = item.get("media_type") if item else None
        if media_type in {"image", "raw"} and path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            return RedirectResponse(url=f"/inline/{asset_id}", status_code=307)
        return FileResponse(path)

    @app.get("/poster/{asset_id}")
    def poster(asset_id: str):
        current_config, resolved, db_path, managed_library_dir = _current_config()
        path, item = _resolve_poster_path(
            db_path,
            current_config.derivatives_dir(resolved),
            managed_library_dir,
            asset_id,
        )
        if path and path.exists():
            if item and item.get("media_type") in {"image", "raw"} and path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                return RedirectResponse(url=f"/inline/{asset_id}", status_code=307)
            return FileResponse(path)
        if item and item.get("managed_path") and item.get("media_type") in {"image", "raw"}:
            return RedirectResponse(url=f"/inline/{asset_id}", status_code=307)
        svg = """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 640">
          <defs>
            <linearGradient id="g" x1="0" x2="1" y1="0" y2="1">
              <stop offset="0%" stop-color="#eceef3"/>
              <stop offset="100%" stop-color="#d8dde7"/>
            </linearGradient>
          </defs>
          <rect width="640" height="640" rx="52" fill="url(#g)"/>
          <circle cx="220" cy="220" r="54" fill="#ffffff" opacity="0.55"/>
          <path d="M120 470l124-136 83 88 66-58 127 106H120z" fill="#ffffff" opacity="0.72"/>
          <text x="320" y="565" text-anchor="middle" font-family="-apple-system,BlinkMacSystemFont,Helvetica Neue,sans-serif" font-size="28" fill="#6e6e73">Preview unavailable</text>
        </svg>
        """.strip()
        return Response(content=svg, media_type="image/svg+xml")

    @app.get("/face-crop/{face_id}")
    def face_crop(face_id: str):
        _config, _resolved, db_path, managed_library_dir = _current_config()
        face = manifest.get_face(db_path, face_id)
        if not face:
            raise HTTPException(status_code=404, detail="Face not found")
        try:
            bbox = json.loads(face.get("bbox_json") or "{}")
        except Exception:
            bbox = {}
        source_image: Optional[Path] = None
        if face.get("media_type") == "video" and face.get("thumbnail_path"):
            source_image = Path(face["thumbnail_path"])
        elif face.get("managed_path"):
            source_image = managed_library_dir / str(face["managed_path"])
        elif face.get("thumbnail_path"):
            source_image = Path(face["thumbnail_path"])
        if not source_image or not source_image.exists():
            raise HTTPException(status_code=404, detail="Face image missing")
        x = int(bbox.get("x", 0))
        y = int(bbox.get("y", 0))
        w = int(bbox.get("w", 0))
        h = int(bbox.get("h", 0))
        with Image.open(source_image) as img:
            img = img.convert("RGB")
            image_width = int(bbox.get("image_width") or img.width)
            image_height = int(bbox.get("image_height") or img.height)
            scale_x = img.width / max(1, image_width)
            scale_y = img.height / max(1, image_height)
            left = max(0, int((x - w * 0.25) * scale_x))
            top = max(0, int((y - h * 0.35) * scale_y))
            right = min(img.width, int((x + w * 1.25) * scale_x))
            bottom = min(img.height, int((y + h * 1.15) * scale_y))
            crop = img.crop((left, top, right, bottom))
            crop.thumbnail((420, 420))
            payload = BytesIO()
            crop.save(payload, format="JPEG", quality=90)
        return Response(content=payload.getvalue(), media_type="image/jpeg")

    @app.get("/managed/{asset_id}")
    def managed(asset_id: str):
        _config, _resolved, db_path, managed_library_dir = _current_config()
        managed_rel = manifest.managed_path_for_asset(db_path, asset_id)
        if not managed_rel:
            raise HTTPException(status_code=404, detail="Managed asset not found")
        path = managed_library_dir / managed_rel
        if not path.exists():
            raise HTTPException(status_code=404, detail="Managed file missing")
        return FileResponse(path)

    @app.get("/download/{asset_id}")
    def download(asset_id: str):
        _config, _resolved, db_path, managed_library_dir = _current_config()
        managed_rel = manifest.managed_path_for_asset(db_path, asset_id)
        item = manifest.get_asset(db_path, asset_id)
        if not managed_rel or not item:
            raise HTTPException(status_code=404, detail="Managed asset not found")
        path = managed_library_dir / managed_rel
        if not path.exists():
            raise HTTPException(status_code=404, detail="Managed file missing")
        return FileResponse(path, filename=item.get("orig_filename") or path.name)

    @app.get("/inline/{asset_id}")
    def inline(asset_id: str):
        _config, _resolved, db_path, managed_library_dir = _current_config()
        managed_rel = manifest.managed_path_for_asset(db_path, asset_id)
        if not managed_rel:
            raise HTTPException(status_code=404, detail="Managed asset not found")
        path = managed_library_dir / managed_rel
        if not path.exists():
            raise HTTPException(status_code=404, detail="Managed file missing")
        guessed, _ = mimetypes.guess_type(str(path))
        return FileResponse(path, media_type=guessed or "application/octet-stream")

    @app.get("/app/actions/{action_name}")
    def run_action(action_name: str):
        current_config, _resolved, _db_path, _managed = _current_config()
        if action_name == "toggle-auto":
            updated = AppConfig(
                workspace_root=current_config.workspace_root,
                sources=current_config.sources,
                paths=current_config.paths,
                tools=current_config.tools,
                pipeline=current_config.pipeline.__class__(
                    batch_size=current_config.pipeline.batch_size,
                    max_workers=current_config.pipeline.max_workers,
                    image_thumbnail_size=current_config.pipeline.image_thumbnail_size,
                    video_preview_offset_seconds=current_config.pipeline.video_preview_offset_seconds,
                    managed_naming=current_config.pipeline.managed_naming,
                    auto_sync_enabled=not current_config.pipeline.auto_sync_enabled,
                    auto_sync_interval_seconds=current_config.pipeline.auto_sync_interval_seconds,
                ),
                thresholds=current_config.thresholds,
            )
            save_config(updated, config_path)
            state = "enabled" if updated.pipeline.auto_sync_enabled else "disabled"
            return RedirectResponse(url=f"/?message=Auto+Sync+{state}", status_code=303)
        try:
            _start_background(action_name)
        except ValueError as exc:
            return RedirectResponse(url=f"/?message={escape(str(exc)).replace(' ', '+')}", status_code=303)
        message = "Started+safe+batch+library+run"
        if action_name == "detect-faces":
            message = "Started+face+detection+batch+refresh"
        return RedirectResponse(url=f"/?message={message}", status_code=303)

    @app.get("/app/settings/save")
    def save_settings(
        workspace_root: str,
        sources_text: str = "",
        db_path_value: str = "",
        managed_library_dir_value: str = "",
        derivatives_dir_value: str = "",
        logs_dir_value: str = "",
        temp_dir_value: str = "",
        batch_size: int = 500,
        image_thumbnail_size: int = 512,
        video_preview_offset_seconds: int = 1,
        managed_naming: str = "{YYYY}/{YYYY-MM}/{YYYYMMDD}_{hhmmss}_{shortid}",
        exiftool: str = "",
        ffmpeg: str = "",
        auto_sync_interval_seconds: int = 300,
    ):
        current_config, _resolved, _db_path, _managed = _current_config()
        sources = [line.strip() for line in sources_text.splitlines() if line.strip()]
        updated = AppConfig(
            workspace_root=workspace_root.strip() or ".",
            sources=sources,
            paths=current_config.paths.__class__(
                db_path=db_path_value.strip() or current_config.paths.db_path,
                managed_library_dir=managed_library_dir_value.strip() or current_config.paths.managed_library_dir,
                derivatives_dir=derivatives_dir_value.strip() or current_config.paths.derivatives_dir,
                logs_dir=logs_dir_value.strip() or current_config.paths.logs_dir,
                temp_dir=temp_dir_value.strip() or current_config.paths.temp_dir,
            ),
            tools=current_config.tools.__class__(
                exiftool=exiftool.strip() or None,
                ffmpeg=ffmpeg.strip() or None,
                whisper_model=current_config.tools.whisper_model,
                clip_model=current_config.tools.clip_model,
                face_model=current_config.tools.face_model,
            ),
            pipeline=current_config.pipeline.__class__(
                batch_size=batch_size,
                max_workers=current_config.pipeline.max_workers,
                image_thumbnail_size=image_thumbnail_size,
                video_preview_offset_seconds=video_preview_offset_seconds,
                managed_naming=managed_naming.strip() or current_config.pipeline.managed_naming,
                auto_sync_enabled=current_config.pipeline.auto_sync_enabled,
                auto_sync_interval_seconds=max(30, auto_sync_interval_seconds),
            ),
            thresholds=current_config.thresholds,
        )
        save_config(updated, config_path)
        return RedirectResponse(url="/app/settings?message=Settings+saved", status_code=303)

    @app.get("/", response_class=HTMLResponse)
    def dashboard(message: Optional[str] = None):
        current_config, resolved, db_path, _managed = _current_config()
        overview = manifest.get_overview(db_path)
        metadata_overview = manifest.get_metadata_overview(db_path)
        resolved_sources = current_config.resolved_sources(resolved)
        source_list = "".join(f"<li>{escape(str(path))}</li>" for path in resolved_sources) or "<li class='muted'>No sources configured yet.</li>"
        latest_jobs = manifest.list_jobs(db_path, limit=8)
        job_rows = "".join(
            f"<tr><td>{escape(row['job_type'])}</td><td><span class='badge'>{escape(row['status'])}</span></td><td>{escape(row['created_at'])}</td></tr>"
            for row in latest_jobs
        ) or "<tr><td colspan='3' class='muted'>No jobs yet.</td></tr>"
        flash = f"<div class='flash'>{escape(message)}</div>" if message else ""
        auto_label = "On" if current_config.pipeline.auto_sync_enabled else "Off"
        auto_desc = (
            f"Watching {len(resolved_sources)} source folder(s) every {current_config.pipeline.auto_sync_interval_seconds} seconds and processing new items automatically."
            if current_config.pipeline.auto_sync_enabled
            else "Automatic background processing is currently paused."
        )
        running_label = "Running now" if auto_sync_state["running"] else "Idle"
        last_run = auto_sync_state["last_run_at"] or "never"
        total_assets = overview.get("assets_total", 0)
        with_location = metadata_overview.get("assets_with_location", 0)
        with_people = metadata_overview.get("assets_with_people", 0)
        body = f"""
        <div class="top-rail">
          <div>
            <h1>Photos</h1>
            <p>Your archive stays in sync automatically in the background.</p>
          </div>
          <div class="button-row">
            <a href="/app/assets"><button class="btn" type="button">Open Library</button></a>
            <a href="/app/settings"><button class="btn secondary" type="button">Settings</button></a>
          </div>
        </div>
        {flash}
        <section class="summary-grid">
          <section class="summary-card">
            <h2>Everything in one library</h2>
            <div class="summary-big">{total_assets}</div>
            <div class="summary-copy">photos and videos are already in the library database</div>
          </section>
          <section class="summary-card">
            <h2>Location recovered</h2>
            <div class="summary-big">{with_location}</div>
            <div class="summary-copy">items currently have location coordinates attached</div>
          </section>
          <section class="summary-card">
            <h2>People named</h2>
            <div class="summary-big">{with_people}</div>
            <div class="summary-copy">items already have a person label saved</div>
          </section>
        </section>
        <section class="card section">
          <div class="button-row">
            <span class="status-pill">Auto Sync: {escape(auto_label)}</span>
            <span class="status-pill">Run status: {escape(running_label)}</span>
            <span class="status-pill">Last run: {escape(last_run)}</span>
          </div>
          <p class="muted" style="margin-top:14px;">{escape(auto_desc)}</p>
        </section>
        <section class="card section">
          <div class="section-header">
            <h2>Where your originals come from</h2>
            <p>These folders are watched or scanned by the app.</p>
          </div>
          <ul>{source_list}</ul>
        </section>
        <section class="card section">
          <div class="section-header">
            <h2>Recent activity</h2>
            <p>The newest jobs the app has run.</p>
          </div>
          <table class="table">
            <thead><tr><th>Job</th><th>Status</th><th>Created</th></tr></thead>
            <tbody>{job_rows}</tbody>
          </table>
        </section>
        """
        return _page("Photo Unifier Library", body)

    @app.get("/app/settings", response_class=HTMLResponse)
    def settings_page(message: Optional[str] = None):
        current_config, _resolved, _db_path, _managed = _current_config()
        flash = f"<div class='flash'>{escape(message)}</div>" if message else ""
        sources_text = "\n".join(current_config.sources)
        body = f"""
        <div class="toolbar">
          <div class="title">
            <h1>Settings</h1>
            <p>Point the app at your real archive and choose where the managed library lives.</p>
          </div>
        </div>
        {flash}
        <section class="card">
          <form method="get" action="/app/settings/save">
            <div class="form-grid">
              <label class="field">
                <span>Workspace root</span>
                <input type="text" name="workspace_root" value="{escape(current_config.workspace_root)}">
              </label>
              <label class="field">
                <span>Database path</span>
                <input type="text" name="db_path_value" value="{escape(current_config.paths.db_path)}">
              </label>
              <label class="field">
                <span>Managed library dir</span>
                <input type="text" name="managed_library_dir_value" value="{escape(current_config.paths.managed_library_dir)}">
              </label>
              <label class="field">
                <span>Derivatives dir</span>
                <input type="text" name="derivatives_dir_value" value="{escape(current_config.paths.derivatives_dir)}">
              </label>
              <label class="field">
                <span>Logs dir</span>
                <input type="text" name="logs_dir_value" value="{escape(current_config.paths.logs_dir)}">
              </label>
              <label class="field">
                <span>Temp dir</span>
                <input type="text" name="temp_dir_value" value="{escape(current_config.paths.temp_dir)}">
              </label>
              <label class="field">
                <span>Batch size</span>
                <input type="number" name="batch_size" value="{current_config.pipeline.batch_size}">
              </label>
              <label class="field">
                <span>Thumbnail size</span>
                <input type="number" name="image_thumbnail_size" value="{current_config.pipeline.image_thumbnail_size}">
              </label>
              <label class="field">
                <span>Video preview offset</span>
                <input type="number" name="video_preview_offset_seconds" value="{current_config.pipeline.video_preview_offset_seconds}">
              </label>
              <label class="field">
                <span>Auto sync interval (seconds)</span>
                <input type="number" name="auto_sync_interval_seconds" value="{current_config.pipeline.auto_sync_interval_seconds}">
              </label>
              <label class="field">
                <span>Managed naming pattern</span>
                <input type="text" name="managed_naming" value="{escape(current_config.pipeline.managed_naming)}">
              </label>
              <label class="field">
                <span>Exiftool path</span>
                <input type="text" name="exiftool" value="{escape(current_config.tools.exiftool or '')}">
              </label>
              <label class="field">
                <span>ffmpeg path</span>
                <input type="text" name="ffmpeg" value="{escape(current_config.tools.ffmpeg or '')}">
              </label>
            </div>
            <label class="field" style="margin-top: 16px;">
              <span>Sources, one per line</span>
              <textarea name="sources_text">{escape(sources_text)}</textarea>
            </label>
            <div class="button-row" style="margin-top: 14px;">
              <button class="btn" type="submit">Save Settings</button>
            </div>
          </form>
        </section>
        """
        return _page("Settings", body)

    @app.get("/app/jobs", response_class=HTMLResponse)
    def jobs_page():
        _config, _resolved, db_path, _managed = _current_config()
        rows = manifest.list_jobs(db_path, limit=50)
        body_rows = "".join(
            f"<tr><td>{escape(row['job_type'])}</td><td><span class='badge'>{escape(row['status'])}</span></td>"
            f"<td>{escape(row['created_at'])}</td><td>{escape(row.get('finished_at') or '-')}</td></tr>"
            for row in rows
        ) or "<tr><td colspan='4' class='muted'>No jobs yet.</td></tr>"
        body = f"""
        <div class="toolbar"><div class="title"><h1>Imports</h1><p>Everything the app has done so far.</p></div></div>
        <section class="card">
          <table class="table">
            <thead><tr><th>Job</th><th>Status</th><th>Created</th><th>Finished</th></tr></thead>
            <tbody>{body_rows}</tbody>
          </table>
        </section>
        """
        return _page("Imports", body)

    @app.get("/app/metadata", response_class=HTMLResponse)
    def metadata_page(field: Optional[str] = None):
        _config, _resolved, db_path, _managed = _current_config()
        overview = manifest.get_metadata_overview(db_path)
        audit = manifest.get_metadata_audit_overview(db_path)
        field_summaries = audit.get("field_summaries", [])
        selected_field = field or "captured_at"
        if not any(item["field_name"] == selected_field for item in field_summaries):
            selected_field = field_summaries[0]["field_name"] if field_summaries else "captured_at"
        missing_assets = manifest.list_assets_missing_metadata(db_path, selected_field, limit=24)
        cards = "".join(
            f'<section class="card"><div class="metric-label">{escape(key.replace("_", " "))}</div><div class="metric-value">{value}</div></section>'
            for key, value in overview.items()
        )
        audit_cards = f"""
        <section class="card"><div class="metric-label">Average metadata score</div><div class="metric-value">{audit.get("average_metadata_score", 0)}</div></section>
        <section class="card"><div class="metric-label">Assets meeting 90 score</div><div class="metric-value">{audit.get("assets_meeting_90_score", 0)}</div></section>
        """
        field_rows = "".join(
            f"""
            <tr>
              <td><a href="/app/metadata?field={escape(item['field_name'])}">{escape(item['label'])}</a></td>
              <td>{item['weight']}</td>
              <td>{item['present_assets']} / {item['applicable_assets']}</td>
              <td>{item['missing_assets']}</td>
              <td>{item['completeness_pct']}%</td>
            </tr>
            """
            for item in sorted(field_summaries, key=lambda row: (row["completeness_pct"], row["missing_assets"]), reverse=False)
        ) or "<tr><td colspan='5' class='muted'>No metadata audit yet.</td></tr>"
        missing_rows = "".join(
            f"""
            <tr>
              <td><a href="/app/assets/{escape(row['id'])}">{escape(row.get('orig_filename') or row['id'])}</a></td>
              <td>{escape(str(row.get('media_type') or '-'))}</td>
              <td>{escape(str(row.get('dt_original') or '-'))}</td>
              <td>{escape(str(row.get('status') or '-'))}</td>
            </tr>
            """
            for row in missing_assets
        ) or "<tr><td colspan='4' class='muted'>No missing items for this field.</td></tr>"
        body = f"""
        <div class="toolbar">
          <div class="title">
            <h1>Metadata</h1>
            <p>This shows how much information we have recovered so far. It is a progress view, not a guarantee that every item is perfect.</p>
          </div>
        </div>
        <section class="metrics">{cards}</section>
        <section class="metrics">{audit_cards}</section>
        <section class="card">
          <div class="section-header">
            <h2>Canonical metadata standard</h2>
            <p>Each asset is measured against one provider-agnostic metadata standard, so we can track completeness instead of guessing.</p>
          </div>
          <table class="table">
            <thead><tr><th>Field</th><th>Weight</th><th>Present</th><th>Missing</th><th>Coverage</th></tr></thead>
            <tbody>{field_rows}</tbody>
          </table>
        </section>
        <section class="card">
          <div class="section-header">
            <h2>Missing right now: {escape(selected_field.replace('_', ' '))}</h2>
            <p>These are example assets that still need this field.</p>
          </div>
          <table class="table">
            <thead><tr><th>Asset</th><th>Type</th><th>Captured</th><th>Status</th></tr></thead>
            <tbody>{missing_rows}</tbody>
          </table>
        </section>
        <section class="card">
          <div class="section-header">
            <h2>What this means</h2>
            <p>Short version in plain English.</p>
          </div>
          <div class="help-list">
            <div class="help-item"><strong>Strong timestamps</strong> means the time probably came from embedded photo/video metadata, not just the filename or file modified time.</div>
            <div class="help-item"><strong>Assets with location</strong> means the app has actual GPS coordinates saved for that item.</div>
            <div class="help-item"><strong>Assets with people</strong> means a person label has been saved. It does not mean face recognition is fully complete yet.</div>
            <div class="help-item"><strong>Important:</strong> we cannot honestly guarantee that every photo and video already has absolutely all metadata. This page is here to show coverage clearly.</div>
          </div>
        </section>
        """
        return _page("Metadata", body)

    @app.get("/app/faces", response_class=HTMLResponse)
    def faces_page(message: Optional[str] = None):
        _config, _resolved, db_path, _managed = _current_config()
        rows = manifest.list_faces(db_path, limit=200)
        overview = manifest.get_face_overview(db_path)
        identities = manifest.list_face_identities(db_path, limit=100)
        identity_options = "".join(
            f"<option value=\"{escape(item['label'])}\">{escape(item['label'])}</option>"
            for item in identities
        )
        identity_summary = "".join(
            f"<span class='status-pill'>{escape(item['label'])}: {item['face_count']}</span>"
            for item in identities[:12]
            if item.get("face_count")
        ) or "<span class='status-pill'>No named people yet</span>"
        cards = []
        for row in rows:
            thumb = f"/face-crop/{row['id']}"
            bbox = {}
            try:
                bbox = json.loads(row.get("bbox_json") or "{}")
            except Exception:
                bbox = {}
            badge = row.get("identity_label") or "Unlabeled"
            cards.append(
                f"""
                <section class="card">
                  <div class="button-row" style="justify-content:space-between; align-items:center;">
                    <strong>{escape(row.get('orig_filename') or row['asset_id'])}</strong>
                    <span class="badge">{escape(badge)}</span>
                  </div>
                  <div style="margin-top:12px;">
                    <img class="asset-thumb" style="aspect-ratio:1/1; object-fit:cover;" src="{thumb}" alt="{escape(row.get('orig_filename') or 'face asset')}" loading="lazy" decoding="async">
                  </div>
                  <div class="asset-meta" style="margin-top:10px;">{escape(row.get('media_type') or 'unknown')} • {escape(json.dumps(bbox, sort_keys=True))}</div>
                  <form class="searchbar" method="get" action="/app/faces/assign" style="margin-top:10px;">
                    <input type="hidden" name="face_id" value="{escape(row['id'])}">
                    <input type="text" name="identity_label" list="identity-labels" placeholder="Name this person">
                    <button class="btn secondary" type="submit">Save Label</button>
                  </form>
                  <div class="button-row" style="margin-top:10px;">
                    <a href="/app/assets/{escape(row['asset_id'])}"><button class="btn secondary" type="button">Open Photo</button></a>
                    <a href="/app/faces/reject?face_id={escape(row['id'])}"><button class="btn secondary" type="button">Hide Detection</button></a>
                  </div>
                </section>
                """
            )
        flash = f"<div class='flash'>{escape(message)}</div>" if message else ""
        summary_cards = "".join(
            f'<section class="card"><div class="metric-label">{escape(key.replace("_", " "))}</div><div class="metric-value">{value}</div></section>'
            for key, value in overview.items()
        )
        body = f"""
        <div class="toolbar">
          <div class="title">
            <h1>People</h1>
            <p>Local face detection with cropped previews, manual naming, and false-positive cleanup.</p>
          </div>
          <div class="button-row">
            <a href="/app/actions/detect-faces"><button class="btn secondary" type="button">Refresh Face Detection</button></a>
          </div>
        </div>
        {flash}
        <datalist id="identity-labels">{identity_options}</datalist>
        <section class="metrics">{summary_cards}</section>
        <section class="card section" style="margin-bottom:20px;">
          <div class="section-header">
            <h2>Named People</h2>
            <p>Once you name someone, the label is reused as provider-agnostic metadata.</p>
          </div>
          <div class="button-row">{identity_summary}</div>
        </section>
        <section class="asset-grid" style="grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));">
          {''.join(cards) or "<section class='card muted'>No faces detected yet. Run face detection first.</section>"}
        </section>
        """
        return _page("People", body)

    @app.get("/app/assets", response_class=HTMLResponse)
    def assets_page(q: Optional[str] = None, show_hidden: int = 0, page: int = 1, per_page: int = 120):
        _config, _resolved, db_path, _managed = _current_config()
        per_page = max(24, min(int(per_page or 120), 300))
        page = max(1, int(page or 1))
        offset = (page - 1) * per_page
        items = manifest.list_assets(
            db_path,
            limit=per_page + 1,
            offset=offset,
            query=q,
            include_hidden=bool(show_hidden),
        )
        has_next = len(items) > per_page
        items = items[:per_page]
        moments = _group_assets(items)
        grids = []
        for label, group_items in moments:
            grids.append(
                f"""
                <section class="moment">
                  <h3>{escape(label)}</h3>
                  <div class="asset-grid">
                    {''.join(_asset_thumb_html(
                        asset_id=item['id'],
                        filename=item['orig_filename'],
                        managed_path=item.get('managed_path'),
                        dt_original=item.get('dt_original'),
                        media_type=item.get('media_type') or 'unknown',
                        status=(item.get('duplicate_decision') or item.get('status') or 'unknown'),
                    ) for item in group_items)}
                  </div>
                </section>
                """
            )
        hidden_toggle = 0 if show_hidden else 1
        hidden_label = "Hide Hidden Duplicates" if show_hidden else "Show Hidden Duplicates"
        query_suffix = f"&q={escape(q)}" if q else ""
        prev_page = max(1, page - 1)
        next_page = page + 1
        paging = """
        <div class="button-row" style="margin-bottom: 16px;">
          {prev_link}
          {next_link}
        </div>
        """.format(
            prev_link=(
                f'<a href="/app/assets?page={prev_page}&per_page={per_page}{query_suffix}&show_hidden={show_hidden}">'
                f'<button class="btn secondary" type="button">Previous</button></a>'
                if page > 1
                else ""
            ),
            next_link=(
                f'<a href="/app/assets?page={next_page}&per_page={per_page}{query_suffix}&show_hidden={show_hidden}">'
                f'<button class="btn secondary" type="button">Next</button></a>'
                if has_next
                else ""
            ),
        )
        body = f"""
        <div class="toolbar">
          <div class="title">
            <h1>Library</h1>
            <p>Your archive in a gallery-first view, grouped by month like a photo library.</p>
          </div>
        </div>
        <form class="searchbar" method="get" action="/app/assets">
          <input type="search" name="q" placeholder="Search filenames, titles, descriptions" value="{escape(q or '')}">
          <input type="hidden" name="show_hidden" value="{hidden_toggle}">
          <input type="hidden" name="per_page" value="{per_page}">
          <button class="btn secondary" type="submit">Search</button>
          <a href="/app/assets?show_hidden={hidden_toggle}{query_suffix}"><button class="btn secondary" type="button">{hidden_label}</button></a>
        </form>
        {paging}
        {''.join(grids) or "<section class='card muted'>No assets found.</section>"}
        {paging}
        """
        return _page("Library", body)

    @app.get("/app/assets/{asset_id}", response_class=HTMLResponse)
    def asset_detail_page(asset_id: str):
        _config, _resolved, db_path, _managed = _current_config()
        item = manifest.get_asset(db_path, asset_id)
        if not item:
            raise HTTPException(status_code=404, detail="Asset not found")
        metadata_rows = manifest.list_asset_metadata(db_path, asset_id)
        artifacts = manifest.list_asset_artifacts(db_path, asset_id)
        viewer = _viewer_html(asset_id, item)
        fields = [
            ("Filename", item.get("orig_filename")),
            ("Media type", item.get("media_type")),
            ("Source", item.get("source")),
            ("Source kind", item.get("source_kind")),
            ("Captured", item.get("dt_original")),
            ("Status", item.get("status")),
            ("Managed status", item.get("managed_status")),
            ("Managed path", item.get("managed_path")),
            ("SHA256", item.get("sha256")),
            ("Description", item.get("description")),
            ("Error / warning", item.get("error_msg")),
        ]
        rows = "".join(
            f"<tr><th>{escape(label)}</th><td>{escape(str(value or '-'))}</td></tr>"
            for label, value in fields
        )
        canonical_rows = "".join(
            f"<tr><th>{escape(row['field_name'])}</th><td><pre style='margin:0;white-space:pre-wrap;font:inherit;'>{_render_metadata_value(row.get('value'))}</pre></td><td>{escape(str(row.get('source_field') or row.get('source_name') or '-'))}</td><td>{escape(str(row.get('confidence') or '-'))}</td></tr>"
            for row in metadata_rows
            if row.get("is_canonical")
        )
        provenance_rows = "".join(
            f"<tr><th>{escape(row['field_name'])}</th><td><pre style='margin:0;white-space:pre-wrap;font:inherit;'>{_render_metadata_value(row.get('value'))}</pre></td><td>{escape(str(row.get('source_name') or '-'))}</td><td>{escape(str(row.get('source_field') or '-'))}</td><td>{escape(str(row.get('confidence') or '-'))}</td></tr>"
            for row in metadata_rows
        )
        artifact_rows = "".join(
            f"<tr><th>{escape(artifact['artifact_type'])}</th><td>{escape(artifact['status'])}</td><td>{escape(artifact['path'])}</td></tr>"
            for artifact in artifacts
        )
        body = f"""
        <div class="toolbar">
          <div class="title">
            <h1>{escape(item.get('orig_filename') or 'Asset')}</h1>
            <p>A smoother library-style viewer with normalized metadata and portable provenance.</p>
          </div>
        </div>
        <section class="section" style="display:grid;grid-template-columns:minmax(0,1.35fr) minmax(320px,.9fr);gap:18px;">
          {viewer}
          <section class="card">
            <div class="section-header">
              <h2>File Details</h2>
              <p>Core information for this asset.</p>
            </div>
            <table class="table"><tbody>{rows}</tbody></table>
          </section>
        </section>
        <section class="card section">
          <div class="section-header">
            <h2>Normalized Metadata</h2>
            <p>Provider-agnostic fields the app will carry forward.</p>
          </div>
          <table class="table">
            <thead><tr><th>Field</th><th>Value</th><th>Chosen From</th><th>Confidence</th></tr></thead>
            <tbody>{canonical_rows or "<tr><td colspan='4' class='muted'>No normalized metadata yet.</td></tr>"}</tbody>
          </table>
        </section>
        <section class="card section">
          <div class="section-header">
            <h2>Metadata Provenance</h2>
            <p>Where each normalized field came from.</p>
          </div>
          <table class="table">
            <thead><tr><th>Field</th><th>Value</th><th>Source</th><th>Detail</th><th>Confidence</th></tr></thead>
            <tbody>{provenance_rows or "<tr><td colspan='5' class='muted'>No provenance records yet.</td></tr>"}</tbody>
          </table>
        </section>
        <section class="card section">
          <div class="section-header">
            <h2>Derived Files</h2>
            <p>Sidecars and other generated artifacts attached to this asset.</p>
          </div>
          <table class="table">
            <thead><tr><th>Type</th><th>Status</th><th>Path</th></tr></thead>
            <tbody>{artifact_rows or "<tr><td colspan='3' class='muted'>No derived files yet.</td></tr>"}</tbody>
          </table>
        </section>
        """
        return _page("Asset Detail", body)

    @app.get("/app/duplicates/{group_id}/resolve")
    def resolve_duplicate_group(group_id: str, canonical_asset_id: str):
        _config, _resolved, db_path, _managed = _current_config()
        items = manifest.list_duplicate_group_items(db_path, group_id)
        if not items:
            raise HTTPException(status_code=404, detail="Duplicate group not found")
        try:
            manifest.resolve_duplicate_group(db_path, group_id, canonical_asset_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return RedirectResponse(url="/app/duplicates?message=Duplicate+group+resolved", status_code=303)

    @app.get("/app/duplicates/{group_id}/keep-all")
    def keep_all_duplicate_group(group_id: str):
        _config, _resolved, db_path, _managed = _current_config()
        try:
            manifest.keep_all_duplicate_group(db_path, group_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return RedirectResponse(url="/app/duplicates?message=Marked+duplicate+group+as+keep+all", status_code=303)

    @app.get("/app/duplicates/{group_id}/skip")
    def skip_duplicate_group(group_id: str):
        _config, _resolved, db_path, _managed = _current_config()
        try:
            manifest.skip_duplicate_group(db_path, group_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return RedirectResponse(url="/app/duplicates?message=Skipped+duplicate+group", status_code=303)

    @app.get("/app/duplicates", response_class=HTMLResponse)
    def duplicates_page(message: Optional[str] = None):
        _config, _resolved, db_path, _managed = _current_config()
        groups = manifest.list_duplicate_groups(db_path, limit=100)
        blocks = []
        for group in groups:
            items = manifest.list_duplicate_group_items(db_path, group["id"])
            previews = "".join(
                _asset_thumb_html(
                    asset_id=item["asset_id"],
                    filename=item["orig_filename"],
                    managed_path=item.get("managed_path"),
                    dt_original=item.get("dt_original"),
                    media_type=item.get("media_type") or "unknown",
                    status=item.get("keep_decision") or "candidate",
                )
                for item in items
            )
            choices = "".join(
                f"""
                <div class="card" style="padding:12px;">
                  <div><strong>{escape(item['orig_filename'])}</strong></div>
                  <div class="asset-meta">{escape(item.get('dt_original') or 'unknown time')} · {escape(str(item.get('orig_size') or 0))} bytes</div>
                  <div class="button-row" style="margin-top:10px;">
                    <a href="/app/duplicates/{escape(group['id'])}/resolve?canonical_asset_id={escape(item['asset_id'])}"><button class="btn secondary" type="button">Choose As Canonical</button></a>
                  </div>
                </div>
                """
                for item in items
            )
            blocks.append(
                f"""
                <section class="card section">
                  <div class="section-header">
                    <h2>{escape(group.get('group_type') or 'Duplicate Group')}</h2>
                    <p>Status: {escape(group.get('status') or 'OPEN')} · Items: {group.get('item_count', 0)}</p>
                  </div>
                  <div class="dupe-actions">
                    <a href="/app/duplicates/{escape(group['id'])}/keep-all"><button class="btn secondary" type="button">Keep Both / All</button></a>
                    <a href="/app/duplicates/{escape(group['id'])}/skip"><button class="btn secondary" type="button">Skip Group</button></a>
                  </div>
                  <div class="asset-grid" style="margin-bottom:14px;">{previews}</div>
                  <div class="form-grid">{choices}</div>
                </section>
                """
            )
        flash = f"<div class='flash'>{escape(message)}</div>" if message else ""
        body = f"""
        <div class="toolbar">
          <div class="title">
            <h1>Duplicates</h1>
            <p>Review exact duplicate candidates and decide what the main library should show.</p>
          </div>
        </div>
        {flash}
        {''.join(blocks) or "<section class='card muted'>No duplicate groups yet. Run a manual library pass or turn on Auto Sync first.</section>"}
        """
        return _page("Duplicates", body)

    @app.get("/app/cleanup", response_class=HTMLResponse)
    def cleanup_page():
        _config, _resolved, db_path, _managed = _current_config()
        groups = manifest.list_duplicate_groups(db_path, limit=200)
        exact_groups = [group for group in groups if group.get("group_type") == "EXACT_SHA256"]
        near_groups = [group for group in groups if group.get("group_type") == "NEAR_AHASH"]
        screenshots = manifest.list_assets_with_flag(db_path, "is_screenshot", limit=100)
        blurry = manifest.list_assets_with_flag(db_path, "is_blurry", limit=100)
        exact_rows = "".join(
            f"<tr><td><a href='/app/duplicates'>{escape(group.get('canonical_filename') or group['id'])}</a></td><td>{group.get('item_count', 0)}</td><td>{escape(group.get('status') or 'OPEN')}</td></tr>"
            for group in exact_groups
        ) or "<tr><td colspan='3' class='muted'>No exact duplicate groups yet.</td></tr>"
        near_rows = "".join(
            f"<tr><td><a href='/app/duplicates'>{escape(group.get('canonical_filename') or group['id'])}</a></td><td>{group.get('item_count', 0)}</td><td>{escape(group.get('status') or 'OPEN')}</td></tr>"
            for group in near_groups
        ) or "<tr><td colspan='3' class='muted'>No near-duplicate groups yet.</td></tr>"
        screenshot_rows = "".join(
            f"<tr><td><a href='/app/assets/{escape(row['id'])}'>{escape(row.get('orig_filename') or row['id'])}</a></td><td>{escape(str(row.get('media_type') or '-'))}</td><td>{escape(str(row.get('dt_original') or '-'))}</td></tr>"
            for row in screenshots
        ) or "<tr><td colspan='3' class='muted'>No screenshot flags yet.</td></tr>"
        blurry_rows = "".join(
            f"<tr><td><a href='/app/assets/{escape(row['id'])}'>{escape(row.get('orig_filename') or row['id'])}</a></td><td>{escape(str(row.get('media_type') or '-'))}</td><td>{escape(str(row.get('dt_original') or '-'))}</td></tr>"
            for row in blurry
        ) or "<tr><td colspan='3' class='muted'>No blurry flags yet.</td></tr>"
        body = f"""
        <div class="toolbar">
          <div class="title">
            <h1>Cleanup</h1>
            <p>Fast review lanes for the main kinds of rubbish we can detect automatically.</p>
          </div>
        </div>
        <section class="card section">
          <div class="section-header">
            <h2>Exact duplicates</h2>
            <p>Files with identical managed SHA-256 hashes.</p>
          </div>
          <table class="table">
            <thead><tr><th>Group</th><th>Items</th><th>Status</th></tr></thead>
            <tbody>{exact_rows}</tbody>
          </table>
        </section>
        <section class="card section">
          <div class="section-header">
            <h2>Near duplicates</h2>
            <p>Likely duplicates detected with perceptual hashing.</p>
          </div>
          <table class="table">
            <thead><tr><th>Group</th><th>Items</th><th>Status</th></tr></thead>
            <tbody>{near_rows}</tbody>
          </table>
        </section>
        <section class="card section">
          <div class="section-header">
            <h2>Screenshots</h2>
            <p>Items flagged as likely screenshots.</p>
          </div>
          <table class="table">
            <thead><tr><th>Asset</th><th>Type</th><th>Captured</th></tr></thead>
            <tbody>{screenshot_rows}</tbody>
          </table>
        </section>
        <section class="card section">
          <div class="section-header">
            <h2>Blurry items</h2>
            <p>Items flagged by the image sharpness heuristic.</p>
          </div>
          <table class="table">
            <thead><tr><th>Asset</th><th>Type</th><th>Captured</th></tr></thead>
            <tbody>{blurry_rows}</tbody>
          </table>
        </section>
        """
        return _page("Cleanup", body)

    return app
