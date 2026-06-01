from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from html import escape
from io import BytesIO
from math import log, pi, radians, tan
from pathlib import Path
from typing import Optional
import json
import mimetypes
import threading
import time
from urllib.parse import quote_plus

from PIL import Image

try:  # optional at import time so core tests can run without FastAPI installed
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
except Exception:  # pragma: no cover - optional dependency fallback
    FastAPI = None  # type: ignore[assignment]
    HTTPException = Query = None  # type: ignore[assignment]
    FileResponse = HTMLResponse = RedirectResponse = Response = None  # type: ignore[assignment]

from .config import DEFAULT_CONFIG_PATH, AppConfig, load_config, save_config
from . import intelligence
from .derivatives import _build_image_thumbnail
from .metadata import manifest
from .pipeline import run_face_detection, run_full_pipeline
from .utils.location import reverse_geocode_address
from .tooling import build_tool_stack_report


def _page(title: str, body: str, *, history_html: str = "", body_class: str = "") -> str:
    body_class_attr = f' class="{escape(body_class)}"' if body_class else ""
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{escape(title)}</title>
    <style>
      :root {{
        --bg: #eef2f7;
        --panel: rgba(255,255,255,.56);
        --panel-strong: rgba(255,255,255,.74);
        --ink: #101114;
        --muted: #667085;
        --line: rgba(15,23,42,.08);
        --accent: #0a84ff;
        --accent-soft: rgba(10,132,255,.12);
        --shadow: 0 20px 44px rgba(15,23,42,.10);
        --shadow-soft: 0 12px 24px rgba(15,23,42,.08);
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        color: var(--ink);
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", sans-serif;
        overflow: hidden;
        background:
          radial-gradient(circle at 12% 8%, rgba(10,132,255,.20), transparent 18rem),
          radial-gradient(circle at 88% 14%, rgba(120,120,255,.10), transparent 20rem),
          radial-gradient(circle at 50% 100%, rgba(255,255,255,.82), transparent 28rem),
          linear-gradient(180deg, #fbfcff 0%, #e7edf6 100%);
        background-attachment: fixed;
      }}
      a {{ color: inherit; text-decoration: none; }}
      .app {{
        display: grid;
        grid-template-columns: 240px minmax(0, 1fr);
        position: fixed;
        inset: 0;
        width: 100vw;
        height: 100vh;
        overflow: hidden;
      }}
      .sidebar {{
        border-right: 1px solid rgba(255,255,255,.55);
        background: linear-gradient(180deg, rgba(255,255,255,.38), rgba(255,255,255,.24));
        backdrop-filter: blur(30px) saturate(170%);
        padding: 26px 16px;
        position: relative;
        height: 100%;
        overflow: hidden;
        box-shadow: inset -1px 0 0 rgba(255,255,255,.45);
        display: flex;
        flex-direction: column;
      }}
      .brand {{
        font-size: 1.05rem;
        font-weight: 700;
        margin: 0 10px 24px;
        letter-spacing: -.02em;
      }}
      .nav-group {{
        display: grid;
        gap: 6px;
        margin-bottom: 18px;
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
        border-radius: 18px;
        font-weight: 550;
        color: #243041;
        transition: background .15s ease, transform .15s ease, box-shadow .15s ease, backdrop-filter .15s ease;
      }}
      .nav-item:hover {{
        background: rgba(255,255,255,.62);
        backdrop-filter: blur(18px) saturate(150%);
        color: #0b61bb;
        transform: translateY(-1px);
        box-shadow: var(--shadow-soft);
      }}
      .nav-item.active {{
        background: rgba(255,255,255,.72);
        box-shadow: var(--shadow-soft);
        color: #0a4fa0;
      }}
      .sidebar-foot {{
        margin-top: 18px;
        padding: 0 10px;
        color: var(--muted);
        font-size: .86rem;
      }}
      .sidebar-foot a {{
        color: #0b61bb;
      }}
      .main {{
        padding: 26px 28px 42px;
        min-width: 0;
        height: 100%;
        overflow-y: auto;
        overflow-x: hidden;
        overscroll-behavior: contain;
      }}
      .home-page .main {{
        display: grid;
        grid-template-rows: min-content min-content min-content minmax(0, 1fr);
        gap: 12px;
        overflow: hidden;
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
      .sidebar-history {{
        margin-top: auto;
        padding: 8px 10px 0;
      }}
      .history-list {{
        display: grid;
        gap: 8px;
      }}
      .history-entry {{
        display: grid;
        gap: 3px;
        padding: 10px 12px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,.58);
        background: rgba(255,255,255,.44);
        box-shadow: var(--shadow-soft);
        transition: transform .15s ease, background .15s ease, box-shadow .15s ease;
      }}
      .history-entry:hover {{
        transform: translateY(-1px);
        background: rgba(255,255,255,.66);
        box-shadow: var(--shadow);
      }}
      .history-entry-top {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        font-size: .84rem;
        font-weight: 700;
        color: #17324f;
      }}
      .history-entry-summary {{
        color: var(--muted);
        font-size: .8rem;
        line-height: 1.35;
      }}
      .history-entry-time {{
        color: #0b61bb;
        font-size: .76rem;
        font-weight: 600;
      }}
      .history-empty {{
        padding: 10px 12px;
        border-radius: 16px;
        background: rgba(255,255,255,.30);
        color: var(--muted);
        font-size: .84rem;
        border: 1px dashed rgba(255,255,255,.48);
      }}
      .hero-card {{
        display: grid;
        grid-template-columns: minmax(0, 1.35fr) minmax(300px, .95fr);
        gap: 18px;
        margin-bottom: 20px;
      }}
      .home-page .toolbar {{
        margin-bottom: 14px;
      }}
      .home-page .hero-card {{
        grid-template-columns: minmax(0, 1fr);
        gap: 14px;
        margin-bottom: 0;
      }}
      .hero-panel {{
        background: linear-gradient(180deg, rgba(255,255,255,.68), rgba(255,255,255,.38));
        border: 1px solid rgba(255,255,255,.62);
        border-radius: 34px;
        padding: 24px;
        backdrop-filter: blur(30px) saturate(165%);
        box-shadow: 0 26px 60px rgba(15,23,42,.12);
      }}
      .home-page .hero-panel {{
        padding: 18px;
      }}
      .hero-copy {{
        display: grid;
        gap: 10px;
        align-content: start;
      }}
      .hero-copy h1 {{
        margin: 0;
        font-size: clamp(2.2rem, 4vw, 4rem);
        line-height: .96;
        letter-spacing: -.05em;
      }}
      .home-page .hero-copy h1 {{
        font-size: clamp(1.9rem, 3vw, 2.9rem);
      }}
      .hero-copy p {{
        margin: 0;
        color: var(--muted);
        max-width: 54ch;
        font-size: 1rem;
      }}
      .hero-media {{
        display: grid;
        gap: 12px;
      }}
      .hero-strip {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
      }}
      .home-page .hero-strip {{
        gap: 8px;
      }}
      .btn {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border: 0;
        border-radius: 999px;
        padding: 11px 16px;
        background: linear-gradient(180deg, rgba(17,24,39,.95), rgba(17,24,39,.88));
        color: #fff;
        font: inherit;
        cursor: pointer;
        box-shadow: var(--shadow);
      }}
      .btn.secondary {{
        background: linear-gradient(180deg, rgba(255,255,255,.72), rgba(255,255,255,.52));
        color: #0f172a;
        border: 1px solid rgba(255,255,255,.72);
        box-shadow: 0 12px 26px rgba(15,23,42,.08);
      }}
      .flash {{
        margin-bottom: 16px;
        padding: 14px 16px;
        border-radius: 16px;
        background: #eaf3ff;
        color: #0b61bb;
        border: 1px solid rgba(10,132,255,.18);
      }}
      .sync-banner {{
        margin-bottom: 16px;
        padding: 14px 16px;
        border-radius: 16px;
        background: #fff4d8;
        color: #8a4b00;
        border: 1px solid rgba(196, 118, 0, .18);
      }}
      .metrics {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 14px;
        margin-bottom: 18px;
      }}
      .home-page .metrics {{
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 10px;
        margin-bottom: 0;
      }}
      .home-panels {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
        min-height: 0;
      }}
      .home-panel {{
        min-height: 0;
        height: 100%;
        display: grid;
        gap: 0;
        overflow: hidden;
      }}
      .home-page .home-panel {{
        background: transparent;
        border: 0;
        box-shadow: none;
        padding: 0;
      }}
      .home-page .home-panel .section-header {{
        display: none;
      }}
      .home-page .hero-card {{
        grid-template-columns: minmax(0, 1.12fr) minmax(260px, .88fr);
        gap: 10px;
        margin-bottom: 0;
      }}
      .home-page .hero-copy {{
        min-height: 92px;
        gap: 4px;
      }}
      .home-page .hero-copy p {{
        display: none;
      }}
      .home-page .hero-copy h1 {{
        font-size: clamp(1.5rem, 2.2vw, 2.1rem);
      }}
      .home-page .hero-panel {{
        padding: 10px;
      }}
      .home-page .hero-people-panel {{
        background: transparent;
        border: 0;
        box-shadow: none;
        padding: 0;
        display: grid;
        align-items: center;
      }}
      .featured-people {{
        display: flex;
        align-items: stretch;
        gap: 5px;
        width: 100%;
        height: clamp(58px, 7vw, 78px);
        overflow: hidden;
      }}
      .featured-person {{
        display: block;
        flex: 1 1 0;
        min-width: 0;
        aspect-ratio: 1 / 1;
        overflow: hidden;
        border-radius: 14px;
        background: rgba(255,255,255,.06);
      }}
      .featured-person img {{
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
      }}
      .home-carousel {{
        position: relative;
        width: 100%;
        height: 100%;
        min-height: 0;
        border-radius: 0;
        overflow: hidden;
      }}
      .carousel-stage {{
        position: relative;
        width: 100%;
        height: 100%;
        min-height: 0;
        border-radius: 0;
        overflow: hidden;
        background: transparent;
      }}
      .carousel-slide {{
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        opacity: 0;
        transition: opacity .45s ease;
      }}
      .carousel-slide.is-active {{
        opacity: 1;
      }}
      .carousel-dots {{
        position: absolute;
        left: 0;
        right: 0;
        bottom: 12px;
        display: flex;
        justify-content: center;
        gap: 6px;
        z-index: 2;
      }}
      .carousel-dot {{
        width: 6px;
        height: 6px;
        border: 0;
        border-radius: 999px;
        background: rgba(255,255,255,.46);
        opacity: .9;
      }}
      .carousel-dot.is-active {{
        width: 18px;
        background: rgba(255,255,255,.96);
      }}
      .home-map {{
        width: 100%;
        height: 100%;
        min-height: 0;
        border-radius: 0;
        overflow: hidden;
        background: transparent;
      }}
      .home-map svg {{
        display: block;
        width: 100%;
        height: 100%;
        min-height: 0;
      }}
      .home-page .home-panels {{
        gap: 10px;
        height: 100%;
        min-height: 0;
        align-self: stretch;
      }}
      .home-page .home-panel {{
        padding: 0;
      }}
      .home-page .summary-card {{
        padding: 12px;
      }}
      .home-page .summary-big {{
        font-size: 1.6rem;
      }}
      .home-page .metrics {{
        gap: 8px;
        margin-bottom: 0;
      }}
      .home-page .button-row {{
        gap: 8px;
      }}
      .home-page .btn {{
        padding: 8px 12px;
        font-size: .92rem;
      }}
      .card {{
        background: linear-gradient(180deg, rgba(255,255,255,.60), rgba(255,255,255,.40));
        backdrop-filter: blur(28px) saturate(160%);
        border: 1px solid rgba(255,255,255,.66);
        border-radius: 32px;
        padding: 18px;
        box-shadow: 0 18px 42px rgba(15,23,42,.10);
      }}
      .home-page .summary-card {{
        padding: 16px;
      }}
      .home-page .summary-big {{
        font-size: 2rem;
      }}
      .card.soft {{
        background: rgba(255,255,255,.36);
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
        table-layout: fixed;
      }}
      .table th, .table td {{
        text-align: left;
        padding: 12px 10px;
        border-bottom: 1px solid var(--line);
        vertical-align: top;
        word-break: break-word;
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
      .album-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
        gap: 12px;
      }}
      .review-queue {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 14px;
      }}
      .review-page .toolbar {{
        display: none;
      }}
      .review-page .main {{
        padding-top: 12px;
      }}
      .review-shell {{
        display: grid;
        gap: 12px;
      }}
      .review-actions {{
        display: flex;
        justify-content: flex-end;
        margin-bottom: 2px;
      }}
      .review-actions .btn {{
        padding: 14px 18px;
        font-size: 1rem;
      }}
      .review-group {{
        display: grid;
        gap: 10px;
      }}
      .review-group.featured {{
        gap: 12px;
      }}
      .review-sections {{
        display: grid;
        gap: 26px;
      }}
      .review-section {{
        display: grid;
        gap: 12px;
      }}
      .review-section-head {{
        display: flex;
        align-items: end;
        justify-content: space-between;
        gap: 12px;
      }}
      .review-section-head-actions {{
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
      }}
      .review-section-head h2 {{
        margin: 0;
        font-size: 1.45rem;
        letter-spacing: -.03em;
      }}
      .review-section-head p {{
        margin: 4px 0 0;
        color: var(--muted);
      }}
      .review-group-head {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 0 2px;
      }}
      .review-group-head h3 {{
        margin: 0;
        font-size: 1.02rem;
        letter-spacing: -.02em;
      }}
      .review-group-head .review-group-subtitle {{
        margin-top: 4px;
        color: var(--muted);
        font-size: .86rem;
      }}
      .review-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
        gap: 14px;
      }}
      .review-group.featured .review-grid {{
        grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      }}
      .review-card-tile {{
        display: grid;
        gap: 10px;
        min-width: 0;
      }}
      .review-card-media {{
        position: relative;
        overflow: hidden;
        border-radius: 22px;
        background: rgba(255,255,255,.18);
        box-shadow: 0 8px 18px rgba(15,23,42,.08);
        transition: transform .15s ease, box-shadow .15s ease;
      }}
      .review-card-media:hover {{
        transform: translateY(-1px);
        box-shadow: 0 12px 26px rgba(15,23,42,.12);
      }}
      .review-card-media.primary {{
        box-shadow: 0 0 0 2px rgba(10,132,255,.78), 0 8px 18px rgba(15,23,42,.08);
      }}
      .review-card-media img,
      .review-card-media video {{
        width: 100%;
        height: 100%;
        aspect-ratio: 1 / 1;
        object-fit: cover;
        display: block;
        background: rgba(15,23,42,.04);
      }}
      .review-card-media video {{
        background: #000;
        cursor: pointer;
      }}
      .review-card-actions {{
        position: absolute;
        top: 12px;
        left: 12px;
        right: 12px;
        display: flex;
        justify-content: space-between;
        gap: 8px;
        pointer-events: none;
      }}
      .review-card-action {{
        pointer-events: auto;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,.34);
        background: rgba(15,23,42,.58);
        color: #fff;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 22px rgba(15,23,42,.18);
        font-size: .95rem;
      }}
      .review-card-action.keep {{
        background: rgba(10,132,255,.66);
      }}
      .review-card-action.delete {{
        background: rgba(220,38,38,.68);
      }}
      .review-card-badge {{
        position: absolute;
        top: 12px;
        left: 50%;
        transform: translateX(-50%);
        padding: 5px 10px;
        border-radius: 999px;
        background: rgba(255,255,255,.82);
        color: #0f172a;
        border: 1px solid rgba(255,255,255,.72);
        font-size: .78rem;
        font-weight: 700;
        box-shadow: 0 8px 18px rgba(15,23,42,.08);
      }}
      .review-card-header {{
        display: grid;
        gap: 8px;
        padding: 0 4px 4px;
      }}
      .review-card-title {{
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 10px;
      }}
      .review-card-filename {{
        font-size: .95rem;
        font-weight: 700;
        line-height: 1.3;
        letter-spacing: -.02em;
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }}
      .review-card-meta {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        color: var(--muted);
        font-size: .84rem;
      }}
      .review-card-meta span {{
        padding: 4px 10px;
        border-radius: 999px;
        background: rgba(255,255,255,.50);
        border: 1px solid rgba(255,255,255,.62);
      }}
      .review-card-header .review-card-meta span {{
        background: rgba(15,23,42,.05);
        border-color: rgba(15,23,42,.08);
      }}
      .review-hero {{
        display: grid;
        grid-template-columns: minmax(0, 1.15fr) minmax(300px, .85fr);
        gap: 16px;
        margin-bottom: 18px;
      }}
      .review-hero-panel {{
        display: grid;
        gap: 14px;
        padding: 22px;
      }}
      .review-hero-copy {{
        display: grid;
        gap: 10px;
      }}
      .review-hero-copy h1 {{
        margin: 0;
        font-size: clamp(2.1rem, 3.6vw, 3.3rem);
        letter-spacing: -.05em;
        line-height: .96;
      }}
      .review-hero-copy p {{
        margin: 0;
        max-width: 58ch;
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.5;
      }}
      .review-automation {{
        display: grid;
        gap: 10px;
      }}
      .review-automation h2 {{
        margin: 0;
        font-size: 1.02rem;
      }}
      .review-automation-list {{
        display: grid;
        gap: 8px;
        margin: 0;
        padding: 0;
        list-style: none;
      }}
      .review-automation-list li {{
        display: flex;
        align-items: flex-start;
        gap: 10px;
        color: #243041;
        line-height: 1.4;
      }}
      .review-automation-list li::before {{
        content: "•";
        color: #0b61bb;
        font-weight: 800;
      }}
      .review-hero-actions {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }}
      .review-rail {{
        display: grid;
        gap: 12px;
      }}
      .review-rail .system-card {{
        padding: 18px;
      }}
      .review-rail h3 {{
        margin: 0 0 8px;
        font-size: .94rem;
        color: var(--muted);
        font-weight: 650;
      }}
      .review-mini-stat {{
        display: grid;
        gap: 4px;
      }}
      .review-mini-stat strong {{
        font-size: 1.55rem;
        letter-spacing: -.04em;
      }}
      .review-mini-stat span {{
        color: var(--muted);
        font-size: .88rem;
      }}
      .review-progress {{
        display: grid;
        gap: 8px;
      }}
      .review-progress-row {{
        display: flex;
        justify-content: space-between;
        gap: 12px;
        color: var(--muted);
        font-size: .86rem;
      }}
      .review-progress-row strong {{
        color: #0f172a;
      }}
      .review-progress-fill {{
        background: linear-gradient(90deg, #0a84ff, #7ab7ff);
      }}
      .review-card {{
        display: grid;
        gap: 14px;
        padding: 18px;
      }}
      .review-card-head {{
        display: grid;
        gap: 8px;
      }}
      .review-card-topline {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        flex-wrap: wrap;
      }}
      .review-card-topline .badge {{
        background: rgba(255,255,255,.76);
      }}
      .review-card h3 {{
        margin: 0;
        font-size: 1.08rem;
        letter-spacing: -.02em;
      }}
      .review-card-summary {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        color: var(--muted);
        font-size: .88rem;
      }}
      .review-card-summary span {{
        padding: 5px 10px;
        border-radius: 999px;
        background: rgba(255,255,255,.52);
        border: 1px solid rgba(255,255,255,.6);
      }}
      .review-card-note {{
        color: var(--muted);
        font-size: .88rem;
        line-height: 1.45;
      }}
      .review-card-actions {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }}
      .review-card-actions .btn {{
        justify-content: center;
      }}
      .review-card-actions .btn.secondary {{
        box-shadow: none;
      }}
      .review-card-actions .btn.primary {{
        background: linear-gradient(180deg, #0a84ff, #0a67d1);
      }}
      .review-card-actions .btn.ghost {{
        background: rgba(255,255,255,.58);
      }}
      .review-lanes {{
        display: grid;
        gap: 16px;
      }}
      .review-strip {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(146px, 1fr));
        gap: 10px;
        align-items: stretch;
      }}
      .review-strip a {{
        display: grid;
        gap: 6px;
        min-width: 0;
      }}
      .review-strip .album-cover {{
        width: 100%;
        height: 146px;
        aspect-ratio: 1 / 1;
      }}
      .duplicate-card {{
        display: grid;
        gap: 10px;
      }}
      .duplicate-tile {{
        position: relative;
        display: grid;
        gap: 8px;
        min-width: 0;
      }}
      .duplicate-tile .asset-frame {{
        overflow: hidden;
        border-radius: 20px;
        aspect-ratio: 1 / 1;
      }}
      .duplicate-tile .asset-thumb {{
        border-radius: 20px;
        width: 100%;
        height: 100%;
        object-fit: cover;
      }}
      .duplicate-tile-actions {{
        position: absolute;
        left: 12px;
        right: 12px;
        bottom: 12px;
        display: flex;
        justify-content: space-between;
        gap: 8px;
        pointer-events: none;
      }}
      .duplicate-action {{
        pointer-events: auto;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 34px;
        height: 34px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,.32);
        background: rgba(15,23,42,.55);
        color: #fff;
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 22px rgba(15,23,42,.18);
        font-size: .92rem;
      }}
      .duplicate-action.keep {{
        background: rgba(10,132,255,.58);
      }}
      .duplicate-action.delete {{
        background: rgba(16,24,40,.58);
      }}
      .duplicate-item-footer {{
        display: flex;
        justify-content: space-between;
        gap: 10px;
        align-items: center;
      }}
      .compare-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 16px;
      }}
      .review-title-link {{
        color: inherit;
        text-decoration: none;
      }}
      .review-title-link:hover {{
        text-decoration: underline;
      }}
      .icon-btn {{
        width: 44px;
        min-width: 44px;
        padding: 0;
        justify-content: center;
        font-size: 1.05rem;
      }}
      .review-stats {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
      }}
      .review-stack {{
        display: grid;
        gap: 28px;
      }}
      .review-stat {{
        border: 1px solid var(--line);
        border-radius: 20px;
        background: rgba(255,255,255,.58);
        padding: 16px;
        box-shadow: var(--shadow-soft);
      }}
      .review-stat h3 {{
        margin: 0 0 6px;
        font-size: .92rem;
        color: var(--muted);
        font-weight: 600;
      }}
      .review-stat .big {{
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -.03em;
      }}
      .review-empty {{
        padding: 18px;
        border-radius: 20px;
        border: 1px dashed rgba(15,23,42,.14);
        background: rgba(255,255,255,.42);
        color: var(--muted);
      }}
      .system-top {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
      }}
      .system-meta {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
      }}
      .system-stack {{
        display: grid;
        gap: 40px;
      }}
      .stack-groups {{
        display: grid;
        gap: 12px;
      }}
      .stack-group {{
        display: grid;
        gap: 8px;
      }}
      .stack-group h3 {{
        margin: 0;
        font-size: .92rem;
        color: var(--muted);
        font-weight: 600;
      }}
      .stack-pills {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
      }}
      .stack-chip {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,.68);
        background: rgba(255,255,255,.58);
        backdrop-filter: blur(16px) saturate(150%);
        box-shadow: var(--shadow-soft);
        font-size: .88rem;
      }}
      .stack-chip.ready {{
        color: #065f46;
        background: rgba(236,253,245,.72);
      }}
      .stack-chip.missing {{
        color: #991b1b;
        background: rgba(254,242,242,.72);
      }}
      .stack-chip small {{
        opacity: .78;
        font-size: .78rem;
      }}
      .system-card {{
        border: 1px solid var(--line);
        border-radius: 20px;
        background: rgba(255,255,255,.58);
        padding: 16px;
        box-shadow: var(--shadow-soft);
      }}
      .system-card h3 {{
        margin: 0 0 6px;
        font-size: .92rem;
        color: var(--muted);
        font-weight: 600;
      }}
      .system-card .big {{
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -.03em;
      }}
      .album-card {{
        display: grid;
        gap: 10px;
        padding: 14px;
      }}
      .people-strap {{
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        color: var(--muted);
        font-size: .92rem;
        margin-bottom: 14px;
      }}
      .people-strap span {{
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,.72);
        border: 1px solid rgba(255,255,255,.6);
      }}
      .people-empty {{
        color: var(--muted);
        padding: 16px 4px;
      }}
      .people-person-tile {{
        display: block;
        position: relative;
        min-width: 0;
        text-decoration: none;
      }}
      .people-person-tile .album-cover-shell {{
        position: relative;
      }}
      .people-person-tile img,
      .people-photo-image {{
        width: 100%;
        aspect-ratio: 1 / 1;
        object-fit: cover;
        border-radius: 26px;
        background: linear-gradient(135deg, #eceef3, #d8dde7);
        border: 1px solid rgba(255,255,255,.65);
        display: block;
      }}
      .people-person-meta {{
        display: flex;
        justify-content: flex-start;
        margin-top: 8px;
      }}
      .people-photo-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
        gap: 14px;
      }}
      .people-photo-tile {{
        display: block;
      }}
      .people-detail {{
        border: 0;
        background: transparent;
        box-shadow: none;
        padding: 0;
      }}
      .people-detail .section-header h2 {{
        margin-bottom: 0;
      }}
      .people-detail-main {{
        background: rgba(255,255,255,.52);
        border: 1px solid rgba(255,255,255,.55);
        border-radius: 28px;
        padding: 20px;
        box-shadow: var(--shadow-soft);
      }}
      .people-detail-rail .card.section,
      .people-detail-rail .card.person-name-form {{
        background: rgba(255,255,255,.58);
        border: 1px solid rgba(255,255,255,.55);
        border-radius: 24px;
        box-shadow: var(--shadow-soft);
      }}
      .album-cover-placeholder {{
        aspect-ratio: 1 / 1;
        display: grid;
        place-items: center;
        border-radius: 26px;
        background: linear-gradient(135deg, #eceef3, #d8dde7);
        border: 1px solid rgba(255,255,255,.65);
        font-size: 3rem;
        font-weight: 700;
        color: rgba(17,24,39,.72);
      }}
      .album-cover-shell {{
        position: relative;
      }}
      .album-cover-link {{
        display: block;
      }}
      .album-cover {{
        width: 100%;
        aspect-ratio: 1 / 1;
        object-fit: cover;
        border-radius: 22px;
        background: linear-gradient(135deg, #eceef3, #d8dde7);
        border: 1px solid rgba(255,255,255,.65);
      }}
      .album-cover-badge {{
        position: absolute;
        left: 10px;
        top: 10px;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(17,24,39,.72);
        color: #fff;
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .01em;
        box-shadow: 0 10px 20px rgba(15,23,42,.15);
      }}
      .album-head {{
        display: flex;
        align-items: start;
        justify-content: space-between;
        gap: 10px;
      }}
      .album-name-form {{
        display: grid;
        gap: 8px;
      }}
      .album-name-form input[type="text"] {{
        min-width: 0;
      }}
      .asset-card {{
        display: grid;
        gap: 8px;
        position: relative;
        touch-action: none;
      }}
      .asset-select-toggle {{
        position: absolute;
        top: 10px;
        right: 10px;
        width: 34px;
        height: 34px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,.75);
        background: rgba(17,24,39,.78);
        color: #fff;
        display: none;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        z-index: 3;
        cursor: pointer;
        box-shadow: 0 10px 22px rgba(15,23,42,.18);
      }}
      .library-select-mode .asset-select-toggle {{
        display: flex;
      }}
      .asset-card.is-selected .asset-select-toggle {{
        display: flex;
        background: rgba(10,132,255,.92);
      }}
      .asset-card.is-selected .asset-frame::after {{
        box-shadow: inset 0 0 0 3px rgba(10,132,255,.78);
      }}
      .library-select-bar {{
        display: none;
      }}
      .library-select-bar.active {{
        display: grid;
        gap: 12px;
        position: fixed;
        left: 264px;
        right: 28px;
        bottom: 20px;
        z-index: 18;
        margin: 0;
        padding: 14px 16px;
        border-radius: 24px;
        border: 1px solid rgba(255,255,255,.62);
        background: rgba(255,255,255,.62);
        backdrop-filter: blur(24px) saturate(150%);
        box-shadow: 0 20px 44px rgba(15,23,42,.15);
      }}
      .library-select-fab {{
        position: fixed;
        top: 18px;
        right: 28px;
        z-index: 19;
        border: 1px solid rgba(255,255,255,.62);
        background: rgba(255,255,255,.64);
        backdrop-filter: blur(24px) saturate(150%);
        box-shadow: 0 16px 34px rgba(15,23,42,.12);
      }}
      .library-select-bar-head {{
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
      }}
      .library-select-bar-count {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 88px;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(15,23,42,.06);
        border: 1px solid rgba(15,23,42,.08);
        color: #0f172a;
        font-size: .84rem;
        font-weight: 600;
        backdrop-filter: blur(10px);
      }}
      .library-name-suggestions {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        max-height: 96px;
        overflow: auto;
        padding-right: 4px;
      }}
      .library-name-chip {{
        border: 1px solid rgba(10,132,255,.18);
        background: rgba(10,132,255,.08);
        color: #0b61bb;
        border-radius: 999px;
        padding: 7px 11px;
        font-size: .84rem;
        font-weight: 600;
        cursor: pointer;
      }}
      .library-name-chip:hover {{
        background: rgba(10,132,255,.14);
      }}
      .library-select-bar input[type="text"] {{
        min-width: 220px;
        flex: 1;
      }}
      .library-load-more {{
        display: grid;
        place-items: center;
        margin: 24px 0 120px;
        color: var(--muted);
        font-size: .9rem;
      }}
      .library-load-more .spinner {{
        width: 18px;
        height: 18px;
        border-radius: 50%;
        border: 2px solid rgba(10,132,255,.18);
        border-top-color: rgba(10,132,255,.9);
        animation: spin .9s linear infinite;
        margin-bottom: 8px;
      }}
      .library-infinite-root {{
        display: grid;
        gap: 18px;
        padding-bottom: 128px;
      }}
      @keyframes spin {{
        to {{ transform: rotate(360deg); }}
      }}
      .people-detail {{
        display: grid;
        grid-template-columns: minmax(0, 1.45fr) minmax(320px, .85fr);
        gap: 20px;
        align-items: start;
      }}
      .people-detail-main, .people-detail-rail {{
        min-width: 0;
      }}
      .people-detail-rail {{
        display: grid;
        gap: 16px;
      }}
      .person-name-form {{
        display: grid;
        gap: 10px;
      }}
      .person-name-form.compact {{
        gap: 8px;
      }}
      .inline-form-note {{
        color: var(--muted);
        font-size: .84rem;
      }}
      .person-candidate-card {{
        display: grid;
        gap: 10px;
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
      .detail-form {{
        display: grid;
        gap: 12px;
      }}
      .detail-form label {{
        display: grid;
        gap: 6px;
      }}
      .detail-form textarea {{
        min-height: 90px;
      }}
      .asset-detail-page .main {{
        display: grid;
        grid-template-rows: min-content minmax(0, 1fr);
        gap: 18px;
        overflow: hidden;
      }}
      .asset-detail-workspace {{
        display: grid;
        grid-template-columns: minmax(0, 1.55fr) minmax(360px, .92fr);
        gap: 18px;
        min-height: 0;
      }}
      .asset-viewer-panel,
      .asset-detail-panel {{
        min-width: 0;
        min-height: 0;
      }}
      .asset-detail-page .viewer-card {{
        height: 100%;
        display: grid;
        grid-template-rows: minmax(0, 1fr) auto;
      }}
      .asset-detail-page .viewer-stage {{
        min-height: 0;
        height: 100%;
      }}
      .asset-detail-page .viewer-preview-link {{
        display: block;
        width: 100%;
        height: 100%;
      }}
      .asset-detail-page .viewer-stage img,
      .asset-detail-page .viewer-stage video {{
        height: 100%;
        max-height: none;
      }}
      .asset-detail-shell {{
        height: 100%;
        display: grid;
        grid-template-rows: min-content min-content min-content min-content;
        gap: 16px;
        align-content: start;
      }}
      .asset-header-card {{
        display: grid;
        gap: 6px;
      }}
      .asset-header-card h2 {{
        margin: 0;
        font-size: 1.2rem;
      }}
      .asset-header-card p {{
        margin: 0;
        color: var(--muted);
        line-height: 1.45;
      }}
      .asset-meta-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
      }}
      .asset-field {{
        display: grid;
        gap: 6px;
      }}
      .asset-field.full {{
        grid-column: 1 / -1;
      }}
      .asset-field-label {{
        color: var(--muted);
        font-size: .8rem;
        font-weight: 700;
        letter-spacing: .05em;
        text-transform: uppercase;
      }}
      .asset-field input[readonly] {{
        color: #475467;
        background: rgba(242,244,247,.96);
      }}
      .asset-inline-form {{
        display: grid;
        gap: 8px;
      }}
      .asset-tag-strip {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
      }}
      .asset-tag-strip .status-pill {{
        color: #17324f;
      }}
      .asset-tag-strip .status-pill.empty {{
        color: var(--muted);
      }}
      .asset-actions-card {{
        display: grid;
        gap: 12px;
      }}
      .asset-actions-card .section-header {{
        margin-bottom: 0;
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
      .import-picker {{
        display: grid;
        gap: 8px;
      }}
      .section {{
        margin-bottom: 24px;
      }}
      .home-page .section {{
        margin-bottom: 0;
      }}
      .page-nav {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
        margin-bottom: 16px;
      }}
      .page-num {{
        min-width: 42px;
        justify-content: center;
      }}
      .page-num.active {{
        background: linear-gradient(180deg, rgba(17,24,39,.95), rgba(17,24,39,.88));
        color: #fff;
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
        background: linear-gradient(180deg, rgba(255,255,255,.72), rgba(255,255,255,.48));
        border: 1px solid rgba(255,255,255,.64);
        border-radius: 24px;
        padding: 22px;
        box-shadow: 0 16px 34px rgba(15,23,42,.08);
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
        background: rgba(255,255,255,.46);
        border: 1px solid rgba(255,255,255,.60);
        backdrop-filter: blur(16px) saturate(150%);
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
      .progress-shell {{
        display: grid;
        gap: 10px;
      }}
      .progress-track {{
        height: 12px;
        border-radius: 999px;
        background: rgba(15,23,42,.08);
        overflow: hidden;
      }}
      .progress-fill {{
        height: 100%;
        border-radius: inherit;
        background: linear-gradient(90deg, #0a84ff, #7ab7ff);
      }}
      .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,.58);
        border: 1px solid rgba(255,255,255,.68);
        color: var(--muted);
        backdrop-filter: blur(16px) saturate(150%);
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
          radial-gradient(circle at top left, rgba(10,132,255,.18), transparent 18rem),
          radial-gradient(circle at bottom right, rgba(255,255,255,.9), transparent 18rem),
          linear-gradient(180deg, #f8fbff 0%, #e6edf7 100%);
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
          position: static;
          inset: auto;
          width: auto;
          height: auto;
          overflow: visible;
        }}
        .sidebar {{
          position: static;
          height: auto;
          overflow: visible;
          border-right: 0;
          border-bottom: 1px solid var(--line);
        }}
        .main {{
          height: auto;
          overflow: visible;
        }}
        .asset-detail-page .main {{
          height: auto;
          overflow: visible;
        }}
        .asset-detail-workspace {{
          grid-template-columns: 1fr;
        }}
        .asset-meta-grid {{
          grid-template-columns: 1fr;
        }}
        .hero-card {{
          grid-template-columns: 1fr;
        }}
        .hero-strip {{
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }}
        .library-select-bar.active {{
          left: 12px;
          right: 12px;
          bottom: 12px;
        }}
        .library-select-fab {{
          top: 12px;
          right: 12px;
        }}
      }}
    </style>
  </head>
  <body{body_class_attr}>
    <div class="app">
      <aside class="sidebar">
        <a class="brand" href="/">Photo Unifier</a>
        <div class="nav-group">
          <a class="nav-item" href="/app/assets">Library</a>
          <a class="nav-item" href="/app/people">People</a>
          <a class="nav-item" href="/app/review">Review</a>
          <a class="nav-item" href="/app/system">System</a>
        </div>
        {history_html}
        <div class="sidebar-foot"><a href="/docs">API Docs</a></div>
      </aside>
      <main class="main">
        {body}
      </main>
    </div>
    <script>
      (function() {{
        const current = window.location.pathname + window.location.search;
        document.querySelectorAll('[data-history-seq]').forEach((el) => {{
          const seq = el.getAttribute('data-history-seq');
          if (seq) {{
            el.href = '/app/history/' + encodeURIComponent(seq) + '/restore?return_to=' + encodeURIComponent(current);
          }}
        }});
        document.querySelectorAll('form[data-auto-submit="true"]').forEach((form) => {{
          form.querySelectorAll('select').forEach((field) => {{
            field.addEventListener('change', () => form.requestSubmit());
          }});
          form.querySelectorAll('input[type="search"], input[type="text"]').forEach((field) => {{
            field.addEventListener('keydown', (event) => {{
              if (event.key === 'Enter') {{
                event.preventDefault();
                form.requestSubmit();
              }}
            }});
          }});
        }});
        document.querySelectorAll('[data-home-carousel="true"]').forEach((carousel) => {{
          const slides = Array.from(carousel.querySelectorAll('[data-carousel-slide]'));
          const dots = Array.from(carousel.querySelectorAll('[data-carousel-dot]'));
          if (!slides.length) return;
          let index = 0;
          const show = (nextIndex) => {{
            index = (nextIndex + slides.length) % slides.length;
            slides.forEach((slide, i) => slide.classList.toggle('is-active', i === index));
            dots.forEach((dot, i) => dot.classList.toggle('is-active', i === index));
          }};
          dots.forEach((dot, i) => {{
            dot.addEventListener('click', () => show(i));
          }});
          show(0);
          window.setInterval(() => show(index + 1), parseInt(carousel.dataset.interval || '5000', 10));
        }});
      }})();
    </script>
  </body>
</html>"""


def _format_history_stamp(value: Optional[str]) -> str:
    if not value:
        return ""
    try:
        return datetime.fromisoformat(value).strftime("%H:%M")
    except Exception:
        return str(value)


def _friendly_history_label(action_type: str) -> str:
    labels = {
        "SET_ASSET_REVIEW": "Quick Rank",
        "ADD_ASSET_PERSON": "Quick person tagger",
        "REMOVE_ASSET_PERSON": "Quick person tagger",
        "RESOLVE_DUPLICATE_GROUP": "Choose canonical",
        "KEEP_ALL_DUPLICATE_GROUP": "Keep all",
        "SKIP_DUPLICATE_GROUP": "Skip group",
        "HIDE_DUPLICATE_ITEM": "Remove duplicate item",
    }
    return labels.get(action_type, action_type.replace("_", " ").title())


PERSON_DETAIL_ASSET_LIMIT = 5000


def _mercator_point(lat: float, lon: float) -> tuple[float, float]:
    x = (lon + 180.0) / 360.0
    lat = max(min(lat, 85.0), -85.0)
    lat_rad = radians(lat)
    merc_n = log(tan((pi / 4.0) + (lat_rad / 2.0)))
    y = (1.0 - (merc_n / pi)) / 2.0
    return (x, y)


def _render_home_recent_panel(recent_assets: list[dict]) -> str:
    if not recent_assets:
        return """
        <div class="home-carousel" aria-label="Recent media carousel">
          <div class="carousel-stage"></div>
        </div>
        """
    slides = recent_assets[:20]
    slide_imgs = []
    dots = []
    for index, row in enumerate(slides):
        active = " is-active" if index == 0 else ""
        slide_imgs.append(
            f'<img class="carousel-slide{active}" src="/poster/{escape(row["id"])}" alt="{escape(row.get("orig_filename") or row["id"])}" loading="lazy" decoding="async" data-carousel-slide="{index}">'
        )
        dots.append(f'<button class="carousel-dot{" is-active" if index == 0 else ""}" type="button" aria-label="Show item {index + 1}" data-carousel-dot="{index}"></button>')
    return f"""
    <div class="home-carousel" data-home-carousel="true" data-interval="5000" aria-label="Recent media carousel">
      <div class="carousel-stage">
        {''.join(slide_imgs)}
      </div>
      <div class="carousel-dots">{''.join(dots)}</div>
    </div>
    """


def _render_home_map_panel(db_path: Path) -> str:
    rows = manifest.list_geotagged_assets(db_path, limit=24)
    if not rows:
        recent_assets = manifest.list_assets(db_path, limit=12)
        fallback_dots = []
        for index, row in enumerate(recent_assets):
            asset_id = str(row.get("id") or "")
            if len(asset_id) < 16:
                continue
            x_seed = int(asset_id[:8], 16) / 0xFFFFFFFF
            y_seed = int(asset_id[8:16], 16) / 0xFFFFFFFF
            cx = 10 + (x_seed * 80)
            cy = 10 + (y_seed * 80)
            opacity = max(0.18, 0.72 - (index * 0.04))
            fallback_dots.append(
                f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{max(1.6, 3.2 - index * 0.06):.2f}" fill="rgba(255,255,255,{opacity:.2f})" />'
            )
        return """
        <div class="home-map" aria-label="Location map">
          <svg viewBox="0 0 100 100" preserveAspectRatio="none">
            <defs>
              <linearGradient id="mapGlow" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="#0a84ff" stop-opacity=".16" />
                <stop offset="100%" stop-color="#ffffff" stop-opacity=".04" />
              </linearGradient>
            </defs>
            <rect x="0" y="0" width="100" height="100" fill="url(#mapGlow)"/>
            <g stroke="rgba(255,255,255,.10)" stroke-width=".35">
              {''.join(f'<line x1="{x}" y1="0" x2="{x}" y2="100" />' for x in (20, 40, 60, 80))}
              {''.join(f'<line x1="0" y1="{y}" x2="100" y2="{y}" />' for y in (20, 40, 60, 80))}
            </g>
            <g fill="none" stroke="rgba(255,255,255,.14)" stroke-width=".7">
              <path d="M10,24 C18,20 24,18 32,19 C38,20 44,18 51,20 C57,22 63,24 70,23 C76,22 82,20 90,22" />
              <path d="M12,64 C18,61 24,59 30,60 C37,61 44,58 49,57 C56,56 62,58 68,60 C76,62 82,64 89,63" />
              <path d="M18,82 C22,79 28,77 34,78 C41,79 47,77 53,76 C60,75 67,77 73,79 C79,81 84,83 88,82" />
            </g>
            <g>
              {{DOTS}}
            </g>
          </svg>
        </div>
        """.replace("{{DOTS}}", "".join(fallback_dots))
    dots = []
    for index, row in enumerate(rows):
        lat = float(row.get("gps_lat") or 0)
        lon = float(row.get("gps_lon") or 0)
        x, y = _mercator_point(lat, lon)
        cx = 10 + (x * 80)
        cy = 10 + (y * 80)
        opacity = max(0.28, 0.96 - (index * 0.03))
        dots.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{max(1.8, 3.4 - index * 0.05):.2f}" fill="rgba(255,255,255,{opacity:.2f})" />'
        )
    return f"""
    <div class="home-map" aria-label="Location map">
      <svg viewBox="0 0 100 100" preserveAspectRatio="none">
          <defs>
            <linearGradient id="mapGlow" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stop-color="#0a84ff" stop-opacity=".18" />
              <stop offset="100%" stop-color="#ffffff" stop-opacity=".06" />
            </linearGradient>
          </defs>
          <rect x="0" y="0" width="100" height="100" fill="url(#mapGlow)"/>
          <g stroke="rgba(255,255,255,.12)" stroke-width=".4">
            {''.join(f'<line x1="{x}" y1="0" x2="{x}" y2="100" />' for x in (20, 40, 60, 80))}
            {''.join(f'<line x1="0" y1="{y}" x2="100" y2="{y}" />' for y in (20, 40, 60, 80))}
          </g>
          <g fill="none" stroke="rgba(255,255,255,.16)" stroke-width=".7">
            <path d="M10,24 C18,20 24,18 32,19 C38,20 44,18 51,20 C57,22 63,24 70,23 C76,22 82,20 90,22" />
            <path d="M12,64 C18,61 24,59 30,60 C37,61 44,58 49,57 C56,56 62,58 68,60 C76,62 82,64 89,63" />
            <path d="M18,82 C22,79 28,77 34,78 C41,79 47,77 53,76 C60,75 67,77 73,79 C79,81 84,83 88,82" />
          </g>
          <g>
            {''.join(dots)}
          </g>
      </svg>
    </div>
    """


def _render_home_people_panel(db_path: Path) -> str:
    people = manifest.list_person_albums(db_path, limit=4)
    tiles = []
    for row in people:
        cover_asset = row.get("cover_asset_id")
        cover_face = row.get("cover_face_id")
        cover_src = ""
        if cover_asset:
            cover_src = f"/poster/{escape(str(cover_asset))}"
        elif cover_face:
            cover_src = f"/face-crop/{escape(str(cover_face))}"
        if not cover_src:
            continue
        label = row.get("label") or "Person"
        tiles.append(
            f"""
            <a class="featured-person" href="/app/people?identity_id={escape(str(row['id']))}" title="{escape(label)}">
              <img src="{cover_src}" alt="{escape(label)}" loading="lazy" decoding="async">
            </a>
            """
        )
    if len(tiles) < 4:
        recent_assets = manifest.list_assets(db_path, limit=4)
        used_ids = {row.get("cover_asset_id") for row in people if row.get("cover_asset_id")}
        for row in recent_assets:
            asset_id = row.get("id")
            if not asset_id or asset_id in used_ids:
                continue
            tiles.append(
                f"""
                <a class="featured-person" href="/app/assets/{escape(asset_id)}" title="{escape(row.get('orig_filename') or asset_id)}">
                  <img src="/poster/{escape(asset_id)}" alt="{escape(row.get('orig_filename') or asset_id)}" loading="lazy" decoding="async">
                </a>
                """
            )
            if len(tiles) >= 4:
                break
    return f"""
    <div class="featured-people" aria-label="Featured people">
      {''.join(tiles)}
    </div>
    """


def _history_sidebar_html(db_path: Optional[Path]) -> str:
    if not db_path:
        return ""
    rows = [row for row in manifest.get_mutation_history(db_path, limit=6) if not row.get("undone_at")]
    if not rows:
        return """
        <div class="sidebar-history">
          <div class="nav-label">History</div>
          <div class="history-empty">No photo changes yet.</div>
        </div>
        """
    items = []
    for row in rows:
        label = _friendly_history_label(str(row.get("action_type") or ""))
        summary = str(row.get("summary") or "")
        stamp = _format_history_stamp(str(row.get("created_at") or ""))
        items.append(
            f"""
            <a class="history-entry" href="/app/history/{row['seq']}/restore" data-history-seq="{row['seq']}" title="{escape(summary)}">
              <div class="history-entry-top"><span>{escape(label)}</span><span class="history-entry-time">{escape(stamp)}</span></div>
              <div class="history-entry-summary">{escape(summary)}</div>
            </a>
            """
        )
    return f"""
    <div class="sidebar-history">
      <div class="nav-label">History</div>
      <div class="history-list">
        {''.join(items)}
      </div>
    </div>
    """


def _format_moment_label(value: Optional[str]) -> str:
    if not value:
        return "Unknown Date"
    try:
        dt = datetime.strptime(value[:19], "%Y-%m-%dT%H:%M:%S")
        return dt.strftime("%B %Y")
    except Exception:
        return "Unknown Date"


def _human_file_size(value: Optional[object]) -> str:
    try:
        size = float(value or 0)
    except Exception:
        return "-"
    if size <= 0:
        return "-"
    units = ["bytes", "KB", "MB", "GB", "TB"]
    index = 0
    while size >= 1024 and index < len(units) - 1:
        size /= 1024.0
        index += 1
    if index == 0:
        return f"{int(size)} {units[index]}"
    return f"{size:.1f} {units[index]}"


def _location_address_from_metadata(metadata_rows: list[dict]) -> str:
    for row in metadata_rows:
        if row.get("field_name") != "location":
            continue
        value = row.get("value")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _resolve_asset_address(db_path, asset_id: str, item: dict, metadata_rows: list[dict]) -> str:
    address = _location_address_from_metadata(metadata_rows)
    if address:
        return address
    lat = item.get("gps_lat")
    lon = item.get("gps_lon")
    if lat is None or lon is None:
        return ""
    resolved = reverse_geocode_address(float(lat), float(lon))
    if resolved:
        manifest.set_metadata_field(
            db_path,
            asset_id,
            field_name="location",
            value=resolved,
            source_name="reverse_geocoder",
            source_field="Address",
            is_canonical=True,
        )
        return resolved
    return ""


def _asset_thumb_html(asset_id: str, filename: str, managed_path: Optional[str], dt_original: Optional[str], media_type: str, status: str) -> str:
    thumb_src = f"/poster/{asset_id}"
    return f"""
    <article class="asset-card" data-asset-id="{escape(asset_id)}">
      <button class="asset-select-toggle" type="button" aria-label="Select {escape(filename)}">+</button>
      <a class="asset-link" href="/app/assets/{asset_id}" aria-label="{escape(filename)}">
        <div class="asset-frame">
          <img class="asset-thumb" src="{thumb_src}" alt="{escape(filename)}" loading="lazy" decoding="async">
          <span class="media-badge">{escape(media_type)}</span>
        </div>
      </a>
    </article>
    """


def _search_result_card(row: dict, *, query: Optional[str] = None, similar_to: Optional[str] = None) -> str:
    reasons = ", ".join(row.get("reasons") or [])
    similar_link = f"/app/search?similar_to={escape(row['id'])}"
    compare_link = ""
    if similar_to:
        compare_link = f"<a href='/app/compare?left={escape(similar_to)}&right={escape(row['id'])}'><button class='btn secondary' type='button'>Compare</button></a>"
    rating = row.get("user_rating")
    rating_badge = f"{rating}★" if rating else "Unrated"
    score = row.get("score")
    return f"""
    <article class="asset-card">
      <a class="asset-link" href="/app/assets/{escape(row['id'])}">
        <div class="asset-frame">
          <img class="asset-thumb" src="/poster/{escape(row['id'])}" alt="{escape(row.get('orig_filename') or row['id'])}" loading="lazy" decoding="async">
          <span class="media-badge">{escape(str(rating_badge))}</span>
        </div>
      </a>
      <div class="asset-name"><a href="/app/assets/{escape(row['id'])}">{escape(row.get('orig_filename') or row['id'])}</a></div>
      <div class="asset-meta">{escape(str(row.get('media_type') or 'unknown'))} · score {escape(str(score if score is not None else '-'))}</div>
      <div class="asset-meta">{escape(reasons or 'no explanation yet')}</div>
      <div class="button-row">
        <a href="{similar_link}"><button class="btn secondary" type="button">More Like This</button></a>
        {compare_link}
      </div>
    </article>
    """


def _duplicate_tile_html(group_id: str, item: dict, *, badge: str = "") -> str:
    asset_id = item["asset_id"]
    filename = item.get("orig_filename") or asset_id
    media_kind = _guess_media_kind(item.get("managed_path"), item.get("media_type"))
    inline_url = f"/inline/{escape(asset_id)}"
    poster_url = f"/poster/{escape(asset_id)}"
    detail_bits = [
        str(item.get("dt_original") or "unknown time"),
        _human_file_size(item.get("orig_size")) or "Unknown size",
        str(item.get("media_type") or "unknown"),
    ]
    detail = " · ".join(part for part in detail_bits if part)
    keep_link = f"/app/duplicates/{escape(group_id)}/keep?asset_id={escape(asset_id)}"
    hide_link = f"/app/duplicates/{escape(group_id)}/hide?asset_id={escape(asset_id)}"
    compare_link = f"/app/duplicates/{escape(group_id)}/review"
    media_html = (
        f"""
        <video controls preload="metadata" poster="{poster_url}">
          <source src="{inline_url}">
        </video>
        """
        if media_kind == "video"
        else f'<img src="{poster_url}" alt="{escape(filename)}" loading="lazy" decoding="async">'
    )
    badge_html = escape(badge) if badge else ""
    return f"""
    <article class="duplicate-tile review-card-tile card">
      <div class="review-card-media primary">
        {media_html}
        <div class="review-card-actions">
          <a class="review-card-action keep" href="{keep_link}" title="Keep this one" aria-label="Keep this one">✓</a>
          <a class="review-card-action delete" href="{hide_link}" title="Remove this one" aria-label="Remove this one">🗑</a>
        </div>
        {f'<div class="review-card-badge">{badge_html}</div>' if badge_html else ''}
      </div>
      <div class="review-card-footer">
        <div class="review-card-title">
          <a class="review-title-link" href="{compare_link}"><strong>{escape(filename)}</strong></a>
        </div>
        <div class="review-card-meta">
          <span>{escape(detail)}</span>
        </div>
      </div>
    </article>
    """


def _review_sequence_score_label(group: dict) -> str:
    group_type = str(group.get("group_type") or "")
    if group_type == "EXACT_SHA256":
        return "100% duplicate"
    return f"{int(round(float(group.get('match_score_pct') or 0)))}% near duplicate"


def _friendly_job_label(job_type: str) -> str:
    labels = {
        "ingest": "Import media",
        "plan_library": "Prepare library",
        "build_library": "Build library copies",
        "build_derivatives": "Build thumbnails",
        "repair_metadata": "Fix metadata",
        "extract_content": "Read text and make embeddings",
        "detect_faces": "Find faces",
        "dedupe_exact": "Find exact duplicates",
        "dedupe_near": "Find near duplicates",
        "pilot_run": "Quick check",
        "audit": "Run report",
        "validate_library": "Check library files",
    }
    return labels.get(job_type, job_type.replace("_", " ").strip().title())


def _job_summary(job_type: str) -> str:
    hints = {
        "ingest": "Scan new folders and add files to the archive.",
        "plan_library": "Work out where library copies should go.",
        "build_library": "Copy files into the library.",
        "build_derivatives": "Create thumbnails and previews.",
        "repair_metadata": "Fill in missing dates, places, and names.",
        "extract_content": "Read text and create search signals.",
        "detect_faces": "Look for faces and face clusters.",
        "dedupe_exact": "Group files that are identical.",
        "dedupe_near": "Group files that look almost the same.",
        "pilot_run": "Run a small end-to-end sanity check.",
        "audit": "Summarize what looks good and what still needs work.",
        "validate_library": "Check that library files are present.",
    }
    return hints.get(job_type, "Background task for Photo Unifier.")


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
    poster_url = f"/poster/{asset_id}"
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
            <div class="file-pill">Library file pending</div>
            <div class="muted">This item has been indexed, but the managed copy is not ready yet.</div>
          </div>
        </section>
        """
    if media_kind == "image":
        return f"""
        <section class="card viewer-card">
          <div class="viewer-stage">
            <a class="viewer-preview-link" href="{inline_url}" title="Open full resolution">
              <img src="{poster_url}" alt="{escape(filename)}" loading="eager" decoding="async" fetchpriority="high">
            </a>
          </div>
          {actions}
        </section>
        """
    if media_kind == "video":
        return f"""
        <section class="card viewer-card">
          <div class="viewer-stage">
            <video controls autoplay muted playsinline preload="metadata" poster="{thumb_url}">
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
            <audio controls preload="none">
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
        <div class="muted">You can open the library file directly, download it, or use the generated preview if one exists.</div>
        <div style="justify-self:center; width:min(420px, 100%);">
          <img class="asset-thumb" style="aspect-ratio:4/3;" src="/poster/{asset_id}" alt="{escape(filename)}" loading="lazy" decoding="async">
        </div>
      </div>
      {actions}
    </section>
    """


def _rating_controls_html(asset_id: str, item: dict) -> str:
    rating = int(item.get("user_rating") or 0)
    buttons = []
    for value in range(1, 6):
        active = "btn" if rating == value else "btn secondary"
        buttons.append(
            f'<a href="/app/assets/{asset_id}/review?rating={value}"><button class="{active}" type="button">{value}★</button></a>'
        )
    return "".join(buttons)


def _keyboard_shortcuts_script(prev_id: Optional[str], next_id: Optional[str], asset_id: str) -> str:
    prev_url = f"/app/assets/{prev_id}" if prev_id else ""
    next_url = f"/app/assets/{next_id}" if next_id else ""
    return f"""
    <script>
      (function() {{
        const prevUrl = {json.dumps(prev_url)};
        const nextUrl = {json.dumps(next_url)};
        const assetId = {json.dumps(asset_id)};
        document.addEventListener('keydown', function(ev) {{
          if (ev.target && ['INPUT', 'TEXTAREA'].includes(ev.target.tagName)) return;
          if (ev.key === 'ArrowLeft' && prevUrl) window.location.href = prevUrl;
          if (ev.key === 'ArrowRight' && nextUrl) window.location.href = nextUrl;
          if (/^[1-5]$/.test(ev.key)) window.location.href = `/app/assets/${{assetId}}/review?rating=${{ev.key}}`;
          if (ev.key.toLowerCase() === 'r') window.location.href = `/app/assets/${{assetId}}/review?state=reviewed`;
        }});
      }})();
    </script>
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
                width, height = _build_image_thumbnail(managed_file, candidate, 320)
                manifest.record_thumbnail(db_path, asset_id, "primary", str(candidate), "READY", width=width, height=height)
                return candidate, item
            except Exception:
                return None, item
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
        try:
            manifest.init_db(current_db_path)
        except Exception as exc:
            if "readonly database" not in str(exc).lower():
                raise
        return current_config, resolved, current_db_path, current_managed

    def _render_dashboard(current_config: AppConfig, db_path: Path, resolved: Path, message: Optional[str] = None) -> str:
        overview = manifest.get_overview(db_path)
        metadata_overview = manifest.get_metadata_overview(db_path)
        import_progress = manifest.get_import_progress(db_path)
        face_overview = manifest.get_face_overview(db_path)
        recent_assets = []
        for row in manifest.list_assets(db_path, limit=120, sort="recent"):
            if row.get("managed_path"):
                recent_assets.append(row)
            if len(recent_assets) >= 20:
                break
        if not recent_assets:
            recent_assets = manifest.list_assets(db_path, limit=20, sort="recent")
        flash = f"<div class='flash'>{escape(message)}</div>" if message else ""
        body = f"""
        <section class="hero-card">
            <section class="hero-panel hero-copy">
            <span class="badge">Photo library</span>
            <h1>Everything, in one calm place.</h1>
            <p>Browse the archive, jump into people, review cleanup queues, and keep the system healthy without feeling like you are in settings.</p>
            <div class="button-row">
              <a href="/app/assets"><button class="btn" type="button">Open Library</button></a>
              <a href="/app/people"><button class="btn secondary" type="button">People</button></a>
              <a href="/app/review"><button class="btn secondary" type="button">Review</button></a>
              <a href="/app/system"><button class="btn secondary" type="button">System</button></a>
            </div>
          </section>
          <section class="hero-panel hero-people-panel">
            {_render_home_people_panel(db_path)}
          </section>
        </section>
        {flash}
        <section class="metrics">
          <section class="summary-card">
            <h2>Library</h2>
            <div class="summary-big">{overview.get('assets_total', 0)}</div>
            <div class="summary-copy">items ready to browse</div>
          </section>
          <section class="summary-card">
            <h2>Imports</h2>
            <div class="summary-big">{import_progress.get('completion_pct', 0)}%</div>
            <div class="summary-copy">{import_progress.get('ready_assets', 0)} of {import_progress.get('total_assets', 0)} fully through the pipeline</div>
          </section>
          <section class="summary-card">
            <h2>People</h2>
            <div class="summary-big">{face_overview.get('confirmed_identities', 0)}</div>
            <div class="summary-copy">named people</div>
          </section>
          <section class="summary-card">
            <h2>Metadata</h2>
            <div class="summary-big">{metadata_overview.get('average_metadata_score', 0)}</div>
            <div class="summary-copy">average metadata coverage score</div>
          </section>
        </section>
        <section class="home-panels">
          {_render_home_recent_panel(recent_assets)}
          {_render_home_map_panel(db_path)}
        </section>
        """
        return _page("Photos", body, history_html=_history_sidebar_html(db_path), body_class="home-page")

    def _render_people_page(
        db_path: Path,
        message: Optional[str] = None,
        person: Optional[str] = None,
        identity_id: Optional[str] = None,
        show_review: bool = False,
        view: str = "confirmed",
    ) -> str:
        active_jobs = [
            job for job in manifest.list_jobs(db_path, limit=20)
            if str(job.get("status") or "") in {"QUEUED", "RUNNING"}
            and str(job.get("job_type") or "") in {"face_detection", "face_clustering", "full_pipeline"}
        ]
        sync_banner = ""
        if active_jobs:
            labels = ", ".join(str(job.get("job_type") or "sync").replace("_", " ") for job in active_jobs[:2])
            if len(active_jobs) > 2:
                labels += f" and {len(active_jobs) - 2} more"
            sync_banner = f"<div class='sync-banner'>Syncing in progress: {escape(labels)}. People clusters may change while this finishes.</div>"
        query = (person or "").strip() or None
        selected_identity = manifest.get_face_identity(db_path, identity_id) if identity_id else None
        overview = manifest.get_face_overview(db_path)
        confirmed_albums = manifest.list_person_albums(db_path, limit=24, query=query, status="CONFIRMED")
        clustered_albums = manifest.list_person_albums(db_path, limit=48, query=query, status="CLUSTERED")
        people_by_id: OrderedDict[str, dict[str, object]] = OrderedDict()
        for row in confirmed_albums + clustered_albums:
            people_by_id[str(row["id"])] = row
        named_albums = list(people_by_id.values())
        active_view = "confirmed"
        people_return_to = (
            f"/app/people?identity_id={quote_plus(str(identity_id))}"
            if identity_id
            else "/app/people"
        )

        def _person_tile(row: dict[str, object]) -> str:
            cover_src = ""
            if row.get("cover_asset_id"):
                cover_src = f"/poster/{escape(str(row['cover_asset_id']))}"
            elif row.get("cover_face_id"):
                cover_src = f"/face-crop/{escape(str(row['cover_face_id']))}"
            else:
                tagged_assets = manifest.list_assets_tagged_with_person(db_path, str(row.get("label") or ""), limit=1)
                if tagged_assets:
                    cover_src = f"/poster/{escape(str(tagged_assets[0]['id']))}"
            label = str(row.get("label") or "Person")
            status = str(row.get("status") or "")
            initials = escape((label[:1] or "?").upper())
            href = f"/app/people?identity_id={escape(str(row['id']))}"
            cover_html = f'<img src="{cover_src}" alt="{escape(label)}" loading="lazy" decoding="async">' if cover_src else f'<div class="album-cover-placeholder">{initials}</div>'
            status_badge = (
                '<span class="album-cover-chip">Cluster</span>'
                if status == "CLUSTERED"
                else ""
            )
            return f"""
            <a class="people-person-tile" href="{href}">
              <div class="album-cover-shell">
                {cover_html}
                <span class="album-cover-badge">{escape(label)}</span>
                {status_badge}
              </div>
            </a>
            """

        named_cards = "".join(card for card in (_person_tile(row) for row in named_albums) if card) or "<div class='people-empty'>No named people yet.</div>"
        flash = f"<div class='flash'>{escape(message)}</div>" if message else ""
        body_parts = [
            f"""
        <div class="toolbar">
          <div class="title">
            <h1>People</h1>
            <p>Open a person to browse confirmed photos and merge strong related clusters.</p>
          </div>
        </div>
            """,
            flash,
            sync_banner,
            f"""
        <div class="people-strap">
          <span>{overview.get('confirmed_identities', 0)} named people</span>
          <span>{overview.get('clustered_identities', 0)} clusters</span>
          <span>{overview.get('total_faces', 0)} faces</span>
        </div>
            """,
        ]

        if selected_identity:
            selected_label = str(selected_identity.get("label") or "Person")
            face_assets = manifest.list_assets_for_identity(
                db_path,
                str(selected_identity["id"]),
                limit=PERSON_DETAIL_ASSET_LIMIT,
            )
            tagged_assets = manifest.list_assets_tagged_with_person(
                db_path,
                selected_label,
                limit=PERSON_DETAIL_ASSET_LIMIT,
            )
            selected_assets_map = OrderedDict()
            for row in face_assets + tagged_assets:
                selected_assets_map[str(row["id"])] = row
            selected_assets = list(selected_assets_map.values())
            related_clusters = manifest.list_related_identity_candidates(
                db_path,
                str(selected_identity["id"]),
                limit=12,
                threshold=0.80,
            )
            merged_cluster_count = 0
            for cluster in related_clusters:
                cluster_id = str(cluster.get("id") or "")
                if not cluster_id:
                    continue
                score = float(cluster.get("score") or 0.0)
                if score < 0.80:
                    continue
                try:
                    manifest.merge_face_identities(db_path, cluster_id, str(selected_identity["id"]))
                    merged_cluster_count += 1
                except Exception as exc:
                    print(f"[people] auto-merge failed source_identity_id={cluster_id} target_identity_id={selected_identity['id']}: {exc}")
            if merged_cluster_count:
                manifest.apply_identity_to_assets(db_path, str(selected_identity["id"]))
                manifest.refresh_people_for_identity(db_path, str(selected_identity["id"]))
                face_assets = manifest.list_assets_for_identity(
                    db_path,
                    str(selected_identity["id"]),
                    limit=PERSON_DETAIL_ASSET_LIMIT,
                )
                tagged_assets = manifest.list_assets_tagged_with_person(
                    db_path,
                    selected_label,
                    limit=PERSON_DETAIL_ASSET_LIMIT,
                )
                selected_assets_map = OrderedDict()
                for row in face_assets + tagged_assets:
                    selected_assets_map[str(row["id"])] = row
                selected_assets = list(selected_assets_map.values())
                related_clusters = manifest.list_related_identity_candidates(
                    db_path,
                    str(selected_identity["id"]),
                    limit=12,
                    threshold=0.80,
                )
            confirmed_grid = "".join(
                f"""
                <a class="people-photo-tile" href="/app/assets/{escape(str(row['id']))}">
                  <img class="people-photo-image" src="/poster/{escape(str(row['id']))}" alt="{escape(str(row.get('orig_filename') or selected_label))}" loading="lazy" decoding="async">
                </a>
                """
                for row in selected_assets
            ) or "<div class='people-empty'>No confirmed photos yet.</div>"
            related_cluster_cards = []
            for cluster in related_clusters:
                cluster_id = str(cluster.get("id") or "")
                if not cluster_id:
                    continue
                cover_asset_id = str(cluster.get("cover_asset_id") or "")
                cover_face_id = str(cluster.get("cover_face_id") or "")
                cluster_label = str(cluster.get("label") or "Cluster")
                score_pct = round(float(cluster.get("score") or 0.0) * 100, 1)
                face_count = int(cluster.get("face_count") or 0)
                asset_count = int(cluster.get("asset_count") or 0)
                merge_link = (
                    f"/app/people/merge?source_identity_id={escape(cluster_id)}"
                    f"&target_identity_id={escape(str(selected_identity['id']))}"
                    f"&return_to={quote_plus(people_return_to)}"
                )
                if cover_asset_id:
                    cover_html = f'<img class="people-photo-image" src="/poster/{escape(cover_asset_id)}" alt="{escape(cluster_label)}" loading="lazy" decoding="async">'
                elif cover_face_id:
                    cover_html = f'<img class="people-photo-image" src="/face-crop/{escape(cover_face_id)}" alt="{escape(cluster_label)}" loading="lazy" decoding="async">'
                else:
                    cover_html = f'<div class="album-cover-placeholder">{escape((cluster_label[:1] or "?").upper())}</div>'
                related_cluster_cards.append(
                    f"""
                    <section class="people-photo-tile" style="position:relative;">
                      <a href="/app/people?identity_id={escape(cluster_id)}">
                        {cover_html}
                      </a>
                      <div style="position:absolute;top:10px;left:10px;right:10px;display:flex;justify-content:space-between;align-items:center;gap:8px;">
                        <span class="badge">{escape(str(score_pct))}% cluster</span>
                        <a href="{merge_link}"><button class="btn secondary" type="button">Merge</button></a>
                      </div>
                      <div style="position:absolute;left:10px;right:10px;bottom:10px;">
                        <span class="badge">{escape(cluster_label)} · {face_count} faces · {asset_count} assets</span>
                      </div>
                    </section>
                    """
                )
            secondary_grid_parts = []
            if related_cluster_cards:
                secondary_grid_parts.append(
                    """
                    <div class="section-header" style="grid-column:1 / -1;">
                      <h2>Related clusters</h2>
                      <p>Merge only when the whole cluster clearly belongs to this person.</p>
                    </div>
                    """
                )
                secondary_grid_parts.extend(related_cluster_cards)
            secondary_grid = "".join(secondary_grid_parts)
            body_parts.append(
                f"""
        <div class="toolbar">
          <div class="title">
            <h1>{escape(selected_label)}</h1>
            <p>{len(selected_assets)} confirmed photos</p>
          </div>
          <div class="button-row">
            <a href="/app/people"><button class="btn secondary" type="button">Back</button></a>
          </div>
        </div>
        <form class="person-name-form compact" method="get" action="/app/people/rename" data-auto-submit="true" style="margin-bottom:14px;">
          <input type="hidden" name="identity_id" value="{escape(str(selected_identity['id']))}">
          <input type="hidden" name="return_to" value="{escape(people_return_to)}">
          <input type="text" name="label" value="{escape(selected_label)}" placeholder="Rename person">
          <div class="inline-form-note">Press Enter to save the name.</div>
        </form>
        <div class="people-strap" style="margin-bottom:16px;">
          <span>{len(selected_assets)} confirmed photos</span>
          <span>{len(related_cluster_cards)} related clusters</span>
        </div>
        <section class="people-photo-grid">
          {confirmed_grid}
        </section>
        {f'<section class="people-photo-grid" style="margin-top:18px;">{secondary_grid}</section>' if secondary_grid else ''}
                """
            )
        else:
            body_parts.append(
                f"""
        <section class="album-grid">
          {named_cards}
        </section>
        """
            )
        body = "".join(body_parts)
        return _page("People", body, history_html=_history_sidebar_html(db_path))

    def _render_review_page(db_path: Path, message: Optional[str] = None) -> str:
        exact_groups = manifest.list_duplicate_groups(db_path, limit=80, group_type="EXACT_SHA256", status="OPEN")
        sequence_group_types = ("SNAPCHAT_SEQUENCE", "NEAR_VISUAL", "NEAR_AHASH")
        sequence_groups_by_id: dict[str, dict] = {}
        for group_type in sequence_group_types:
            for group in manifest.list_duplicate_groups(
                db_path,
                limit=80,
                group_type=group_type,
                status="OPEN",
            ):
                sequence_groups_by_id.setdefault(group["id"], group)
        sequence_groups = sorted(
            sequence_groups_by_id.values(),
            key=lambda group: (
                -float(group.get("match_score") or 0),
                str(group.get("created_at") or ""),
                str(group.get("id") or ""),
            ),
        )
        blurry_items = manifest.list_assets_with_flag(db_path, "is_blurry", limit=48)
        top_group = (exact_groups + sequence_groups)[0] if (exact_groups or sequence_groups) else None

        def _group_primary_asset_id(group: dict, items: list[dict]) -> str:
            canonical_id = group.get("canonical_asset_id")
            if canonical_id and any(item["asset_id"] == canonical_id for item in items):
                return canonical_id
            return items[0]["asset_id"]

        def _review_tile_html(group: dict, item: dict, *, primary: bool = False) -> str:
            asset_id = item["asset_id"]
            filename = item.get("orig_filename") or asset_id
            primary_class = " primary" if primary else ""
            media_kind = _guess_media_kind(item.get("managed_path"), item.get("media_type"))
            dt_original = str(item.get("dt_original") or "unknown time")
            inline_url = f"/inline/{escape(asset_id)}"
            poster_url = f"/poster/{escape(asset_id)}"
            media_html = (
                f"""
                <video data-review-video controls controlslist="nodownload noplaybackrate" preload="metadata" poster="{poster_url}" muted playsinline>
                  <source src="{inline_url}">
                </video>
                """
                if media_kind == "video"
                else f'<img src="{poster_url}" alt="{escape(str(filename))}" loading="lazy" decoding="async">'
            )
            file_size = _human_file_size(item.get("orig_size")) or "Unknown size"
            match_label = _review_sequence_score_label(group)
            keep_link = f"/app/duplicates/{escape(group['id'])}/keep?asset_id={escape(asset_id)}"
            hide_link = f"/app/duplicates/{escape(group['id'])}/hide?asset_id={escape(asset_id)}"
            group_key = escape(group["id"])
            return f"""
            <article class="review-card-tile card" data-review-group="{group_key}" data-review-media-kind="{escape(media_kind)}">
              <div class="review-card-header">
                <div class="review-card-title">
                  <div class="review-card-filename">{escape(str(filename))}</div>
                  <span class="badge">{escape(match_label)}</span>
                </div>
                <div class="review-card-meta">
                  <span>{escape(file_size)}</span>
                  <span>{escape(dt_original)}</span>
                </div>
              </div>
              <div class="review-card-media{primary_class}" data-review-media-shell>
                {media_html}
                <div class="review-card-actions">
                  <a class="review-card-action keep" href="{keep_link}" title="Keep this one" aria-label="Keep this one">✓</a>
                  <a class="review-card-action delete" href="{hide_link}" title="Remove this one" aria-label="Remove this one">🗑</a>
                </div>
              </div>
            </article>
            """

        def _review_group_html(group: dict, *, featured: bool = False) -> str:
            items = manifest.list_duplicate_group_items(db_path, group["id"])
            if not items:
                return ""
            primary_asset_id = _group_primary_asset_id(group, items)
            tiles = "".join(_review_tile_html(group, item, primary=item["asset_id"] == primary_asset_id) for item in items)
            featured_class = " featured" if featured else ""
            return f"""
            <div class="review-group{featured_class}">
              <div class="review-grid">{tiles}</div>
            </div>
            """

        def _blurry_tile_html(row: dict) -> str:
            asset_id = escape(str(row["id"]))
            filename = escape(str(row.get("orig_filename") or row["id"]))
            file_size = escape(_human_file_size(row.get("orig_size")) or "Unknown size")
            dt_original = escape(str(row.get("dt_original") or "unknown time"))
            return f"""
            <article class="review-card-tile card">
              <div class="review-card-header">
                <div class="review-card-title">
                  <div class="review-card-filename">{filename}</div>
                  <span class="badge">Blur</span>
                </div>
                <div class="review-card-meta">
                  <span>{file_size}</span>
                  <span>{dt_original}</span>
                </div>
              </div>
              <div class="review-card-media primary">
                <img src="/poster/{asset_id}" alt="{filename}" loading="lazy" decoding="async">
              </div>
            </article>
            """

        exact_queue_html = "".join(
            _review_group_html(group, featured=index == 0)
            for index, group in enumerate(exact_groups)
        )
        sequence_queue_html = "".join(
            _review_group_html(group, featured=False)
            for group in sequence_groups
        )
        exact_delete_all_link = "/app/duplicates/delete-all?group_type=EXACT_SHA256"
        sequence_delete_all_link = "/app/duplicates/delete-all?group_type=NEAR_DUPLICATES"
        blurry_group_html = ""
        if blurry_items:
            blurry_queue_html = "".join(_blurry_tile_html(row) for row in blurry_items)
            blurry_group_html = f"""
          <section class="review-section">
            <div class="review-section-head">
              <div>
                <h2>Blurry items</h2>
                <p>These are flagged for quality review, but not treated as duplicates.</p>
              </div>
              <span class="badge">{len(blurry_items)} items</span>
            </div>
            <div class="review-shell">
              <div class="review-group">
                <div class="review-grid">{blurry_queue_html}</div>
              </div>
            </div>
          </section>
            """
        body_class = "review-page"
        empty_state = """
        <div class="review-empty">
          No review items right now.
        </div>
        """
        body = f"""
        <section class="review-sections">
          <section class="review-section">
            <div class="review-section-head">
              <div>
                <h2>Exact duplicates</h2>
                <p>These are 100% hash matches. Keep one and remove the rest.</p>
              </div>
              <div class="review-section-head-actions">
                <a href="{exact_delete_all_link}"><button class="btn secondary" type="button">Delete all</button></a>
                <span class="badge">{len(exact_groups)} groups</span>
              </div>
            </div>
            <div class="review-shell">
              {exact_queue_html or "<div class='review-empty'>No exact duplicates right now.</div>"}
            </div>
          </section>
          <section class="review-section">
            <div class="review-section-head">
              <div>
                <h2>Near duplicates</h2>
                <p>These are the ambiguous Snapchat-style clips and near matches.</p>
              </div>
              <div class="review-section-head-actions">
                <a href="{sequence_delete_all_link}"><button class="btn secondary" type="button">Delete all</button></a>
                <span class="badge">{len(sequence_groups)} groups</span>
              </div>
            </div>
            <div class="review-shell">
              {sequence_queue_html or "<div class='review-empty'>No near duplicates right now.</div>"}
            </div>
          </section>
          {blurry_group_html}
          {empty_state if not exact_groups and not sequence_groups and not blurry_items else ""}
        </section>
        <script>
          (function() {{
            const groups = new Map();
            document.querySelectorAll('[data-review-group]').forEach((card) => {{
              const groupKey = card.getAttribute('data-review-group');
              const kind = card.getAttribute('data-review-media-kind');
              if (!groups.has(groupKey)) groups.set(groupKey, []);
              if (kind === 'video') {{
                const video = card.querySelector('video');
                if (video) groups.get(groupKey).push(video);
              }}
            }});

            groups.forEach((videos) => {{
              if (!videos.length) return;
              let isSyncing = false;
              const syncPlay = (sourceVideo) => {{
                if (isSyncing) return;
                isSyncing = true;
                const currentTime = Number.isFinite(sourceVideo.currentTime) ? sourceVideo.currentTime : 0;
                videos.forEach((video) => {{
                  try {{
                    video.currentTime = currentTime;
                  }} catch (error) {{}}
                }});
                videos.forEach((video) => {{
                  if (video.paused || video.ended) {{
                    const playPromise = video.play();
                    if (playPromise && typeof playPromise.catch === 'function') playPromise.catch(() => {{}});
                  }}
                }});
                setTimeout(() => {{
                  isSyncing = false;
                }}, 0);
              }};

              videos.forEach((video) => {{
                video.addEventListener('play', () => {{
                  syncPlay(video);
                }});
                video.addEventListener('seeking', () => {{
                  if (isSyncing) return;
                  const currentTime = Number.isFinite(video.currentTime) ? video.currentTime : 0;
                  videos.forEach((otherVideo) => {{
                    if (otherVideo === video) return;
                    try {{
                      otherVideo.currentTime = currentTime;
                    }} catch (error) {{}}
                  }});
                }});
              }});
            }});
          }})();
        </script>
        """
        return _page("Review", body, history_html=_history_sidebar_html(db_path), body_class=body_class)

    def _render_system_page(current_config: AppConfig, db_path: Path, resolved: Path, message: Optional[str] = None) -> str:
        overview = manifest.get_overview(db_path)
        metadata_overview = manifest.get_metadata_overview(db_path)
        import_progress = manifest.get_import_progress(db_path)
        jobs = manifest.list_jobs(db_path, limit=6)
        tool_stack = build_tool_stack_report(current_config.tools)
        flash = f"<div class='flash'>{escape(message)}</div>" if message else ""
        sources_text = "\n".join(current_config.sources)
        stack_groups_html = "".join(
            f"""
            <section class="stack-group">
              <h3>{escape(group.get('name', 'Stack'))}</h3>
              <div class="stack-pills">
                {''.join(
                    f"<span class='stack-chip {'ready' if item.get('ready') else 'missing'}'><strong>{escape(str(item.get('name') or 'item'))}</strong><small>{escape('ready' if item.get('ready') else 'missing')}</small></span>"
                    for item in group.get('items', [])
                )}
              </div>
            </section>
            """
            for group in tool_stack.get("locked_stack", {}).get("groups", [])
        )
        body = f"""
        <div class="toolbar">
          <div class="title">
            <h1>System</h1>
            <p>Settings, metadata, health, and imports in one place.</p>
          </div>
        </div>
        {flash}
        <section class="system-stack">
          <section class="system-top">
            <section class="system-card"><h3>Imports</h3><div class="big">{import_progress.get('completion_pct', 0)}%</div><div class="asset-meta">{import_progress.get('ready_assets', 0)} ready</div></section>
            <section class="system-card"><h3>Coverage</h3><div class="big">{metadata_overview.get('average_metadata_score', 0)}</div><div class="asset-meta">average metadata coverage</div></section>
            <section class="system-card"><h3>Library</h3><div class="big">{overview.get('assets_total', 0)}</div><div class="asset-meta">items in the archive</div></section>
          </section>
          <section class="system-meta">
            <section class="system-card"><h3>Timestamps</h3><div class="big">{metadata_overview.get('timestamped_assets', 0)}</div><div class="asset-meta">items with capture time</div></section>
            <section class="system-card"><h3>Location</h3><div class="big">{metadata_overview.get('assets_with_location', 0)}</div><div class="asset-meta">items with GPS</div></section>
            <section class="system-card"><h3>People</h3><div class="big">{metadata_overview.get('assets_with_people', 0)}</div><div class="asset-meta">items with people metadata</div></section>
          </section>
          <section class="card section">
            <div class="section-header">
              <h2>Stack</h2>
              <p>Locked Photo Unifier tools and engines.</p>
            </div>
            <div class="stack-groups">
              {stack_groups_html or "<div class='muted'>No stack info available.</div>"}
            </div>
          </section>
          <section class="card section">
            <div class="section-header">
              <h2>Settings</h2>
              <p>Pick the folder where new photos and videos arrive.</p>
            </div>
            <form method="get" action="/app/settings/save" id="settings-form">
              <input type="hidden" name="workspace_root" value="{escape(current_config.workspace_root)}">
              <input type="hidden" name="sources_text" id="sources_text" value="{escape(sources_text)}">
              <label class="field" style="grid-column:1 / -1;">
                <span>Import folder</span>
                <div class="import-picker">
                  <div class="button-row">
                    <button class="btn secondary" type="button" id="choose-import-folder">Choose import folder…</button>
                  </div>
                  <div class="asset-meta" id="chosen-import-folder" style="font-size:.92rem;">{escape(sources_text or str(Path(current_config.workspace_root) / 'imports'))}</div>
                  <input type="file" id="import-folder-picker" style="display:none;" webkitdirectory directory multiple>
                </div>
              </label>
              <div class="button-row" style="margin-top: 14px;"><button class="btn" type="submit">Save</button></div>
            </form>
          </section>
          <section class="card section">
            <div class="section-header">
              <h2>Recent jobs</h2>
              <p>Small and readable, not a log dump.</p>
            </div>
            <table class="table">
              <thead><tr><th>Task</th><th>Status</th><th>Created</th><th>Finished</th><th>Error</th></tr></thead>
              <tbody>{''.join(f"<tr><td><strong>{escape(_friendly_job_label(row['job_type']))}</strong><div class='asset-meta'>{escape(_job_summary(row['job_type']))}</div></td><td><span class='badge'>{escape(row['status'])}</span></td><td>{escape(row['created_at'])}</td><td>{escape(row.get('finished_at') or '-')}</td><td>{escape(row.get('error_msg') or '-')}</td></tr>" for row in jobs) or "<tr><td colspan='5' class='muted'>No tasks yet.</td></tr>"}</tbody>
            </table>
          </section>
        </section>
        """
        body += """
        <script>
          (function() {
            const chooser = document.getElementById('choose-import-folder');
            const picker = document.getElementById('import-folder-picker');
            const field = document.getElementById('sources_text');
            const label = document.getElementById('chosen-import-folder');
            const form = document.getElementById('settings-form');
            if (!chooser || !picker || !field || !label || !form) return;
            chooser.addEventListener('click', function() {
              picker.click();
            });
            picker.addEventListener('change', function() {
              const files = Array.from(picker.files || []);
              if (!files.length) return;
              const roots = Array.from(new Set(files.map(file => {
                const rel = file.webkitRelativePath || file.name || '';
                return rel.split('/')[0];
              }).filter(Boolean)));
              const selected = roots[0] || '';
              if (!selected) return;
              field.value = selected;
              label.textContent = selected;
              form.submit();
            });
          })();
        </script>
        """
        return _page("System", body, history_html=_history_sidebar_html(db_path))

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
        _config, _resolved, db_path, _managed = _current_config()
        tool_stack = build_tool_stack_report(_config.tools)
        return {
            "ok": True,
            "pipeline": manifest.get_pipeline_health(db_path),
            "tools": tool_stack,
        }

    @app.get("/status")
    def status():
        _config, _resolved, db_path, _managed = _current_config()
        tool_stack = build_tool_stack_report(_config.tools)
        return {
            "overview": manifest.get_overview(db_path),
            "metadata": manifest.get_metadata_overview(db_path),
            "pipeline": manifest.get_pipeline_health(db_path),
            "tools": tool_stack,
        }

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
    def search(
        q: Optional[str] = None,
        similar_to: Optional[str] = None,
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ):
        _config, _resolved, db_path, _managed = _current_config()
        return intelligence.search_assets_semantic(
            db_path,
            q,
            similar_to=similar_to,
            limit=limit,
            offset=offset,
        )

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
    def assign_face(identity_label: str, face_id: str, return_to: Optional[str] = None):
        _config, _resolved, db_path, _managed = _current_config()
        identities = {item["label"]: item["id"] for item in manifest.list_face_identities(db_path, limit=500)}
        identity_id = identities.get(identity_label)
        if not identity_id:
            identity_id = manifest.create_face_identity(db_path, identity_label)
        manifest.assign_face_identity(db_path, face_id, identity_id)
        return RedirectResponse(url=return_to or "/app/people?message=Face+labeled", status_code=303)

    @app.get("/app/faces/reject")
    def reject_face(face_id: str, return_to: Optional[str] = None):
        _config, _resolved, db_path, _managed = _current_config()
        manifest.reject_face(db_path, face_id)
        return RedirectResponse(url=return_to or "/app/people?message=Detection+hidden", status_code=303)

    @app.get("/app/faces/dismiss")
    def dismiss_face(face_id: str, return_to: Optional[str] = None):
        _config, _resolved, db_path, _managed = _current_config()
        manifest.dismiss_face_suggestion(db_path, face_id)
        return RedirectResponse(url=return_to or "/app/people?message=Suggestion+dismissed", status_code=303)

    @app.get("/app/people/rename")
    def rename_people_identity(identity_id: str, label: str, return_to: Optional[str] = None):
        _config, _resolved, db_path, _managed = _current_config()
        cleaned_label = label.strip()
        target_url = return_to or "/app/people?message=Person+named"
        if not cleaned_label:
            return RedirectResponse(url=return_to or "/app/people?message=Name+required", status_code=303)
        try:
            identities = {item["id"]: item for item in manifest.list_face_identities(db_path, limit=5000)}
            if identity_id not in identities:
                raise HTTPException(status_code=404, detail="Identity not found")
            target_identity = next(
                (item for item in identities.values() if str(item["label"]).lower() == cleaned_label.lower()),
                None,
            )
            if target_identity and target_identity["id"] != identity_id:
                manifest.rename_face_identity(db_path, target_identity["id"], cleaned_label, status="CONFIRMED")
                manifest.merge_face_identities(db_path, identity_id, target_identity["id"])
                manifest.refresh_people_for_identity(db_path, target_identity["id"])
            else:
                manifest.rename_face_identity(db_path, identity_id, cleaned_label, status="CONFIRMED")
                manifest.refresh_people_for_identity(db_path, identity_id)
        except HTTPException:
            raise
        except Exception as exc:
            print(f"[people] rename failed identity_id={identity_id}: {exc}")
            return RedirectResponse(url=return_to or "/app/people?message=Could+not+save+name", status_code=303)
        return RedirectResponse(url=target_url, status_code=303)

    @app.get("/app/people/merge")
    def merge_people_identity(
        source_identity_id: str,
        target_identity_id: Optional[str] = None,
        target_label: Optional[str] = None,
        return_to: Optional[str] = None,
    ):
        _config, _resolved, db_path, _managed = _current_config()
        try:
            identities = manifest.list_face_identities(db_path, limit=5000)
            identity_by_id = {str(item["id"]): item for item in identities}
            identity_by_label = {str(item["label"]).strip().lower(): item for item in identities if str(item.get("label") or "").strip()}
            target = identity_by_id.get(str(target_identity_id)) if target_identity_id else None
            if not target and target_label:
                target = identity_by_label.get(target_label.strip().lower())
            if source_identity_id not in identity_by_id or not target:
                return RedirectResponse(url=return_to or "/app/people?message=Person+not+found", status_code=303)
            if str(target["id"]) == str(source_identity_id):
                return RedirectResponse(url=return_to or f"/app/people?identity_id={source_identity_id}&message=Nothing+to+merge", status_code=303)
            manifest.merge_face_identities(db_path, source_identity_id, str(target["id"]))
            manifest.rename_face_identity(db_path, str(target["id"]), str(target["label"]), status="CONFIRMED")
            manifest.apply_identity_to_assets(db_path, str(target["id"]))
            manifest.refresh_people_for_identity(db_path, str(target["id"]))
        except Exception as exc:
            print(f"[people] merge failed source_identity_id={source_identity_id} target_identity_id={target_identity_id}: {exc}")
            return RedirectResponse(url=return_to or "/app/people?message=Could+not+merge+people", status_code=303)
        target_url = return_to or f"/app/people?identity_id={target['id']}&message=People+merged"
        return RedirectResponse(url=target_url, status_code=303)

    @app.get("/app/people/merge-related")
    def merge_related_people_identities(
        identity_id: str,
        return_to: Optional[str] = None,
        min_score: float = 0.80,
    ):
        _config, _resolved, db_path, _managed = _current_config()
        try:
            target = manifest.get_face_identity(db_path, identity_id)
            if not target:
                return RedirectResponse(url=return_to or "/app/people?message=Person+not+found", status_code=303)
            related = manifest.list_related_identity_candidates(
                db_path,
                identity_id,
                limit=24,
                threshold=min_score,
            )
            merged = 0
            for cluster in related:
                source_identity_id = str(cluster.get("id") or "")
                if not source_identity_id or source_identity_id == identity_id:
                    continue
                manifest.merge_face_identities(db_path, source_identity_id, identity_id)
                merged += 1
            manifest.rename_face_identity(db_path, identity_id, str(target.get("label") or "Person"), status="CONFIRMED")
            manifest.apply_identity_to_assets(db_path, identity_id)
            manifest.refresh_people_for_identity(db_path, identity_id)
        except Exception as exc:
            print(f"[people] merge-related failed identity_id={identity_id}: {exc}")
            return RedirectResponse(url=return_to or "/app/people?message=Could+not+merge+related+clusters", status_code=303)
        target_url = return_to or f"/app/people?identity_id={identity_id}"
        message = "No+strong+related+clusters+found" if merged == 0 else f"Merged+{merged}+related+cluster{'s' if merged != 1 else ''}"
        separator = "&" if "?" in target_url else "?"
        return RedirectResponse(url=f"{target_url}{separator}message={message}", status_code=303)

    @app.get("/app/people/prepare-rerun")
    def prepare_people_rerun(return_to: Optional[str] = None):
        _config, _resolved, db_path, _managed = _current_config()
        try:
            result = manifest.prepare_face_rerun(db_path)
        except Exception as exc:
            print(f"[people] prepare-rerun failed: {exc}")
            target = return_to or "/app/people?message=Could+not+prepare+rerun"
            return RedirectResponse(url=target, status_code=303)
        message = quote_plus(
            "Prepared rerun"
            f" faces={result.get('cleared_faces', 0)}"
            f" clusters={result.get('removed_identities', 0)}"
            f" jobs={result.get('cleared_jobs', 0)}"
        )
        target = return_to or "/app/people"
        separator = "&" if "?" in target else "?"
        return RedirectResponse(url=f"{target}{separator}message={message}", status_code=303)

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
        return FileResponse(path, headers={"Cache-Control": "public, max-age=3600"})

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
            return FileResponse(path, headers={"Cache-Control": "public, max-age=86400, immutable"})
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
        return Response(content=svg, media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400, immutable"})

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
            raise HTTPException(status_code=404, detail="Library file missing")
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
            raise HTTPException(status_code=404, detail="Library file missing")
        return FileResponse(path, filename=item.get("orig_filename") or path.name)

    @app.get("/inline/{asset_id}")
    def inline(asset_id: str):
        _config, _resolved, db_path, managed_library_dir = _current_config()
        managed_rel = manifest.managed_path_for_asset(db_path, asset_id)
        if not managed_rel:
            raise HTTPException(status_code=404, detail="Managed asset not found")
        path = managed_library_dir / managed_rel
        if not path.exists():
            raise HTTPException(status_code=404, detail="Library file missing")
        guessed, _ = mimetypes.guess_type(str(path))
        return FileResponse(path, media_type=guessed or "application/octet-stream", headers={"Cache-Control": "public, max-age=3600"})

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

    @app.get("/app/history/{seq}/restore")
    def restore_history(seq: int, return_to: str = "/app/system"):
        _config, _resolved, db_path, _managed = _current_config()
        target = return_to if return_to.startswith("/") else "/app/system"
        separator = "&" if "?" in target else "?"
        try:
            row = manifest.restore_mutation_history_entry(db_path, seq)
            message = quote_plus(f"Restored: {row.get('summary') or row.get('action_type')}")
        except ValueError as exc:
            message = quote_plus(str(exc))
        return RedirectResponse(url=f"{target}{separator}message={message}", status_code=303)

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
        return RedirectResponse(url="/app/system?message=Settings+saved", status_code=303)

    @app.get("/", response_class=HTMLResponse)
    def dashboard(message: Optional[str] = None):
        current_config, resolved, db_path, _managed = _current_config()
        return _render_dashboard(current_config, db_path, resolved, message)

    @app.get("/app/system", response_class=HTMLResponse)
    def system_page(message: Optional[str] = None):
        current_config, resolved, db_path, _managed = _current_config()
        return _render_system_page(current_config, db_path, resolved, message)

    @app.get("/app/settings", response_class=HTMLResponse)
    def settings_page():
        return RedirectResponse(url="/app/system", status_code=303)

    @app.get("/app/jobs", response_class=HTMLResponse)
    def jobs_page():
        return RedirectResponse(url="/app/system", status_code=303)

    @app.get("/app/metadata", response_class=HTMLResponse)
    def metadata_page(field: Optional[str] = None):
        return RedirectResponse(url="/app/system", status_code=303)

    @app.get("/app/people", response_class=HTMLResponse)
    def people_page(
        message: Optional[str] = None,
        person: Optional[str] = None,
        identity_id: Optional[str] = None,
        show_review: int = 0,
        view: str = "confirmed",
    ):
        _config, _resolved, db_path, _managed = _current_config()
        return _render_people_page(db_path, message, person, identity_id, bool(show_review), view)

    @app.get("/app/faces", response_class=HTMLResponse)
    def faces_page():
        return RedirectResponse(url="/app/people", status_code=303)

    @app.get("/app/assets", response_class=HTMLResponse)
    def assets_page(
        q: Optional[str] = None,
        show_hidden: int = 0,
        page: int = 1,
        per_page: int = 100,
        sort: str = "recent",
        media_type: Optional[str] = None,
        review_state: Optional[str] = None,
        favorite_only: int = 0,
        partial: int = 0,
    ):
        _config, _resolved, db_path, _managed = _current_config()
        per_page = max(24, min(int(per_page or 100), 300))
        page = max(1, int(page or 1))
        partial_mode = bool(partial)
        total_count = 0
        total_pages = 1
        items: list[dict[str, object]]
        if q:
            search_data = intelligence.search_assets_semantic(
                db_path,
                q,
                limit=per_page + 1,
                offset=(page - 1) * per_page,
                include_hidden=True,
                media_type=media_type or None,
            )
            items = list(search_data.get("items", []))
            total_count = int(search_data.get("count", len(items)))
            total_pages = max(1, (total_count + per_page - 1) // per_page) if total_count else 1
            if total_count and page > total_pages:
                page = total_pages
                search_data = intelligence.search_assets_semantic(
                    db_path,
                    q,
                    limit=per_page + 1,
                    offset=(page - 1) * per_page,
                    include_hidden=True,
                    media_type=media_type or None,
                )
                items = list(search_data.get("items", []))
                total_count = int(search_data.get("count", len(items)))
                total_pages = max(1, (total_count + per_page - 1) // per_page) if total_count else 1
        else:
            total_count = manifest.count_assets(
                db_path,
                query=None,
                include_hidden=True,
                media_type=media_type or None,
                favorite_only=bool(favorite_only),
            )
            total_pages = max(1, (total_count + per_page - 1) // per_page) if total_count else 1
            page = min(page, total_pages)
            offset = (page - 1) * per_page
            items = manifest.list_assets(
                db_path,
                limit=per_page + 1,
                offset=offset,
                query=None,
                include_hidden=True,
                sort=sort,
                media_type=media_type or None,
                review_state=review_state or None,
                favorite_only=bool(favorite_only),
            )
        if q:
            total_pages = max(1, (total_count + per_page - 1) // per_page) if total_count else 1
            offset = (page - 1) * per_page
        known_people = [row["label"] for row in manifest.list_face_identities(db_path, limit=200, status="CONFIRMED")]
        if media_type:
            items = [row for row in items if row.get("media_type") == media_type]
        if not q and sort == "rated":
            items = sorted(items, key=lambda row: (row.get("user_rating") or 0, row.get("score") or 0), reverse=True)
        elif not q and sort == "recent":
            items = sorted(items, key=lambda row: (row.get("dt_original") or "", row.get("score") or 0), reverse=True)
        has_next = len(items) > per_page
        items = items[:per_page]
        visible_count = len(items)
        if total_count == 0 and visible_count:
            total_count = visible_count
            total_pages = max(1, (total_count + per_page - 1) // per_page)
        start_item = 0 if not visible_count else offset + 1
        end_item = offset + visible_count
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
        def _asset_query_url(*, page_number: int, include_partial: bool = False, include_page: bool = True) -> str:
            params = []
            if include_page:
                params.append(f"page={page_number}")
            params.append(f"per_page={per_page}")
            if q:
                params.append(f"q={quote_plus(str(q))}")
            if sort:
                params.append(f"sort={quote_plus(str(sort))}")
            if media_type:
                params.append(f"media_type={quote_plus(str(media_type))}")
            if review_state:
                params.append(f"review_state={quote_plus(str(review_state))}")
            if favorite_only:
                params.append("favorite_only=1")
            if include_partial:
                params.append("partial=1")
            return "/app/assets?" + "&".join(params)

        base_return_url = _asset_query_url(page_number=1, include_page=False)
        sort_options = """
        <option value="recent" {recent}>Recent</option>
        <option value="rated" {rated}>Rated</option>
        """.format(
            recent="selected" if sort == "recent" else "",
            rated="selected" if sort == "rated" else "",
        )
        summary_text = (
            f"Showing {start_item:,}-{end_item:,} of {total_count:,}"
            if total_count
            else "No items found"
        )
        load_more_label = "Scroll to load more" if has_next else "All assets loaded"
        chunk_html = "".join(grids) or "<section class='card muted'>No assets found.</section>"
        if partial_mode:
            headers = {"X-Next-Page": str(page + 1) if has_next else "0"}
            return HTMLResponse(chunk_html, headers=headers)
        people_suggestions = "".join(
            f'<button class="library-name-chip" type="button" data-library-name="{escape(str(name))}">{escape(str(name))}</button>'
            for name in known_people[:16]
        )
        body = f"""
        <div class="toolbar">
          <div class="title">
            <h1>Library</h1>
            <p>Your archive in a gallery-first view, grouped by month like a photo library.</p>
            <div class="status-pill">{escape(summary_text)}</div>
          </div>
          <div class="button-row">
            <button class="btn secondary library-select-fab" type="button" id="library-select-mode-toggle">Select</button>
          </div>
        </div>
        <form class="library-select-bar" id="library-select-bar" method="get" action="/app/library/tag-people">
          <input type="hidden" name="asset_ids" id="library-selected-asset-ids" value="">
          <input type="hidden" name="return_to" value="{escape(base_return_url)}">
          <div class="library-select-bar-head">
            <span class="library-select-bar-count" id="library-selected-count">0 selected</span>
            <input type="text" name="person" id="library-selected-person" list="library-people-suggestions" placeholder="Type a name and press Enter">
            <button class="btn secondary" type="button" id="library-clear-selection">Clear</button>
          </div>
          <div class="library-name-suggestions">
            {people_suggestions or "<span class='inline-form-note'>No people labels yet.</span>"}
          </div>
          <datalist id="library-people-suggestions">{''.join(f'<option value="{escape(name)}">' for name in known_people)}</datalist>
        </form>
        <form class="searchbar" method="get" action="/app/assets" data-auto-submit="true">
          <input type="search" name="q" placeholder="Search by dog, receipt, London Bridge, people, or places" value="{escape(q or '')}">
          <input type="hidden" name="per_page" value="{per_page}">
          <select name="media_type" style="border:1px solid var(--line);border-radius:16px;background:rgba(255,255,255,.92);padding:12px 14px;font:inherit;">
            <option value="">All Media</option>
            <option value="image" {"selected" if media_type == "image" else ""}>Images</option>
            <option value="raw" {"selected" if media_type == "raw" else ""}>RAW</option>
            <option value="video" {"selected" if media_type == "video" else ""}>Video</option>
            <option value="audio" {"selected" if media_type == "audio" else ""}>Audio</option>
          </select>
          <select name="sort" style="border:1px solid var(--line);border-radius:16px;background:rgba(255,255,255,.92);padding:12px 14px;font:inherit;">
            {sort_options}
          </select>
        </form>
        <div id="library-infinite-root" class="library-infinite-root" data-next-page="{page + 1 if has_next else 0}">
          {chunk_html}
        </div>
        <div class="library-load-more" id="library-load-more" data-has-next="{1 if has_next else 0}">
          <div class="spinner" aria-hidden="true"></div>
          <div>{escape(load_more_label)}</div>
        </div>
        <script>
          (function() {{
            const toggle = document.getElementById('library-select-mode-toggle');
            const clear = document.getElementById('library-clear-selection');
            const bar = document.getElementById('library-select-bar');
            const hiddenInput = document.getElementById('library-selected-asset-ids');
            const count = document.getElementById('library-selected-count');
            const personInput = document.getElementById('library-selected-person');
            const nameChips = Array.from(document.querySelectorAll('[data-library-name]'));
            const root = document.getElementById('library-infinite-root');
            const sentinel = document.getElementById('library-load-more');
            const cards = new Set();
            if (!toggle || !bar || !hiddenInput || !count || !root) return;
            const selected = new Set();
            let pressCard = null;
            let pressPoint = null;
            let dragging = false;
            let dragShouldSelect = true;
            let lastDraggedId = '';
            let suppressNextClick = false;
            let selectMode = false;
            let nextPage = Number(root.dataset.nextPage || '0');
            let loadingMore = false;

            function refreshCards() {{
              cards.clear();
              root.querySelectorAll('.asset-card[data-asset-id]').forEach((card) => cards.add(card));
            }}

            function findCardFromPoint(x, y) {{
              const el = document.elementFromPoint(x, y);
              return el ? el.closest('.asset-card[data-asset-id]') : null;
            }}

            function selectCard(card, nextValue) {{
              if (!card) return;
              const assetId = card.dataset.assetId || '';
              if (!assetId) return;
              if (nextValue) selected.add(assetId);
              else selected.delete(assetId);
              sync();
            }}

            function sync() {{
              hiddenInput.value = Array.from(selected).join(',');
              count.textContent = `${{selected.size}} selected`;
              cards.forEach((card) => {{
                const selectedNow = selected.has(card.dataset.assetId || '');
                card.classList.toggle('is-selected', selectedNow);
                const button = card.querySelector('.asset-select-toggle');
                if (button) button.textContent = selectedNow ? '✓' : '+';
              }});
              bar.classList.toggle('active', selectMode);
              document.body.classList.toggle('library-select-mode', selectMode);
              toggle.textContent = selectMode ? 'Done' : 'Select';
              bar.querySelectorAll('button, input').forEach((field) => {{
                field.disabled = !selectMode;
              }});
              if (personInput) personInput.disabled = !selectMode;
            }}

            function applyDragSelection(card) {{
              if (!card) return;
              const assetId = card.dataset.assetId || '';
              if (!assetId || assetId === lastDraggedId) return;
              lastDraggedId = assetId;
              selectCard(card, dragShouldSelect);
            }}

            toggle.addEventListener('click', function() {{
              selectMode = !selectMode;
              if (!selectMode) selected.clear();
              sync();
            }});

            if (clear) {{
              clear.addEventListener('click', function() {{
                selected.clear();
                sync();
              }});
            }}

            if (nameChips.length && personInput) {{
              nameChips.forEach((chip) => {{
                chip.addEventListener('click', () => {{
                  personInput.value = chip.getAttribute('data-library-name') || '';
                  personInput.focus();
                }});
              }});
            }}

            async function loadMore() {{
              if (!nextPage || loadingMore) return;
              loadingMore = true;
              const url = new URL(window.location.href);
              url.searchParams.set('page', String(nextPage));
              url.searchParams.set('partial', '1');
              url.searchParams.set('per_page', '{per_page}');
              try {{
                const response = await fetch(url.toString(), {{ headers: {{ 'X-Requested-With': 'fetch' }} }});
                if (!response.ok) return;
                const html = await response.text();
                const template = document.createElement('template');
                template.innerHTML = html;
                template.content.childNodes.forEach((node) => {{
                  root.appendChild(node);
                }});
                refreshCards();
                nextPage = Number(response.headers.get('X-Next-Page') || '0');
                root.dataset.nextPage = String(nextPage || 0);
                if (!nextPage && observer) observer.disconnect();
              }} finally {{
                loadingMore = false;
              }}
            }}

            const observer = ('IntersectionObserver' in window && sentinel) ? new IntersectionObserver((entries) => {{
              entries.forEach((entry) => {{
                if (entry.isIntersecting) loadMore();
              }});
            }}, {{ rootMargin: '600px 0px' }}) : null;

            if (observer && sentinel) observer.observe(sentinel);

            root.addEventListener('pointerdown', function(event) {{
              if (!selectMode) return;
              const card = event.target.closest('.asset-card[data-asset-id]');
              if (!card) return;
              event.preventDefault();
              pressCard = card;
              pressPoint = {{ x: event.clientX, y: event.clientY }};
              dragging = false;
              lastDraggedId = card.dataset.assetId || '';
              dragShouldSelect = !selected.has(lastDraggedId);
            }});

            document.addEventListener('pointermove', function(event) {{
              if (!selectMode || !pressCard) return;
              const dx = pressPoint ? Math.abs(event.clientX - pressPoint.x) : 0;
              const dy = pressPoint ? Math.abs(event.clientY - pressPoint.y) : 0;
              if (!dragging && dx < 6 && dy < 6) return;
              if (!dragging) {{
                dragging = true;
                selectCard(pressCard, dragShouldSelect);
                lastDraggedId = pressCard.dataset.assetId || '';
              }}
              const card = findCardFromPoint(event.clientX, event.clientY);
              if (card) applyDragSelection(card);
            }});

            document.addEventListener('pointerup', function() {{
              if (!selectMode || !pressCard) return;
              if (!dragging) {{
                const assetId = pressCard.dataset.assetId || '';
                if (assetId) selectCard(pressCard, !selected.has(assetId));
              }}
              pressCard = null;
              pressPoint = null;
              dragging = false;
              lastDraggedId = '';
              suppressNextClick = true;
            }});

            document.addEventListener('click', function(event) {{
              if (!suppressNextClick) return;
              suppressNextClick = false;
              event.preventDefault();
              event.stopPropagation();
            }});

            refreshCards();
            sync();
          }})();
        </script>
        """
        return _page("Library", body, history_html=_history_sidebar_html(db_path))

    @app.get("/app/assets/{asset_id}", response_class=HTMLResponse)
    def asset_detail_page(asset_id: str):
        _config, _resolved, db_path, _managed = _current_config()
        item = manifest.get_asset(db_path, asset_id)
        if not item:
            raise HTTPException(status_code=404, detail="Asset not found")
        neighbors = manifest.list_asset_neighbors(db_path, asset_id, sort="recent")
        similar = intelligence.search_assets_semantic(db_path, similar_to=asset_id, limit=8).get("items", [])
        metadata_rows = manifest.list_asset_metadata(db_path, asset_id)
        text_rows = manifest.list_extraction_results(db_path, asset_id, result_type="ocr")
        artifacts = manifest.list_asset_artifacts(db_path, asset_id)
        try:
            current_people = json.loads(item.get("people_json") or "[]")
        except Exception:
            current_people = []
        if not isinstance(current_people, list):
            current_people = []
        current_people = [str(person) for person in current_people if str(person).strip()]
        known_people = [row["label"] for row in manifest.list_face_identities(db_path, limit=200)]
        viewer = _viewer_html(asset_id, item)
        rating_controls = _rating_controls_html(asset_id, item)
        address_value = _resolve_asset_address(db_path, asset_id, item, metadata_rows)
        shortcut_script = _keyboard_shortcuts_script(
            neighbors.get("prev", {}).get("id") if neighbors.get("prev") else None,
            neighbors.get("next", {}).get("id") if neighbors.get("next") else None,
            asset_id,
        )
        fields = [
            ("Captured", item.get("dt_original") or "Unknown"),
            ("File size", _human_file_size(item.get("orig_size")) or "Unknown"),
        ]
        edit_location_value = address_value
        current_people_links = []
        for name in current_people:
            current_people_links.append(
                f"""
                <a class="status-pill" href="/app/assets/{escape(asset_id)}/people/remove?person={quote_plus(name)}" title="Click to remove {escape(name)}">
                  {escape(name)}
                </a>
                """
            )
        body = f"""
        <div class="toolbar">
          <div class="title">
            <h1>{escape(item.get('title') or 'Asset')}</h1>
            <p>Review the file, clean up the metadata, and tag people without bouncing between panels.</p>
          </div>
          <div class="button-row">
            {f'<a href="/app/assets/{escape(neighbors["prev"]["id"])}"><button class="btn secondary" type="button">Previous</button></a>' if neighbors.get("prev") else ''}
            {f'<a href="/app/assets/{escape(neighbors["next"]["id"])}"><button class="btn secondary" type="button">Next</button></a>' if neighbors.get("next") else ''}
            <a href="/app/search?similar_to={escape(asset_id)}"><button class="btn secondary" type="button">Similar</button></a>
          </div>
        </div>
        <section class="asset-detail-workspace">
          <div class="asset-viewer-panel">
            {viewer}
          </div>
          <section class="card asset-detail-panel">
            <div class="asset-detail-shell">
              <section class="asset-header-card">
                <h2>Details</h2>
                <p>Everything here follows the same pattern: grey fields are reference values, bright fields are yours to edit.</p>
              </section>
              <form class="detail-form" method="get" action="/app/assets/{asset_id}/edit">
                <div class="asset-meta-grid">
                  <label class="asset-field">
                    <span class="asset-field-label">{escape(fields[0][0])}</span>
                    <input type="text" value="{escape(str(fields[0][1]))}" readonly>
                  </label>
                  <label class="asset-field">
                    <span class="asset-field-label">{escape(fields[1][0])}</span>
                    <input type="text" value="{escape(str(fields[1][1]))}" readonly>
                  </label>
                  <label class="asset-field full">
                    <span class="asset-field-label">Title</span>
                    <input type="text" name="title" placeholder="Add a title" value="{escape(item.get('title') or '')}">
                  </label>
                  <label class="asset-field full">
                    <span class="asset-field-label">Address</span>
                    <input type="text" name="address" placeholder="10 Downing Street, London" value="{escape(edit_location_value)}">
                  </label>
                  <label class="asset-field full">
                    <span class="asset-field-label">Description</span>
                    <textarea name="description" rows="4" placeholder="Add a description">{escape(item.get('description') or '')}</textarea>
                  </label>
                </div>
                <div class="button-row" style="margin-top: 2px;">
                  <button class="btn" type="submit">Save details</button>
                </div>
              </form>
              <section class="asset-actions-card">
                <div class="section-header">
                  <h2>Quick Rank</h2>
                  <p>One to five is the whole review flow here.</p>
                </div>
                <div class="button-row">{rating_controls}</div>
              </section>
              <section class="asset-actions-card">
                <div class="section-header">
                  <h2>People</h2>
                  <p>Press Enter to tag someone. Click an existing tag to remove it.</p>
                </div>
                <form class="asset-inline-form" method="get" action="/app/assets/{asset_id}/people" data-auto-submit="true">
                  <input type="search" name="person" list="people-suggestions-{asset_id}" placeholder="Type a name like Avery" value="">
                  <datalist id="people-suggestions-{asset_id}">
                    {''.join(f'<option value="{escape(name)}">' for name in known_people)}
                  </datalist>
                </form>
                <div class="asset-tag-strip">
                  {''.join(current_people_links) or "<span class='status-pill empty'>No people tagged yet.</span>"}
                </div>
              </section>
            </div>
          </section>
        </section>
        {shortcut_script}
        """
        return _page("Asset Detail", body, history_html=_history_sidebar_html(db_path), body_class="asset-detail-page")

    @app.get("/app/assets/{asset_id}/review")
    def review_asset(asset_id: str, rating: Optional[int] = None, favorite: Optional[int] = None, state: Optional[str] = None):
        _config, _resolved, db_path, _managed = _current_config()
        item = manifest.get_asset(db_path, asset_id)
        if not item:
            raise HTTPException(status_code=404, detail="Asset not found")
        manifest.set_asset_review(
            db_path,
            asset_id,
            user_rating=rating,
            review_state=state,
            favorite=bool(int(favorite)) if favorite is not None else None,
        )
        return RedirectResponse(url=f"/app/assets/{asset_id}?message=Rank+saved", status_code=303)

    @app.get("/app/assets/{asset_id}/people")
    def tag_person(asset_id: str, person: str):
        _config, _resolved, db_path, _managed = _current_config()
        item = manifest.get_asset(db_path, asset_id)
        if not item:
            raise HTTPException(status_code=404, detail="Asset not found")
        manifest.add_asset_person(db_path, asset_id, person)
        return RedirectResponse(url=f"/app/assets/{asset_id}?message=Person+tagged", status_code=303)

    @app.get("/app/library/tag-people")
    def bulk_tag_people(asset_ids: str, person: str, return_to: Optional[str] = None):
        _config, _resolved, db_path, _managed = _current_config()
        cleaned_person = str(person or "").strip()
        if not cleaned_person:
            return RedirectResponse(url=return_to or "/app/assets?message=Name+required", status_code=303)
        ids = [item.strip() for item in str(asset_ids or "").split(",") if item.strip()]
        if not ids:
            return RedirectResponse(url=return_to or "/app/assets?message=Nothing+selected", status_code=303)
        manifest.create_face_identity(db_path, cleaned_person, status="CONFIRMED")
        tagged = 0
        for asset_id in ids:
            item = manifest.get_asset(db_path, asset_id)
            if not item:
                continue
            manifest.add_asset_person(db_path, asset_id, cleaned_person)
            tagged += 1
        target = return_to or "/app/assets"
        separator = "&" if "?" in target else "?"
        return RedirectResponse(url=f"{target}{separator}message=Tagged+{tagged}", status_code=303)

    @app.get("/app/assets/{asset_id}/people/remove")
    def untag_person(asset_id: str, person: str):
        _config, _resolved, db_path, _managed = _current_config()
        item = manifest.get_asset(db_path, asset_id)
        if not item:
            raise HTTPException(status_code=404, detail="Asset not found")
        manifest.remove_asset_person(db_path, asset_id, person)
        return RedirectResponse(url=f"/app/assets/{asset_id}?message=Person+removed", status_code=303)

    @app.get("/app/assets/{asset_id}/edit")
    def edit_asset_details(asset_id: str, title: Optional[str] = None, description: Optional[str] = None, address: Optional[str] = None):
        _config, _resolved, db_path, _managed = _current_config()
        item = manifest.get_asset(db_path, asset_id)
        if not item:
            raise HTTPException(status_code=404, detail="Asset not found")
        kwargs: dict[str, object] = {}
        if title is not None:
            kwargs["title"] = title.strip() or None
        if description is not None:
            kwargs["description"] = description.strip() or None
        if kwargs:
            manifest.apply_metadata_updates(db_path, asset_id, source_name="manual_edit", **kwargs)
        if address is not None:
            cleaned_address = address.strip()
            if cleaned_address:
                manifest.set_metadata_field(
                    db_path,
                    asset_id,
                    field_name="location",
                    value=cleaned_address,
                    source_name="manual_edit",
                    source_field="Address",
                    is_canonical=True,
                )
            else:
                manifest.delete_metadata_field(
                    db_path,
                    asset_id,
                    field_name="location",
                    source_name="manual_edit",
                    source_field="Address",
                )
        return RedirectResponse(url=f"/app/assets/{asset_id}?message=Details+saved", status_code=303)

    @app.get("/app/search", response_class=HTMLResponse)
    def search_page(
        q: Optional[str] = None,
        similar_to: Optional[str] = None,
        sort: str = "relevance",
        media_type: Optional[str] = None,
        review_state: Optional[str] = None,
    ):
        _config, _resolved, db_path, _managed = _current_config()
        results = intelligence.search_assets_semantic(db_path, q, similar_to=similar_to, limit=120).get("items", [])
        if media_type:
            results = [row for row in results if row.get("media_type") == media_type]
        if review_state:
            results = [row for row in results if (row.get("review_state") or "new") == review_state]
        if sort == "rated":
            results = sorted(results, key=lambda row: (row.get("user_rating") or 0, row.get("score") or 0), reverse=True)
        elif sort == "recent":
            results = sorted(results, key=lambda row: (row.get("dt_original") or "", row.get("score") or 0), reverse=True)
        source_label = "similar items" if similar_to else "semantic matches"
        ref = manifest.get_asset(db_path, similar_to) if similar_to else None
        if similar_to:
            title = "Similar photos"
        else:
            title = "Search"
        cards = "".join(_search_result_card(row, query=q, similar_to=similar_to) for row in results[:80])
        saved_searches = manifest.list_saved_searches(db_path, limit=12)
        saved_cards_parts: list[str] = []
        for row in saved_searches:
            try:
                row_params = json.loads(row.get("params_json") or "{}")
            except Exception:
                row_params = {}
            search_parts: list[str] = []
            for key in ("q", "similar_to", "sort", "media_type", "review_state"):
                value = row_params.get(key)
                if value:
                    search_parts.append(f"{key}={escape(str(value))}")
            href = "/app/search"
            if search_parts:
                href += "?" + "&".join(search_parts)
            saved_cards_parts.append(
                f'<a href="{href}"><button class="btn secondary" type="button">{escape(row.get("label") or "Saved Search")}</button></a>'
            )
        saved_cards = "".join(saved_cards_parts)
        save_link = f"/app/search/save?q={escape(q or '')}&similar_to={escape(similar_to or '')}&sort={escape(sort)}"
        if media_type:
            save_link += f"&media_type={escape(media_type)}"
        if review_state:
            save_link += f"&review_state={escape(review_state)}"
        source_panel = ""
        if similar_to:
            source_name = escape(ref.get("orig_filename") if ref else similar_to)
            source_panel = f"""
            <section class="card section" style="display:grid;grid-template-columns:100px minmax(0,1fr) auto;gap:14px;align-items:center;margin-bottom:18px;">
              <img class="album-cover" src="/poster/{escape(similar_to)}" alt="{source_name}" style="width:100px;height:100px;">
              <div>
                <div class="asset-name" style="font-size:1rem;">Similar to {source_name}</div>
                <div class="asset-meta" style="margin-top:4px;">Open the source photo or keep narrowing the match.</div>
              </div>
              <a href="/app/assets/{escape(similar_to)}"><button class="btn secondary" type="button">Open photo</button></a>
            </section>
            """
        body = f"""
        <div class="toolbar">
          <div class="title">
            <h1>{escape(title)}</h1>
            <p>{escape(source_label)} with ranked results, ratings, and compare actions.</p>
          </div>
        </div>
        {source_panel}
        <form class="searchbar" method="get" action="/app/search" data-auto-submit="true">
          <input type="search" name="q" placeholder="Search filenames, OCR text, people, captions, places" value="{escape(q or '')}">
          <input type="hidden" name="similar_to" value="{escape(similar_to or '')}">
          <select name="media_type" style="border:1px solid var(--line);border-radius:16px;background:rgba(255,255,255,.92);padding:12px 14px;font:inherit;">
            <option value="">All Media</option>
            <option value="image" {"selected" if media_type == "image" else ""}>Images</option>
            <option value="raw" {"selected" if media_type == "raw" else ""}>RAW</option>
            <option value="video" {"selected" if media_type == "video" else ""}>Video</option>
            <option value="audio" {"selected" if media_type == "audio" else ""}>Audio</option>
          </select>
          <select name="review_state" style="border:1px solid var(--line);border-radius:16px;background:rgba(255,255,255,.92);padding:12px 14px;font:inherit;">
            <option value="">Any Review State</option>
            <option value="reviewed" {"selected" if review_state == "reviewed" else ""}>Reviewed</option>
            <option value="review" {"selected" if review_state == "review" else ""}>Needs Review</option>
            <option value="new" {"selected" if review_state == "new" else ""}>New</option>
          </select>
          <select name="sort" style="border:1px solid var(--line);border-radius:16px;background:rgba(255,255,255,.92);padding:12px 14px;font:inherit;">
            <option value="relevance" {"selected" if sort == "relevance" else ""}>Relevance</option>
            <option value="recent" {"selected" if sort == "recent" else ""}>Recent</option>
            <option value="rated" {"selected" if sort == "rated" else ""}>Rated</option>
          </select>
        </form>
        <section class="button-row" style="margin-bottom:16px;">
          <a href="/app/review"><button class="btn secondary" type="button">Review Queue</button></a>
          <a href="/app/assets"><button class="btn secondary" type="button">Library</button></a>
          <a href="{save_link}"><button class="btn secondary" type="button">Save Search</button></a>
        </section>
        {'' if similar_to else f'''<section class="card section">
          <div class="section-header">
            <h2>Saved Searches</h2>
            <p>Fast return points for common queries and smart albums.</p>
          </div>
          <div class="button-row">
            {saved_cards or "<span class='muted'>No saved searches yet.</span>"}
          </div>
        </section>'''}
        <section class="asset-grid">
          {cards or "<section class='card muted'>No results yet. Try a broader query or search from a specific asset.</section>"}
        </section>
        """
        return _page("Search", body, history_html=_history_sidebar_html(db_path))

    @app.get("/app/search/save")
    def save_search(
        q: Optional[str] = None,
        similar_to: Optional[str] = None,
        sort: str = "relevance",
        media_type: Optional[str] = None,
        review_state: Optional[str] = None,
    ):
        _config, _resolved, db_path, _managed = _current_config()
        params = {
            "q": q or "",
            "similar_to": similar_to or "",
            "sort": sort or "relevance",
            "media_type": media_type or "",
            "review_state": review_state or "",
        }
        label_parts = [part for part in [q, media_type, review_state] if part]
        label = " / ".join(label_parts) if label_parts else "Saved Search"
        manifest.save_saved_search(db_path, label=label, params=params)
        suffix = []
        if q:
            suffix.append(f"q={escape(q)}")
        if similar_to:
            suffix.append(f"similar_to={escape(similar_to)}")
        if sort:
            suffix.append(f"sort={escape(sort)}")
        if media_type:
            suffix.append(f"media_type={escape(media_type)}")
        if review_state:
            suffix.append(f"review_state={escape(review_state)}")
        query_string = "&".join(suffix)
        return RedirectResponse(url=f"/app/search?{query_string}&message=Search+saved", status_code=303)

    @app.get("/app/review", response_class=HTMLResponse)
    def review_page(message: Optional[str] = None):
        _config, _resolved, db_path, _managed = _current_config()
        return _render_review_page(db_path, message=message)

    @app.get("/app/duplicates/{group_id}/review", response_class=HTMLResponse)
    def duplicate_group_review(group_id: str):
        return RedirectResponse(url="/app/review", status_code=303)

    @app.get("/app/compare", response_class=HTMLResponse)
    def compare_page(left: str, right: str):
        return RedirectResponse(url="/app/review", status_code=303)

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
        return RedirectResponse(url="/app/review?message=Duplicate+group+resolved", status_code=303)

    @app.get("/app/duplicates/{group_id}/keep-all")
    def keep_all_duplicate_group(group_id: str):
        _config, _resolved, db_path, _managed = _current_config()
        try:
            manifest.keep_all_duplicate_group(db_path, group_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return RedirectResponse(url="/app/review?message=Marked+duplicate+group+as+keep+all", status_code=303)

    @app.get("/app/duplicates/delete-all")
    def delete_all_duplicate_groups(group_type: str):
        _config, _resolved, db_path, _managed = _current_config()
        normalized = str(group_type or "").strip().upper()
        if normalized == "EXACT_SHA256":
            target_types = ("EXACT_SHA256",)
        elif normalized in {"NEAR_DUPLICATES", "NEAR_VISUAL", "NEAR_AHASH", "SNAPCHAT_SEQUENCE"}:
            target_types = ("SNAPCHAT_SEQUENCE", "NEAR_VISUAL", "NEAR_AHASH")
        else:
            raise HTTPException(status_code=400, detail="Unknown duplicate group type")
        groups_deleted = 0
        for target_type in target_types:
            for group in manifest.list_duplicate_groups(db_path, limit=500, group_type=target_type, status="OPEN"):
                try:
                    manifest.skip_duplicate_group(db_path, group["id"])
                    groups_deleted += 1
                except ValueError:
                    continue
        return RedirectResponse(
            url=f"/app/review?message=Removed+{groups_deleted}+groups+from+review",
            status_code=303,
        )

    @app.get("/app/duplicates/{group_id}/skip")
    def skip_duplicate_group(group_id: str):
        _config, _resolved, db_path, _managed = _current_config()
        try:
            manifest.skip_duplicate_group(db_path, group_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return RedirectResponse(url="/app/review?message=Skipped+duplicate+group", status_code=303)

    @app.get("/app/duplicates/{group_id}/keep")
    def keep_duplicate_item(group_id: str, asset_id: str):
        _config, _resolved, db_path, _managed = _current_config()
        items = manifest.list_duplicate_group_items(db_path, group_id)
        if not any(item["asset_id"] == asset_id for item in items):
            raise HTTPException(status_code=404, detail="Duplicate item not found")
        try:
            manifest.resolve_duplicate_group(db_path, group_id, asset_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return RedirectResponse(url="/app/review?message=Duplicate+kept", status_code=303)

    @app.get("/app/duplicates/{group_id}/hide")
    def hide_duplicate_item(group_id: str, asset_id: str):
        _config, _resolved, db_path, _managed = _current_config()
        items = manifest.list_duplicate_group_items(db_path, group_id)
        if not any(item["asset_id"] == asset_id for item in items):
            raise HTTPException(status_code=404, detail="Duplicate item not found")
        try:
            manifest.hide_duplicate_group_item(db_path, group_id, asset_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        remaining = manifest.list_duplicate_group_items(db_path, group_id)
        if len(remaining) <= 1:
            return RedirectResponse(url="/app/review?message=Duplicate+item+removed", status_code=303)
        return RedirectResponse(url=f"/app/duplicates/{group_id}/review?message=Duplicate+item+removed", status_code=303)

    @app.get("/app/duplicates", response_class=HTMLResponse)
    def duplicates_page(message: Optional[str] = None):
        return RedirectResponse(url=f"/app/review{('?message=' + quote_plus(message)) if message else ''}", status_code=303)

    @app.get("/app/cleanup", response_class=HTMLResponse)
    def cleanup_page():
        return RedirectResponse(url="/app/review", status_code=303)

    return app
