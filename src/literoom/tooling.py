from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Dict, List, Optional


CORE_BINARY_TOOLS = {
    "exiftool": "exiftool",
    "ffmpeg": "ffmpeg",
    "tesseract": "tesseract",
}

LOCKED_STACK_GROUPS = (
    (
        "Metadata",
        (
            {"name": "exiftool", "kind": "binary", "fallback": "exiftool", "config_field": "exiftool"},
        ),
    ),
    (
        "Media",
        (
            {"name": "ffmpeg", "kind": "binary", "fallback": "ffmpeg", "config_field": "ffmpeg"},
            {"name": "libvips", "kind": "binary", "fallback": "vips", "config_field": "vips"},
        ),
    ),
    (
        "Faces",
        (
            {"name": "OpenCV", "kind": "python", "module": "cv2", "detail": "OpenCV image/video helpers"},
            {"name": "InsightFace", "kind": "python", "module": "insightface", "model_field": "face_model"},
        ),
    ),
    (
        "Objects + meaning",
        (
            {"name": "OpenCLIP", "kind": "python", "module": "open_clip", "model_field": "clip_model"},
        ),
    ),
    (
        "Search",
        (
            {"name": "SQLite FTS + vector search", "kind": "builtin", "detail": "Built into Literoom"},
            {"name": "FAISS", "kind": "python", "module": "faiss"},
        ),
    ),
)

OPTIONAL_PYTHON_TOOLS = {
    "open_clip": "open_clip",
    "faiss": "faiss",
    "insightface": "insightface",
    "cv2": "cv2",
}


def resolve_binary_tool(explicit_path: Optional[str], fallback_name: str) -> Optional[str]:
    if explicit_path:
        path = Path(explicit_path)
        if path.exists():
            return str(path)
    return shutil.which(fallback_name)


def binary_tool_status(explicit_path: Optional[str], fallback_name: str) -> Dict[str, Any]:
    resolved = resolve_binary_tool(explicit_path, fallback_name)
    return {
        "kind": "binary",
        "configured_path": explicit_path,
        "resolved_path": resolved,
        "ready": bool(resolved),
    }


@lru_cache(maxsize=None)
def _import_status(module_name: str) -> tuple[bool, Optional[str]]:
    try:
        if find_spec(module_name) is None:
            return False, f"No module named '{module_name}'"
        return True, None
    except Exception as exc:
        return False, str(exc)


def python_tool_status(module_name: str, *, detail: Optional[str] = None) -> Dict[str, Any]:
    ready, error = _import_status(module_name)
    payload: Dict[str, Any] = {
        "kind": "python",
        "module": module_name,
        "ready": ready,
    }
    if detail:
        payload["detail"] = detail
    if error:
        payload["error"] = error
    return payload


def _sqlite_stack_status() -> Dict[str, Any]:
    ready = True
    detail = "Built into Literoom"
    try:
        sqlite3.connect(":memory:").close()
    except Exception:
        ready = False
        detail = "sqlite3 unavailable"
    return {
        "kind": "builtin",
        "ready": ready,
        "detail": detail,
    }


def discover_default_binary_paths() -> Dict[str, Optional[str]]:
    paths = {name: shutil.which(fallback) for name, fallback in CORE_BINARY_TOOLS.items()}
    paths["libvips"] = shutil.which("vips")
    return paths


def _config_value(config_tools: Any, field_name: str) -> Optional[str]:
    if hasattr(config_tools, "__dataclass_fields__"):
        return getattr(config_tools, field_name, None)
    if isinstance(config_tools, dict):
        return config_tools.get(field_name)
    return getattr(config_tools, field_name, None)


def _locked_stack_item(config_tools: Any, spec: Dict[str, Any]) -> Dict[str, Any]:
    kind = spec["kind"]
    name = spec["name"]
    if kind == "binary":
        configured_path = _config_value(config_tools, spec["config_field"])
        status = binary_tool_status(configured_path, spec["fallback"])
        status.update({
            "name": name,
            "detail": spec.get("detail") or status["resolved_path"] or "missing",
        })
        return status
    if kind == "python":
        detail = spec.get("detail")
        model_field = spec.get("model_field")
        if model_field:
            model_value = _config_value(config_tools, model_field)
            if model_value:
                detail = f"model={model_value}"
        status = python_tool_status(spec["module"], detail=detail)
        status.update({
            "name": name,
            "resolved_path": spec["module"] if status["ready"] else None,
        })
        return status
    if kind == "builtin":
        status = _sqlite_stack_status()
        status.update({
            "name": name,
            "resolved_path": "sqlite3 (stdlib)" if status["ready"] else None,
        })
        return status
    raise ValueError(f"Unknown locked stack item kind: {kind}")


def build_tool_stack_report(config_tools: Any) -> Dict[str, Any]:
    data = asdict(config_tools) if hasattr(config_tools, "__dataclass_fields__") else dict(config_tools)
    required = {
        name: binary_tool_status(data.get(name), fallback)
        for name, fallback in CORE_BINARY_TOOLS.items()
    }
    optional = {
        "libvips": binary_tool_status(data.get("vips"), "vips"),
    }
    python_tools = {
        name: python_tool_status(module_name)
        for name, module_name in OPTIONAL_PYTHON_TOOLS.items()
    }

    locked_stack_groups: List[Dict[str, Any]] = []
    locked_stack_index: Dict[str, Dict[str, Any]] = {}
    for group_name, specs in LOCKED_STACK_GROUPS:
        items = [_locked_stack_item(config_tools, spec) for spec in specs]
        for item in items:
            locked_stack_index[item["name"]] = item
        locked_stack_groups.append({"name": group_name, "items": items})

    core_ready = all(item["ready"] for item in required.values())
    locked_stack_ready = all(item["ready"] for item in locked_stack_index.values())

    report = {
        "core": required,
        "required": required,
        "optional": optional,
        "python": python_tools,
        "locked_stack": {
            "groups": locked_stack_groups,
            "items": locked_stack_index,
        },
        "core_ready": core_ready,
        "required_ready": core_ready,
        "locked_stack_ready": locked_stack_ready,
        "exiftool": required["exiftool"],
        "ffmpeg": required["ffmpeg"],
        "tesseract": required["tesseract"],
        "libvips": optional["libvips"],
        "OpenCV": locked_stack_index["OpenCV"],
        "InsightFace": locked_stack_index["InsightFace"],
        "OpenCLIP": locked_stack_index["OpenCLIP"],
        "SQLite FTS + vector search": locked_stack_index["SQLite FTS + vector search"],
        "open_clip": python_tools["open_clip"],
        "insightface": python_tools["insightface"],
        "cv2": python_tools["cv2"],
    }
    return report


def _status_label(item: Dict[str, Any]) -> str:
    if item.get("ready"):
        return "ready"
    return "missing"


def format_tool_stack_report(report: Dict[str, Any]) -> str:
    lines = [
        f"Core runtime: {'ready' if report.get('core_ready') else 'missing'}",
        f"Locked stack: {'ready' if report.get('locked_stack_ready') else 'missing'}",
        "",
        "Core tools:",
    ]
    for name, item in report.get("core", {}).items():
        lines.append(f"- {name}: {_status_label(item)}")
    if report.get("optional"):
        lines.append("")
        lines.append("Media helpers:")
        for name, item in report.get("optional", {}).items():
            lines.append(f"- {name}: {_status_label(item)}")
    lines.append("")
    lines.append("Locked stack:")
    for group in report.get("locked_stack", {}).get("groups", []):
        lines.append(f"{group.get('name', 'Stack')}:")
        for item in group.get("items", []):
            detail = item.get("detail")
            suffix = f" ({detail})" if detail and detail not in {"ready", "missing"} else ""
            lines.append(f"- {item.get('name', 'item')}: {_status_label(item)}{suffix}")
    return "\n".join(lines)
