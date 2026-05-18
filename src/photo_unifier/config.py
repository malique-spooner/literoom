from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    yaml = None  # type: ignore[assignment]


DEFAULT_CONFIG_PATH = Path(".photo_unifier/config.yaml")


@dataclass
class WorkspacePaths:
    db_path: str = ".photo_unifier/manifest.sqlite"
    managed_library_dir: str = ".photo_unifier/library"
    derivatives_dir: str = ".photo_unifier/derived"
    logs_dir: str = ".photo_unifier/logs"
    temp_dir: str = ".photo_unifier/tmp"


@dataclass
class ToolPaths:
    exiftool: Optional[str] = None
    ffmpeg: Optional[str] = None
    whisper_model: Optional[str] = None
    clip_model: Optional[str] = None
    face_model: Optional[str] = None


@dataclass
class PipelineConfig:
    batch_size: int = 500
    max_workers: int = 4
    image_thumbnail_size: int = 512
    video_preview_offset_seconds: int = 1
    managed_naming: str = "{YYYY}/{YYYY-MM}/{YYYYMMDD}_{hhmmss}_{shortid}"
    auto_sync_enabled: bool = False
    auto_sync_interval_seconds: int = 300


@dataclass
class ThresholdConfig:
    image_phash_distance: int = 8
    video_frame_phash_distance: int = 10


@dataclass
class AppConfig:
    workspace_root: str = "."
    sources: List[str] = field(default_factory=list)
    paths: WorkspacePaths = field(default_factory=WorkspacePaths)
    tools: ToolPaths = field(default_factory=ToolPaths)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)

    def resolve_root(self, config_path: Path) -> Path:
        configured = Path(self.workspace_root)
        if configured.is_absolute():
            return configured
        return (config_path.parent / configured).resolve()

    def resolve_path(self, config_path: Path, value: str) -> Path:
        p = Path(value)
        if p.is_absolute():
            return p
        return (self.resolve_root(config_path) / p).resolve()

    def db_path(self, config_path: Path) -> Path:
        return self.resolve_path(config_path, self.paths.db_path)

    def managed_library_dir(self, config_path: Path) -> Path:
        return self.resolve_path(config_path, self.paths.managed_library_dir)

    def derivatives_dir(self, config_path: Path) -> Path:
        return self.resolve_path(config_path, self.paths.derivatives_dir)

    def logs_dir(self, config_path: Path) -> Path:
        return self.resolve_path(config_path, self.paths.logs_dir)

    def temp_dir(self, config_path: Path) -> Path:
        return self.resolve_path(config_path, self.paths.temp_dir)

    def resolved_sources(self, config_path: Path, extra_sources: Optional[Iterable[str]] = None) -> List[Path]:
        raw = list(self.sources)
        if extra_sources:
            raw.extend(str(item) for item in extra_sources)
        out: List[Path] = []
        for item in raw:
            p = Path(item)
            if not p.is_absolute():
                p = self.resolve_root(config_path) / p
            out.append(p.resolve())
        return out

    def ensure_workspace_dirs(self, config_path: Path) -> None:
        self.db_path(config_path).parent.mkdir(parents=True, exist_ok=True)
        self.managed_library_dir(config_path).mkdir(parents=True, exist_ok=True)
        self.derivatives_dir(config_path).mkdir(parents=True, exist_ok=True)
        self.logs_dir(config_path).mkdir(parents=True, exist_ok=True)
        self.temp_dir(config_path).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _merge_dataclass(cls, raw: Dict[str, Any]):
    valid = {key: raw[key] for key in raw if key in cls.__dataclass_fields__}
    return cls(**valid)


def _simple_scalar(value: str) -> Any:
    text = value.strip()
    if text == "":
        return ""
    if text.lower() in {"null", "none", "~"}:
        return None
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if re.fullmatch(r"-?\d+", text):
        try:
            return int(text)
        except Exception:
            return text
    if re.fullmatch(r"-?\d+\.\d+", text):
        try:
            return float(text)
        except Exception:
            return text
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    return text


def _fallback_load_config(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: List[tuple[int, Any]] = [(-1, root)]
    current_key: Optional[str] = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1] if stack else root
        if stripped.startswith("- "):
            item = _simple_scalar(stripped[2:])
            if isinstance(parent, list):
                parent.append(item)
            elif current_key and isinstance(root.get(current_key), list):
                root[current_key].append(item)
            continue
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                container: Any = []
                if key in {"paths", "tools", "pipeline", "thresholds"}:
                    container = {}
                elif key == "sources":
                    container = []
                else:
                    container = {}
                if isinstance(parent, dict):
                    parent[key] = container
                current_key = key
                stack.append((indent, container))
            else:
                if isinstance(parent, dict):
                    parent[key] = _simple_scalar(value)
                current_key = key
    return root


def _fallback_dump_value(value: Any, indent: int = 0) -> List[str]:
    pad = " " * indent
    if isinstance(value, dict):
        lines: List[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.extend(_fallback_dump_value(item, indent + 2))
            else:
                lines.append(f"{pad}{key}: {_fallback_dump_scalar(item)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.extend(_fallback_dump_value(item, indent + 2))
            else:
                lines.append(f"{pad}- {_fallback_dump_scalar(item)}")
        return lines
    return [f"{pad}{_fallback_dump_scalar(value)}"]


def _fallback_dump_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or re.search(r"[:#\n\r\t]|^\s|\s$", text):
        return json.dumps(text)
    return text


def default_config() -> AppConfig:
    return AppConfig()


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> tuple[AppConfig, Path]:
    path = Path(config_path).resolve()
    if not path.exists():
        cfg = default_config()
        return cfg, path

    raw_text = path.read_text(encoding="utf-8")
    if yaml is not None:
        payload = yaml.safe_load(raw_text) or {}
    else:
        payload = _fallback_load_config(raw_text)
    cfg = AppConfig(
        workspace_root=payload.get("workspace_root", "."),
        sources=list(payload.get("sources", [])),
        paths=_merge_dataclass(WorkspacePaths, payload.get("paths", {})),
        tools=_merge_dataclass(ToolPaths, payload.get("tools", {})),
        pipeline=_merge_dataclass(PipelineConfig, payload.get("pipeline", {})),
        thresholds=_merge_dataclass(ThresholdConfig, payload.get("thresholds", {})),
    )
    return cfg, path


def write_default_config(config_path: Path | str = DEFAULT_CONFIG_PATH, *, overwrite: bool = False) -> Path:
    cfg = default_config()
    path = Path(config_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return path
    if yaml is not None:
        path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
    else:
        path.write_text("\n".join(_fallback_dump_value(cfg.to_dict())) + "\n", encoding="utf-8")
    return path


def save_config(config: AppConfig, config_path: Path | str = DEFAULT_CONFIG_PATH) -> Path:
    path = Path(config_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
    else:
        path.write_text("\n".join(_fallback_dump_value(config.to_dict())) + "\n", encoding="utf-8")
    return path


__all__ = [
    "AppConfig",
    "DEFAULT_CONFIG_PATH",
    "PipelineConfig",
    "ThresholdConfig",
    "ToolPaths",
    "WorkspacePaths",
    "default_config",
    "load_config",
    "save_config",
    "write_default_config",
]
