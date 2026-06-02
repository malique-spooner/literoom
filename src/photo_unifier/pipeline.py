from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from .config import DEFAULT_CONFIG_PATH, load_config
from .derivatives import build_derivatives
from .export import export_built_assets
from . import intelligence
from .metadata_repair import repair_metadata
from . import dedupe
from . import faces
from .metadata import gather, manifest, master


def load_runtime(config_path: Path | str = DEFAULT_CONFIG_PATH):
    config, resolved = load_config(config_path)
    config.prepare_runtime_environment(resolved)
    config.ensure_workspace_dirs(resolved)
    db_path = config.db_path(resolved)
    manifest.init_db(db_path)
    return config, resolved, db_path


def _complete_job(db_path, job_id: str, stage: str, result: dict) -> dict:
    payload = {"stage": stage, **result}
    manifest.complete_job(db_path, job_id, payload)
    return payload


def _fail_job(db_path, job_id: str, stage: str, exc: Exception, *, metrics: Optional[dict] = None) -> None:
    message = str(exc)
    retryable = True
    if any(token in message.lower() for token in ["no ingest sources configured", "not found", "missing config"]):
        retryable = False
    manifest.fail_job(db_path, job_id, f"{stage}: {message}", retryable=retryable, metrics=metrics)


def run_ingest(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    *,
    source_tag: Optional[str] = None,
    sources: Optional[Sequence[Path | str]] = None,
) -> dict:
    config, resolved, db_path = load_runtime(config_path)
    resolved_sources = [Path(p).resolve() for p in sources] if sources else config.resolved_sources(resolved)
    if not resolved_sources:
        raise ValueError("No ingest sources configured.")
    job_id = manifest.create_job(db_path, "ingest", {"sources": [str(p) for p in resolved_sources]})
    manifest.start_job(db_path, job_id)
    try:
        total = gather.run(
            list(resolved_sources),
            db_path=db_path,
            source_hint=source_tag,
            batch_size=config.pipeline.batch_size,
            job_id=job_id,
        )
        snapchat_sequence_groups = manifest.build_snapchat_sequence_groups(db_path)
        return _complete_job(
            db_path,
            job_id,
            "ingest",
            {
                "job_id": job_id,
                "rows_upserted": total,
                "snapchat_sequence_groups": snapchat_sequence_groups,
            },
        )
    except Exception as exc:
        _fail_job(db_path, job_id, "ingest", exc, metrics={"sources": [str(p) for p in resolved_sources]})
        raise


def run_plan_library(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict:
    config, resolved, db_path = load_runtime(config_path)
    job_id = manifest.create_job(db_path, "plan_library", {"naming": config.pipeline.managed_naming})
    manifest.start_job(db_path, job_id)
    try:
        planned = manifest.plan_targets(db_path, naming=config.pipeline.managed_naming)
        return _complete_job(db_path, job_id, "plan_library", {"job_id": job_id, "planned_assets": planned})
    except Exception as exc:
        _fail_job(db_path, job_id, "plan_library", exc, metrics={"naming": config.pipeline.managed_naming})
        raise


def run_build_library(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    *,
    limit: Optional[int] = None,
    force: bool = False,
) -> dict:
    config, resolved, db_path = load_runtime(config_path)
    job_id = manifest.create_job(db_path, "build_library", {"limit": limit, "force": force})
    manifest.start_job(db_path, job_id)
    try:
        count = master.build(
            master_dir=config.managed_library_dir(resolved),
            db_path=db_path,
            exiftool_path=Path(config.tools.exiftool) if config.tools.exiftool else None,
            source_roots=config.resolved_sources(resolved),
            force=force,
            limit=limit,
            job_id=job_id,
        )
        return _complete_job(db_path, job_id, "build_library", {"job_id": job_id, "processed_assets": count, "force": force})
    except Exception as exc:
        _fail_job(db_path, job_id, "build_library", exc, metrics={"limit": limit, "force": force})
        raise


def run_build_derivatives(config_path: Path | str = DEFAULT_CONFIG_PATH, *, limit: Optional[int] = None) -> dict:
    config, resolved, db_path = load_runtime(config_path)
    job_id = manifest.create_job(db_path, "build_derivatives", {"limit": limit})
    manifest.start_job(db_path, job_id)
    try:
        result = build_derivatives(
            db_path=db_path,
            managed_library_dir=config.managed_library_dir(resolved),
            derivatives_dir=config.derivatives_dir(resolved),
            image_thumbnail_size=config.pipeline.image_thumbnail_size,
            video_preview_offset_seconds=config.pipeline.video_preview_offset_seconds,
            ffmpeg_path=config.tools.ffmpeg,
            limit=limit,
        )
        return _complete_job(db_path, job_id, "build_derivatives", {"job_id": job_id, **result})
    except Exception as exc:
        _fail_job(db_path, job_id, "build_derivatives", exc, metrics={"limit": limit})
        raise


def run_dedupe_exact(config_path: Path | str = DEFAULT_CONFIG_PATH, *, limit: Optional[int] = None) -> dict:
    config, resolved, db_path = load_runtime(config_path)
    job_id = manifest.create_job(db_path, "dedupe_exact", {"limit": limit})
    manifest.start_job(db_path, job_id)
    try:
        result = dedupe.run_exact(
            db_path=db_path,
            managed_library_dir=config.managed_library_dir(resolved),
            limit=limit,
        )
        return _complete_job(db_path, job_id, "dedupe_exact", {"job_id": job_id, **result})
    except Exception as exc:
        _fail_job(db_path, job_id, "dedupe_exact", exc, metrics={"limit": limit})
        raise


def run_dedupe_near(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    *,
    limit: Optional[int] = None,
    max_distance: int = 6,
) -> dict:
    config, resolved, db_path = load_runtime(config_path)
    job_id = manifest.create_job(db_path, "dedupe_near", {"limit": limit, "max_distance": max_distance})
    manifest.start_job(db_path, job_id)
    try:
        result = dedupe.run_near(
            db_path=db_path,
            managed_library_dir=config.managed_library_dir(resolved),
            derivatives_dir=config.derivatives_dir(resolved),
            limit=limit,
            max_distance=max_distance,
            clip_model=config.tools.clip_model,
        )
        return _complete_job(db_path, job_id, "dedupe_near", {"job_id": job_id, **result})
    except Exception as exc:
        _fail_job(db_path, job_id, "dedupe_near", exc, metrics={"limit": limit, "max_distance": max_distance})
        raise


def run_metadata_repair(config_path: Path | str = DEFAULT_CONFIG_PATH, *, limit: Optional[int] = None) -> dict:
    config, resolved, db_path = load_runtime(config_path)
    job_id = manifest.create_job(db_path, "metadata_repair", {"limit": limit})
    manifest.start_job(db_path, job_id)
    try:
        result = repair_metadata(
            db_path=db_path,
            managed_library_dir=config.managed_library_dir(resolved),
            exiftool_path=Path(config.tools.exiftool) if config.tools.exiftool else None,
            source_roots=config.resolved_sources(resolved),
            limit=limit,
        )
        return _complete_job(db_path, job_id, "metadata_repair", {"job_id": job_id, **result})
    except Exception as exc:
        _fail_job(db_path, job_id, "metadata_repair", exc, metrics={"limit": limit})
        raise


def run_content_extraction(config_path: Path | str = DEFAULT_CONFIG_PATH, *, limit: Optional[int] = None) -> dict:
    config, resolved, db_path = load_runtime(config_path)
    job_id = manifest.create_job(db_path, "content_extraction", {"limit": limit})
    manifest.start_job(db_path, job_id)
    try:
        result = intelligence.extract_asset_content(
            db_path=db_path,
            managed_library_dir=config.managed_library_dir(resolved),
            limit=limit,
            whisper_model=config.tools.whisper_model,
        )
        return _complete_job(db_path, job_id, "content_extraction", {"job_id": job_id, **result})
    except Exception as exc:
        _fail_job(db_path, job_id, "content_extraction", exc, metrics={"limit": limit})
        raise


def run_export_library(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    *,
    destination: Path | str,
    limit: Optional[int] = None,
    overwrite: bool = False,
) -> dict:
    config, resolved, db_path = load_runtime(config_path)
    destination_dir = Path(destination).expanduser().resolve()
    job_id = manifest.create_job(
        db_path,
        "export_library",
        {"destination": str(destination_dir), "limit": limit, "overwrite": overwrite},
    )
    manifest.start_job(db_path, job_id)
    try:
        result = export_built_assets(
            db_path=db_path,
            managed_library_dir=config.managed_library_dir(resolved),
            destination_dir=destination_dir,
            limit=limit,
            overwrite=overwrite,
        )
        return _complete_job(db_path, job_id, "export_library", {"job_id": job_id, **result})
    except Exception as exc:
        _fail_job(
            db_path,
            job_id,
            "export_library",
            exc,
            metrics={"destination": str(destination_dir), "limit": limit, "overwrite": overwrite},
        )
        raise


def run_full_pipeline(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    *,
    source_tag: Optional[str] = None,
    sources: Optional[Sequence[Path | str]] = None,
    batch_limit: Optional[int] = None,
) -> dict:
    ingest = run_ingest(config_path, source_tag=source_tag, sources=sources)
    planned = run_plan_library(config_path)
    built = run_build_library(config_path, limit=batch_limit)
    derivatives = run_build_derivatives(config_path, limit=batch_limit)
    metadata = run_metadata_repair(config_path, limit=batch_limit)
    content = run_content_extraction(config_path, limit=batch_limit)
    dedupe = run_dedupe_exact(config_path, limit=batch_limit)
    near_dedupe = run_dedupe_near(config_path, limit=batch_limit)
    return {
        "ingest": ingest,
        "plan": planned,
        "build": built,
        "derivatives": derivatives,
        "metadata": metadata,
        "content": content,
        "dedupe": dedupe,
        "near_dedupe": near_dedupe,
    }


def run_pilot_pipeline(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    *,
    source_tag: Optional[str] = None,
    sources: Optional[Sequence[Path | str]] = None,
    batch_limit: Optional[int] = 250,
) -> dict:
    before_config, resolved, db_path = load_runtime(config_path)
    before_metadata = manifest.get_metadata_overview(db_path)
    before_assets = manifest.get_overview(db_path)
    before_duplicates = manifest.list_duplicate_groups(db_path, limit=5000)
    before_faces = manifest.get_face_overview(db_path)
    ingest = run_ingest(config_path, source_tag=source_tag, sources=sources)
    planned = run_plan_library(config_path)
    built = run_build_library(config_path, limit=batch_limit)
    derivatives = run_build_derivatives(config_path, limit=batch_limit)
    metadata = run_metadata_repair(config_path, limit=batch_limit)
    content = run_content_extraction(config_path, limit=batch_limit)
    dedupe_exact = run_dedupe_exact(config_path, limit=batch_limit)
    dedupe_near = run_dedupe_near(config_path, limit=batch_limit)
    face_result = run_face_detection(config_path, limit=batch_limit, force=True)
    after_metadata = manifest.get_metadata_overview(db_path)
    after_assets = manifest.get_overview(db_path)
    after_duplicates = manifest.list_duplicate_groups(db_path, limit=5000)
    after_faces = manifest.get_face_overview(db_path)
    return {
        "before": {"metadata": before_metadata, "assets": before_assets},
        "after": {"metadata": after_metadata, "assets": after_assets},
        "before_counts": {"duplicates": len(before_duplicates), "faces": before_faces},
        "after_counts": {"duplicates": len(after_duplicates), "faces": after_faces},
        "ingest": ingest,
        "plan": planned,
        "build": built,
        "derivatives": derivatives,
        "metadata": metadata,
        "content": content,
        "dedupe_exact": dedupe_exact,
        "dedupe_near": dedupe_near,
        "faces": face_result,
    }


def run_face_detection(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    *,
    limit: Optional[int] = None,
    force: bool = False,
) -> dict:
    config, resolved, db_path = load_runtime(config_path)
    job_id = manifest.create_job(db_path, "face_detection", {"limit": limit, "force": force})
    manifest.start_job(db_path, job_id)
    try:
        result = faces.detect_faces(
            db_path=db_path,
            managed_library_dir=config.managed_library_dir(resolved),
            limit=limit,
            force=force,
            face_model=config.tools.face_model,
        )
        return _complete_job(db_path, job_id, "face_detection", {"job_id": job_id, **result})
    except Exception as exc:
        _fail_job(
            db_path,
            job_id,
            "face_detection",
            exc,
            metrics={"limit": limit, "force": force, "face_model": config.tools.face_model},
        )
        raise
