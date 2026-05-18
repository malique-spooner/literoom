from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from .config import DEFAULT_CONFIG_PATH, load_config
from .derivatives import build_derivatives
from .metadata_repair import repair_metadata
from . import dedupe
from . import faces
from .metadata import gather, manifest, master


def load_runtime(config_path: Path | str = DEFAULT_CONFIG_PATH):
    config, resolved = load_config(config_path)
    config.ensure_workspace_dirs(resolved)
    db_path = config.db_path(resolved)
    manifest.init_db(db_path)
    return config, resolved, db_path


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
        result = {"job_id": job_id, "rows_upserted": total}
        manifest.complete_job(db_path, job_id, result)
        return result
    except Exception as exc:
        manifest.fail_job(db_path, job_id, str(exc))
        raise


def run_plan_library(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict:
    config, resolved, db_path = load_runtime(config_path)
    job_id = manifest.create_job(db_path, "plan_library", {"naming": config.pipeline.managed_naming})
    manifest.start_job(db_path, job_id)
    try:
        planned = manifest.plan_targets(db_path, naming=config.pipeline.managed_naming)
        result = {"job_id": job_id, "planned_assets": planned}
        manifest.complete_job(db_path, job_id, result)
        return result
    except Exception as exc:
        manifest.fail_job(db_path, job_id, str(exc))
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
        )
        result = {"job_id": job_id, "processed_assets": count}
        manifest.complete_job(db_path, job_id, result)
        return result
    except Exception as exc:
        manifest.fail_job(db_path, job_id, str(exc))
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
        result = {"job_id": job_id, **result}
        manifest.complete_job(db_path, job_id, result)
        return result
    except Exception as exc:
        manifest.fail_job(db_path, job_id, str(exc))
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
        result = {"job_id": job_id, **result}
        manifest.complete_job(db_path, job_id, result)
        return result
    except Exception as exc:
        manifest.fail_job(db_path, job_id, str(exc))
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
        )
        result = {"job_id": job_id, **result}
        manifest.complete_job(db_path, job_id, result)
        return result
    except Exception as exc:
        manifest.fail_job(db_path, job_id, str(exc))
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
        result = {"job_id": job_id, **result}
        manifest.complete_job(db_path, job_id, result)
        return result
    except Exception as exc:
        manifest.fail_job(db_path, job_id, str(exc))
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
    dedupe = run_dedupe_exact(config_path, limit=batch_limit)
    near_dedupe = run_dedupe_near(config_path, limit=batch_limit)
    return {
        "ingest": ingest,
        "plan": planned,
        "build": built,
        "derivatives": derivatives,
        "metadata": metadata,
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
        )
        result = {"job_id": job_id, **result}
        manifest.complete_job(db_path, job_id, result)
        return result
    except Exception as exc:
        manifest.fail_job(db_path, job_id, str(exc))
        raise
