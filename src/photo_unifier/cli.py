#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import click

from .config import DEFAULT_CONFIG_PATH, load_config, write_default_config
from .metadata import manifest
from .metadata import master
from .audit import build_audit_report, format_audit_report
from .tooling import build_tool_stack_report, format_tool_stack_report
from .pipeline import (
    load_runtime as _load_runtime,
    run_build_derivatives,
    run_build_library,
    run_dedupe_exact,
    run_dedupe_near,
    run_content_extraction,
    run_face_detection,
    run_ingest,
    run_metadata_repair,
    run_plan_library,
    run_pilot_pipeline,
)


def _load_runtime_config(config_path: Path) -> tuple[object, Path]:
    config, resolved, _db_path = _load_runtime(config_path)
    return config, resolved


@click.group()
def app():
    """Photo Unifier CLI."""


@app.command("init-config")
@click.option("--config", "config_path", default=str(DEFAULT_CONFIG_PATH), type=click.Path(path_type=Path))
@click.option("--overwrite/--no-overwrite", default=False, show_default=True)
def init_config_cmd(config_path: Path, overwrite: bool):
    path = write_default_config(config_path, overwrite=overwrite)
    click.echo(f"Wrote config: {path}")


@app.command("status")
@click.option("--config", "config_path", default=str(DEFAULT_CONFIG_PATH), type=click.Path(path_type=Path))
def status_cmd(config_path: Path):
    config, resolved = _load_runtime_config(config_path)
    overview = manifest.get_overview(config.db_path(resolved))
    for key, value in overview.items():
        click.echo(f"{key}: {value}")


@app.command("doctor")
@click.option("--config", "config_path", default=str(DEFAULT_CONFIG_PATH), type=click.Path(path_type=Path))
def doctor_cmd(config_path: Path):
    config, resolved = _load_runtime_config(config_path)
    report = build_tool_stack_report(config.tools)
    click.echo(format_tool_stack_report(report))
    if not report.get("required_ready", False):
        raise click.ClickException("Required media tools are missing.")


@app.command("jobs")
@click.option("--config", "config_path", default=str(DEFAULT_CONFIG_PATH), type=click.Path(path_type=Path))
@click.option("--limit", default=20, show_default=True)
def jobs_cmd(config_path: Path, limit: int):
    config, resolved = _load_runtime_config(config_path)
    for row in manifest.list_jobs(config.db_path(resolved), limit=limit):
        click.echo(
            f"{row['id']}  {row['job_type']}  {row['status']}  "
            f"created={row['created_at']} finished={row['finished_at'] or '-'}"
        )


@app.command("ingest")
@click.argument("paths", nargs=-1, type=click.Path(path_type=Path, exists=True))
@click.option("--config", "config_path", default=str(DEFAULT_CONFIG_PATH), type=click.Path(path_type=Path))
@click.option("--source-tag", default=None)
def ingest_cmd(paths: Tuple[Path, ...], config_path: Path, source_tag: Optional[str]):
    try:
        result = run_ingest(config_path, source_tag=source_tag, sources=list(paths) or None)
        click.echo(f"Ingested rows: {result['rows_upserted']}")
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@app.command("plan-library")
@click.option("--config", "config_path", default=str(DEFAULT_CONFIG_PATH), type=click.Path(path_type=Path))
def plan_library_cmd(config_path: Path):
    try:
        result = run_plan_library(config_path)
        click.echo(f"Planned library items: {result['planned_assets']}")
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@app.command("build-library")
@click.option("--config", "config_path", default=str(DEFAULT_CONFIG_PATH), type=click.Path(path_type=Path))
@click.option("--limit", default=None, type=int)
@click.option("--force/--no-force", default=False, show_default=True)
def build_library_cmd(config_path: Path, limit: Optional[int], force: bool):
    try:
        result = run_build_library(config_path, limit=limit, force=force)
        click.echo(f"Built library items: {result['processed_assets']}")
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@app.command("build-previews")
@click.option("--config", "config_path", default=str(DEFAULT_CONFIG_PATH), type=click.Path(path_type=Path))
@click.option("--limit", default=None, type=int)
def build_previews_cmd(config_path: Path, limit: Optional[int]):
    try:
        result = run_build_derivatives(config_path, limit=limit)
        click.echo(
            f"Previews built={result['built']} failed={result['failed']} skipped={result['skipped']}"
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@app.command("build-derivatives")
@click.option("--config", "config_path", default=str(DEFAULT_CONFIG_PATH), type=click.Path(path_type=Path))
@click.option("--limit", default=None, type=int)
def build_derivatives_cmd(config_path: Path, limit: Optional[int]):
    try:
        result = run_build_derivatives(config_path, limit=limit)
        click.echo(
            f"Derivatives built={result['built']} failed={result['failed']} skipped={result['skipped']}"
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@app.command("repair-metadata")
@click.option("--config", "config_path", default=str(DEFAULT_CONFIG_PATH), type=click.Path(path_type=Path))
@click.option("--limit", default=None, type=int)
def repair_metadata_cmd(config_path: Path, limit: Optional[int]):
    try:
        result = run_metadata_repair(config_path, limit=limit)
        click.echo(f"Metadata repair examined={result['examined']} repaired={result['repaired']}")
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@app.command("extract-content")
@click.option("--config", "config_path", default=str(DEFAULT_CONFIG_PATH), type=click.Path(path_type=Path))
@click.option("--limit", default=None, type=int)
def extract_content_cmd(config_path: Path, limit: Optional[int]):
    try:
        result = run_content_extraction(config_path, limit=limit)
        click.echo(
            f"Content extraction processed={result['processed']} ocr={result['ocr_saved']} "
            f"transcripts={result['transcript_saved']} text_embeddings={result['text_saved']} "
            f"visual_embeddings={result['visual_saved']}"
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@app.command("detect-faces")
@click.option("--config", "config_path", default=str(DEFAULT_CONFIG_PATH), type=click.Path(path_type=Path))
@click.option("--limit", default=None, type=int)
def detect_faces_cmd(config_path: Path, limit: Optional[int]):
    try:
        result = run_face_detection(config_path, limit=limit)
        click.echo(f"Face detection processed={result['processed']} detected={result['detected']} failed={result['failed']}")
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@app.command("audit")
@click.option("--config", "config_path", default=str(DEFAULT_CONFIG_PATH), type=click.Path(path_type=Path))
@click.option("--sample-size", default=100, show_default=True, type=int)
@click.option("--source", "sources", multiple=True)
def audit_cmd(config_path: Path, sample_size: int, sources: tuple[str, ...]):
    try:
        _config, resolved, db_path = _load_runtime(config_path)
        report = build_audit_report(db_path, sample_size=sample_size, sources=sources or None)
        click.echo(format_audit_report(report))
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@app.command("validate-library")
@click.option("--config", "config_path", default=str(DEFAULT_CONFIG_PATH), type=click.Path(path_type=Path))
@click.option("--limit", default=None, type=int)
def validate_library_cmd(config_path: Path, limit: Optional[int]):
    config, resolved = _load_runtime_config(config_path)
    result = master.validate(
        master_dir=config.managed_library_dir(resolved),
        db_path=config.db_path(resolved),
        limit=limit,
    )
    click.echo(
        f"Validated library files: checked={result['checked']} present={result['present']} missing={result['missing']}"
    )


@app.command("dedupe-exact")
@click.option("--config", "config_path", default=str(DEFAULT_CONFIG_PATH), type=click.Path(path_type=Path))
@click.option("--limit", default=None, type=int)
def dedupe_exact_cmd(config_path: Path, limit: Optional[int]):
    try:
        result = run_dedupe_exact(config_path, limit=limit)
        click.echo(
            f"Exact dedupe processed={result['processed']} hashed={result['hashed']} "
            f"missing={result['missing']} groups={result['groups']}"
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@app.command("dedupe-near")
@click.option("--config", "config_path", default=str(DEFAULT_CONFIG_PATH), type=click.Path(path_type=Path))
@click.option("--limit", default=None, type=int)
@click.option("--max-distance", default=6, show_default=True, type=int)
def dedupe_near_cmd(config_path: Path, limit: Optional[int], max_distance: int):
    try:
        result = run_dedupe_near(config_path, limit=limit, max_distance=max_distance)
        click.echo(
            f"Near dedupe processed={result['processed']} hashed={result['hashed']} "
            f"missing={result['missing']} groups={result['groups']}"
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@app.command("pilot-run")
@click.option("--config", "config_path", default=str(DEFAULT_CONFIG_PATH), type=click.Path(path_type=Path))
@click.option("--limit", default=250, show_default=True, type=int)
def pilot_run_cmd(config_path: Path, limit: int):
    try:
        result = run_pilot_pipeline(config_path, batch_limit=limit)
        before = result["before"]
        after = result["after"]
        before_counts = result["before_counts"]
        after_counts = result["after_counts"]
        click.echo(
            "Pilot complete: "
            f"metadata score {before['metadata']['average_metadata_score']} -> {after['metadata']['average_metadata_score']}, "
            f"location {before['metadata']['assets_with_location']} -> {after['metadata']['assets_with_location']}, "
            f"people {before['metadata']['assets_with_people']} -> {after['metadata']['assets_with_people']}, "
            f"duplicate groups {before_counts['duplicates']} -> {after_counts['duplicates']}, "
            f"faces {before_counts['faces']['total_faces']} -> {after_counts['faces']['total_faces']}"
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@app.command("serve-api")
@click.option("--config", "config_path", default=str(DEFAULT_CONFIG_PATH), type=click.Path(path_type=Path))
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
def serve_api_cmd(config_path: Path, host: str, port: int):
    try:
        import uvicorn
    except ImportError as exc:
        raise click.ClickException("uvicorn is required for serve-api. Install requirements first.") from exc
    from .api import create_app

    app_instance = create_app(config_path)
    uvicorn.run(app_instance, host=host, port=port)


def main():
    app(prog_name="photo-unifier")


if __name__ == "__main__":
    main()
