from __future__ import annotations

import json
import os
import shutil
import time
import zipfile
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Iterable, Optional

from . import manifest
from ..utils.exiftool import write_fields
from ..utils.hashing import sha256_file

SIDECAR_SUFFIX = ".literoom.json"


def _json_list(val):
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return []
    return val or []


def _write_metadata_sidecar(db_path: Path, asset_id: str, dest: Path) -> None:
    payload = manifest.export_asset_metadata_payload(db_path, asset_id)
    payload["managed_file"] = str(dest)
    sidecar_path = dest.with_suffix(dest.suffix + SIDECAR_SUFFIX)
    sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    manifest.record_artifact(db_path, asset_id, "metadata_sidecar", str(sidecar_path), "READY")


def _set_windows_created_time(path: Path, epoch_secs: float) -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        import ctypes.wintypes as wt

        handle = ctypes.windll.kernel32.CreateFileW(
            str(path),
            0x0100,
            0,
            None,
            3,
            0x02000000,
            None,
        )
        if handle == -1:
            return
        try:
            unix_epoch_start = 11644473600
            intervals = int((epoch_secs + unix_epoch_start) * 10_000_000)
            c_time = wt.FILETIME(intervals & 0xFFFFFFFF, intervals >> 32)
            ctypes.windll.kernel32.SetFileTime(handle, ctypes.byref(c_time), None, None)
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    except Exception:
        pass


def _set_fs_times(dest: Path, dt_iso: Optional[str]) -> None:
    if not dt_iso:
        return
    try:
        d = time.strptime(dt_iso[:19], "%Y-%m-%dT%H:%M:%S")
        epoch = time.mktime(d)
        os.utime(dest, (epoch, epoch))
        _set_windows_created_time(dest, epoch)
    except Exception:
        pass


def _copy_asset(row: dict, dest: Path) -> None:
    source_kind = row.get("source_kind") or "zip"
    if source_kind == "zip":
        src_zip = Path(row["source_locator"] or row["abs_zip"])
        inner = row["source_path"] or row["zip_path"]
        with zipfile.ZipFile(src_zip, "r") as zp:
            with zp.open(inner, "r") as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
        return

    src = Path(row["source_locator"] or row["abs_zip"])
    shutil.copy2(src, dest)


def _build_source_index(source_roots: Iterable[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in source_roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                index.setdefault(path.name.lower(), path)
    return index


def _locator_name(locator: str) -> str:
    if "\\" in locator:
        return PureWindowsPath(locator).name.lower()
    return Path(locator).name.lower()


def _resolve_source_file(row: dict, source_index: dict[str, Path]) -> Optional[Path]:
    current = Path(row.get("source_locator") or row.get("abs_zip") or "")
    if current.exists():
        return current
    locator = str(row.get("source_locator") or row.get("abs_zip") or "")
    key = _locator_name(locator)
    if not key:
        return None
    return source_index.get(key)


def build(
    master_dir: Path,
    db_path: Path,
    exiftool_path: Optional[Path] = None,
    source_roots: Optional[Iterable[Path]] = None,
    backup_ext: str = ".bak",
    force: bool = False,
    limit: int | None = None,
    set_fs_times: bool = True,
    log_every: int = 50,
    job_id: Optional[str] = None,
) -> int:
    master_dir = Path(master_dir)
    master_dir.mkdir(parents=True, exist_ok=True)

    total = manifest.count_for_master(db_path, limit=limit)
    count = copied = existed = embed_ok = embed_fail = 0
    source_index = _build_source_index(source_roots or [])
    for row in manifest.iter_for_master(db_path=db_path, limit=limit):
        aid = row["id"]
        rel = row.get("target_relpath") or row.get("managed_relpath") or ""
        fname = row.get("target_filename") or row.get("managed_filename")
        if not fname:
            manifest.update_status(db_path, aid, "ERROR", error="missing target filename")
            continue

        dest_dir = master_dir / rel
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / fname

        if dest.exists() and not force:
            h = sha256_file(dest)
            manifest.mark_embedded(db_path, aid, h, embed_hash=None, tool_version=None)
            try:
                _write_metadata_sidecar(db_path, aid, dest)
            except Exception as exc:
                manifest.record_artifact(db_path, aid, "metadata_sidecar", str(dest.with_suffix(dest.suffix + SIDECAR_SUFFIX)), "FAILED", error=str(exc))
            existed += 1
            count += 1
            if job_id and (count % 10 == 0 or count == total):
                manifest.attach_job_metrics(
                    db_path,
                    job_id,
                    {
                        "stage": "build_library",
                        "total_assets": total,
                        "processed_assets": count,
                        "remaining_assets": max(total - count, 0),
                        "copied_assets": copied,
                        "existing_assets": existed,
                        "embedded_assets": embed_ok,
                        "failed_assets": embed_fail,
                    },
                )
            continue

        try:
            resolved_source = _resolve_source_file(row, source_index)
            if resolved_source is not None and str(resolved_source) != str(row.get("source_locator") or row.get("abs_zip") or ""):
                row["source_locator"] = str(resolved_source)
                row["abs_zip"] = str(resolved_source)
                source_root = next((str(root) for root in (source_roots or []) if resolved_source.is_relative_to(Path(root))), None) if hasattr(resolved_source, "is_relative_to") else None
                manifest.update_source_locator(
                    db_path,
                    aid,
                    source_locator=str(resolved_source),
                    source_root=source_root,
                )
            _copy_asset(row, dest)
            copied += 1
        except Exception as exc:
            manifest.update_status(db_path, aid, "ERROR", error=f"copy failed: {exc}")
            count += 1
            if job_id and (count % 10 == 0 or count == total):
                manifest.attach_job_metrics(
                    db_path,
                    job_id,
                    {
                        "stage": "build_library",
                        "total_assets": total,
                        "processed_assets": count,
                        "remaining_assets": max(total - count, 0),
                        "copied_assets": copied,
                        "existing_assets": existed,
                        "embedded_assets": embed_ok,
                        "failed_assets": embed_fail,
                    },
                )
            continue

        dt = row.get("dt_original") or row.get("src_mtime")
        fields = {
            "title": row.get("title"),
            "description": row.get("description"),
            "dt_original": dt,
            "gps_lat": row.get("gps_lat"),
            "gps_lon": row.get("gps_lon"),
            "gps_alt": row.get("gps_alt"),
            "keywords": _json_list(row.get("keywords_json")),
            "people": _json_list(row.get("people_json")),
        }

        try:
            ok, version = write_fields(dest, fields, exiftool_path=exiftool_path, backup_ext=backup_ext)
        except Exception as exc:
            h = sha256_file(dest)
            manifest.mark_copied(db_path, aid, h, tool_version=None, warning=str(exc))
            try:
                _write_metadata_sidecar(db_path, aid, dest)
            except Exception as sidecar_exc:
                manifest.record_artifact(
                    db_path,
                    aid,
                    "metadata_sidecar",
                    str(dest.with_suffix(dest.suffix + SIDECAR_SUFFIX)),
                    "FAILED",
                    error=str(sidecar_exc),
                )
            embed_fail += 1
            count += 1
            if job_id and (count % 10 == 0 or count == total):
                manifest.attach_job_metrics(
                    db_path,
                    job_id,
                    {
                        "stage": "build_library",
                        "total_assets": total,
                        "processed_assets": count,
                        "remaining_assets": max(total - count, 0),
                        "copied_assets": copied,
                        "existing_assets": existed,
                        "embedded_assets": embed_ok,
                        "failed_assets": embed_fail,
                    },
                )
            continue

        if ok:
            h = sha256_file(dest)
            manifest.mark_embedded(db_path, aid, h, embed_hash=None, tool_version=version)
            embed_ok += 1
            if set_fs_times:
                _set_fs_times(dest, dt)
        else:
            h = sha256_file(dest)
            manifest.mark_copied(db_path, aid, h, tool_version=version, warning="exiftool embed failed")
            embed_fail += 1

        try:
            _write_metadata_sidecar(db_path, aid, dest)
        except Exception as exc:
            manifest.record_artifact(
                db_path,
                aid,
                "metadata_sidecar",
                str(dest.with_suffix(dest.suffix + SIDECAR_SUFFIX)),
                "FAILED",
                error=str(exc),
            )

        count += 1
        if job_id and (count % 10 == 0 or count == total):
            manifest.attach_job_metrics(
                db_path,
                job_id,
                {
                    "stage": "build_library",
                    "total_assets": total,
                    "processed_assets": count,
                    "remaining_assets": max(total - count, 0),
                    "copied_assets": copied,
                    "existing_assets": existed,
                    "embedded_assets": embed_ok,
                    "failed_assets": embed_fail,
                },
            )
        if count % log_every == 0:
            print(f"[build] {count}: processed {dest.name}")

    if job_id:
        manifest.attach_job_metrics(
            db_path,
            job_id,
            {
                "stage": "build_library",
                "total_assets": total,
                "processed_assets": count,
                "remaining_assets": max(total - count, 0),
                "copied_assets": copied,
                "existing_assets": existed,
                "embedded_assets": embed_ok,
                "failed_assets": embed_fail,
            },
        )
    print(f"[build] Done. items={count} copied={copied} exist={existed} embed_ok={embed_ok} embed_fail={embed_fail}")
    return count


def validate(master_dir: Path, db_path: Path, limit: int | None = None) -> dict:
    master_dir = Path(master_dir)
    checked = missing = present = 0
    for row in manifest.iter_assets_for_derivatives(db_path=db_path, limit=limit):
        dest = master_dir / row["managed_path"]
        checked += 1
        if dest.exists():
            present += 1
        else:
            missing += 1
            manifest.update_status(db_path, row["id"], "ERROR", error="managed file missing")
    return {"checked": checked, "present": present, "missing": missing}
