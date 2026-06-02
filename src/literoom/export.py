from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from .metadata import manifest
from .utils.hashing import sha256_file


def export_built_assets(
    *,
    db_path: Path,
    managed_library_dir: Path,
    destination_dir: Path,
    limit: Optional[int] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = destination_dir / "literoom-export.jsonl"
    summary_path = destination_dir / "literoom-export-summary.json"
    started_at = manifest.utcnow_iso()

    counts = {
        "planned": 0,
        "exported": 0,
        "skipped": 0,
        "missing": 0,
        "failed": 0,
    }

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for row in manifest.iter_built_assets(db_path, limit=limit):
            counts["planned"] += 1
            managed_rel = row.get("managed_path")
            asset_id = row.get("id")
            if not managed_rel:
                counts["failed"] += 1
                manifest_file.write(
                    json.dumps(
                        {
                            "asset_id": asset_id,
                            "status": "failed",
                            "reason": "missing managed path",
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                continue

            src = managed_library_dir / managed_rel
            dest = destination_dir / managed_rel
            dest.parent.mkdir(parents=True, exist_ok=True)

            if not src.exists():
                counts["missing"] += 1
                manifest_file.write(
                    json.dumps(
                        {
                            "asset_id": asset_id,
                            "managed_path": managed_rel,
                            "status": "missing",
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                continue

            if src.resolve() == dest.resolve():
                status = "skipped"
                counts["skipped"] += 1
            elif dest.exists():
                if overwrite:
                    shutil.copy2(src, dest)
                    status = "exported"
                    counts["exported"] += 1
                else:
                    src_hash = row.get("sha256") or sha256_file(src)
                    dest_hash = sha256_file(dest)
                    if src_hash and dest_hash == src_hash:
                        status = "skipped"
                        counts["skipped"] += 1
                    else:
                        status = "failed"
                        counts["failed"] += 1
                        manifest_file.write(
                            json.dumps(
                                {
                                    "asset_id": asset_id,
                                    "managed_path": managed_rel,
                                    "status": status,
                                    "reason": "destination exists and does not match source",
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        continue
            else:
                shutil.copy2(src, dest)
                status = "exported"
                counts["exported"] += 1

            manifest_file.write(
                json.dumps(
                    {
                        "asset_id": asset_id,
                        "orig_filename": row.get("orig_filename"),
                        "media_type": row.get("media_type"),
                        "dt_original": row.get("dt_original"),
                        "sha256": row.get("sha256"),
                        "managed_path": managed_rel,
                        "exported_path": str(dest.relative_to(destination_dir)),
                        "status": status,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    summary = {
        "exported_at": started_at,
        "destination": str(destination_dir),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        **counts,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
