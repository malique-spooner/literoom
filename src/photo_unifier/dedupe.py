from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .metadata import manifest
from .utils.hashing import average_hash, hamming_distance, sha256_file


def _canonical_asset(items: List[Dict]) -> str:
    ranked = sorted(
        items,
        key=lambda item: (
            -(item.get("orig_size") or 0),
            item.get("dt_original") or "",
            item["id"],
        ),
    )
    return ranked[0]["id"]


def run_exact(
    db_path: Path,
    managed_library_dir: Path,
    *,
    limit: Optional[int] = None,
) -> Dict[str, int]:
    managed_library_dir = Path(managed_library_dir)
    by_hash: Dict[str, List[Dict]] = defaultdict(list)
    processed = hashed = missing = 0

    for row in manifest.iter_built_assets(db_path=db_path, limit=limit):
        processed += 1
        managed_rel = row.get("managed_path")
        if not managed_rel:
            missing += 1
            continue
        asset_path = managed_library_dir / managed_rel
        if not asset_path.exists():
            missing += 1
            continue

        digest = row.get("sha256")
        if not digest:
            digest = sha256_file(asset_path)
            manifest.record_hash(db_path, row["id"], "sha256", "managed", digest)
            hashed += 1
        row["sha256"] = digest
        by_hash[digest].append(row)

    groups: List[Dict] = []
    for digest, items in by_hash.items():
        if len(items) < 2:
            continue
        canonical_asset_id = _canonical_asset(items)
        groups.append(
            {
                "canonical_asset_id": canonical_asset_id,
                "items": [
                    {
                        "asset_id": item["id"],
                        "score": 1.0,
                        "rationale": f"exact sha256 match {digest}",
                    }
                    for item in items
                ],
            }
        )

    created = manifest.replace_duplicate_groups(
        db_path,
        group_type="EXACT_SHA256",
        groups=groups,
    )
    return {
        "processed": processed,
        "hashed": hashed,
        "missing": missing,
        "groups": created,
    }


def _preview_path(
    row: Dict,
    managed_library_dir: Path,
    derivatives_dir: Optional[Path] = None,
) -> Optional[Path]:
    if row.get("media_type") in {"image", "raw"} and row.get("managed_path"):
        candidate = managed_library_dir / row["managed_path"]
        if candidate.exists():
            return candidate
    if row.get("media_type") == "video":
        thumbnail = row.get("thumbnail_path")
        if thumbnail and Path(thumbnail).exists():
            return Path(thumbnail)
        if derivatives_dir:
            candidate = Path(derivatives_dir) / "video_preview" / f"{row['id']}.jpg"
            if candidate.exists():
                return candidate
    if row.get("managed_path"):
        candidate = managed_library_dir / row["managed_path"]
        if candidate.exists():
            return candidate
    return None


def run_near(
    db_path: Path,
    managed_library_dir: Path,
    *,
    derivatives_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    max_distance: int = 6,
) -> Dict[str, int]:
    managed_library_dir = Path(managed_library_dir)
    derivatives_dir = Path(derivatives_dir) if derivatives_dir else None
    processed = hashed = missing = 0
    buckets: Dict[tuple[str, str], List[Dict]] = defaultdict(list)

    for row in manifest.iter_built_assets(db_path=db_path, limit=limit):
        processed += 1
        preview = _preview_path(row, managed_library_dir, derivatives_dir)
        if not preview or not preview.exists():
            missing += 1
            continue
        try:
            digest = average_hash(preview)
        except Exception:
            missing += 1
            continue
        hashed += 1
        manifest.record_hash(db_path, row["id"], "ahash64", "preview", digest)
        day_bucket = (row.get("dt_original") or "")[:10] or "unknown"
        buckets[(row.get("media_type") or "unknown", day_bucket)].append({**row, "ahash64": digest, "preview_path": str(preview)})

    groups: List[Dict] = []
    for (_media_type, _day_bucket), items in buckets.items():
        if len(items) < 2:
            continue
        parent = {item["id"]: item["id"] for item in items}

        def find(node: str) -> str:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left: str, right: str) -> None:
            root_left = find(left)
            root_right = find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        for i, left in enumerate(items):
            for right in items[i + 1 :]:
                if hamming_distance(left["ahash64"], right["ahash64"]) <= max_distance:
                    union(left["id"], right["id"])

        clusters: Dict[str, List[Dict]] = defaultdict(list)
        for item in items:
            clusters[find(item["id"])].append(item)

        for cluster_items in clusters.values():
            if len(cluster_items) < 2:
                continue
            canonical_asset_id = _canonical_asset(cluster_items)
            groups.append(
                {
                    "canonical_asset_id": canonical_asset_id,
                    "items": [
                        {
                            "asset_id": item["id"],
                            "score": round(1.0 - (hamming_distance(cluster_items[0]["ahash64"], item["ahash64"]) / 64.0), 3),
                            "rationale": f"near match ahash64 distance={hamming_distance(cluster_items[0]['ahash64'], item['ahash64'])}",
                        }
                        for item in cluster_items
                    ],
                }
            )

    created = manifest.replace_duplicate_groups(
        db_path,
        group_type="NEAR_AHASH",
        groups=groups,
    )
    return {
        "processed": processed,
        "hashed": hashed,
        "missing": missing,
        "groups": created,
    }
