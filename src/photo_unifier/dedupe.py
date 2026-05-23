from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .metadata import manifest
from .utils.hashing import average_hash, difference_hash, hamming_distance, sha256_file


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


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _pair_score(left: Dict, right: Dict) -> tuple[float, str, float]:
    ah_left = left.get("ahash64")
    ah_right = right.get("ahash64")
    dh_left = left.get("dhash64")
    dh_right = right.get("dhash64")
    ah_distance = hamming_distance(ah_left, ah_right) if ah_left and ah_right else 64
    dh_distance = hamming_distance(dh_left, dh_right) if dh_left and dh_right else 64
    hash_score = 1.0 - ((ah_distance * 0.6 + dh_distance * 0.4) / 64.0)

    dt_left = _parse_dt(left.get("dt_original"))
    dt_right = _parse_dt(right.get("dt_original"))
    time_score = 0.0
    time_note = "time=unknown"
    if dt_left and dt_right:
        delta_seconds = abs((dt_left - dt_right).total_seconds())
        if delta_seconds <= 30:
            time_score = 1.0
        elif delta_seconds <= 300:
            time_score = 0.85
        elif delta_seconds <= 1800:
            time_score = 0.65
        elif delta_seconds <= 43200:
            time_score = 0.35
        else:
            time_score = 0.1
        time_note = f"time_delta={int(delta_seconds)}s"

    source_score = 0.0
    source_bits: list[str] = []
    if left.get("source") and left.get("source") == right.get("source"):
        source_score += 0.12
        source_bits.append("same source")
    if left.get("source_kind") and left.get("source_kind") == right.get("source_kind"):
        source_score += 0.08
        source_bits.append("same source kind")
    if left.get("source_root") and left.get("source_root") == right.get("source_root"):
        source_score += 0.08
        source_bits.append("same source root")

    media_bias = 0.05 if left.get("media_type") == right.get("media_type") else -0.2
    score = (hash_score * 0.72) + (time_score * 0.18) + source_score + media_bias
    average_distance = (ah_distance * 0.6) + (dh_distance * 0.4)
    rationale = (
        f"ahash={ah_distance} dhash={dh_distance} {time_note}"
        + (f" {'; '.join(source_bits)}" if source_bits else "")
    )
    return score, rationale, average_distance


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
            diff_digest = difference_hash(preview)
        except Exception:
            missing += 1
            continue
        hashed += 1
        manifest.record_hash(db_path, row["id"], "ahash64", "preview", digest)
        manifest.record_hash(db_path, row["id"], "dhash64", "preview", diff_digest)
        day_bucket = (row.get("dt_original") or "")[:10] or "unknown"
        buckets[(row.get("media_type") or "unknown", day_bucket)].append(
            {**row, "ahash64": digest, "dhash64": diff_digest, "preview_path": str(preview)}
        )

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
                score, rationale, avg_distance = _pair_score(left, right)
                if avg_distance <= max_distance or (score >= 0.82 and avg_distance <= max_distance + 4):
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
                            "score": round(_pair_score(cluster_items[0], item)[0], 3),
                            "rationale": _pair_score(cluster_items[0], item)[1],
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
