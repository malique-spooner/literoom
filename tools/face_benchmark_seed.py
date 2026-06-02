from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from literoom.config import load_config
from literoom.metadata import manifest


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    limit = min(len(a), len(b))
    if limit <= 0:
        return 0.0
    dot = sum(float(a[i]) * float(b[i]) for i in range(limit))
    an = sum(float(a[i]) * float(a[i]) for i in range(limit)) ** 0.5 or 1.0
    bn = sum(float(b[i]) * float(b[i]) for i in range(limit)) ** 0.5 or 1.0
    return dot / (an * bn)


def _centroid(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = max(len(vector) for vector in vectors)
    out = [0.0] * dim
    for vector in vectors:
        for idx, value in enumerate(vector[:dim]):
            out[idx] += float(value)
    scale = float(len(vectors)) or 1.0
    return [value / scale for value in out]


def _embedding_rows(db_path: Path, model_name: str) -> list[dict]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            SELECT f.id AS face_id, f.asset_id, f.identity_id, f.status AS face_status,
                   i.label AS identity_label, i.status AS identity_status,
                   a.orig_filename, e.model_name, e.payload_json
            FROM faces f
            JOIN embeddings e ON e.id = f.embedding_ref
            LEFT JOIN face_identities i ON i.id = f.identity_id
            LEFT JOIN assets a ON a.id = f.asset_id
            WHERE e.embedding_type = 'face'
              AND e.model_name = ?
              AND f.status <> 'REJECTED'
            """,
            (model_name,),
        ).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            try:
                payload = json.loads(item.get("payload_json") or "{}")
            except Exception:
                payload = {}
            vector = payload.get("vector") or []
            if vector:
                item["vector"] = [float(value) for value in vector]
                items.append(item)
        return items
    finally:
        con.close()


def build_seed_benchmark(
    config_path: Path,
    *,
    positives_per_identity: int = 24,
    hard_negatives_per_identity: int = 16,
    easy_negatives_per_identity: int = 12,
) -> list[dict]:
    config, resolved = load_config(config_path)
    db_path = config.db_path(resolved)
    identities = manifest.list_face_identities(db_path, limit=200, status="CONFIRMED")
    rows: list[dict] = []

    for identity in identities:
        identity_id = str(identity["id"])
        identity_label = str(identity["label"])
        model_name = manifest.get_identity_model_name(db_path, identity_id)
        if not model_name:
            continue
        embeddings = _embedding_rows(db_path, model_name)
        positives = [item for item in embeddings if str(item.get("identity_id") or "") == identity_id]
        if len(positives) < 4:
            continue
        centroid = _centroid([item["vector"] for item in positives])

        scored_positives = []
        for item in positives:
            score = _cosine_similarity(item["vector"], centroid)
            scored_positives.append({**item, "score": score})
        scored_positives.sort(key=lambda item: (float(item["score"]), str(item.get("face_id") or "")), reverse=True)

        other_rows = [item for item in embeddings if str(item.get("identity_id") or "") != identity_id]
        scored_negatives = []
        for item in other_rows:
            score = _cosine_similarity(item["vector"], centroid)
            status = str(item.get("identity_status") or "")
            if status == "CONFIRMED":
                seed_type = "confirmed_negative"
                weight = 1.0
            elif status == "CLUSTERED":
                seed_type = "cluster_negative"
                weight = 0.4
            elif score <= 0.45:
                seed_type = "easy_negative"
                weight = 1.0
            else:
                seed_type = "heuristic_negative"
                weight = 0.25
            scored_negatives.append({**item, "score": score, "seed_type": seed_type, "weight": weight})

        hard_negatives = [
            item for item in scored_negatives
            if item["seed_type"] in {"confirmed_negative", "cluster_negative", "heuristic_negative"}
            and float(item["score"]) >= 0.60
        ]
        hard_negatives.sort(key=lambda item: (float(item["score"]), str(item.get("face_id") or "")), reverse=True)

        easy_negatives = [
            item for item in scored_negatives
            if item["seed_type"] == "easy_negative"
        ]
        easy_negatives.sort(key=lambda item: (float(item["score"]), str(item.get("face_id") or "")))

        for item in scored_positives[:positives_per_identity]:
            rows.append(
                {
                    "identity_id": identity_id,
                    "identity_label": identity_label,
                    "model_name": model_name,
                    "face_id": str(item["face_id"]),
                    "asset_id": str(item["asset_id"]),
                    "orig_filename": item.get("orig_filename"),
                    "score": round(float(item["score"]), 6),
                    "label": 1,
                    "weight": 1.0,
                    "seed_type": "confirmed_positive",
                    "note": "Confirmed identity member.",
                }
            )
        for item in hard_negatives[:hard_negatives_per_identity]:
            rows.append(
                {
                    "identity_id": identity_id,
                    "identity_label": identity_label,
                    "model_name": model_name,
                    "face_id": str(item["face_id"]),
                    "asset_id": str(item["asset_id"]),
                    "orig_filename": item.get("orig_filename"),
                    "score": round(float(item["score"]), 6),
                    "label": 0,
                    "weight": float(item["weight"]),
                    "seed_type": item["seed_type"],
                    "note": "High-similarity negative seed. Review manually before treating as gold.",
                }
            )
        for item in easy_negatives[:easy_negatives_per_identity]:
            rows.append(
                {
                    "identity_id": identity_id,
                    "identity_label": identity_label,
                    "model_name": model_name,
                    "face_id": str(item["face_id"]),
                    "asset_id": str(item["asset_id"]),
                    "orig_filename": item.get("orig_filename"),
                    "score": round(float(item["score"]), 6),
                    "label": 0,
                    "weight": 1.0,
                    "seed_type": item["seed_type"],
                    "note": "Low-similarity negative seed.",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a seeded face benchmark from confirmed identities.")
    parser.add_argument("--config", default="literoom.local.yaml")
    parser.add_argument("--output", default="benchmarks/face-benchmark.seed.jsonl")
    parser.add_argument("--positives-per-identity", type=int, default=24)
    parser.add_argument("--hard-negatives-per-identity", type=int, default=16)
    parser.add_argument("--easy-negatives-per-identity", type=int, default=12)
    args = parser.parse_args()

    rows = build_seed_benchmark(
        Path(args.config),
        positives_per_identity=args.positives_per_identity,
        hard_negatives_per_identity=args.hard_negatives_per_identity,
        easy_negatives_per_identity=args.easy_negatives_per_identity,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
