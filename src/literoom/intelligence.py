from __future__ import annotations

import json
import math
import hashlib
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from PIL import Image, ImageOps

from .metadata import manifest

TEXT_EMBED_DIM = 96
IMAGE_EMBED_DIM = 88


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 1]


def _normalize_vector(vector: Sequence[float]) -> list[float]:
    values = [float(item) for item in vector]
    norm = math.sqrt(sum(item * item for item in values)) or 1.0
    return [item / norm for item in values]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    limit = min(len(a), len(b))
    if limit == 0:
        return 0.0
    dot = sum(float(a[i]) * float(b[i]) for i in range(limit))
    an = math.sqrt(sum(float(a[i]) * float(a[i]) for i in range(limit))) or 1.0
    bn = math.sqrt(sum(float(b[i]) * float(b[i]) for i in range(limit))) or 1.0
    return dot / (an * bn)


def text_embedding(text: str, *, dim: int = TEXT_EMBED_DIM) -> list[float]:
    tokens = _tokenize(text)
    if not tokens:
        return [0.0] * dim
    vector = [0.0] * dim
    for token in tokens:
        digest = re.sub(r"[^a-z0-9]", "", token)
        if not digest:
            continue
        token_hash = hashlib.sha1(digest.encode("utf-8")).digest()
        idx = int.from_bytes(token_hash[:4], "big") % dim
        sign = -1.0 if (token_hash[4] & 1) else 1.0
        vector[idx] += sign
    return _normalize_vector(vector)


def image_embedding(image_path: Path) -> list[float]:
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        small = img.resize((8, 8), Image.Resampling.BILINEAR).convert("L")
        gray = [float(value) / 255.0 for value in small.tobytes()]
        hist: list[float] = []
        for band in img.split():
            bins = [0.0] * 8
            for value in band.tobytes():
                bins[min(7, int(value) // 32)] += 1.0
            total = float(sum(bins)) or 1.0
            hist.extend([value / total for value in bins])
        vector = gray + hist
    return _normalize_vector(vector[:IMAGE_EMBED_DIM])


def _clean_text(*values: Any) -> str:
    parts: list[str] = []
    for value in values:
        if value in (None, "", [], {}):
            continue
        if isinstance(value, (list, tuple, set)):
            parts.extend(str(item) for item in value if str(item).strip())
            continue
        if isinstance(value, dict):
            parts.extend(str(item) for item in value.values() if str(item).strip())
            continue
        parts.append(str(value))
    return " ".join(part.strip() for part in parts if str(part).strip())


def _read_text_asset_rows(db_path: Path) -> list[dict[str, Any]]:
    con = manifest.connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT a.id, a.orig_filename, a.title, a.description, a.people_json,
                   a.keywords_json, a.dt_original, a.source, a.source_kind, a.media_type,
                   a.gps_lat, a.gps_lon, a.gps_alt, m.managed_path,
                   COALESCE((
                     SELECT group_concat(i.label, ' ')
                     FROM faces f
                     JOIN face_identities i ON i.id = f.identity_id
                     WHERE f.asset_id = a.id AND f.status <> 'REJECTED'
                   ), '') AS face_labels,
                   COALESCE((
                     SELECT group_concat(er.text_content, ' ')
                     FROM extraction_results er
                     WHERE er.asset_id = a.id AND er.status = 'READY'
                   ), '') AS extracted_text
                   ,COALESCE((
                     SELECT group_concat(mf.value_text, ' ')
                     FROM metadata_fields mf
                     WHERE mf.asset_id = a.id
                       AND mf.field_name = 'location'
                       AND COALESCE(mf.value_text, '') <> ''
                   ), '') AS location_text
            FROM assets a
            LEFT JOIN managed_assets m ON m.asset_id = a.id
            ORDER BY a.dt_original DESC, a.id DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()


def _ocr_text_from_image(image_path: Path) -> tuple[Optional[str], str]:
    if not image_path.exists():
        return None, "missing image"
    try:
        import pytesseract  # type: ignore

        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img).convert("L")
            text = pytesseract.image_to_string(img)
            text = text.strip()
            return (text or None), "pytesseract"
    except Exception:
        pass
    tesseract_bin = shutil.which("tesseract")
    if not tesseract_bin:
        return None, "tesseract unavailable"
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img).convert("L")
            img.save(tmp_path)
        proc = subprocess.run(
            [tesseract_bin, str(tmp_path), "stdout", "--psm", "6"],
            capture_output=True,
            text=True,
        )
        text = (proc.stdout or "").strip()
        if proc.returncode != 0:
            return None, proc.stderr.strip() or "tesseract failed"
        return (text or None), "tesseract-cli"
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def _extract_audio_track(video_path: Path) -> Optional[Path]:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin or not video_path.exists():
        return None
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)
    proc = subprocess.run(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(wav_path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not wav_path.exists():
        try:
            wav_path.unlink(missing_ok=True)
        except Exception:
            pass
        return None
    return wav_path


def _transcribe_video(video_path: Path, *, model_name: Optional[str] = None) -> tuple[Optional[str], str]:
    audio_path = _extract_audio_track(video_path)
    if not audio_path:
        return None, "audio unavailable"
    try:
        import whisper  # type: ignore

        model = whisper.load_model(model_name or "base")
        result = model.transcribe(str(audio_path))
        text = (result.get("text") or "").strip()
        return (text or None), f"whisper:{model_name or 'base'}"
    except Exception as exc:
        return None, f"transcription unavailable: {exc}"
    finally:
        try:
            audio_path.unlink(missing_ok=True)
        except Exception:
            pass


def _asset_text_profile(asset: dict[str, Any], extraction_rows: list[dict[str, Any]]) -> str:
    extracted_text = [row.get("text_content") for row in extraction_rows if row.get("status") == "READY" and row.get("text_content")]
    return _clean_text(
        asset.get("orig_filename"),
        asset.get("title"),
        asset.get("description"),
        asset.get("location_text"),
        asset.get("people_json"),
        asset.get("keywords_json"),
        asset.get("face_labels"),
        asset.get("dt_original"),
        asset.get("source"),
        asset.get("source_kind"),
        asset.get("media_type"),
        asset.get("extracted_text"),
        extracted_text,
    )


def index_asset_embeddings(
    db_path: Path,
    managed_library_dir: Path,
    *,
    asset_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, int]:
    rows = manifest.iter_built_assets(db_path, limit=limit)
    processed = text_saved = visual_saved = 0
    for row in rows:
        if asset_id and row["id"] != asset_id:
            continue
        asset = manifest.get_asset(db_path, row["id"])
        if not asset:
            continue
        extraction_rows = manifest.list_extraction_results(db_path, row["id"])
        metadata_rows = manifest.list_asset_metadata(db_path, row["id"])
        search_blob = _asset_text_profile(
            {**asset, **row, "location_text": _location_text_from_metadata_rows(metadata_rows)},
            extraction_rows,
        )
        manifest.replace_embedding(
            db_path,
            asset_id=row["id"],
            embedding_type="asset_text",
            model_name="hashed-text-v1",
            vector=text_embedding(search_blob),
            payload={
                "search_blob": search_blob,
                "source_count": len(extraction_rows),
            },
        )
        text_saved += 1
        visual_path: Optional[Path] = None
        if asset.get("media_type") in {"image", "raw"} and asset.get("managed_path"):
            candidate = managed_library_dir / str(asset["managed_path"])
            if candidate.exists():
                visual_path = candidate
        else:
            thumb = manifest.thumbnail_for_asset(db_path, row["id"])
            if thumb:
                candidate = Path(thumb)
                if candidate.exists():
                    visual_path = candidate
        if visual_path:
            manifest.replace_embedding(
                db_path,
                asset_id=row["id"],
                embedding_type="asset_visual",
                model_name="simple-image-v1",
                vector=image_embedding(visual_path),
                vector_ref=str(visual_path),
                payload={"path": str(visual_path)},
            )
            visual_saved += 1
        processed += 1
    return {"processed": processed, "text_saved": text_saved, "visual_saved": visual_saved}


def extract_asset_content(
    db_path: Path,
    managed_library_dir: Path,
    *,
    asset_id: Optional[str] = None,
    limit: Optional[int] = None,
    whisper_model: Optional[str] = None,
) -> Dict[str, int]:
    processed = ocr_saved = transcript_saved = 0
    for row in manifest.iter_built_assets(db_path, limit=limit):
        if asset_id and row["id"] != asset_id:
            continue
        asset = manifest.get_asset(db_path, row["id"])
        if not asset:
            continue
        processed += 1
        ocr_source: Optional[Path] = None
        if asset.get("media_type") in {"image", "raw"} and asset.get("managed_path"):
            candidate = managed_library_dir / str(asset["managed_path"])
            if candidate.exists():
                ocr_source = candidate
        elif asset.get("thumbnail_path"):
            candidate = Path(asset["thumbnail_path"])
            if candidate.exists():
                ocr_source = candidate
        if ocr_source:
            ocr_text, ocr_tool = _ocr_text_from_image(ocr_source)
            manifest.record_extraction_result(
                db_path,
                row["id"],
                result_type="ocr",
                text_content=ocr_text,
                payload={"tool": ocr_tool, "source_path": str(ocr_source)},
                status="READY" if ocr_text else "SKIPPED",
            )
            if ocr_text:
                manifest.set_metadata_field(
                    db_path,
                    row["id"],
                    field_name="ocr_text",
                    value=ocr_text,
                    source_name="ocr",
                    source_field=ocr_tool,
                    is_canonical=False,
                    confidence=0.9,
                )
                ocr_saved += 1
        if asset.get("media_type") == "video" and asset.get("managed_path"):
            video_path = managed_library_dir / str(asset["managed_path"])
            transcript_text, tool_name = _transcribe_video(video_path, model_name=whisper_model)
            manifest.record_extraction_result(
                db_path,
                row["id"],
                result_type="transcript",
                text_content=transcript_text,
                payload={"tool": tool_name, "source_path": str(video_path)},
                status="READY" if transcript_text else "SKIPPED",
            )
            if transcript_text:
                manifest.set_metadata_field(
                    db_path,
                    row["id"],
                    field_name="transcript",
                    value=transcript_text,
                    source_name="transcript",
                    source_field=tool_name,
                    is_canonical=False,
                    confidence=0.85,
                )
                transcript_saved += 1
    index_result = index_asset_embeddings(db_path, managed_library_dir, asset_id=asset_id, limit=limit)
    return {
        "processed": processed,
        "ocr_saved": ocr_saved,
        "transcript_saved": transcript_saved,
        **index_result,
    }


def _asset_search_text(row: dict[str, Any]) -> str:
    keywords = []
    people = []
    try:
        keywords = json.loads(row.get("keywords_json") or "[]")
    except Exception:
        keywords = []
    try:
        people = json.loads(row.get("people_json") or "[]")
    except Exception:
        people = []
    return _clean_text(
        row.get("orig_filename"),
        row.get("title"),
        row.get("description"),
        row.get("location_text"),
        keywords,
        people,
        row.get("face_labels"),
        row.get("extracted_text"),
        row.get("dt_original"),
        row.get("source"),
        row.get("source_kind"),
        row.get("media_type"),
    )


def _location_text_from_metadata_rows(metadata_rows: list[dict[str, Any]]) -> str:
    location_parts: list[str] = []
    for row in metadata_rows:
        if row.get("field_name") != "location":
            continue
        value = row.get("value")
        if isinstance(value, str) and value.strip():
            location_parts.append(value.strip())
    return " ".join(location_parts).strip()


def search_assets_semantic(
    db_path: Path,
    query: Optional[str] = None,
    *,
    similar_to: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    include_hidden: bool = False,
    media_type: Optional[str] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    q = (query or "").strip()
    q_tokens = set(_tokenize(q))
    q_vec = text_embedding(q) if q else None
    con = manifest.connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT a.id, a.source, a.source_kind, a.media_type, a.orig_filename, a.dt_original,
                   a.title, a.description, a.keywords_json, a.people_json, a.gps_lat, a.gps_lon,
                   a.status, a.user_rating, a.review_state, a.review_score, a.favorite, m.managed_path,
                   COALESCE((
                     SELECT group_concat(mf.value_text, ' ')
                     FROM metadata_fields mf
                     WHERE mf.asset_id = a.id
                       AND mf.field_name = 'location'
                       AND COALESCE(mf.value_text, '') <> ''
                   ), '') AS location_text,
                   COALESCE((
                     SELECT group_concat(i.label, ' ')
                     FROM faces f
                     JOIN face_identities i ON i.id = f.identity_id
                     WHERE f.asset_id = a.id AND f.status <> 'REJECTED'
                   ), '') AS face_labels,
                   COALESCE((
                     SELECT group_concat(er.text_content, ' ')
                     FROM extraction_results er
                     WHERE er.asset_id = a.id AND er.status = 'READY'
                   ), '') AS extracted_text
            FROM assets a
            LEFT JOIN managed_assets m ON m.asset_id = a.id
            """
        ).fetchall()
        records = [dict(row) for row in rows]
        if not include_hidden:
            records = [row for row in records if not _is_hidden_asset(con, row["id"])]
        if media_type:
            records = [row for row in records if row.get("media_type") == media_type]
        if source:
            records = [row for row in records if row.get("source") == source]
        embedding_map = _prefetch_embeddings(con, {"asset_text", "asset_visual"})
        ref_text_vec: Optional[list[float]] = None
        ref_visual_vec: Optional[list[float]] = None
        ref_tokens: set[str] = set()
        if similar_to:
            ref_rows = [row for row in records if row["id"] == similar_to]
            if ref_rows:
                ref = ref_rows[0]
                ref_tokens = set(_tokenize(_asset_search_text(ref)))
                ref_text_vec = embedding_map.get((similar_to, "asset_text")) or text_embedding(_asset_search_text(ref))
                ref_visual_vec = embedding_map.get((similar_to, "asset_visual"))
                if ref_visual_vec is None:
                    ref_visual_vec = embedding_map.get((similar_to, "asset_text"))
        scored: list[dict[str, Any]] = []
        for row in records:
            if row["id"] == similar_to:
                continue
            text_blob = _asset_search_text(row)
            tokens = set(_tokenize(text_blob))
            people_text = _clean_text(row.get("people_json"), row.get("face_labels"))
            location_text = _clean_text(row.get("location_text"))
            people_tokens = set(_tokenize(people_text))
            location_tokens = set(_tokenize(location_text))
            score = 0.0
            reasons: list[str] = []
            stored_text_vec = embedding_map.get((row["id"], "asset_text")) or text_embedding(text_blob)
            if q_vec is not None:
                text_similarity = _cosine(q_vec, stored_text_vec)
                token_overlap = len(q_tokens & tokens) / max(1, len(q_tokens))
                score += text_similarity * 0.62 + token_overlap * 0.26
                if text_similarity > 0.2:
                    reasons.append(f"text={text_similarity:.2f}")
                if token_overlap > 0:
                    reasons.append(f"tokens={token_overlap:.2f}")
                people_overlap = len(q_tokens & people_tokens) / max(1, len(q_tokens))
                if people_overlap > 0:
                    score += 0.32 + people_overlap * 0.28
                    reasons.append("people match")
                location_overlap = len(q_tokens & location_tokens) / max(1, len(q_tokens))
                if location_overlap > 0:
                    score += 0.28 + location_overlap * 0.24
                    reasons.append("location match")
                if row.get("extracted_text") and q_tokens & set(_tokenize(str(row.get("extracted_text")))):
                    score += 0.14
                    reasons.append("ocr/transcript match")
            if ref_text_vec is not None:
                ref_text_score = _cosine(ref_text_vec, stored_text_vec)
                score += ref_text_score * 0.25
                if ref_text_score > 0.2:
                    reasons.append(f"ref-text={ref_text_score:.2f}")
            if ref_visual_vec is not None:
                cand_visual_vec = embedding_map.get((row["id"], "asset_visual"))
                if cand_visual_vec:
                    ref_visual_score = _cosine(ref_visual_vec, cand_visual_vec)
                    score += ref_visual_score * 0.22
                    if ref_visual_score > 0.2:
                        reasons.append(f"visual={ref_visual_score:.2f}")
            if row.get("dt_original") and q and row["dt_original"][:10] in q:
                score += 0.05
                reasons.append("date match")
            if row.get("gps_lat") is not None and row.get("gps_lon") is not None and q_tokens & {"photo", "map", "location", "where", "place", "places"}:
                score += 0.05
                reasons.append("location present")
            if row.get("user_rating"):
                rating_boost = min(0.12, float(row["user_rating"]) * 0.02)
                score += rating_boost
                reasons.append(f"rating={row['user_rating']}")
            if row.get("review_score") is not None:
                review_boost = min(0.08, max(0.0, float(row["review_score"])) * 0.01)
                score += review_boost
                reasons.append(f"review={float(row['review_score']):.2f}")
            if str(row.get("review_state") or "").upper() in {"REVIEWED", "RANKED", "KEPT"}:
                score += 0.03
                reasons.append(f"state={row['review_state']}")
            if not q and similar_to is None:
                score += 0.01
            if score > 0:
                scored.append(
                    {
                        **row,
                        "score": round(score, 4),
                        "reasons": reasons[:4],
                    }
                )
        scored.sort(key=lambda item: (item["score"], item.get("dt_original") or ""), reverse=True)
        paged = scored[offset : offset + limit]
        return {"query": q, "similar_to": similar_to, "items": paged, "count": len(scored)}
    finally:
        con.close()


def _stored_embedding(con, asset_id: str, embedding_type: str) -> Optional[list[float]]:
    row = con.execute(
        """
        SELECT payload_json
        FROM embeddings
        WHERE asset_id = ? AND embedding_type = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (asset_id, embedding_type),
    ).fetchone()
    if not row:
        return None
    try:
        payload = json.loads(row["payload_json"] or "{}")
    except Exception:
        payload = {}
    vector = payload.get("vector") or []
    return [float(item) for item in vector]


def _prefetch_embeddings(con, embedding_types: set[str]) -> dict[tuple[str, str], list[float]]:
    if not embedding_types:
        return {}
    placeholders = ", ".join("?" for _ in embedding_types)
    rows = con.execute(
        f"""
        SELECT asset_id, embedding_type, payload_json
        FROM embeddings
        WHERE embedding_type IN ({placeholders})
        ORDER BY created_at DESC, id DESC
        """,
        tuple(sorted(embedding_types)),
    ).fetchall()
    out: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (row["asset_id"], row["embedding_type"])
        if key in out:
            continue
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except Exception:
            payload = {}
        vector = payload.get("vector") or []
        out[key] = [float(item) for item in vector]
    return out


def _is_hidden_asset(con, asset_id: str) -> bool:
    row = con.execute(
        """
        SELECT 1
        FROM duplicate_items
        WHERE asset_id = ? AND keep_decision = 'HIDE'
        LIMIT 1
        """,
        (asset_id,),
    ).fetchone()
    return bool(row)
