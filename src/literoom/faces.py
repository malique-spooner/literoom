from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .metadata import manifest

MIN_CLUSTER_ASSET_EVIDENCE = 10
AUTO_ASSIGN_BASE_THRESHOLD = 0.95
AUTO_ASSIGN_FLOOR = 0.50
AUTO_ASSIGN_DROP_PER_FACE = 0.01
INITIAL_CLUSTER_LINK_FLOOR = 0.82
INITIAL_CLUSTER_PAIRWISE_FLOOR = 0.78
FINCH_NEIGHBOR_FLOOR = 0.72
FINCH_CLUSTER_PAIRWISE_FLOOR = 0.67
FINCH_COMPONENT_MERGE_FLOOR = 0.995
CLUSTER_MERGE_FLOOR = 0.58
CLUSTER_MERGE_CENTROID_FLOOR = 0.78
CLUSTER_MERGE_CENTROID_SLACK = 0.03
CLUSTER_MERGE_CROSS_MEAN_FLOOR = 0.76
CLUSTER_MERGE_DIRECTIONAL_MEAN_FLOOR = 0.72
CLUSTER_MERGE_SUPPORT_FLOOR = 0.72
CLUSTER_MERGE_STRONG_PAIR_FLOOR = 0.84
CLUSTER_MERGE_COMBINED_COHESION_FLOOR = 0.70
CLUSTER_MERGE_COHESION_DROP_LIMIT = 0.16
CLUSTER_MERGE_HIGH_CONFIDENCE_FLOOR = 0.80
CLUSTER_MERGE_HIGH_CONFIDENCE_PAIR_FLOOR = 0.88
CLUSTER_MERGE_HIGH_CONFIDENCE_TOP_MEAN_FLOOR = 0.79


class FaceRecognitionQualityError(RuntimeError):
    """Raised when a requested face recognition run cannot meet the quality bar."""


def _cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency fallback
        raise RuntimeError("OpenCV is required for face detection") from exc
    return cv2


def _insightface_home() -> Path:
    configured = os.environ.get("INSIGHTFACE_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.cwd() / ".literoom" / "cache" / "insightface").resolve()


def _insightface_model_dir(model_name: str) -> Path:
    return (_insightface_home() / "models" / model_name).resolve()


def _insightface_nested_dir(model_name: str) -> Path:
    return (_insightface_model_dir(model_name) / model_name).resolve()


def _flatten_insightface_pack(model_name: str) -> Path:
    model_dir = _insightface_model_dir(model_name)
    nested_dir = _insightface_nested_dir(model_name)
    if not nested_dir.exists():
        return model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    existing_onnx = list(model_dir.glob("*.onnx"))
    if existing_onnx:
        return model_dir
    nested_onnx = [path for path in sorted(nested_dir.glob("*.onnx")) if not path.name.startswith("._")]
    if not nested_onnx:
        return model_dir
    for source in nested_onnx:
        target = model_dir / source.name
        if target.exists():
            continue
        try:
            shutil.copy2(source, target)
        except Exception:
            try:
                os.symlink(source, target)
            except Exception:
                continue
    return model_dir


@lru_cache(maxsize=4)
def _insightface_app(model_name: str):
    try:
        from insightface.app import FaceAnalysis  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency fallback
        raise RuntimeError("InsightFace is required for face analysis") from exc
    _flatten_insightface_pack(model_name)
    app = FaceAnalysis(name=model_name, root=str(_insightface_home()), providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def _ensure_insightface_ready(model_name: str) -> None:
    try:
        from insightface.app import FaceAnalysis  # noqa: F401
    except Exception as exc:  # pragma: no cover - dependency environment specific
        raise FaceRecognitionQualityError(
            f"InsightFace is required for face recognition quality with model '{model_name}'. "
            "Install the 'insightface' Python package to enable face detection."
        ) from exc


def _insightface_detections(image: np.ndarray, *, model_name: str) -> List[Dict[str, object]]:
    app = _insightface_app(model_name)
    detections: List[Dict[str, object]] = []
    faces = app.get(image) or []
    for face in faces:
        bbox = getattr(face, "bbox", None)
        if bbox is None:
            continue
        try:
            x1, y1, x2, y2 = [float(value) for value in bbox]
        except Exception:
            continue
        x = max(0, int(round(min(x1, x2))))
        y = max(0, int(round(min(y1, y2))))
        w = max(1, int(round(abs(x2 - x1))))
        h = max(1, int(round(abs(y2 - y1))))
        crop = image[max(0, y): max(0, y + h), max(0, x): max(0, x + w)]
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            embedding = getattr(face, "embedding", None)
        vector = []
        if embedding is not None:
            try:
                vector = np.asarray(embedding, dtype="float32").reshape(-1).tolist()
            except Exception:
                vector = []
        det_score = float(getattr(face, "det_score", 0.0) or 0.0)
        quality = _face_quality(crop) if crop.size else det_score * 100.0
        if quality < 28 and det_score < 0.35:
            continue
        detections.append(
            {
                "bbox": {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "image_width": int(image.shape[1]),
                    "image_height": int(image.shape[0]),
                    "quality": round(max(quality, det_score * 100.0), 2),
                },
                "embedding_vector": vector or _compute_embedding(crop),
            }
        )
    return detections


def _detectors() -> list[cv2.CascadeClassifier]:
    cv2 = _cv2()
    cascade_names = [
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_profileface.xml",
    ]
    candidate_dirs: list[Path] = []
    data_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if data_dir:
        candidate_dirs.append(Path(str(data_dir)))
    candidate_dirs.extend(
        [
            Path("/opt/homebrew/opt/opencv/share/opencv4/haarcascades"),
            Path("/opt/homebrew/Cellar/opencv/4.13.0_10/share/opencv4/haarcascades"),
            Path("/usr/local/opt/opencv/share/opencv4/haarcascades"),
        ]
    )
    module_path = Path(getattr(cv2, "__file__", "")).resolve()
    candidate_dirs.extend(
        [
            module_path.parent / "data",
            module_path.parent.parent / "share" / "opencv4" / "haarcascades",
            module_path.parent.parent.parent / "share" / "opencv4" / "haarcascades",
        ]
    )
    seen: set[str] = set()
    detectors: list[cv2.CascadeClassifier] = []
    for name in cascade_names:
        for base_dir in candidate_dirs:
            if not base_dir:
                continue
            candidate = (base_dir / name).resolve()
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if not candidate.exists():
                continue
            detector = cv2.CascadeClassifier(str(candidate))
            if not detector.empty():
                detectors.append(detector)
                break
    if not detectors:
        raise RuntimeError("OpenCV face cascades could not be loaded")
    return detectors


def _load_detection_image(asset: Dict, managed_library_dir: Path) -> tuple[Optional[str], Optional[int]]:
    if asset.get("media_type") == "video" and asset.get("thumbnail_path"):
        return asset["thumbnail_path"], 1000
    managed_path = asset.get("managed_path")
    if managed_path:
        return str((managed_library_dir / managed_path).resolve()), None
    return None, None


def _video_frame_offsets(duration_seconds: Optional[float]) -> List[float]:
    if not duration_seconds or duration_seconds <= 0:
        return [0.75, 1.5, 3.0]
    anchors = [0.15, 0.5, 0.85]
    offsets: List[float] = []
    for anchor in anchors:
        offset = max(0.5, min(duration_seconds * anchor, max(0.5, duration_seconds - 0.5)))
        if all(abs(offset - existing) > 0.35 for existing in offsets):
            offsets.append(offset)
    return offsets or [max(0.5, duration_seconds / 2.0)]


def _extract_video_frame(video_path: Path, offset_seconds: float) -> Optional[str]:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin or not video_path.exists():
        return None
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        frame_path = tmp.name
    proc = subprocess.run(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{offset_seconds:.2f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            frame_path,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not Path(frame_path).exists():
        try:
            Path(frame_path).unlink(missing_ok=True)
        except Exception:
            pass
        return None
    return frame_path


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    a_area = aw * ah
    b_area = bw * bh
    union = a_area + b_area - inter_area
    return inter_area / union if union else 0.0


def _filter_boxes(boxes: List[Tuple[int, int, int, int]], width: int, height: int) -> List[Tuple[int, int, int, int]]:
    image_area = max(1, width * height)
    kept: List[Tuple[int, int, int, int]] = []
    for x, y, w, h in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
        area_fraction = (w * h) / image_area
        aspect_ratio = (w / h) if h else 0
        if area_fraction < 0.012 or area_fraction > 0.7:
            continue
        if aspect_ratio < 0.55 or aspect_ratio > 1.6:
            continue
        if any(_iou((x, y, w, h), existing) > 0.3 for existing in kept):
            continue
        kept.append((x, y, w, h))
        if len(kept) >= 16:
            break
    return kept


def _face_quality(crop: np.ndarray) -> float:
    if crop.size == 0:
        return 0.0
    cv2 = _cv2()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    brightness = float(gray.mean())
    size_bonus = min(gray.shape[0], gray.shape[1]) / 160.0
    return sharpness * 0.45 + contrast * 1.2 + brightness * 0.1 + size_bonus * 100.0


def _compute_embedding(crop: np.ndarray) -> List[float]:
    cv2 = _cv2()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    normalized = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(normalized, (3, 3), 0)

    def _feature_map(img: np.ndarray) -> np.ndarray:
        resized = cv2.resize(img, (12, 12), interpolation=cv2.INTER_AREA)
        pooled = resized.astype("float32").reshape(-1)
        pooled -= float(pooled.mean())
        pooled_std = float(pooled.std()) or 1.0
        pooled /= pooled_std

        grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        angle = (cv2.phase(grad_x, grad_y, angleInDegrees=False) % (2.0 * math.pi)) / (2.0 * math.pi)
        edge_hist = np.zeros(8, dtype="float32")
        for idx in range(8):
            mask = (angle >= idx / 8.0) & (angle < (idx + 1) / 8.0)
            edge_hist[idx] = float(magnitude[mask].sum()) if np.any(mask) else 0.0
        edge_hist_sum = float(edge_hist.sum()) or 1.0
        edge_hist /= edge_hist_sum

        intensity_hist = cv2.calcHist([img], [0], None, [16], [0, 256]).astype("float32").reshape(-1)
        intensity_hist_sum = float(intensity_hist.sum()) or 1.0
        intensity_hist /= intensity_hist_sum

        combined = np.concatenate([pooled, edge_hist, intensity_hist], axis=0)
        return combined

    vector = (_feature_map(normalized) + _feature_map(cv2.flip(blurred, 1))) / 2.0
    norm = float(np.linalg.norm(vector)) or 1.0
    vector /= norm
    return vector.astype("float32").tolist()


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    aa = np.array(a, dtype="float32")
    bb = np.array(b, dtype="float32")
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb)) or 1.0
    return float(np.dot(aa, bb) / denom)


def _vector_length(item: Dict[str, object]) -> int:
    vector = item.get("vector") or []
    return len(vector) if isinstance(vector, list) else 0


def _dominant_vector_length(items: List[Dict[str, object]]) -> int:
    lengths: Dict[int, int] = {}
    for item in items:
        length = _vector_length(item)
        if length > 0:
            lengths[length] = lengths.get(length, 0) + 1
    if not lengths:
        return 0
    return max(lengths.items(), key=lambda pair: (pair[1], pair[0]))[0]


def _filter_vector_length(items: List[Dict[str, object]], vector_length: int) -> List[Dict[str, object]]:
    if vector_length <= 0:
        return [item for item in items if _vector_length(item) > 0]
    return [item for item in items if _vector_length(item) == vector_length]


def _detect_boxes(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    cv2 = _cv2()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    collected: List[Tuple[int, int, int, int]] = []
    for detector in _detectors():
        boxes = detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=6, minSize=(56, 56))
        collected.extend(tuple(map(int, box)) for box in boxes)
    return _filter_boxes(collected, gray.shape[1], gray.shape[0])


def _cluster_label(index: int) -> str:
    return f"Cluster {index:03d}"


def _pairwise_similarity(items: List[Dict[str, object]]) -> float:
    if len(items) < 2:
        return 1.0
    scores: List[float] = []
    for i, left in enumerate(items):
        for right in items[i + 1 :]:
            scores.append(_cosine_similarity(left["vector"], right["vector"]))
    return float(sum(scores) / len(scores)) if scores else 0.0


def _centroid_from_vectors(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    arr = np.array(vectors, dtype="float32")
    mean = arr.mean(axis=0)
    norm = float(np.linalg.norm(mean)) or 1.0
    return (mean / norm).astype("float32").tolist()


def _average_centroid_similarity(vectors: List[List[float]]) -> float:
    if len(vectors) < 2:
        return 1.0
    centroid = _centroid_from_vectors(vectors)
    if not centroid:
        return 0.0
    scores = [_cosine_similarity(vector, centroid) for vector in vectors]
    return float(sum(scores) / len(scores)) if scores else 0.0


def _normalized_vector_array(vectors: List[List[float]]) -> np.ndarray:
    if not vectors:
        return np.empty((0, 0), dtype="float32")
    arr = np.array(vectors, dtype="float32")
    if arr.ndim != 2:
        return np.empty((0, 0), dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return arr / norms


def _cluster_merge_support_needed(cluster_size: int) -> int:
    return max(2, min(6, int(math.ceil(max(1, cluster_size) * 0.2))))


def _cluster_merge_evidence(
    left: Dict[str, object],
    right: Dict[str, object],
    *,
    similarity_threshold: float,
) -> Optional[Dict[str, float]]:
    left_vectors = left.get("vectors") or []
    right_vectors = right.get("vectors") or []
    if not isinstance(left_vectors, list) or not isinstance(right_vectors, list):
        return None
    if len(left_vectors) < 2 or len(right_vectors) < 2:
        return None
    left_assets = set(left.get("asset_ids") or set())
    right_assets = set(right.get("asset_ids") or set())
    if left_assets and right_assets and left_assets.intersection(right_assets):
        return None

    left_centroid = _centroid_from_vectors(left_vectors)
    right_centroid = _centroid_from_vectors(right_vectors)
    centroid_score = _cosine_similarity(left_centroid, right_centroid)
    centroid_floor = max(CLUSTER_MERGE_CENTROID_FLOOR, similarity_threshold)
    if centroid_score < CLUSTER_MERGE_FLOOR:
        return None

    left_arr = _normalized_vector_array(left_vectors)
    right_arr = _normalized_vector_array(right_vectors)
    if left_arr.size == 0 or right_arr.size == 0 or left_arr.shape[1] != right_arr.shape[1]:
        return None
    cross_scores = left_arr @ right_arr.T
    best_pair = float(cross_scores.max(initial=0.0))
    top_count = max(1, min(12, int(math.ceil(min(len(left_vectors), len(right_vectors)) * 0.35))))
    top_scores = np.partition(cross_scores.reshape(-1), -top_count)[-top_count:]
    top_mean = float(top_scores.mean()) if top_scores.size else 0.0
    left_best = cross_scores.max(axis=1)
    right_best = cross_scores.max(axis=0)
    left_directional_mean = float(left_best.mean()) if left_best.size else 0.0
    right_directional_mean = float(right_best.mean()) if right_best.size else 0.0
    directional_mean = min(left_directional_mean, right_directional_mean)
    high_confidence_merge = (
        centroid_score >= CLUSTER_MERGE_HIGH_CONFIDENCE_FLOOR
        and best_pair >= CLUSTER_MERGE_HIGH_CONFIDENCE_PAIR_FLOOR
        and top_mean >= CLUSTER_MERGE_HIGH_CONFIDENCE_TOP_MEAN_FLOOR
        and directional_mean >= CLUSTER_MERGE_DIRECTIONAL_MEAN_FLOOR - 0.02
    )
    left_support = int((left_best >= CLUSTER_MERGE_SUPPORT_FLOOR).sum())
    right_support = int((right_best >= CLUSTER_MERGE_SUPPORT_FLOOR).sum())
    support_needed = _cluster_merge_support_needed(min(len(left_vectors), len(right_vectors)))
    if not high_confidence_merge and (left_support < support_needed or right_support < support_needed):
        return None
    if directional_mean < CLUSTER_MERGE_DIRECTIONAL_MEAN_FLOOR:
        return None

    left_cohesion = _average_centroid_similarity(left_vectors)
    right_cohesion = _average_centroid_similarity(right_vectors)
    combined_vectors = left_vectors + right_vectors
    combined_cohesion = _average_centroid_similarity(combined_vectors)
    cohesion_drop = min(left_cohesion, right_cohesion) - combined_cohesion
    if combined_cohesion < CLUSTER_MERGE_COMBINED_COHESION_FLOOR and not (
        high_confidence_merge and combined_cohesion >= CLUSTER_MERGE_COMBINED_COHESION_FLOOR - 0.03
    ):
        return None
    if cohesion_drop > CLUSTER_MERGE_COHESION_DROP_LIMIT and combined_cohesion < centroid_floor:
        return None

    near_centroid_match = (
        centroid_score >= centroid_floor - CLUSTER_MERGE_CENTROID_SLACK
        and top_mean >= CLUSTER_MERGE_CROSS_MEAN_FLOOR
    )
    strong_bridge_match = (
        centroid_score >= max(CLUSTER_MERGE_FLOOR, centroid_floor - 0.12)
        and best_pair >= CLUSTER_MERGE_STRONG_PAIR_FLOOR
        and top_mean >= CLUSTER_MERGE_CROSS_MEAN_FLOOR
    )
    if not near_centroid_match and not strong_bridge_match:
        return None

    score = (centroid_score * 0.45) + (top_mean * 0.35) + (directional_mean * 0.20)
    return {
        "score": score,
        "centroid_score": centroid_score,
        "top_mean": top_mean,
        "directional_mean": directional_mean,
        "combined_cohesion": combined_cohesion,
    }


def _cluster_auto_assign_threshold(
    identity: Dict[str, object],
    vectors: List[List[float]],
) -> float:
    face_count = max(int(identity.get("face_count") or 0), len(vectors))
    return max(AUTO_ASSIGN_FLOOR, AUTO_ASSIGN_BASE_THRESHOLD - max(0, face_count - 5) * AUTO_ASSIGN_DROP_PER_FACE)


def _confirmed_auto_assign_threshold(
    identity: Dict[str, object],
    vectors: List[List[float]],
) -> float:
    face_count = max(int(identity.get("face_count") or 0), len(vectors))
    return max(AUTO_ASSIGN_FLOOR, AUTO_ASSIGN_BASE_THRESHOLD - max(0, face_count - 5) * AUTO_ASSIGN_DROP_PER_FACE)


def _match_unlabeled_faces(
    db_path: Path,
    items: List[Dict[str, object]],
    *,
    identity_index: Dict[str, Dict[str, object]],
    confirmed_identity_ids: set[str],
    centroids: Dict[str, List[List[float]]],
    clarification_threshold: float,
) -> Tuple[int, List[Dict[str, object]], set[str]]:
    def centroid_for(identity_id: str) -> List[float]:
        return _centroid_from_vectors(centroids.get(identity_id) or [])

    assigned = 0
    unmatched: List[Dict[str, object]] = []
    identities_to_propagate: set[str] = set()
    for item in items:
        if not item.get("vector"):
            continue
        current_identity = item.get("identity_id")
        if current_identity:
            continue
        best_identity: Optional[str] = None
        best_score = 0.0
        for identity_id in centroids:
            if identity_id == current_identity:
                continue
            score = _cosine_similarity(item["vector"], centroid_for(identity_id))
            if score > best_score:
                best_score = score
                best_identity = identity_id
        if not best_identity:
            unmatched.append(item)
            continue

        best_is_confirmed = best_identity in confirmed_identity_ids
        identity_vectors = centroids.get(best_identity) or []
        if best_is_confirmed:
            auto_assign_threshold = _confirmed_auto_assign_threshold(
                identity_index.get(best_identity, {}),
                identity_vectors,
            )
        else:
            auto_assign_threshold = _cluster_auto_assign_threshold(
                identity_index.get(best_identity, {}),
                identity_vectors,
            )
        if best_score >= auto_assign_threshold:
            manifest.update_face_cluster(
                db_path,
                item["face_id"],
                best_identity,
                labeled=identity_index.get(best_identity, {}).get("status") == "CONFIRMED",
            )
            manifest.resolve_face_clarification(db_path, item["face_id"], status="RESOLVED")
            centroids.setdefault(best_identity, []).append(item["vector"])
            if best_identity in identity_index:
                identity_index[best_identity]["face_count"] = int(identity_index[best_identity].get("face_count") or 0) + 1
            assigned += 1
            if best_is_confirmed:
                identities_to_propagate.add(best_identity)
            continue
        unmatched.append(item)
    return assigned, unmatched, identities_to_propagate


def merge_similar_cluster_identities(
    db_path: Path,
    *,
    similarity_threshold: float = 0.88,
    model_name: Optional[str] = None,
) -> Dict[str, int]:
    embeddings = manifest.list_face_embeddings(db_path, model_name=model_name)
    embeddings = _filter_vector_length(embeddings, _dominant_vector_length(embeddings))
    identities = manifest.list_face_identities(db_path, limit=5000, status="CLUSTERED")
    if len(identities) < 2:
        return {"merged": 0}

    identity_index = {item["id"]: item for item in identities}
    vectors_by_identity: dict[str, list[List[float]]] = {}
    asset_ids_by_identity: dict[str, set[str]] = {}
    for item in embeddings:
        identity_id = item.get("identity_id")
        if not identity_id or identity_id not in identity_index:
            continue
        vector = item.get("vector") or []
        if vector:
            vectors_by_identity.setdefault(identity_id, []).append(vector)
            asset_id = item.get("asset_id")
            if asset_id:
                asset_ids_by_identity.setdefault(identity_id, set()).add(str(asset_id))

    active_profiles: dict[str, Dict[str, object]] = {}
    for identity_id, identity in identity_index.items():
        vectors = vectors_by_identity.get(identity_id) or []
        if len(vectors) < 2:
            continue
        active_profiles[identity_id] = {
            "id": identity_id,
            "label": str(identity.get("label") or identity_id),
            "vectors": list(vectors),
            "asset_ids": set(asset_ids_by_identity.get(identity_id) or set()),
            "count": len(vectors),
        }
    if len(active_profiles) < 2:
        return {"merged": 0}

    merge_plan: list[tuple[str, str]] = []
    while True:
        best_pair: Optional[tuple[str, str, Dict[str, float]]] = None
        ids = sorted(active_profiles)
        for i, left_id in enumerate(ids):
            for right_id in ids[i + 1 :]:
                evidence = _cluster_merge_evidence(
                    active_profiles[left_id],
                    active_profiles[right_id],
                    similarity_threshold=similarity_threshold,
                )
                if not evidence:
                    continue
                if not best_pair or (
                    evidence["score"],
                    evidence["centroid_score"],
                    evidence["combined_cohesion"],
                ) > (
                    best_pair[2]["score"],
                    best_pair[2]["centroid_score"],
                    best_pair[2]["combined_cohesion"],
                ):
                    best_pair = (left_id, right_id, evidence)
        if not best_pair:
            break

        left_id, right_id, _ = best_pair
        left = active_profiles[left_id]
        right = active_profiles[right_id]
        ordered = sorted(
            [left, right],
            key=lambda profile: (-int(profile.get("count") or 0), str(profile.get("label") or "")),
        )
        target_id = str(ordered[0]["id"])
        source_id = str(ordered[1]["id"])
        target = active_profiles[target_id]
        source = active_profiles[source_id]
        merge_plan.append((source_id, target_id))
        target["vectors"] = list(target.get("vectors") or []) + list(source.get("vectors") or [])
        target["asset_ids"] = set(target.get("asset_ids") or set()).union(set(source.get("asset_ids") or set()))
        target["count"] = int(target.get("count") or 0) + int(source.get("count") or 0)
        del active_profiles[source_id]

    merged = 0
    for source, target in merge_plan:
        manifest.merge_face_identities(db_path, source, target)
        merged += 1
    if merged:
        identities = manifest.list_face_identities(db_path, limit=5000, status="CLUSTERED")
        for identity in identities:
            manifest.refresh_people_for_identity(db_path, identity["id"])
    return {"merged": merged}


def cluster_faces(
    db_path: Path,
    *,
    similarity_threshold: float = 0.9,
    model_name: Optional[str] = None,
) -> Dict[str, int]:
    embeddings = manifest.list_face_embeddings(db_path, model_name=model_name)
    embeddings = _filter_vector_length(embeddings, _dominant_vector_length(embeddings))
    identities = manifest.list_face_identities(db_path, limit=5000)
    identity_index = {item["id"]: dict(item) for item in identities}
    confirmed_identity_ids = {
        item["id"]
        for item in identities
        if item.get("status") == "CONFIRMED"
    }

    centroids: dict[str, List[List[float]]] = {}
    for item in embeddings:
        identity_id = item.get("identity_id")
        vector = item.get("vector") or []
        if identity_id and vector:
            centroids.setdefault(identity_id, []).append(vector)

    assigned = created = 0
    identities_to_propagate: set[str] = set()
    clarification_threshold = similarity_threshold

    assigned_now, unmatched, propagate_now = _match_unlabeled_faces(
        db_path,
        embeddings,
        identity_index=identity_index,
        confirmed_identity_ids=confirmed_identity_ids,
        centroids=centroids,
        clarification_threshold=clarification_threshold,
    )
    assigned += assigned_now
    identities_to_propagate.update(propagate_now)

    if unmatched:
        finch_neighbor_threshold = max(FINCH_NEIGHBOR_FLOOR, similarity_threshold - 0.20)
        unmatched_vectors = np.array([item["vector"] for item in unmatched], dtype="float32")
        unmatched_norms = np.linalg.norm(unmatched_vectors, axis=1, keepdims=True)
        unmatched_norms[unmatched_norms == 0.0] = 1.0
        unmatched_vectors = unmatched_vectors / unmatched_norms
        similarity_matrix = unmatched_vectors @ unmatched_vectors.T
        np.fill_diagonal(similarity_matrix, -1.0)
        asset_ids = [str(item.get("asset_id") or "") for item in unmatched]
        asset_id_array = np.array(asset_ids, dtype=object)
        for left_index, asset_id in enumerate(asset_ids):
            if not asset_id:
                continue
            same_asset = asset_id_array == asset_id
            similarity_matrix[left_index, same_asset] = -1.0
        graph = {}
        primary_neighbor_indexes = np.argmax(similarity_matrix, axis=1)
        primary_neighbor_scores = similarity_matrix[np.arange(len(unmatched)), primary_neighbor_indexes]
        followers_by_index: dict[int, list[int]] = {}
        active_indexes: set[int] = set()
        for idx, best_index in enumerate(primary_neighbor_indexes.tolist()):
            best_score = float(primary_neighbor_scores[idx])
            if best_score < finch_neighbor_threshold:
                continue
            graph.setdefault(idx, set()).add(best_index)
            graph.setdefault(best_index, set()).add(idx)
            active_indexes.add(idx)
            active_indexes.add(best_index)
            followers_by_index.setdefault(best_index, []).append(idx)

        for _, follower_indexes in followers_by_index.items():
            if len(follower_indexes) < 2:
                continue
            head = follower_indexes[0]
            for follower_index in follower_indexes[1:]:
                graph.setdefault(head, set()).add(follower_index)
                graph.setdefault(follower_index, set()).add(head)
                active_indexes.add(head)
                active_indexes.add(follower_index)

        clusters: dict[int, list[dict]] = {}
        seen: set[int] = set()
        for item_index, item in enumerate(unmatched):
            if item_index not in active_indexes or item_index in seen:
                continue
            stack = [item_index]
            component: list[int] = []
            while stack:
                node = stack.pop()
                if node in seen:
                    continue
                seen.add(node)
                component.append(node)
                stack.extend(int(neighbor) for neighbor in graph.get(node, set()) if int(neighbor) not in seen)
            clusters[item_index] = [unmatched[index] for index in component]

        cluster_keys = list(clusters)
        cluster_centroids: dict[int, List[float]] = {}
        for cluster_key, cluster_items in clusters.items():
            vectors = np.array([item["vector"] for item in cluster_items], dtype="float32")
            mean = vectors.mean(axis=0)
            norm = float(np.linalg.norm(mean)) or 1.0
            cluster_centroids[cluster_key] = (mean / norm).astype("float32").tolist()
        merged_keys: set[int] = set()
        merged_clusters: dict[int, list[dict]] = {}
        for cluster_key in cluster_keys:
            if cluster_key in merged_keys:
                continue
            combined = list(clusters[cluster_key])
            merged_keys.add(cluster_key)
            for other_key in cluster_keys:
                if other_key in merged_keys:
                    continue
                if _cosine_similarity(cluster_centroids[cluster_key], cluster_centroids[other_key]) >= FINCH_COMPONENT_MERGE_FLOOR:
                    combined.extend(clusters[other_key])
                    merged_keys.add(other_key)
            merged_clusters[cluster_key] = combined
        clusters = merged_clusters

        existing_cluster_numbers = [
            int(item["label"].split()[-1])
            for item in identities
            if str(item.get("label") or "").startswith("Cluster ") and str(item["label"]).split()[-1].isdigit()
        ]
        next_cluster = max(existing_cluster_numbers, default=0) + 1
        for cluster_items in clusters.values():
            unique_assets = {item.get("asset_id") for item in cluster_items if item.get("asset_id")}
            if len(cluster_items) < 2:
                continue
            if len(unique_assets) < 2 and len(cluster_items) < 3:
                continue
            if len(unique_assets) < MIN_CLUSTER_ASSET_EVIDENCE:
                continue
            if _pairwise_similarity(cluster_items) < FINCH_CLUSTER_PAIRWISE_FLOOR:
                continue
            identity_id = manifest.create_face_identity(db_path, _cluster_label(next_cluster), status="CLUSTERED")
            next_cluster += 1
            identity_index[identity_id] = {"id": identity_id, "label": _cluster_label(next_cluster - 1), "status": "CLUSTERED", "face_count": 0}
            for item in cluster_items:
                manifest.update_face_cluster(db_path, item["face_id"], identity_id, labeled=False)
                manifest.resolve_face_clarification(db_path, item["face_id"], status="RESOLVED")
                centroids.setdefault(identity_id, []).append(item["vector"])
                identity_index[identity_id]["face_count"] = int(identity_index[identity_id].get("face_count") or 0) + 1
            created += 1

        if created:
            pending = unmatched
            for _ in range(6):
                assigned_now, pending, propagate_now = _match_unlabeled_faces(
                    db_path,
                    pending,
                    identity_index=identity_index,
                    confirmed_identity_ids=confirmed_identity_ids,
                    centroids=centroids,
                    clarification_threshold=clarification_threshold,
                )
                assigned += assigned_now
                identities_to_propagate.update(propagate_now)
                if assigned_now == 0:
                    break
            unmatched = pending

    for _ in range(8):
        consolidated = merge_similar_cluster_identities(
            db_path,
            similarity_threshold=max(CLUSTER_MERGE_FLOOR, similarity_threshold - 0.20),
            model_name=model_name,
        )
        if not consolidated.get("merged", 0):
            break
        refreshed_embeddings = manifest.list_face_embeddings(db_path, model_name=model_name)
        refreshed_embeddings = _filter_vector_length(refreshed_embeddings, _dominant_vector_length(refreshed_embeddings))
        refreshed_identities = manifest.list_face_identities(db_path, limit=5000)
        refreshed_identity_index = {item["id"]: dict(item) for item in refreshed_identities}
        refreshed_centroids: dict[str, List[List[float]]] = {}
        for item in refreshed_embeddings:
            identity_id = item.get("identity_id")
            vector = item.get("vector") or []
            if identity_id and vector:
                refreshed_centroids.setdefault(identity_id, []).append(vector)
        extra_assigned, _, extra_propagate = _match_unlabeled_faces(
            db_path,
            refreshed_embeddings,
            identity_index=refreshed_identity_index,
            confirmed_identity_ids=confirmed_identity_ids,
            centroids=refreshed_centroids,
            clarification_threshold=clarification_threshold,
        )
        assigned += extra_assigned
        identities_to_propagate.update(extra_propagate)
        if extra_assigned == 0:
            continue

    for identity_id in identities_to_propagate:
        manifest.apply_identity_to_assets(db_path, identity_id)

    return {"assigned": assigned, "clusters_created": created, "clarified": 0}


def detect_faces(
    db_path: Path,
    managed_library_dir: Path,
    *,
    limit: int | None = None,
    force: bool = False,
    face_model: Optional[str] = None,
) -> Dict[str, int]:
    active_model = (face_model or "antelopev2").strip() if face_model else "antelopev2"
    _ensure_insightface_ready(active_model)
    processed = detected = failed = embedded = 0
    for asset in manifest.iter_assets_for_face_detection(db_path=db_path, limit=limit, include_existing=force):
        processed += 1
        image_path, frame_time_ms = _load_detection_image(asset, managed_library_dir)
        source_paths: List[tuple[str, Optional[int]]] = []
        if image_path and Path(image_path).exists():
            source_paths.append((image_path, frame_time_ms))
        if asset.get("media_type") == "video":
            managed_path = asset.get("managed_path")
            video_path = (managed_library_dir / managed_path).resolve() if managed_path else None
            duration_seconds = asset.get("duration_seconds")
            if video_path and video_path.exists():
                for offset in _video_frame_offsets(float(duration_seconds) if duration_seconds is not None else None):
                    extracted = _extract_video_frame(video_path, offset)
                    if extracted:
                        source_paths.append((extracted, int(offset * 1000)))
        if not source_paths:
            failed += 1
            continue
        cv2 = _cv2()
        detections = []
        seen_boxes: List[Tuple[int, int, int, int]] = []
        extracted_temp_paths = [Path(path) for path, _ in source_paths if path not in {image_path}]
        try:
            for candidate_path, candidate_frame_ms in source_paths:
                image = cv2.imread(candidate_path)
                if image is None:
                    continue
                image_detections = _insightface_detections(image, model_name=active_model)
                for detection in image_detections:
                    bbox = detection["bbox"]
                    x = int(bbox["x"])
                    y = int(bbox["y"])
                    w = int(bbox["w"])
                    h = int(bbox["h"])
                    if any(_iou((x, y, w, h), existing) > 0.7 for existing in seen_boxes):
                        continue
                    seen_boxes.append((x, y, w, h))
                    face_id = uuid.uuid4().hex[:20]
                    detections.append(
                        {
                            "id": face_id,
                            "bbox": bbox,
                            "frame_time_ms": candidate_frame_ms,
                            "embedding_vector": detection["embedding_vector"],
                            "embedding_model_name": detection.get("embedding_model_name") or f"insightface_{active_model}",
                        }
                    )
        finally:
            for temp_path in extracted_temp_paths:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass
        source_name = f"insightface_{active_model}"
        count = manifest.replace_faces_for_asset(db_path, asset["id"], detections, source_name=source_name)
        detected += count
        for detection in detections:
            manifest.save_face_embedding(
                db_path,
                asset_id=asset["id"],
                face_id=detection["id"],
                vector=detection["embedding_vector"],
                model_name=detection.get("embedding_model_name") or source_name,
            )
            embedded += 1
    clustered = cluster_faces(db_path, model_name=source_name)
    return {
        "processed": processed,
        "detected": detected,
        "failed": failed,
        "embedded": embedded,
        **clustered,
    }
