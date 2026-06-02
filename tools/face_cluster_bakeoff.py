from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import networkx as nx
import numpy as np

try:
    from sklearn.cluster import AgglomerativeClustering, DBSCAN
    SKLEARN_AVAILABLE = True
except Exception:
    AgglomerativeClustering = None  # type: ignore[assignment]
    DBSCAN = None  # type: ignore[assignment]
    SKLEARN_AVAILABLE = False

from literoom.config import load_config
from literoom.metadata import manifest


MIN_ASSET_EVIDENCE = 10


@dataclass
class Dataset:
    face_ids: list[str]
    asset_ids: list[str]
    vectors: np.ndarray
    similarity: np.ndarray


def _dominant_vector_length(items: list[dict]) -> int:
    counts: dict[int, int] = {}
    for item in items:
        vector = item.get("vector") or []
        length = len(vector)
        if length > 0:
            counts[length] = counts.get(length, 0) + 1
    if not counts:
        return 0
    return max(counts.items(), key=lambda pair: (pair[1], pair[0]))[0]


def load_dataset(config_path: Path, model_name: str) -> Dataset:
    config, resolved = load_config(config_path)
    db_path = config.db_path(resolved)
    rows = manifest.list_face_embeddings(db_path, model_name=model_name)
    dominant = _dominant_vector_length(rows)
    rows = [row for row in rows if len(row.get("vector") or []) == dominant and not row.get("identity_id")]
    face_ids = [str(row["face_id"]) for row in rows]
    asset_ids = [str(row["asset_id"]) for row in rows]
    vectors = np.asarray([row["vector"] for row in rows], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    vectors = vectors / norms
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, 1.0)
    return Dataset(face_ids=face_ids, asset_ids=asset_ids, vectors=vectors, similarity=similarity)


def labels_from_components(components: Iterable[Iterable[int]], count: int) -> np.ndarray:
    labels = np.full(count, -1, dtype=np.int32)
    for cluster_id, nodes in enumerate(components):
        for node in nodes:
            labels[int(node)] = cluster_id
    return labels


def method_threshold_cc(data: Dataset) -> np.ndarray:
    link_threshold = 0.82
    pairwise_threshold = 0.78
    graph = nx.Graph()
    graph.add_nodes_from(range(len(data.face_ids)))
    hits = np.argwhere(np.triu(data.similarity >= link_threshold, k=1))
    graph.add_edges_from((int(i), int(j)) for i, j in hits)
    kept: list[list[int]] = []
    for component in nx.connected_components(graph):
        nodes = sorted(component)
        if len({data.asset_ids[index] for index in nodes}) < MIN_ASSET_EVIDENCE:
            continue
        block = data.similarity[np.ix_(nodes, nodes)]
        if len(nodes) > 1:
            tri = block[np.triu_indices(len(nodes), k=1)]
            if float(tri.mean()) < pairwise_threshold:
                continue
        kept.append(nodes)
    return labels_from_components(kept, len(data.face_ids))


def method_dbscan(data: Dataset) -> np.ndarray:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is not installed")
    distance = np.clip(1.0 - data.similarity, 0.0, 2.0)
    return DBSCAN(eps=0.22, min_samples=3, metric="precomputed").fit_predict(distance)


def method_agglomerative_complete(data: Dataset) -> np.ndarray:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is not installed")
    distance = np.clip(1.0 - data.similarity, 0.0, 2.0)
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="complete",
        distance_threshold=0.24,
    )
    return model.fit_predict(distance)


def method_finch_like(data: Dataset) -> np.ndarray:
    similarity = data.similarity.copy()
    np.fill_diagonal(similarity, -1.0)
    first = np.argmax(similarity, axis=1)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(data.face_ids)))
    for idx, nbr in enumerate(first):
        if data.similarity[idx, nbr] >= 0.75:
            graph.add_edge(int(idx), int(nbr))
    groups: dict[int, list[int]] = {}
    for idx, nbr in enumerate(first):
        if data.similarity[idx, nbr] < 0.75:
            continue
        groups.setdefault(int(nbr), []).append(int(idx))
    for nodes in groups.values():
        if len(nodes) > 1:
            head = nodes[0]
            for node in nodes[1:]:
                graph.add_edge(head, node)
    return labels_from_components(nx.connected_components(graph), len(data.face_ids))


def method_chinese_whispers(data: Dataset) -> np.ndarray:
    k = 10
    threshold = 0.76
    graph = nx.Graph()
    graph.add_nodes_from(range(len(data.face_ids)))
    for idx in range(len(data.face_ids)):
        order = np.argsort(data.similarity[idx])[-(k + 1) :]
        for nbr in order:
            if nbr == idx:
                continue
            weight = float(data.similarity[idx, nbr])
            if weight >= threshold:
                graph.add_edge(int(idx), int(nbr), weight=weight)
    labels = {node: node for node in graph.nodes}
    rng = np.random.default_rng(7)
    nodes = list(graph.nodes)
    for _ in range(20):
        rng.shuffle(nodes)
        changed = False
        for node in nodes:
            scores: dict[int, float] = {}
            for nbr, payload in graph[node].items():
                label = labels[nbr]
                scores[label] = scores.get(label, 0.0) + float(payload.get("weight", 1.0))
            if not scores:
                continue
            best_label = max(scores.items(), key=lambda item: (item[1], -item[0]))[0]
            if labels[node] != best_label:
                labels[node] = best_label
                changed = True
        if not changed:
            break
    groups: dict[int, list[int]] = {}
    for node, label in labels.items():
        groups.setdefault(int(label), []).append(int(node))
    return labels_from_components(groups.values(), len(data.face_ids))


def method_knn_density(data: Dataset) -> np.ndarray:
    k = 12
    edge_threshold = 0.74
    density_threshold = 0.77
    graph = nx.Graph()
    graph.add_nodes_from(range(len(data.face_ids)))
    neighbors: list[list[int]] = []
    local_density = np.zeros(len(data.face_ids), dtype=np.float32)
    for idx in range(len(data.face_ids)):
        order = np.argsort(data.similarity[idx])[-(k + 1) :]
        chosen = [int(nbr) for nbr in order if nbr != idx and data.similarity[idx, nbr] >= edge_threshold]
        neighbors.append(chosen)
        if chosen:
            local_density[idx] = float(np.mean([data.similarity[idx, nbr] for nbr in chosen]))
    for idx, nbrs in enumerate(neighbors):
        if local_density[idx] < density_threshold:
            continue
        for nbr in nbrs:
            if local_density[nbr] < density_threshold:
                continue
            graph.add_edge(int(idx), int(nbr), weight=float(data.similarity[idx, nbr]))
    return labels_from_components(nx.connected_components(graph), len(data.face_ids))


def cluster_stats(name: str, data: Dataset, labels: np.ndarray, runtime_s: float) -> dict:
    clusters: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        if int(label) < 0:
            continue
        clusters.setdefault(int(label), []).append(idx)
    qualified: list[list[int]] = []
    for nodes in clusters.values():
        unique_assets = {data.asset_ids[index] for index in nodes}
        if len(unique_assets) >= MIN_ASSET_EVIDENCE:
            qualified.append(nodes)
    cluster_sizes = [len(nodes) for nodes in qualified]
    asset_sizes = [len({data.asset_ids[index] for index in nodes}) for nodes in qualified]
    covered_faces = sum(cluster_sizes)
    covered_assets = len({data.asset_ids[index] for nodes in qualified for index in nodes})
    mean_similarity = 0.0
    p10_similarity = 0.0
    nearest_other_centroid = 0.0
    if qualified:
        pair_scores: list[float] = []
        centroids: list[np.ndarray] = []
        for nodes in qualified:
            if len(nodes) > 1:
                block = data.similarity[np.ix_(nodes, nodes)]
                pair_scores.extend(block[np.triu_indices(len(nodes), k=1)].tolist())
            centroid = data.vectors[nodes].mean(axis=0)
            norm = np.linalg.norm(centroid) or 1.0
            centroids.append((centroid / norm).astype(np.float32))
        if pair_scores:
            mean_similarity = float(np.mean(pair_scores))
            p10_similarity = float(np.percentile(pair_scores, 10))
        if len(centroids) > 1:
            centroid_matrix = np.stack(centroids, axis=0)
            centroid_sim = centroid_matrix @ centroid_matrix.T
            np.fill_diagonal(centroid_sim, -1.0)
            nearest_other_centroid = float(np.max(centroid_sim, axis=1).mean())
    return {
        "method": name,
        "runtime_s": round(runtime_s, 2),
        "qualified_clusters": len(qualified),
        "covered_faces": covered_faces,
        "covered_assets": covered_assets,
        "mean_cluster_faces": round(float(np.mean(cluster_sizes)) if cluster_sizes else 0.0, 2),
        "max_cluster_faces": max(cluster_sizes, default=0),
        "mean_cluster_assets": round(float(np.mean(asset_sizes)) if asset_sizes else 0.0, 2),
        "max_cluster_assets": max(asset_sizes, default=0),
        "mean_pairwise_similarity": round(mean_similarity, 4),
        "p10_pairwise_similarity": round(p10_similarity, 4),
        "mean_nearest_other_centroid_similarity": round(nearest_other_centroid, 4),
    }


def run_bakeoff(config_path: Path, model_name: str) -> list[dict]:
    data = load_dataset(config_path, model_name)
    methods: list[tuple[str, Callable[[Dataset], np.ndarray]]] = [
        ("threshold_cc_relaxed", method_threshold_cc),
        ("dbscan_cosine", method_dbscan),
        ("agglomerative_complete", method_agglomerative_complete),
        ("finch_like", method_finch_like),
        ("chinese_whispers_knn", method_chinese_whispers),
        ("knn_density_graph", method_knn_density),
    ]
    results: list[dict] = []
    for name, fn in methods:
        started = time.perf_counter()
        try:
            labels = fn(data)
            runtime_s = time.perf_counter() - started
            results.append(cluster_stats(name, data, labels, runtime_s))
        except Exception as exc:
            runtime_s = time.perf_counter() - started
            results.append(
                {
                    "method": name,
                    "runtime_s": round(runtime_s, 2),
                    "error": str(exc),
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="literoom.local.yaml")
    parser.add_argument("--model-name", default="insightface_antelopev2")
    args = parser.parse_args()
    results = run_bakeoff(Path(args.config), args.model_name)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
