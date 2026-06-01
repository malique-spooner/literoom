from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def evaluate_threshold(rows: list[dict], threshold: float) -> dict:
    tp = fp = tn = fn = 0.0
    for row in rows:
        score = float(row.get("score") or 0.0)
        label = int(row.get("label") or 0)
        weight = float(row.get("weight") or 1.0)
        predicted = 1 if score >= threshold else 0
        if predicted == 1 and label == 1:
            tp += weight
        elif predicted == 1 and label == 0:
            fp += weight
        elif predicted == 0 and label == 0:
            tn += weight
        else:
            fn += weight
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    false_positive_cost = fp * 5.0
    false_negative_cost = fn * 1.0
    total_cost = false_positive_cost + false_negative_cost
    return {
        "threshold": round(threshold, 4),
        "tp": round(tp, 3),
        "fp": round(fp, 3),
        "tn": round(tn, 3),
        "fn": round(fn, 3),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "cost": round(total_cost, 4),
    }


def tune(rows: list[dict], *, start: float = 0.60, stop: float = 0.99, step: float = 0.005) -> dict:
    candidates = []
    current = start
    while current <= stop + 1e-9:
        candidates.append(evaluate_threshold(rows, current))
        current += step
    best = min(
        candidates,
        key=lambda item: (item["cost"], -item["precision"], -item["recall"], item["threshold"]),
    )
    recommended_clarification_floor = max(0.68, round(best["threshold"] - 0.08, 4))
    recommended_candidate_floor = max(0.58, round(recommended_clarification_floor - 0.12, 4))
    return {
        "rows": len(rows),
        "recommended_auto_assign_threshold": best["threshold"],
        "recommended_clarification_floor": recommended_clarification_floor,
        "recommended_candidate_floor": recommended_candidate_floor,
        "best_metrics": best,
        "top_candidates": sorted(candidates, key=lambda item: (item["cost"], -item["precision"], -item["recall"]))[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune face thresholds from a benchmark JSONL.")
    parser.add_argument("--benchmark", default="benchmarks/face-benchmark.seed.jsonl")
    parser.add_argument("--output", default="benchmarks/face-thresholds.json")
    args = parser.parse_args()

    rows = load_rows(Path(args.benchmark))
    result = tune(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
