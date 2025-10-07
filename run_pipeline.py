import argparse
import csv
import os
import time
from pathlib import Path
from datetime import datetime

# keep writes off C:
os.environ.setdefault("TEMP", r"F:\_tmp")
os.environ.setdefault("TMP",  r"F:\_tmp")

from subprocess import run, CalledProcessError, PIPE
from photo_unifier.embed import apply_sidecars  # we call your code directly

DEFAULT_LOG_DIR = Path(r"F:\_unifier_state\logs")
DEFAULT_SHARD_DIR = Path(r"F:\_unifier_state\shards")

def log_line(log_path: Path, msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} | {msg}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8", newline="") as fh:
        fh.write(line + "\n")

def run_index_if_requested(log_path: Path, roots: list[str], manifest: Path, venv_python: Path | None):
    if not roots:
        return
    # Use the current python to run the Click CLI (simplest/most robust)
    py = str(venv_python) if venv_python else "python"
    cmd = [py, "-m", "photo_unifier.cli", "index", *roots, "--manifest", str(manifest)]
    log_line(log_path, f"Index START: {' | '.join(roots)}")
    try:
        p = run(cmd, stdout=PIPE, stderr=PIPE, text=True, check=True)
        if p.stdout: log_line(log_path, p.stdout.strip())
        if p.stderr: log_line(log_path, p.stderr.strip())
    except CalledProcessError as e:
        if e.stdout: log_line(log_path, e.stdout.strip())
        if e.stderr: log_line(log_path, e.stderr.strip())
        raise
    log_line(log_path, "Index DONE")

def shard_manifest(manifest: Path, shard_dir: Path, shard_size: int, start_batch: int = 0):
    shard_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest, "r", encoding="utf-8", newline="") as fh:
        rdr = csv.reader(fh)
        header = next(rdr, None)
        if not header:
            return  # empty file
        batch = 0
        rows = []
        for row in rdr:
            rows.append(row)
            if len(rows) >= shard_size:
                if batch >= start_batch:
                    out = shard_dir / f"manifest_shard_{batch:05d}.csv"
                    with open(out, "w", encoding="utf-8", newline="") as outfh:
                        w = csv.writer(outfh)
                        w.writerow(header)
                        w.writerows(rows)
                    yield out, len(rows)
                batch += 1
                rows = []
        if rows:
            if batch >= start_batch:
                out = shard_dir / f"manifest_shard_{batch:05d}.csv"
                with open(out, "w", encoding="utf-8", newline="") as outfh:
                    w = csv.writer(outfh)
                    w.writerow(header)
                    w.writerows(rows)
                yield out, len(rows)

def count_kpis(dest_root: Path):
    media = 0
    xmp = 0
    unknown = 0
    for p in dest_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() == ".xmp":
            xmp += 1
            continue
        media += 1
        if "\\Unknown\\Unknown\\Unknown\\" in str(p):
            unknown += 1
    return media, xmp, unknown

def main():
    ap = argparse.ArgumentParser(description="Chunked runner for photo_unifier (index + embed in safe batches).")
    ap.add_argument("--manifest", default=r"F:\_unifier_state\manifest.csv")
    ap.add_argument("--dest", default=r"F:\_library")
    ap.add_argument("--batch-size", type=int, default=5000)
    ap.add_argument("--sleep-sec", type=int, default=15)
    ap.add_argument("--max-batches", type=int, default=0, help="0 = all")
    ap.add_argument("--start-batch", type=int, default=0, help="skip first N batches (resume)")
    ap.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    ap.add_argument("--shard-dir", default=str(DEFAULT_SHARD_DIR))
    ap.add_argument("--index-roots", nargs="*", default=[], help="optional: roots to reindex before embedding")
    ap.add_argument("--venv-python", default=str(Path(".") / ".venv" / "Scripts" / "python.exe"))
    args = ap.parse_args()

    manifest = Path(args.manifest)
    dest_root = Path(args.dest)
    log_dir = Path(args.log_dir)
    shard_dir = Path(args.shard_dir)
    venv_python = Path(args.venv_python)

    dest_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_line(log_path, f"Pipeline START | dest={dest_root} | batch_size={args.batch_size} | sleep={args.sleep_sec}s")

    # Optional: reindex
    if args.index_roots:
        run_index_if_requested(log_path, args.index_roots, manifest, venv_python)

    # Shard + process
    batch_count = 0
    for shard_csv, rows in shard_manifest(manifest, shard_dir, args.batch_size, args.start_batch):
        if args.max_batches and batch_count >= args.max_batches:
            break
        batch_count += 1
        log_line(log_path, f"Batch {args.start_batch + batch_count} START | rows={rows} | shard={shard_csv.name}")
        try:
            # copy + sidecars only; limit=0 to process entire shard
            processed, copied, sidecars = apply_sidecars(str(shard_csv), str(dest_root), limit=None)
            log_line(log_path, f"Batch {args.start_batch + batch_count} DONE | processed={processed} copied={copied} sidecars={sidecars}")
        except Exception as e:
            log_line(log_path, f"Batch {args.start_batch + batch_count} ERROR: {e!r}")
        media, xmp, unknown = count_kpis(dest_root)
        log_line(log_path, f"KPI | media={media} xmp={xmp} unknown={unknown}")
        time.sleep(max(0, args.sleep_sec))

    log_line(log_path, "Pipeline FINISHED")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
