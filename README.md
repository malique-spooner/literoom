# Photo Unifier

Photo Unifier is a local-first media archive pipeline and local web app. It ingests photos and videos from folders, drives, and ZIP takeouts into a SQLite-backed catalog, plans a normalized managed library, copies canonical assets without touching originals, repairs metadata, prepares previews, and exposes review workflows through a local browser UI.

## Current Snapshot

The project now has a cleaner separation between code and media:

- Code lives in `/Users/maliquespooner/Desktop/Coding/photo-unifier`
- Media and runtime data live on the SSD under `/Volumes/Extreme SSD/MSp/Photo Unifier`
- The old SSD code tree has been removed, so the Desktop repo is the source of truth
- The `src` tree is simplified around `metadata/`, `faces.py`, `dedupe.py`, and `utils/` instead of the old `phase_*` naming

The pipeline is being run in a correctness-first way:

- timestamps are recovered from the strongest verified source available
- location is only promoted when a valid latitude/longitude pair exists
- Apple and Google export parsing now handles more nested and odd-shaped sidecars
- face clustering is conservative rather than overly aggressive
- thumbnails, duplicates, blur, and screenshot signals are available for cleanup and review

Recent pilot runs on small and widened samples completed cleanly, with strong timestamp recovery and high location coverage on real sample data.

## What Exists Now

- SQLite is the source of truth for assets, jobs, managed copies, hashes, thumbnails, duplicates, faces, embeddings, and extraction records.
- Ingest supports Apple/Google ZIP takeouts plus loose files and directories.
- Managed-library planning is deterministic and resumable.
- Managed-library builds copy assets into a normalized destination and attempt metadata embedding with `exiftool`.
- Derivative generation creates image thumbnails and video preview frames when `ffmpeg` is available.
- Metadata repair normalizes timestamps, location data, screenshots, and provider-agnostic sidecars.
- Exact duplicate detection and first-pass face detection are implemented.
- A local API and browser UI exist for status, jobs, assets, duplicates, and people review.

## Working Approach

The current workflow is intentionally incremental:

1. Ingest or point at a small sample first.
2. Verify metadata and previews on that sample.
3. Widen only if the sample looks correct.
4. Keep canonical metadata conservative so incorrect data does not get promoted.

That approach matters most for time, GPS, and people data, where false positives are more costly than missing values.

## Quick Start

1. Use Python `3.11.9` and install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Write a default config:

   ```bash
   python -m photo_unifier.cli init-config
   ```

3. Ingest sources:

   ```bash
   python -m photo_unifier.cli ingest /path/to/takeout.zip /path/to/media/folder
   ```

4. Plan and build the managed library:

   ```bash
   python -m photo_unifier.cli plan-library
   python -m photo_unifier.cli build-library
   python -m photo_unifier.cli build-derivatives
   ```

5. Inspect status:

   ```bash
   python -m photo_unifier.cli status
   python -m photo_unifier.cli jobs
   ```

6. Run the local API and app:

   ```bash
   python -m photo_unifier.cli serve-api --host 127.0.0.1 --port 8000
   ```

   Then open:

   - `http://127.0.0.1:8000/`
   - `http://127.0.0.1:8000/app/assets`
   - `http://127.0.0.1:8000/app/duplicates`
   - `http://127.0.0.1:8000/app/faces`

## Config

Default config path: `.photo_unifier/config.yaml`

The config controls:

- workspace root
- source paths
- SQLite database location
- managed library destination
- derivative output directory
- tool paths for `exiftool` and `ffmpeg`
- batch size, workers, naming template, and preview settings

## Current Gaps

This repo now has a working ingest/library/review foundation, but the following are still not fully implemented:

- near-duplicate detection
- face clustering and identity propagation
- speech/OCR/scene extraction
- embeddings and semantic ranking
- a more polished Apple/Google-Photos-style browsing experience

The remaining work is mostly in the intelligence and polish layers rather than the original CSV-era foundation.
