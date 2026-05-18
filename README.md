# Photo Unifier

Photo Unifier is a local-first media archive pipeline and local web app. It ingests photos and videos from folders, drives, and ZIP takeouts into a SQLite-backed catalog, plans a normalized managed library, copies canonical assets without touching originals, repairs metadata, prepares previews, and exposes review workflows through a local browser UI.

## What Exists Now

- SQLite is the source of truth for assets, jobs, managed copies, hashes, thumbnails, duplicates, faces, embeddings, and extraction records.
- Ingest supports Apple/Google ZIP takeouts plus loose files and directories.
- Managed-library planning is deterministic and resumable.
- Managed-library builds copy assets into a normalized destination and attempt metadata embedding with `exiftool`.
- Derivative generation creates image thumbnails and video preview frames when `ffmpeg` is available.
- Metadata repair normalizes timestamps, location data, screenshots, and provider-agnostic sidecars.
- Exact duplicate detection and first-pass face detection are implemented.
- A local API and browser UI exist for status, jobs, assets, duplicates, and people review.

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
