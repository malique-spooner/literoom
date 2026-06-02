# Literoom

Literoom is a local-first media archive pipeline and local web app. It ingests photos and videos from folders, drives, and ZIP takeouts into a SQLite-backed catalog, plans a clean library, copies canonical assets without touching originals, repairs metadata, prepares previews, and exposes review workflows through a local browser UI.

The intended operating model is:

1. keep raw imports untouched on the external drive
2. ingest them into the catalog
3. build a clean managed library
4. export that managed library to Apple Photos, Google Photos, or a backup destination

## Current Snapshot

The project now has a cleaner separation between code and media:

- Code lives in `/Users/maliquespooner/Desktop/Coding/photo-unifier`
- Media and runtime data live on the external drive under `/Volumes/Extreme SSD/MSp/Photo Unifier`
- The external drive workspace resolves these relative paths:
  - `imports/` for new media
  - `library/` for the organized final copies
  - `previews/` for thumbnails and video stills
  - `logs/` for run logs
  - `tmp/` for temporary files
- In other words, those folders live under `/Volumes/Extreme SSD/MSp/Photo Unifier`, not in the code checkout
- The SQLite database still lives under `.photo_unifier/manifest.sqlite`
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

- SQLite is the source of truth for assets, jobs, library copies, hashes, previews, duplicates, faces, embeddings, and extraction records.
- Ingest supports Apple/Google ZIP takeouts plus loose files and directories.
- Library planning is deterministic and resumable.
- Library builds copy assets into the library destination and attempt metadata embedding with `exiftool`.
- Preview generation creates image thumbnails and video preview frames with `ffmpeg`.
- Metadata repair normalizes timestamps, location data, screenshots, and provider-agnostic sidecars.
- Exact duplicate detection and first-pass face detection are implemented.
- A local API and browser UI exist for Library, People, Review, and System navigation.

## Working Approach

The current workflow is intentionally incremental:

1. Ingest or point at a small sample first.
2. Verify metadata and previews on that sample.
3. Widen only if the sample looks correct.
4. Keep canonical metadata conservative so incorrect data does not get promoted.

That approach matters most for time, GPS, and people data, where false positives are more costly than missing values.

## Quick Start

0. Make sure the native image helper is installed:

   ```bash
   brew install libvips
   ```

1. Create a virtual environment, then install the package in editable mode:

   ```bash
   python -m venv .venv
   ./.venv/bin/pip install -e .
   ```

   Or use:

   ```bash
   make setup
   ```

2. Write a default config:

   ```bash
   literoom init-config
   ```

3. Ingest sources:

   ```bash
   literoom ingest /path/to/takeout.zip /path/to/media/folder
   ```

4. Plan and build the library:

   ```bash
   literoom plan-library
   literoom build-library
   literoom build-previews
   literoom extract-content
   literoom export-library /path/to/export
   ```

5. Inspect status:

   ```bash
   literoom status
   literoom jobs
   literoom doctor
   ```

6. Run the local API and app:

   ```bash
   literoom serve-api --host 127.0.0.1 --port 8000
   ```

   Then open:

   - `http://127.0.0.1:8000/`
   - `http://127.0.0.1:8000/app/assets`
   - `http://127.0.0.1:8000/app/people`
   - `http://127.0.0.1:8000/app/review`
   - `http://127.0.0.1:8000/app/system`
   - `http://127.0.0.1:8000/healthz`

7. Run tests:

   ```bash
   ./.venv/bin/python -m unittest discover -s tests -t . -q
   ```

   Or use:

   ```bash
   make test
   ```

## Convenience Commands

Once the virtual environment is created, the `Makefile` gives you short commands for the common workflows:

- `make doctor`
- `make status`
- `make jobs`
- `make serve`
- `make test`

## Config

Default config path: `literoom.local.yaml`

The config controls:

- workspace root
- source paths
- SQLite database location
- library destination
- preview output directory
- the locked Literoom stack:
  - `exiftool`
  - `ffmpeg`
  - `libvips`
  - `PaddleOCR`
  - `Whisper`
  - `YuNet`
  - `InsightFace`
  - `YOLO`
  - `OpenCLIP`
  - `SQLite FTS + vector search`
- batch size, workers, naming template, and preview settings

## Current Gaps

This repo now has a working ingest/library/review foundation, and the intelligence layer is now implemented. The remaining work is mostly in release hardening, export polish, and deeper UX refinement:

- a more polished Apple/Google-Photos-style browsing experience
- richer review workflows for duplicates, people, and cleanup queues
- additional smoke tests and pilot validation on large real libraries
- more enrichment support for the locked media stack when packages are installed
- export presets for Apple Photos upload and cloud backup targets

The remaining work is now primarily polish and operational hardening rather than core intelligence gaps.
