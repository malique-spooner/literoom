# Literoom

Literoom is a local-first photo and video library manager. It ingests media from folders and takeouts, builds a structured library, repairs metadata, generates previews, and gives you a browser-based review workflow for cleanup, people, duplicates, and system health.

## What It Does

Literoom is designed for personal archives and large imports where you want control over the data and the storage layout.

- Imports from folders, drives, and ZIP takeouts
- Keeps the original media untouched
- Builds a SQLite-backed catalog of assets, jobs, hashes, previews, and review state
- Repairs timestamps, location data, and sidecar metadata when the source supports it
- Generates thumbnails and video previews
- Surfaces duplicate, face, and review workflows in a local web app

## Why It Exists

The goal is to make a media archive that is:

- local-first
- deterministic and resumable
- conservative about metadata promotion
- practical for large imports
- pleasant to review in a browser

## Repository Layout

- `src/literoom/` contains the application code
- `tests/` contains the unit, integration, and browser-backed regression tests
- `ROADMAP.md` tracks the remaining release work
- `.literoom/` is the local runtime cache and database area

## Getting Started

### Prerequisites

- Python 3.10 or newer
- `exiftool`
- `ffmpeg`
- `libvips`

Optional features also benefit from packages such as `insightface`, `open_clip_torch`, `ultralytics`, `paddleocr`, `whisper`, and `faiss-cpu`, but the app degrades gracefully when some optional tools are missing.

### Install

```bash
git clone https://github.com/malique-spooner/literoom.git
cd literoom
python -m venv .venv
./.venv/bin/pip install -e .
```

You can also use the repo’s `Makefile` if you prefer the shorter wrapper commands.

### First Run

```bash
literoom init-config
literoom desktop
```

On first launch, Literoom opens a setup screen if no import sources are configured yet. That screen is used to pick the workspace and import folder before ingest begins.

### Common Commands

```bash
literoom ingest /path/to/takeout.zip /path/to/media-folder
literoom plan-library
literoom build-library
literoom build-previews
literoom extract-content
literoom status
literoom jobs
literoom doctor
```

### Desktop UI

```bash
literoom desktop --host 127.0.0.1 --port 8000
```

Then open the local app in your browser and use:

- Library
- People
- Review
- System

## Configuration

The default config file is `literoom.local.yaml`.

The config controls:

- workspace root
- source paths
- SQLite database location
- library destination
- preview output directory
- tool settings for the media stack
- ingest and review behavior

## Testing

Run the full test suite with:

```bash
./.venv/bin/python -m unittest -q
```

Or use:

```bash
make test
```

The current suite covers:

- config and workspace behavior
- ingest and metadata repair
- dedupe and face workflows
- the local API and review UI
- filesystem and branding regressions

## Project Status

Literoom is actively being cleaned up for packaging and release. The core app is working, the test suite is in place, and the remaining work is focused on release polish, desktop packaging, and packaged end-to-end validation.

See [`ROADMAP.md`](./ROADMAP.md) for the current milestone breakdown.

## License

MIT. See [`LICENSE`](./LICENSE).
