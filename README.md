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

For the full media stack, install `insightface`, `open_clip_torch`, and `faiss-cpu` alongside the core dependencies.

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

On first launch, Literoom opens a setup screen if no import sources are configured yet. That screen is used to pick the workspace and paste or choose the real import folder before ingest begins. If the folder lives outside the workspace root, paste the absolute path directly.

### Common Commands

```bash
literoom ingest /path/to/takeout.zip /path/to/media-folder
literoom test-ingest /path/to/media-folder
literoom analyze-imports /path/to/media-folder
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

The launcher opens a friendlier local URL (`literoom.localhost`) instead of a raw loopback address.

Then use:

- Library
- People
- Review
- System

The System page also includes a built-in update checker. It compares the installed app against the latest GitHub release and gives you a download link when a newer build is available.
It also includes an import preflight card with an analysis summary and a 10-file test ingest so you can catch broken folder setup before running the full archive.

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

## Packaging

Build the macOS app bundle and ZIP with:

```bash
make package
```

That produces:

- `dist/Literoom.app`
- `dist/Literoom.zip`

The ZIP is the release artifact you can upload to GitHub and hand off directly to someone else. The app bundle is only an intermediate build output.

GitHub Releases are the versioned delivery channel for Literoom:

- `main` stays the source of truth for code
- each stable build is tagged and published as a release
- users install the ZIP once and can check for newer releases from inside the app

## Project Status

Literoom now has a stable desktop release flow, an in-app update checker, and a source preflight path for testing import setup before the full ingest. The core app is working, the test suite is in place, and the remaining work is focused on future release polish and feature work.

See [`ROADMAP.md`](./ROADMAP.md) for the current milestone breakdown.

## License

MIT. See [`LICENSE`](./LICENSE).
