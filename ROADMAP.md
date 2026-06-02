# Literoom Roadmap to 100%

This is a simple milestone plan to take the project from a solid working foundation to a finished product.

## 1. Definition of done

The project is "done" when the following are true in a fresh checkout:

- `python -m pip install -e .` succeeds
- `python -m photo_unifier.cli init-config` succeeds
- `python -m photo_unifier.cli status` succeeds on a valid config
- `python -m photo_unifier.cli ingest ...` works for local folders, Apple ZIPs, and Google takeouts
- `python -m photo_unifier.cli plan-library` and `build-library` work on ingested data
- `python -m photo_unifier.cli build-derivatives` works with the required media tools installed
- `python -m photo_unifier.cli repair-metadata` runs without manual intervention on normal samples
- `python -m photo_unifier.cli dedupe-exact` and `dedupe-near` both work on real sample data
- `python -m photo_unifier.cli detect-faces` works on sample media and review actions behave correctly
- `python -m photo_unifier.cli audit` produces a useful report
- `python -m photo_unifier.cli export-library` produces a clean export tree for upload or backup
- the browser UI supports the main review flows without workarounds
- the test suite passes in a standard local dev setup
- the README matches the actual commands and prerequisites

Release gate:

- a representative sample library can be ingested, normalized, deduped, reviewed, and audited without manual database repair

### Proposed Definition Of Done

The project is done when all of these are true:

- a fresh clone can be installed, configured, and run without manual repo surgery
- the ingest, plan, build, repair, dedupe, face, and audit flows all work on a representative library
- the web UI supports status checks and the main review workflows
- the important data enrichment paths are present and dependable enough for day-to-day use
- the command-line interface is stable and documented
- the test suite passes in the standard developer workflow
- the README reflects the actual supported workflows and setup

### Current Gaps To Close First

These are the most visible blockers against that definition:

- standard test execution still depends on `PYTHONPATH=src`
- the README still describes a smaller surface than the code now exposes
- advanced enrichment features are still incomplete
- the review UI is functional but not yet polished
- the roadmap still needs to be translated into concrete tasks and owners

## 2. Finish the intelligence layer

This layer is now implemented in the repo and should be treated as complete unless future pilot runs expose new quality issues.

- near-duplicate detection uses perceptual hashes plus time and source hints
- face clustering is automatic with a clarification queue for uncertain matches
- OCR extraction is wired into the catalog
- speech or transcription extraction is wired in when local tooling is available
- embeddings are stored consistently in SQLite
- semantic search and similarity retrieval are available through the API

Exit criterion:

- a real sample library can be searched, clustered, and reviewed without manual cleanup of the core data model

## 3. Tighten the review experience

Make the app feel closer to Google Photos plus Lightroom:

- fast grid browsing with lazy/paged thumbnails
- keyboard-first review for next/previous, rating, reject, and people labeling
- quick search and filtering by people, OCR text, captions, date, location, media type, screenshots, blurry items, and review state
- ranking stored back onto assets as metadata, not just returned as a transient sort order
- compare mode for similar photos, near duplicates, and face clarifications
- lightweight review queues for items that need a decision
- responsive UI that stays quick on a normal laptop without heavy client-side machinery
- caching that speeds up thumbnails, previews, and recent searches without turning into a background drain

Exit criterion:

- an hour of review work feels straightforward instead of fiddly

## 4. Harden the pipeline

Make the system trustworthy on a full archive.

- make ingestion and repair idempotent and resumable
- improve job and status reporting
- make failures easier to diagnose
- make sample runs and full runs behave the same way
- cover more odd exports and edge cases
- verify metadata correctness on more real-world samples
- validate the locked Literoom stack end to end:
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

Exit criterion:

- repeated runs do not create drift, corruption, or surprising changes
- the full locked stack is connected, visible in health/status, and behaves predictably on representative media

## 5. Fix the developer experience

Right now the repo works, but the setup flow still has friction.

- make `python -m venv .venv` plus `./.venv/bin/pip install -e .` the documented setup path
- make the common `make setup`, `make doctor`, `make test`, and `make serve` targets work cleanly
- make test execution work from the repo root without extra path setup
- verify packaging and install flow against the locked stack
- align the README with the real commands, cache locations, and folder layout
- make the full locked stack part of the expected working setup:
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
- surface missing media tools and model runtimes as setup problems, not optional niceties
- make the default local setup boring, reliable, and reproducible

Exit criterion:

- a fresh clone can be brought up, doctor-checked, and validated with minimal guessing

## 6. Finish with polish and release quality

This is the last stretch.

- improve error messages
- add final docs for common workflows
- clean up naming and command surface
- add smoke tests for the main flows
- run a full pilot on a representative library
- fix anything that shows up in that pilot

Exit criterion:

- the project feels finished, not just functional

## 7. Workflow interface and model upgrade pass

Make the app feel more like Apple Photos and less like a technical console, while also upgrading the core AI models behind the scenes.

- redesign the workflow surface so the home page, Library, People, Review, and System all feel like clear destinations with one job each
- make People a confirmation workflow, not a cluster browser:
  - show confirmed people as the primary album grid
  - open a person into a split detail view with confirmed photos on the left and candidate matches on the right
  - let naming a person propagate back into clustering so future matches get stronger instead of staying static
  - remove inline rename controls from the grid and move naming/merge actions into the person detail workflow
  - hide noisy machine suggestions by default and only surface them as reviewable candidates
- route true duplicate-looking media out of People and into Review/Duplicates, so the same photo or video does not masquerade as a face cluster
- keep Review focused on duplicates, blurry items, and quick cleanup actions
- make search and filters feel instant and obvious, with minimal button-hunting
- keep the layout calm, compact, and non-technical, with fewer words and fewer competing widgets
- upgrade the face and enrichment models toward the best local quality path on Apple Silicon:
  - `YuNet` for face detection
  - `InsightFace antelopev2` for the highest-quality face recognition / clustering pass
  - `PaddleOCR` for text extraction
  - `Whisper` for speech transcription
  - `OpenCLIP` for semantic search and similarity
  - `YOLO` for object tags where it adds value
- prefer quality-first model choices on the M1 Pro Max, even if they are a little heavier, as long as the app remains responsive
- keep the existing locked stack visible in status/doctor so the model layer is always auditable

Exit criterion:

- the app feels like a polished photo product in daily use, People behaves like a named-person workflow instead of a noisy cluster dump, and the model stack is explicitly chosen for quality rather than convenience

## Suggested Order

If we want the shortest path to 100%, do this in order:

1. developer experience and pipeline hardening
2. face, duplicate, and semantic intelligence
3. review UI polish
4. workflow interface and model upgrades
5. docs, smoke tests, and final release cleanup

That order reduces rework because the data model, ingest flow, and local setup are stabilized before the later product work lands.

## Execution Checklist

### Must-have

- `python -m pip install -e .` works from a fresh clone
- `python -m unittest discover -s tests -q` passes from the repo root
- `python -m photo_unifier.cli init-config` works
- `python -m photo_unifier.cli status` works on a valid config
- ingest works for local folders, Apple ZIPs, and Google takeouts
- `plan-library` and `build-library` work on ingested data
- `build-derivatives` works with the expected media tools installed
- `repair-metadata` runs without manual intervention on normal samples
- `dedupe-exact` and `dedupe-near` work on real sample data
- `detect-faces` works on sample media
- `audit` produces a useful report
- the browser UI supports the main review flows

### Should-have

- face clustering and identity propagation work end-to-end
- OCR extraction exists and is stored in the catalog
- speech or transcription extraction works for video
- semantic ranking or embedding-based search exists
- the UI has good filters, keyboard shortcuts, and batch actions for review
- export paths for Apple Photos and cloud backup are documented and repeatable
- ranking metadata is persisted on assets and can be used to drive smart albums or review ordering
- failures are easy to diagnose from job/status output
- README instructions match actual commands

### Nice-to-have

- smart albums or saved searches
- a more polished Apple/Google-Photos-style browsing experience
- richer scene or object enrichment
- improved heuristics for odd exports and edge cases
- more aggressive automation for repetitive review work
- additional smoke tests for non-happy-path flows

### Done When Each Item Is Green

- every must-have item is verified against a real sample library or a reproducible test
- every should-have item is either implemented or explicitly deferred with a reason
- nice-to-have items are isolated so they do not block release
- a final pilot run succeeds without manual database repair or ad hoc fixes
