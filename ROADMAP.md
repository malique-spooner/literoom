# Literoom Roadmap

This roadmap tracks the remaining work to take the current app from “working and renamed” to “finished and distributable.”

## 1. Platform and identity

Status: 100% complete.

- GitHub repository slug is `literoom`
- local git remotes point at `github.com/malique-spooner/literoom.git`
- project metadata is named `literoom`
- installed console commands are `literoom` and `literoom-desktop`
- `Literoom` is the user-facing product name across docs and app metadata

## 2. Codebase migration

Status: 100% complete.

- implementation lives in `src/literoom`
- packaging includes only `literoom*`
- tests and tools import `literoom` directly
- new config and workspace paths default to `.literoom`
- no legacy package namespace is kept in the app
- old namespace and old product-name references are absent from the codebase

## 3. Ingest hardening

- validate ingest on large real libraries, not just small samples
- prove idempotency and resumability across interrupted runs
- harden failure recovery and job reporting for missing or malformed sources
- confirm the 200 GB-scale workflow is reliable before the next big import

## 4. Test coverage

Status: 100% complete.

- browser-level UI flows are covered for the important buttons and actions in the app
- review history, duplicates, near-duplicates, people, system, settings, and ingest workflows are covered
- unit and integration coverage stays aligned with the user-facing behavior
- regression tests protect the actions that would be painful to lose in a release

## 5. Filesystem and cleanup

Status: 100% complete.

- the default workspace structure is clean and predictable
- `imports/`, `library/`, `previews/`, `logs/`, `tmp/`, and `.literoom/cache/` are created consistently
- temporary migration code and old config-key translation have been removed after the transition

## 6. Final release polish

- run a final full-library pilot and fix anything that appears
- tighten docs so setup, launch, ingest, review, and packaging instructions all match reality
- remove any remaining rough edges before release

## 7. Desktop app packaging

- package the app as a real macOS desktop application
- launch the local backend automatically from the desktop shell
- produce a distributable bundle or installer that a non-developer can open and use
- define the later Windows packaging path after the macOS bundle is stable

## 8. Packaged validation

- run the packaged macOS app end to end on a clean machine or clean user profile
- verify first launch, folder selection, ingest, review, and relaunch behavior from the bundle
- fix any packaging-only issues that do not show up in the normal developer workflow
- confirm the distributable is ready for release after the packaged test passes

## Done when

- the repo, package, app, folders, and docs all say Literoom
- the desktop app is distributable
- large ingest runs are trustworthy
- the important UI actions are covered by tests
- old compatibility layers are absent from the codebase
