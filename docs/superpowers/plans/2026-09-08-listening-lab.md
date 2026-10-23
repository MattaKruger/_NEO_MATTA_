# Simple Listening Lab Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking. This document authorizes no implementation by itself.

**Goal:** Listen to existing MuQ search results, record similarity judgments, and review a reproducible summary using a ready-made annotation UI.

**Architecture:** Label Studio Community Edition owns the listening interface, task navigation, and annotation persistence. A small Python CLI exports PostgreSQL search results and bounded audio excerpts into a review batch, then reads Label Studio's JSON export to summarize judgments. No models run during review.

**Tech Stack:** Python 3.11, uv, existing SQLAlchemy Core/pgvector and audio dependencies, Label Studio in an isolated uv tool environment, unittest, Ruff, ty.

**Spec:** The user's listening-first QA discussion and subsequent instruction to implement. The decisions below define v1. The MIR search and worker plans remain parked.

## Package choice

| Option | Supplied functionality | Work remaining | Fit |
|---|---|---|---|
| Label Studio Community Edition | Audio annotation components, task management, saved annotations, JSON export | Batch exporter, declarative labeling configuration, summary reader | Recommended: closest to an existing annotation application |
| Gradio | Audio/waveform components, Python callbacks, session state | Navigation, durable ratings, resume behavior, reporting | Good for a future live search playground |
| Streamlit | Audio player, forms, tables and application state | Annotation workflow and durable ratings | Good for a future analysis dashboard |

These are alternatives, not dependencies to install together. Label Studio has its own application database and built-in account/project setup; accept that existing workflow rather than implementing application authentication. Its internal Django stack stays outside the POC and its dependencies. The POC continues using SQLAlchemy Core and PostgreSQL; no Django integration is introduced into its code.

Sources: [Label Studio audio template](https://labelstud.io/templates/audio_classification), [annotation export](https://labelstud.io/guide/export), [Gradio Audio](https://gradio.app/main/docs/gradio/audio), [Gradio state](https://github.com/gradio-app/gradio/blob/main/guides/03_building-with-blocks/03_state-in-blocks.md), [Streamlit audio](https://docs.streamlit.io/develop/api-reference/media/st.audio).

## Global constraints

- Local, single reviewer, Python 3.11, Apple Silicon, uv management.
- Reuse stored embeddings and exact PostgreSQL cosine search. No inference or GPU requirement.
- No custom JavaScript, player, frontend, REST service, worker, or annotation database tables.
- No new MIR search modes, genre editing, region editing, beat grids, embedding plots, or model training.
- Exported files and Label Studio data are durable local artifacts, not disposable caches.
- Do not bypass sandbox restrictions. Native installation, browser playback, and database access must be measured before claiming success.
- Keep the existing CLI behavior and user-owned untracked files intact.

## Listening task

One task presents two players: **Reference** and **Candidate**. Ask:

> How similar is the candidate to the reference in overall musical sound? Consider groove, timbre, and instrumentation. Judge similarity, not whether you like the music or would mix the tracks together.

Required choices: **Unrelated**, **Somewhat similar**, **Strongly similar**, **Unsure**. An optional note captures explanations. Label Studio's skip action remains available for an unplayable or unjudgeable task. No default answer.

Choose explicit source slice IDs, one available hidden-state layer, and up to five nearest candidates per source. Start with roughly five sources drawn from the album and VA, preferably different songs. These are exploratory judgments of a small collection, not a benchmark of general music understanding.

Randomize task presentation once using a recorded seed. Hide candidate identity, genre predictions, distance, and retrieval rank in the labeling view. This is metadata-hidden listening, not rigorous blinding: familiar music and task-manager metadata can still reveal identity. Preserve all those identifiers in the batch manifest for inspection after review.

Exclude every analysis of the source file before SQL LIMIT, not just its row ID. Keep other songs' multiple slices in v1: the unit being evaluated is a retrieved slice. Report distinct candidate file counts so repeated songs are visible. File-path matching does not recognize duplicate recordings stored under different paths; document that limit.

## Reproducible batch contract

Create `qa/<batch>/` with `manifest.json`, `tasks.json`, and `clips/`. Refuse an existing destination. Write to a temporary sibling and rename only once the batch is complete; failure must not leave an importable partial batch.

The manifest records schema/rubric version, batch UUID, creation time, shuffle seed, source IDs, checkpoint, layer, requested K, same-file exclusion, and actual candidate counts. Each task has a stable UUID and source/candidate snapshots containing database ID, absolute path, original window, and model name, plus original retrieval rank and finite cosine distance. Record the playback policy and SHA-256 of each exported clip. Snapshot search results once; do not re-query when reading ratings.

Each Label Studio task contains `data.reference_audio`, `data.candidate_audio`, `data.task_key`, and `data.batch_id`. URLs refer to exported clips through Label Studio's local-file serving. Task keys, not Label Studio's assigned IDs or array positions, join annotations to the manifest.

Playback policy `muq-input-v1`: reuse `main.load_audio()` at 24,000 Hz mono for the complete stored analysis window. Validate the decoded duration against the stored duration within one output sample. Write WAV PCM_24 using existing SoundFile; do not normalize, stretch, fade, or silently shorten audio. Reject non-finite or out-of-range samples rather than clipping them silently. Reuse a clip for repeated references to the same analysis row. This evaluates the model's mono input, with original loudness differences retained; it is not a loudness-controlled listening experiment.

Fail with the affected row/path on missing or truncated audio, absent source/layer, invalid arguments, unavailable database, or non-finite cosine distance. Zero-norm source vectors must fail clearly; zero-norm candidates are excluded in the SQL query for the lab policy. Export nothing for a batch with no candidates; explain which sources had none.

## Annotation persistence and reporting

Label Studio uses its own default SQLite storage, explicitly located at `qa/label-studio/`. The MIR PostgreSQL schema remains unchanged. Keep both this directory and the exported batch. A downloaded **full JSON** annotation export is a portable artifact saved beside its manifest. Do not use a simplified export that discards task identity or review state.

The summary CLI accepts the manifest and JSON export. Validate batch/task keys and allowed choices. Ignore cancelled/skipped annotations as ratings; show their count separately. Ignore unfinished drafts. For an updated review of the same task by the same reviewer, use its latest completed annotation; reject ambiguous duplicates and unexpected multiple reviewers in this single-reviewer v1. Reject foreign task keys rather than silently dropping them. Compare against the manifest to count tasks omitted by an export.

Show expected tasks, completed ratings, unsure, skipped/cancelled, and unreviewed tasks; separate task counts from annotation-event counts. Show the three similarity categories and per-source results with original retrieval ranks. Report judged relevance as `(somewhat + strongly) / decisive judgments`, always with its denominator. Unsure is not a negative.

Only show precision@K for a source with K retrieved candidates and K decisive judgments, defining relevant as somewhat or strongly similar. Otherwise report coverage and counts, not a misleading precision@5. Do not compute recall from a top-five review or compare layers evaluated on different source sets as if they were controlled experiments. No aggregate score is needed to ship v1.

## Files and implementation tasks

New files: `lab.py` (argparse export/report commands and small helpers), `listening-labels.xml` (Label Studio configuration), `test_lab.py` (focused unittest checks).

Modify `mir/database.py` only for optional lab search filters and retrieving source window metadata. Modify `justfile`, `.gitignore`, and `README.md` for running and preserving the lab. The POC `pyproject.toml` and lock need no Label Studio runtime dependency; install the tool independently with uv. Do not opportunistically update unrelated dependency issues.

### Task 1: Verify the ready-made listening interface

- [ ] Install Label Studio in an isolated uv tool environment targeting Python 3.11. Check its installed version and CLI help; pin that successfully tested version in the eventual `just lab` recipe. Do not assume current docs establish macOS dependency compatibility.
- [ ] Verify built-in local-file serving and a persistent data directory using two short generated WAV clips. Limit the served root to `qa/`, never the whole music library or home directory. Use the release's documented bind setting for localhost; `--host` controls generated URLs and must not be assumed to be a socket bind option.
- [x] Create `listening-labels.xml` using built-in components:

```xml
<View>
  <Header value="Compare overall musical sound: groove, timbre, and instrumentation. Similarity, not personal preference." />
  <Header value="Reference" />
  <Audio name="reference" value="$reference_audio" />
  <Header value="Candidate" />
  <Audio name="candidate" value="$candidate_audio" />
  <Choices name="similarity" toName="candidate" choice="single-radio" required="true">
    <Choice value="Unrelated" />
    <Choice value="Somewhat similar" />
    <Choice value="Strongly similar" />
    <Choice value="Unsure" />
  </Choices>
  <TextArea name="note" toName="candidate" placeholder="Optional: what matched or felt wrong?" />
</View>
```

- [ ] Use the Community Edition project UI to import two tasks and this configuration. Play both clips, save a judgment, skip a task, edit a judgment, restart, and export full JSON. Verify persistence, task-key retention, cancellation behavior, and the actual export structure. Use the players' existing pause controls; do not add custom synchronized playback.
- [ ] Add `just lab` and README setup instructions after this succeeds. Tool startup commands are finalized from verified CLI help, not guessed during planning. Keep this feasibility check small; if incompatible on this Mac, report the exact blocker before replacing the chosen application.

Official references: [installation](https://labelstud.io/guide/install), [startup and default database](https://labelstud.io/guide/start), [local-file storage implementation guide](https://github.com/HumanSignal/label-studio/blob/develop/label_studio/io_storages/localfiles/README.md), [Choices](https://labelstud.io/tags/choices).

### Task 2: Export a fixed retrieval batch

- [x] Write a generated-FLAC export test with two sources and overlapping candidate references. Check exact decoded windows, clip reuse, stable task keys, recorded ranks, seeded ordering, overwrite refusal, and cleanup after a missing-file failure.
- [x] Add optional keyword lab filters to `find_similar_tracks`; retain defaults for existing callers. Apply source-file exclusion, checkpoint equality, and usable-vector filtering in PostgreSQL before ordering/limit. Preserve the source vector SQL subquery. Add a database check where the closest rows belong to the source file and a valid other-file row must still fill the requested result slot.
- [x] Implement `lab.py export` with repeated `--track-id`, required `--layer`, `--limit` default 5 bounded to 1–20, `--seed` default 0, and required `--output` under `qa/`. Deduplicate supplied source IDs while preserving order. Query source metadata through SQLAlchemy Core. Reuse `load_audio`, `create_engine`, and existing error-message conventions.
- [x] Serialize task URLs with proper URL encoding and relative paths under the served root. Use UUID filenames instead of music titles. Write manifest and tasks using standard-library JSON; no SDK or additional web server needed.
- [x] Add `just lab-export +args` forwarding arguments, and ignore `qa/` in Git with an explanatory comment that it contains durable local data. README must explain that database reset does not reset Label Studio and that deleting `qa/` destroys lab artifacts.
- [ ] Run the focused tests, then import a real exported batch in Label Studio. Confirm that reference and candidate playback are the recorded windows, metadata is absent from the labeling screen, and no model is loaded.

Proposed user command after implementation:

```bash
just lab-export --track-id 1 --track-id 42 --layer 12 --limit 5 --output qa/first-listen
just lab
```

IDs and layer are illustrative: select actual stored IDs and an available layer. The exporter prints a concise per-source candidate count so empty or small result sets are obvious. The README explains project creation, configuration paste, JSON import, review, and full JSON export using the tested Label Studio version.

### Task 3: Summarize saved judgments

- [ ] Use the real JSON export from Task 1 to create a tiny anonymized fixture in `test_lab.py`. Cover one decisive rating, one unsure, one cancelled task, one omitted task, and an edited rating. Assert exact denominators and latest-review selection. Add rejection checks for unknown task keys, invalid labels, and multiple reviewers.
- [x] Implement `lab.py report MANIFEST ANNOTATIONS`. It reads files only: no PostgreSQL connection or Label Studio SDK. Keep parsing and summary calculation as small annotated Python functions callable from unittest.
- [x] Add `just lab-report +args`. Show category counts, coverage, per-source ranked judgments, notes, and distinct candidate files. Include a clear message when no decisive judgments exist; never divide by zero.
- [ ] Run a complete real review of at least five candidates, restart Label Studio, export JSON, and reconcile every saved judgment with the report. Confirm re-exporting the same reviews does not multiply counts.

```bash
just lab-report qa/first-listen/manifest.json qa/first-listen/annotations.json
uv run python -m unittest -v test_lab
uv run ruff format --check .
uv run ruff check .
uv run ty check
```

Run existing `just test` and the opt-in PostgreSQL tests when available. Record exactly which native, database, and browser checks were exercised. Do not report generated clips or mocked queries as real collection validation.

## Definition of done

A batch from existing embeddings opens in Label Studio with two working players per task. Judgments survive application restart, export with stable task identity, and produce a correct summary. The normal analysis/classification CLI still works. No custom annotation UI was built.

## Explicitly deferred

Live database browsing in the UI, triplet/layer battles, statistical significance testing, calibrated loudness matching, genre correction, timeline annotation, semantic retrieval, annotation writeback to PostgreSQL, multi-reviewer adjudication, and embedding visualization. These are separate choices once the basic listening workflow proves useful.

## Planning evidence and limits

Inspected `main.py`, `mir/database.py`, `pyproject.toml`, `justfile`, and existing test names on 2026-09-08. Current search excludes only the source row and returns offsets/durations; its keyword filters are proposed changes. Package capabilities were checked against official documentation. Label Studio has not been installed or exercised during this planning turn. No application code or dependency files were changed.


## Implementation evidence — 2026-09-08

Implemented `lab.py`, `listening-labels.xml`, `test_lab.py`, `test_lab_database.py`,
the optional database filters, just recipes, ignore rule, and README walkthrough.
The narrow implementation proceeded while external runtime checks were blocked; the
Label Studio choice was retained. No custom UI, model inference, annotation tables,
or POC dependency additions were introduced.

- `UV_CACHE_DIR=.cache/uv just test`: 11 tests, 9 passed, 2 opt-in PostgreSQL tests skipped.
  Includes real generated FLAC decode/resample/window export, hashes, clip reuse,
  seeded presentation order, atomic failure cleanup, and report reconciliation.
- Report fixtures follow the official full-JSON shape and are explicitly synthetic;
  no live Label Studio export was available. Tests cover edited judgments, unsure,
  cancelled tasks, drafts, omitted tasks, unknown labels/keys, mixed reviewers,
  malformed manifests and duplicate ranks, full precision@5, and no judgments.
- Direct opt-in lab database test and real `just lab-export` both failed on localhost
  `Operation not permitted`. No successful collection export or DB test is claimed.
- `uv tool install` and the actual `just lab` launcher failed at PyPI DNS resolution;
  offline installation also confirmed Label Studio is absent from cache.
- Launch flags were verified against upstream **1.23.0** source, including
  `--internal-host 127.0.0.1`; that release is pinned in `justfile` but is not native-tested.
- `listening-labels.xml` passes XML parsing. Browser rendering, project import, local
  audio serving, rating/skip/edit behavior, restart persistence and live JSON export
  remain unverified. The README leaves exact commands and UI steps for those checks.
- The default `uv sync --locked --offline` check passed but removed the optional
  EffNet environment packages. They were restored through uv from existing cached
  distributions; `essentia-tensorflow` 2.1b6.dev1389 and `six` 1.17.0 are installed.

Outstanding checkboxes above remain open deliberately. Code and generated-data tests
are not evidence that Label Studio has been exercised on this Mac. No goal completion
claim is justified until the live workflow and PostgreSQL checks pass.
