# MuQ Local FLAC Analyzer

A small Python CLI for extracting music representations from local FLAC files,
storing pooled vectors in PostgreSQL, and finding similar recordings.

## Prerequisites and setup

Install **uv** and **Docker Desktop** (or a compatible Docker runtime). Start the
Docker runtime, then run these commands from this directory in your normal macOS Terminal:

```bash
docker compose up -d
docker compose ps
uv sync
```

The project targets **Python 3.11**; uv can obtain the interpreter if necessary.
Compose uses PostgreSQL 18 with pgvector, a health check, and a named volume mounted
at `/var/lib/postgresql` (the PostgreSQL 18 image layout). Port 5432 is bound to
localhost. Wait for the database to report healthy before analyzing.

The default connection works without any environment file:

```text
postgresql+psycopg://mir:mir@localhost:5432/mir
```

To use another PostgreSQL instance, export `DATABASE_URL`:

```bash
export DATABASE_URL='postgresql+psycopg://mir:mir@localhost:5432/mir'
```

`.env.example` documents this setting; Python does not automatically load `.env`.
The database must have pgvector available and the user must be allowed to initialize
it. The application runs `CREATE EXTENSION IF NOT EXISTS vector` and creates tables
and indexes automatically. No manual schema initialization or migrations are needed.

## Just shortcuts

With `just` installed, the same commands are available through the `justfile`:

```bash
just                     # List available recipes
just setup               # uv sync, then start PostgreSQL and wait for health
just check               # Ruff format/lint and ty
just test
just test-db              # Include live PostgreSQL checks
just info                 # Numba version and MPS availability
just analyze "/path/to/song.flac" --device cpu --save-npz
just analyze "/path/to/song.flac" --offset 60 --duration 30
just search --track-id 1 --layer 12 --limit 5
just inspect-npz "/path/to/song.muq.npz"
just db-status
just db-down
just db-reset             # Deletes all stored analyses
```

`just analyze` and `just search` forward their arguments directly to the Python CLI.
The direct `uv` and Docker commands below remain available.

## Analyze

```bash
uv run python main.py analyze /path/to/song.flac
```

By default this loads the first **30 seconds**, as mono float32 audio resampled to
exactly **24,000 Hz**. It reads the file header for the original duration without
decoding the whole recording. Choose a different window with:

```bash
uv run python main.py analyze /path/to/song.flac \
  --offset 60 \
  --duration 30
```

This analyzes seconds 60–90, or the remaining audio if the track ends earlier.
Stored analysis duration reflects the actual loaded samples. Each invocation creates
a new analysis, including when analyzing the same path again. A region too short for
the model can fail during inference; use a longer window in that case.

The CLI prints the absolute path, original duration, actual analysis region, sample
rate, device, model, returned hidden-state count and final shape, pooled dimension,
database track ID, and number of embeddings stored. Returned counts and shapes are
read from model output.

### Apple Silicon and model download

The default `--device auto` selects MPS when available, otherwise CPU. To force CPU:

```bash
uv run python main.py analyze /path/to/song.flac --device cpu
```

To require MPS:

```bash
uv run python main.py analyze /path/to/song.flac --device mps
```

An explicit MPS request fails clearly if unavailable. Codex's macOS sandbox can
report MPS unavailable even when your normal Terminal supports it. Verify from
your normal Terminal:

```bash
uv run python -c 'import torch; print("MPS available:", torch.backends.mps.is_available())'
```

The official `muq` package loads `OpenMuQ/MuQ-large-msd-iter` and automatically
downloads weights from Hugging Face on first analysis. That needs network access,
disk space, and some patience. Later runs use the model cache.

MuQ 0.1.0 passes an `EasyDict` configuration to its Conformer encoder. Transformers
5 expects private attention configuration fields that this object lacks, causing
`AttributeError: 'EasyDict' object has no attribute '_attn_implementation'`.
`pyproject.toml` constrains this transitive dependency to `transformers<5`, and
`uv.lock` resolves to 4.57.6. Run `just sync` to apply the downgrade before retrying
analysis. The downgrade's inference check remains pending in the sandbox because
the required wheel is not cached and PyPI access is blocked.

Inference uses batch size 1, fp32, evaluation mode, and `torch.inference_mode()`.
Each hidden tensor is mean-pooled over time on its inference device; only pooled
vectors move to CPU as float32 NumPy arrays. Frame tensors and the model are released
after pooling. No frame-level data is saved. The 30-second default limits memory
use on an M3 Pro with 16 GB; longer windows cost more memory and time. Actual MPS
compatibility and peak memory still need a real model run on the target machine.

### Optional NPZ export

```bash
uv run python main.py analyze /path/to/song.flac --save-npz
uv run python main.py analyze /path/to/song.flac --save-npz --output /tmp/inspection.npz
```

The default output is `/path/to/song.muq.npz`, next to the input. `--output` requires
`--save-npz`. Archives contain only `hidden_00`, `hidden_01`, … and
`last_hidden_state`, each a one-dimensional float32 pooled vector.

```bash
uv run python -c 'import numpy as np; a = np.load("/tmp/inspection.npz"); print({k: a[k].shape for k in a.files})'
```

PostgreSQL is always used. NPZ export happens after commit; if export fails, the
CLI exits nonzero and reports the already committed track ID. Existing output
archives are overwritten, and the input FLAC cannot be used as the output path.

## Similarity search

Use a track ID printed by `analyze` and a zero-based hidden-state layer:

```bash
uv run python main.py search \
  --track-id 1 \
  --layer 12 \
  --limit 5
```

The source layer must exist. Search compares only the same hidden-state layer,
excludes the source track, and prints nearest track IDs, distances, and paths.
Cosine distance is calculated **inside PostgreSQL** using pgvector; smaller is
closer. Vectors are not normalized in Python or fetched en masse for comparison.
The separately stored final representation is not a search option in this CLI.

Exact search is sufficient for a hobby collection. Approximate nearest-neighbor
indexing (such as an HNSW cosine index for a selected layer) becomes useful when
the collection is large enough that measured query time warrants it. No HNSW or
IVFFlat index is created by default.

## Embeddings and storage

MuQ is a self-supervised music representation model. The stored vectors describe
representations from different Transformer hidden layers. They are **not direct
predictions** of genre, BPM, mood, or key. Different layers can later be evaluated
for similarity, nearest-neighbor retrieval, genre/mood/instrument classification,
clustering, visualization, or downstream supervised models. This project only
extracts and stores representations.

Each analysis creates, in a single transaction:

- One `tracks` row with file metadata, sample rate, original duration, analysis
  offset and actual duration, model name, and creation timestamp.
- One `embeddings` row per returned hidden state, with zero-based layer index.
- One additional `embeddings` row for pooled `last_hidden_state`, with a NULL layer index.

The final representation may equal the last returned hidden state; both are stored
as requested. Foreign keys cascade deletes from a track to its embeddings. A unique
constraint prevents duplicate hidden-state rows per track; a partial unique index
also prevents duplicate final rows despite PostgreSQL's NULL semantics.

The checkpoint's published configuration specifies 1024 dimensions. This fixed
dimension is defined once in `mir/database.py`, used by the pgvector column, and
validated against every actual pooled output. Mismatches and non-finite values
fail clearly; vectors are never truncated or padded. Layer and frame counts are
not hardcoded. Schema creation does not alter existing incompatible tables;
this exploratory project uses a full local reset when schema changes are needed.

## Numba

Numba is a direct dependency, available for custom CPU numerical kernels when
profiling demonstrates a useful speedup. No custom kernel is warranted in this flow:

- PyTorch/MPS handles model inference and temporal pooling.
- PostgreSQL/pgvector handles similarity search.
- NumPy/librosa handle ordinary array and audio operations.
- Numba is reserved for expensive custom CPU loops, with cached compilation when useful.

Numba does not accelerate MPS inference. There are no object-mode kernels,
parallel Numba kernels, normalization helpers, or unused `numeric.py` module.

## Development checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run python -m unittest -v test_main
uv run python -c 'import numba; print(numba.__version__)'
```

The ordinary checks use generated stereo FLAC audio and real PyTorch tensors to
check window loading/resampling, pooling, dimension validation, NPZ contents, CLI
validation, and simulated device availability. They do not download MuQ weights.
They also run a small randomly initialized MuQ model to check Transformers
compatibility through the actual audio/Conformer forward path.
Device selection mocks do not demonstrate real MPS inference.

To additionally exercise PostgreSQL, run:

```bash
RUN_DATABASE_TESTS=1 uv run python -m unittest -v test_main
```

This checks automatic initialization, vector round trips, duplicate prevention,
rollback after a failed embedding insert, cosine ordering, missing tracks/layers,
and cascading deletion. It inserts marked test tracks and removes its own rows
afterward; database sequences can advance. It uses `DATABASE_URL` like the CLI.

## Stop or reset PostgreSQL

Stop while preserving stored tracks:

```bash
docker compose down
```

Full reset (**deletes the named volume and every stored analysis**):

```bash
docker compose down -v
docker compose up -d
```

## Verification status in Codex

Implementation was written in a sandbox where PyPI DNS resolution failed and access
to the Docker socket was denied. The cache also lacked MuQ and enough Python 3.11
dependencies to complete the original offline sync. Dependencies and model weights
have since been installed from Terminal, and `uv.lock` is present. Real CPU model
inference reproduced the Transformers 5 compatibility error described above. The
lock now resolves to Transformers 4.57.6; installation and inference with that version
remain pending. PostgreSQL access is still blocked in the sandbox. Current results
and earlier supplemental checks are recorded in `VERIFICATION.md`.

Run the remaining checks from your normal macOS Terminal:

```bash
uv sync
docker compose up -d --wait
uv run ruff format --check .
uv run ruff check .
uv run ty check
RUN_DATABASE_TESTS=1 uv run python -m unittest -v test_main
uv run python -c 'import numba, torch; print("Numba:", numba.__version__); print("MPS:", torch.backends.mps.is_available())'
uv run python main.py analyze /path/to/song.flac --device cpu --save-npz
uv run python main.py analyze /path/to/song.flac --device auto --offset 10
```

Then use the printed source track ID with `search --track-id ID --layer 0 --limit 5`.
If MPS is available, also run `analyze /path/to/song.flac --device mps` to exercise
that backend. Keep the generated `uv.lock` in version control once resolution succeeds.

## Sources

- [Official MuQ package and inference example](https://github.com/tencent-ailab/MuQ)
- [Checkpoint configuration](https://huggingface.co/OpenMuQ/MuQ-large-msd-iter/blob/main/config.json)
- [pgvector Docker images](https://github.com/pgvector/pgvector#docker)

## Representative slices (grid alignment deferred)

Preview the regions and selected passages first; this does not load MuQ or connect to PostgreSQL:

```bash
just slices 'songs/Troy - Omen(1).flac' --preview
```

Embed the selected passages:

```bash
just slices 'songs/Troy - Omen(1).flac'
# Optional shorter windows or CPU inference:
just slices 'songs/Troy - Omen(1).flac' --slice-duration 20 --device cpu
# Equivalent without just:
uv run python main.py slices /path/to/song.flac --preview
```

This explicitly scans the **whole track** as mono 24 kHz float32 audio on CPU.
It adapts Metta's energy-based region selection and start/peak sampling:

- One-second RMS bins, normalized and squared to emphasize energy contrast.
- Sustained energy changes form regions (16-second minimum, 4-second lookahead,
  normalized contrast threshold 0.15); a short final region merges into its predecessor.
- Each region contributes its start, capped at `--slice-duration` (default 30 seconds).
  Long regions may contribute a second window around the highest-energy second if
  its start is at least 80% of a window beyond the first start.
- Regions shorter than one second are skipped. Silence/constant energy produces
  one region and one start slice, rather than invented boundaries.

**These are unaligned timestamps, not beat/downbeat or bar boundaries.** No BPM-derived
bar grid, phrase snapping, or genre/section labels are carried over. The detector returns
`(start_seconds, end_seconds)` pairs so grid alignment can be addressed separately.
The heuristic can miss changes in timbre that do not change energy. It selects representative
passages, not exhaustive coverage of every second. No spectrogram plot is generated yet.

MuQ loads once and processes one slice at a time in fp32. Only pooled vectors leave the
inference device. The CPU waveform remains in memory for this explicit whole-track scan;
ordinary `analyze` still loads only its requested window.

Each selected slice creates its own existing `tracks` row and embeddings, with the original
file path, full duration, and slice offset/duration. Each slice commits atomically; if a later
slice fails, earlier committed slices remain and their IDs have already been printed.
Rerunning creates new analyses. Region IDs and start/peak roles are printed, not persisted.
NPZ export remains available through `analyze`, not `slices`.

Use a printed Track ID with the existing search command:

```bash
just search --track-id 42 --layer 12 --limit 5
```

Choose a layer present in that analysis. Results now include time ranges. Search excludes
only the source analysis ID, so other slices of the same file can appear as neighbors.
No database reset or schema change is required.

### Process a directory

```bash
just slices songs/ --preview
just slices songs/
# Equivalent Python CLI:
uv run python main.py slices songs/ --slice-duration 20 --device cpu
```

Directories process immediate `.flac` children (case-insensitive extension) in sorted
filename order; subdirectories and other files are ignored. MuQ loads once for the
whole run. Files and slices run sequentially, releasing each waveform before loading
the next. Preview needs neither MuQ nor PostgreSQL. Empty directories produce a clear error.

The run stops on the first error, printing file progress and already-committed slice IDs.
Earlier inserts remain stored. Rerunning creates new analyses; automatic deduplication
is not implemented. `analyze` continues to accept a single FLAC window.
## Optional Discogs EffNet genres

Classify an existing slice without rerunning MuQ:

```bash
uv sync --extra effnet
just classify --track-id 42
just classify --all
```

`--all` processes only stored slices missing EffNet results, sequentially with one
model instance. An explicit ID recomputes and replaces that slice's result. Each
successful slice commits independently; a failure stops the batch, and rerunning
`--all` resumes the remaining work. Source FLACs must still exist at their saved paths.

The official `discogs-effnet-bs64-1` graph and matching labels download on first use
to `.cache/effnet/`. EffNet reads each stored offset/duration as mono float32 at
16 kHz; MuQ remains at 24 kHz. Incomplete spectrogram patches repeat their final
frames, including short slices. Short-window labels may be less informative.
The fixed graph batch is 64 spectrogram patches, with padded batch outputs removed.
Classifier inference is separate from PyTorch/MPS; no MPS acceleration is assumed.

All 400 mean sigmoid scores and the top five labels are saved in a new
`genre_predictions` table keyed by slice ID and model version. The table initializes
automatically; existing data needs no reset. `just search` displays these labels
when present while continuing to rank by MuQ cosine distance. These are model
scores, not calibrated certainty or metadata retrieved from the Discogs API.

The optional runtime is pinned to `essentia-tensorflow==2.1b6.dev1389`, which has a
Python 3.11 Apple Silicon wheel requiring macOS 15+. See the
[official model metadata](https://essentia.upf.edu/models/music-style-classification/discogs-effnet/discogs-effnet-bs64-1.json)
and [patch handling](https://essentia.upf.edu/reference/std_TensorflowPredictEffnetDiscogs.html).

Implementation verification: the Codex sandbox could not reach PyPI to install
EffNet or regenerate `uv.lock`. The optional dependency is declared, but the lock
still needs refreshing with `uv sync --extra effnet` in Terminal. Native import,
real EffNet predictions, and database writes remain unverified here. After syncing:

```bash
just check
just test
just test-db
just classify --track-id 42
just classify --all
```

## Listening lab (Label Studio)

The lab exports a fixed set of existing MuQ search results for listening and rating.
Label Studio provides the players, task navigation, and saved annotations. No MuQ or
EffNet model runs during review. Its Python dependencies are isolated from this project
using `uv tool run`; the launch recipe pins Label Studio 1.23.0.

**Verification status:** exporter/audio and report checks pass with generated audio and
synthetic annotation exports. Label Studio installation was blocked by PyPI DNS access
in Codex, and PostgreSQL connections were blocked by sandbox permissions. Native
installation, real collection export, UI playback, and review persistence still need
verification in your normal macOS Terminal. The launcher flags were checked against
[upstream 1.23.0](https://github.com/HumanSignal/label-studio/blob/1.23.0/label_studio/core/argparser.py).

### Export a listening batch

Start PostgreSQL and choose existing slice IDs and an available hidden-state layer:

```bash
docker compose up -d
just lab-export --track-id 1 --track-id 42 --layer 12 --limit 5 --output qa/first-listen
```

IDs above are examples. You can list stored windows without loading models:

```bash
docker compose exec postgres psql -U mir -d mir -c 'SELECT id, file_name, analysis_offset, analysis_duration FROM tracks ORDER BY id;'
```

Each source gets up to five nearest slices from other file paths using PostgreSQL cosine
distance. Different checkpoint rows and zero vectors are excluded; a zero source vector
is an error. Duplicate recordings under different file paths are not detected. Multiple
slices of another song may appear: this lab evaluates slice retrieval, not unique-song ranking.

`--track-id` can repeat; `--limit` accepts 1–20; `--seed` defaults to 0. Presentation order
is shuffled once and saved. The output directory must be new and inside `qa/`.
Missing/truncated audio fails the batch; completed batches are never overwritten.

The batch contains `tasks.json`, `manifest.json`, and UUID-named WAV clips. Clips use the
complete stored window at 24 kHz mono, matching MuQ's input policy, encoded as PCM_24.
There is no loudness normalization or time stretching. The manifest preserves retrieval
rank, distances, checkpoint, source paths/windows, and clip hashes. Clip hashes identify
exported playback, not the original audio at the time of embedding extraction; if you
replaced an original FLAC since analysis, re-analyze it before making a batch.

### Open and configure Label Studio

```bash
just lab
```

The first invocation downloads Label Studio into an isolated uv tool environment.
Open the displayed localhost URL (normally `http://localhost:8080`). Use Label Studio's
built-in local account setup, then create one project for this batch.

1. In **Settings → Labeling Interface → Code**, paste `listening-labels.xml` and save.
2. In **Settings → Cloud Storage → Add Source Storage**, select **Local Files**.
   Set its absolute path to this batch's `clips` directory, for example
   `/Users/me/projects/neo-metta-v2/qa/first-listen/clips`. Save the storage entry.
   Do **not** sync it: syncing individual WAVs would create unwanted single-audio tasks.
   This entry permits the project's players to access local clips.
3. Import **only** `qa/first-listen/tasks.json` through the project's normal **Import** button.
   Do not import the manifest or re-import the batch into the same project.
4. Review tasks in ascending task-ID order to retain the exported shuffle.
   Play Reference and Candidate, pausing one before starting the other.
5. Choose **Unrelated**, **Somewhat similar**, **Strongly similar**, or **Unsure**;
   optionally add a note, then submit. Use Skip for an unplayable task.

The question concerns overall musical sound (groove, timbre, instrumentation), not personal
preference or DJ compatibility. Identity, genre predictions, distances, and retrieval rank
are absent from the labeling screen. Familiar music can still reveal identity.

The recipe binds to `127.0.0.1` and serves files only beneath `qa/`. If a player returns
403/404, check the project's Local Files storage entry and that the batch has not moved.
See the official [local file serving guide](https://github.com/HumanSignal/label-studio/blob/develop/label_studio/io_storages/localfiles/README.md).

### Save and summarize judgments

In Label Studio choose **Export → JSON** (the full format, not JSON_MIN). Include all
project tasks when available. Save it as `qa/first-listen/annotations.json`, then run:

```bash
just lab-report qa/first-listen/manifest.json qa/first-listen/annotations.json
```

Reporting needs neither PostgreSQL nor Label Studio running. It joins stable task keys,
uses the latest completed review per task, rejects mixed reviewers, and keeps unsure,
skipped/cancelled, and unreviewed counts separate. Tasks omitted from an export count as
unreviewed: export all tasks to preserve skip status. Drafts do not count as judgments.

Judged relevance is `(somewhat + strongly similar) / decisive judgments`, with its denominator.
Precision@K is shown only for sources with K retrieved candidates and K decisive ratings.
A partial batch does not become a perfect precision@5 score just because its one rated result
was good. Re-running the report does not import, mutate, or multiply annotations.

### Persistence and remaining end-to-end check

Label Studio's local database lives in `qa/label-studio/`; audio batches and JSON exports
also live in `qa/`. **These are durable data, not caches.** Back up that directory. Stopping
with Ctrl-C and running `just lab` again should preserve reviews. A PostgreSQL reset does
not delete Label Studio reviews; deleting `qa/` does. An old batch can still be reviewed
after a PostgreSQL reset because it contains snapshots and clips.

From your normal Terminal, verify five real candidate tasks: play both players, submit
ratings including Unsure, skip one, edit one, restart Label Studio, and check persistence.
Export full JSON twice and confirm the report matches the UI and stays unchanged on rerun.
This manual check is still outstanding; synthetic unit tests do not prove browser behavior.

```bash
just check
just test
just test-db
```

The implementation plan and verification boundary are in
[the listening lab plan](docs/superpowers/plans/2026-09-08-listening-lab.md).
