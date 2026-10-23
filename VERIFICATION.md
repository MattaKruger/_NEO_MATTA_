# Implementation verification

## Current checkout — 2026-10-23

Active checkout: `~/mana/_NEO_MATTA_`, remote
`git@github.com:MattaKruger/_NEO_MATTA_.git`. The user ran the migration script
successfully. All three older projects are now siblings under `~/mana`, and the
four original source directories are absent from `~/personal_projects`.
The imported baseline is `7197cd3`; the original V2 Git metadata remains in
ignored `.cache/pre-move.git`.

| Check | Current evidence | Limit |
| --- | --- | --- |
| Relocation | Source files match the imported baseline; clone remote retained. Repaired 21 generated environment launchers/activation scripts referencing the old path. | Local environments and data remain outside Git. |
| Locked commands | `UV_CACHE_DIR=.cache/uv uv lock --check --offline` passes; no package upgrades or new dependencies. | A fresh online installation was not attempted. |
| Full local verification | `UV_CACHE_DIR=.cache/uv UV_OFFLINE=1 NUMBA_CACHE_DIR=.cache/numba OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 just verify` passes Ruff format/lint, ty, and 9 tests; 2 database tests skipped. | Audioread emits deprecation warnings for Python modules scheduled for removal after Python 3.11. |
| MuQ checkpoint | Real cached `OpenMuQ/MuQ-large-msd-iter`, CPU, 3 seconds of Troy audio at 36.5s: 13 hidden states plus final state, all 14 vectors finite, nonzero, dimension 1024; final shape `(1, 75, 1024)`. | Ran before moving the identical source/environment; no database writes or MPS claim. |
| EffNet | Real cached graph and matching 400-label vocabulary classified a 30-second window after relocation; `(30, 400)` frame scores, top label `Electronic---Techno` about 0.9684. | Real 1-, 2-, and 3-second windows fail in the native graph with a reshape error. No universal minimum duration is established. |
| Failure handling | Regression reproduced the uncaught native `RuntimeError`, then passed after converting it to a slice-specific `ValueError`. The test preserves the first successful result and rejects saving a prediction for the failing second slice. A subsequent real native one-second inference confirmed the contextual error and absence of a saved result. | Storage was replaced at the database boundary for that native check; live transactions remain unverified. |
| CI | Added an Ubuntu workflow with an isolated PostgreSQL 18/pgvector service, pinned action revisions, locked dependencies, and all existing tests enabled. | Hosted execution is pending. Linux PyTorch installs CUDA dependencies despite CPU-only test execution. |

The optional EffNet dependency was already locked. A proposed dependency-group
change could not be resolved from incomplete offline index metadata; it was
discarded. The original dependency manifest and resolved package versions remain
unchanged. Routine `uv run --locked` retains installed extras; exact
`uv sync --locked` removes EffNet unless `--extra effnet` is supplied.

Still blocked or unverified:

- Docker API access is denied for the local Docker socket. No database reset or
  data migration was performed, and live SQL tests remain unverified.
- GitHub SSH fails at DNS resolution. No push or hosted CI run is claimed.
- The sandbox marks this checkout's `.git` read-only; the foundation changes may
  require a normal-Terminal commit and push.
- Label Studio installation, real collection export, browser playback, review
  edits, restart persistence, and full JSON/report reconciliation remain pending.
- Real MPS inference and broad musical retrieval quality remain unverified.

CI uses the official [uv GitHub Actions guidance](https://docs.astral.sh/uv/guides/integration/github/)
and [GitHub PostgreSQL service guidance](https://docs.github.com/en/actions/tutorials/use-containerized-services/create-postgresql-service-containers).

## Historical update: representative slices (2026-09-08)

- `UV_CACHE_DIR=.cache/uv just check`: Ruff format, Ruff lint, and ty all pass.
- `UV_CACHE_DIR=.cache/uv uv run python -m unittest -v test_main`: six tests
  pass, database integration test skipped. Includes synthetic region/peak cases,
  model reuse and timestamp orchestration with explicit stand-ins, and a real
  small MuQ model forward pass.
- Real FLAC preview (`songs/Troy - Omen(1).flac`, 293.92 seconds): five energy
  regions, six representative slices. Preview does not load MuQ or connect to DB.
- Real cached pretrained checkpoint, CPU, one second starting at 36.5 seconds:
  inference and pooling pass, producing 14 vectors; final shape `(1, 25, 1024)`.
  This verifies the earlier compatibility fix in the current environment.
- PostgreSQL initialization attempt fails with `Operation not permitted` for
  localhost:5432 inside the sandbox. New slice persistence and the search query
  with timestamps have not been exercised against a live database here.
- Auto selects CPU here. MPS and a complete multi-slice pretrained/database run
  remain unverified. No sandbox bypass was attempted.

Run from normal macOS Terminal:

```bash
just slices 'songs/Troy - Omen(1).flac' --preview
just slices 'songs/Troy - Omen(1).flac'
just test-db
# Substitute an ID printed by slices; layer 12 exists in the checked checkpoint.
just search --track-id 42 --layer 12 --limit 5
```

Grid alignment is deliberately deferred. Regions and windows are in seconds;
no downbeat accuracy is claimed. Each slice is a separate atomic analysis;
earlier slices remain committed if a later one fails.

The records below describe earlier environment states and are historical.

## Earlier update: MuQ / Transformers compatibility

The original verification record below is historical. The user has since installed
the Python 3.11 dependencies and downloaded the checkpoint. `uv.lock` now exists.

- Reproduced the exact `_attn_implementation` AttributeError using the real cached
  checkpoint on CPU: MuQ 0.1.0 / Transformers 5.16.1 / PyTorch 2.14.0.
- Added `test_muq_transformers_compatibility`, which builds a small real MuQ model
  with random weights. Its forward pass reproduces the same exception without
  downloading a checkpoint or connecting to PostgreSQL.
- Added a uv transitive constraint `transformers<5` and regenerated `uv.lock`
  offline. Resolution selected Transformers 4.57.6, huggingface-hub 0.36.2, and
  tokenizers 0.22.2.
- Installing those versions is blocked: online sync fails PyPI DNS resolution;
  offline sync cannot find the Transformers 4.57.6 wheel. The installed environment
  still contains Transformers 5.16.1, so the compatibility test is still red there.
  A passing inference run after the downgrade is **not yet verified**.

Apply and verify the fix from normal Terminal:

```bash
just sync
uv run python -m unittest -v test_main.AnalyzerChecks.test_muq_transformers_compatibility
just analyze 'Troy - Omen(1).flac' --offset 60 --duration 30 --save-npz
```

## Original implementation record

Checked on 2026-09-08. The implementation is present, but the complete specification
is **not yet verified**. In particular, `uv.lock` is still missing because dependency
resolution cannot finish. No successful MuQ inference or database insertion is claimed.

## Environment and limitations

- Checkout: `main`, no commits; all project files were initially untracked.
- The referenced `charter/pedagogy.md` was absent; no replacement guidance was invented.
- Python project requirement and `.python-version`: 3.11; local interpreter: 3.11.15.
- `uv sync` failed resolving PyPI: DNS lookup failed (`nodename nor servname provided`).
- `uv sync --offline` also failed: `muq` is absent from the local cache.
- `docker compose up -d` failed: permission denied accessing
  `/Users/matthijskruger/.docker/run/docker.sock`.
- No sandbox escalation or network workaround was used.

## Checks completed

Commands with `UV_CACHE_DIR=.cache/uv` use a project-local cache to avoid writing
outside the workspace. The normal `uv run` commands still require dependency
resolution, so cached tool executions below used `--no-sync` explicitly.

| Check | Evidence |
| --- | --- |
| Compose syntax | `docker compose config --quiet` passed. This does not prove image startup. |
| Python syntax | `.venv/bin/python -m compileall -q main.py mir test_main.py` passed. |
| Ruff formatting | `UV_CACHE_DIR=.cache/uv uv run --no-sync ruff format --check .` passed. |
| Ruff lint | `UV_CACHE_DIR=.cache/uv uv run --no-sync ruff check .` passed. |
| CLI on Python 3.11 | `.venv/bin/python -m unittest -v test_main.AnalyzerChecks.test_cli_validation` passed; `main.py --help` also passed. |
| Supplemental runtime checks | `NUMBA_CACHE_DIR=.cache/numba .cache/checks/bin/python -m unittest -v test_main` passed 3 checks, with the live database check skipped. |
| Numba availability, supplemental environment | Imported Numba 0.64.0 alongside PyTorch 2.11.0. |
| MPS availability, supplemental environment | `torch.backends.mps.is_available()` returned False. No MPS inference claimed. |
| Schema compilation | SQLAlchemy's PostgreSQL dialect compiled both tables and the partial unique index: BIGSERIAL IDs, timestamptz timestamps, double precision durations, VECTOR(1024), cascading FK, and both uniqueness rules. No server execution claimed. |
| Supplemental type check | `ty check` with cached dependency sources supplied as CLI `--extra-search-path` arguments reported only unresolved `muq`. This is **not a passing full type check**. No diagnostics were suppressed. |

The supplemental `.cache/checks` environment is Python **3.12.13**, created with uv,
using existing compatible cached package sources via a local `.pth` file. It is
ignored by Git and is not the application's environment or a dependency lock.
It exercises real librosa 0.11.0, NumPy 2.4.3, PyTorch 2.11.0, and pgvector/SQLAlchemy
code without downloading weights. Audio checks emitted dependency deprecation
warnings for Python 3.12's `aifc`, `audioop`, and `sunau` modules.

That supplemental environment also lacks a working Psycopg binary/libpq wrapper;
an attempted search could not reach PostgreSQL. The manifest requests
`psycopg[binary]` so a normal successful sync should install the required wrapper.

## Completion criteria audit

“Implemented” below means source is present; it does not imply the missing live
checks passed. Numbers match the original specification's completion criteria.

| Criteria | Implementation and evidence | Remaining verification |
| --- | --- | --- |
| 1 | PostgreSQL 18 pgvector Compose service, named volume, port, health check; configuration passes. | Actual image startup blocked by Docker socket permissions. |
| 2 | Python 3.11 dependency manifest present. | Successful `uv sync` and generated `uv.lock` blocked by PyPI access. |
| 3–4 | Cached Ruff format and lint checks pass. | Repeat exact commands after sync. |
| 5 | No type suppressions; cached-source check only reports missing MuQ. | Full `uv run ty check` after all dependencies install. |
| 6 | Numba direct dependency; actual import works in supplemental environment. | Verify installed Numba under Python 3.11. |
| 7–8 | Argparse accepts FLAC paths; real generated stereo FLAC loads as mono 24 kHz float32 with bounded window. | Repeat audio check under Python 3.11. |
| 9–10 | `load_model` uses the exact checkpoint and `main` calls it with `output_hidden_states=True`, fp32/eval/inference mode. | Download and execute actual MuQ. |
| 11–13 | `extract_embeddings` pools every returned state and final state in PyTorch before CPU conversion; real tensor/NPZ checks pass. | Validate real model output under Python 3.11. |
| 14–15 | `initialize_database` creates extension/tables/indexes; PostgreSQL DDL compilation passes. | Run database integration check. |
| 16–18 | `insert_analysis` inserts track and vectors inside one `engine.begin()` transaction; runnable rollback/round-trip check provided. | Execute live check, including deliberately failed embedding insert. |
| 19 | One configured dimension, supported by checkpoint config; mismatch rejection exercised with real tensors. | Match actual pretrained output against schema on a real run. |
| 20–21 | SQLAlchemy uses pgvector cosine distance against a scalar source subquery, filters same layer, excludes source. | Execute live search and numeric ordering assertions. |
| 22 | NPZ exports only pooled arrays; contents checked using real NumPy. | Repeat full analyze plus NPZ under Python 3.11. |
| 23 | Actual sandbox MPS availability is False; mocked availability tests prove auto selection and explicit failure behavior. | Verify target Terminal MPS inference. |
| 24–25 | No custom Numba kernels, object mode, or parallel kernels; ordinary tensor/array/database operations stay in their libraries. | None for the deliberate absence of kernels. |
| 26 | README contains setup, Docker lifecycle, analysis/search, checks, Numba, embeddings, model download, Apple Silicon, and exact remaining commands. | None for documentation; live results remain pending. |

## Remaining Terminal commands

Use your normal macOS Terminal in the repository:

```bash
uv sync
docker compose up -d --wait
uv run ruff format --check .
uv run ruff check .
uv run ty check
RUN_DATABASE_TESTS=1 uv run python -m unittest -v test_main
uv run python -c 'import numba, torch; print(numba.__version__); print(torch.backends.mps.is_available())'
uv run python main.py analyze /path/to/song.flac --device cpu --save-npz
uv run python main.py analyze /path/to/song.flac --device auto --offset 10
```

Use the printed source ID in:

```bash
uv run python main.py search --track-id TRACK_ID --layer 0 --limit 5
```

If MPS is available, exercise `analyze /path/to/song.flac --device mps` as well.
Retain the generated `uv.lock`. Reassess dependency compatibility if sync or real
inference exposes an upstream issue; none has been hidden with guessed pins.
