# V2 foundation implementation plan

**Goal:** Preserve the approved V2 checkout and make verification repeatable before
resuming the refactor.

**Spec:** The user's instruction: "v2 in git, verify and provide a solid foundation,"
followed by the approved move into `~/mana/_NEO_MATTA_`.

**Architecture:** Keep the current CLI, SQLAlchemy Core/pgvector storage, optional
EffNet classifier, and Label Studio workflow. Make no worker or search redesign.
Use uv's existing lockfile and the existing unittest suite.

## Constraints and review focus

- Preserve music, model caches, QA data, existing IDs, and all four Git histories.
- Never claim mocked checks establish live PostgreSQL or browser behavior.
- Keep optional EffNet optional; ordinary commands retain installed extras.
- CI must not need pretrained weights, a GPU, or the user's database.
- Prevent music/artwork and environment secrets from entering the source commit.
- Keep CI on the existing lockfile and avoid adding runtime dependencies.

## Tasks

- [x] Checkpoint existing V2 source (`7197cd3`) and migrate into the empty clone.
- [x] Confirm clone remote, source files, and the three older sibling folders.
- [x] Repair generated virtualenv text launchers after relocation.
- [x] Add locked just recipes and `verify`; ignore songs and local env variants.
- [x] Add an isolated PostgreSQL CI job running existing checks and tests.
- [x] Convert the observed native EffNet failure to a slice-specific CLI error,
      with a regression proving earlier successful saves are preserved.
- [x] Run lock consistency, formatting/lint/types, unit checks, and real cached model checks.
- [x] Record current evidence and remaining external runtime gates in the docs.
- [x] Review the complete diff independently; correct the two minor documentation findings.
- [ ] Commit/push when Git writes and network access permit.

## Decisions and limits

- Use one Ubuntu CI job and the existing lockfile; no new dependency group or
  dependencies. PyTorch's Linux distribution includes CUDA dependencies, so the
  first install is larger even though all tests run on CPU.
- Docker socket and GitHub DNS access are blocked in this sandbox. CI execution,
  live database integration, and Label Studio UI verification remain separate gates.
- The current environment marks `.git` read-only. Prepare the complete diff even
  if a normal-Terminal commit/push is needed.
