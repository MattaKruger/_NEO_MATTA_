# Neo Metta v2 handover — 2026-10-23

This is the active rewrite. Work in `~/mana/_NEO_MATTA_`; the older `metta`,
`neo-metta`, and `metta-old` folders are siblings for reference. GitHub origin is
`git@github.com:MattaKruger/_NEO_MATTA_.git`. Baseline `7197cd3` preserves the
original implementation. Migration is complete; do not rerun the migration script.

## Current foundation

- Existing CLI: MuQ analysis/slicing/search, optional EffNet classification, and
  Label Studio batch export/reporting.
- `just verify`: locked format/lint/type and local test checks.
- `just sync-effnet`: locked installation including the optional classifier.
- GitHub Actions: all tests, including PostgreSQL, against an isolated database.
- Native EffNet failures identify the stored slice and duration; earlier successful
  results remain saved. Very short windows can fail in the native graph.
- Music/artwork, environment secrets, models/caches, environments, and QA data stay
  out of source control. QA data is durable: back it up.

See [VERIFICATION.md](VERIFICATION.md) for evidence and limitations, and the
[foundation checklist](docs/superpowers/plans/2026-10-23-foundation.md).

## Next gates

1. Commit/push the prepared foundation diff when Git metadata and network access
   permit. Do not include local data or use `git clean -fdx`.
2. Run `docker compose up -d --wait`, then `just test-db` against the intended local
   database. Tests create marked rows and clean up their own data; no reset needed.
3. Verify the first hosted CI run.
4. Export a real batch, review in Label Studio, edit/skip a judgment, restart,
   export full JSON, and reconcile it with `just lab-report`.
5. Resume refactoring only from verified behavior. Worker redesign and additional
   search modes remain parked; [MIR-SEARCH-PLAN.md](MIR-SEARCH-PLAN.md) is a proposal.

The sandbox denied Docker access, Git writes, and GitHub DNS during this session.
The referenced `charter/pedagogy.md` was absent; no replacement instructions were
invented. No Desktop access is needed for this project.
