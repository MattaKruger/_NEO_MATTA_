# MIR search plan

Status: proposed, 2026-09-08. Worker redesign is parked. This document changes no
analysis code, stored data, or runtime configuration.

## Goal

Explore the saved album and VA through sound, genre, and musical sections, then
add natural-language retrieval. Keep each kind of evidence understandable and
auditionable. Use the current CLI and PostgreSQL; no worker or web UI is required.

The following interprets “genre + loop classification” as both tagging slices with
genres and labeling their musical role (intro/drop/breakdown/etc.). These are
separate outputs and should remain separate.

## Grounded inventory

These findings come from the checked-out source, not a live Metta service or a
measurement of search quality.

| Capability | Metta foundation | Transfer decision |
| --- | --- | --- |
| Track genre similarity | Generic vector lookup on `audio_features`, including separate 400-dimensional MAEST and EffNet columns | Reuse the query idea with a clearly defined song profile and consistent classifier. |
| Loop genre similarity | `LoopEmbeddingRepository.search_by_genre`: MAEST profile cosine distance, optional section filter | Use our EffNet scores consistently for the first version; this changes the model, not the retrieval principle. |
| More loops like this | `search_by_loop`: CLaMP3 cosine distance, source-loop and optional same-track exclusion | Our MuQ search already provides the core operation. Add same-song exclusion and grouped results. |
| Semantic loop search | Text encoded by CLaMP3, queried against matching CLaMP3 audio vectors | Requires a new matched audio/text representation and another audio pass. MuQ vectors cannot serve as these candidates. |
| Genre-informed section classification | Handwritten genre profiles plus Viterbi sequence decoding over energy, position, and duration | Adapt the labeling mechanism after persisting region context; evaluate before trusting or filtering by its labels. |
| Genre-filtered semantic search | Track-level query supports metadata genre filters; loop queries support section filters | Combining classifier-score filters with loop retrieval is new work. |
| Audition and query comparison | `LoopEvalView.vue`: playback, prompt A/B, more-like-this, results grouped by source song | Borrow the listening workflow, not the Vue application. |

Source references:

- [Loop queries](../metta/packages/metta-core/metta_core/repositories/loop_embeddings.py)
- [Loop evaluation endpoint](../metta/packages/metta-api/metta_api/api/endpoints/loop_eval.py)
- [Track vector queries](../metta/packages/metta-core/metta_core/repositories/audio_features.py)
- [Genre profile selection](../metta/packages/metta-worker/metta_worker/core/genre_profiles.py)
- [Section labeling](../metta/packages/metta-worker/metta_worker/core/genre_profile_service.py)
- [Slice extraction](../metta/packages/metta-worker/metta_worker/slice_extractor.py)
- [Old worker ordering](../metta/packages/metta-worker/metta_worker/worker.py)
- [Audition/comparison view](../metta/frontend/src/views/LoopEvalView.vue)

Important differences to preserve in the design:

- The old loop endpoint chooses `loop_id`, `genre_vector`, or `prompt` by precedence;
  it does not implement a hybrid ranker. Our interface should reject conflicting
  seed modes rather than silently choosing one.
- Old track genre filtering reads file metadata (`meta.genre` and `meta.ab:genre`),
  not the classifier score vector. A textual tag and a model prediction are not
  interchangeable evidence.
- The old evaluation endpoint counts CLaMP3-bearing loops even for genre searches.
  Report the eligible population for the actual query and filters instead.
- Old section labeling reads previously stored genres before the current run's
  EffNet step. First-run labeling can therefore use a generic profile. Compute the
  current genre context before labeling, without changing the saved boundaries.
- Metta labels `1 - cosine_distance` as similarity. This is a ranking score, not a
  percentage confidence that two recordings match.

## Vocabulary and representation contracts

**Song** means one source FLAC. **Region** is a structural interval in that song.
**Slice** is one selected analysis window. The current POC calls slice rows
`tracks`; preserve their existing IDs and explain this in CLI output.

Use “loop” informally for retrieval, but mark these as unaligned slices until beat
and downbeat alignment is verified. Sound similarity does not prove loopability,
tempo compatibility, key compatibility, or a clean musical transition.

| Representation | Answers | Scope and comparison rules |
| --- | --- | --- |
| MuQ pooled hidden layer | Which slices sound similar? | Compare the same model, layer, and extraction policy. |
| EffNet 400 scores | Which slices have similar predicted styles, or score for a named style? | Exact model/version and ordered label vocabulary must agree. |
| Section label | What role might this region play in its song? | Contextual heuristic, with profile/version and fit score. |
| Matched audio/text embeddings | Which slices match a description? | Both encoders must belong to the same validated checkpoint and preprocessing contract. |

Do not compare MuQ with CLaMP3, or EffNet with MAEST merely because vectors have
the same length. Do not concatenate them or average their raw distances.

## Queries to support

| User question | Candidate unit | Ranking | Optional constraints |
| --- | --- | --- | --- |
| Which songs have a genre profile like this song? | Song | EffNet profile cosine distance | Exclude source song |
| Which slices have a genre profile like this slice? | Slice | EffNet profile cosine distance | Exclude source song; section |
| Find techno slices | Slice | Score for exact `Electronic---Techno` label | Minimum score; duration |
| Find songs containing acid passages | Song, with matching slices | Best slice score for `Electronic---Acid` | Show where the match occurs |
| Find techno that sounds like this slice | Slice | MuQ cosine distance | EffNet genre-score threshold applied before ranking |
| Find similar sound outside this genre | Slice | MuQ cosine distance | Explicit excluded genre and threshold |
| Find breakdowns with a similar genre profile | Slice | EffNet profile cosine distance | Section label, once evaluated |
| Find “dry rolling percussion with little melody” | Slice | Matched text/audio cosine distance | Genre, section, duration |
| Find an acid breakdown matching this description | Slice | Matched text/audio cosine distance | Exact genre-score and section constraints |

Named-genre ranking uses the actual class score. A one-hot query vector plus cosine
is not the same operation. Multi-genre filters require explicit ANY/ALL semantics;
start with one positive and one negative label rather than an arbitrary query language.

Missing classification does not mean “not techno”: exclude unknowns from both
positive and negative genre-constrained queries and report their count. Reject
zero/nonfinite vectors for cosine search. Use full scores, not just the saved top five.

## Phase 1 — Make current results easy to explore

1. Verify real EffNet installation, inference, and storage in the user's Terminal;
   these were sandbox-blocked during implementation. Record actual model coverage.
2. Add a listing command with slice ID, filename, offset, duration, model coverage,
   and available top genres. No manually querying SQL to choose a seed.
3. Add EffNet profile similarity and exact-label lookup using pgvector/SQLAlchemy
   Core. MuQ remains the default sound-similarity mode.
4. Exclude all slices of the source song by default, with an explicit opt-in for
   within-song exploration. Until source IDs exist, use the stored absolute path;
   document that copied/renamed files are not deduplicated by this rule.
5. Add optional grouping by song, ranked by its best matching slice, with timestamps
   retained as evidence. Select top songs across all eligible slices before applying
   the song limit; grouping an already-truncated slice list gives misleading results.
6. Add a small audition command for a stored slice or returned result using macOS
   playback and a temporary decoded clip. No permanent clip library or web UI.

Done when: one seed retrieves both sound neighbors and genre-profile neighbors;
the user can listen to the source and matches; same-song exclusion, missing-data
counts, and grouped limits behave correctly. No new embeddings are needed beyond
finishing EffNet enrichment.

## Phase 2 — Cross genre and sound

Add genre-score constraints to MuQ retrieval. Filter in PostgreSQL before distance
ordering and limiting. Return MuQ distance and the relevant genre score separately.

Keep thresholds explicit and adjustable. A threshold is a retrieval choice, not a
calibrated confidence level. Use the exact Discogs label vocabulary, with a label
listing command; distinguish e.g. a style label from a broad genre category.

For song-to-song genre similarity, first derive a **selected-slice genre profile**:
mean EffNet scores over distinct selected windows for each song, excluding duplicate
analysis rows with identical boundaries and model/policy. Use equal window weighting
as the initial explicit policy and report contributing windows and completeness.
Only fully classified selected sets participate by default so profiles do not drift
merely because a batch is partially complete.

This profile is biased toward the start/peak windows we selected; it is not a
full-song classification. Do not duration-weight overlapping windows as if they
were independent coverage. If listening shows this profile is too biased, evaluate
a separate uniform, non-overlapping song sampling pass before changing the policy.

Keep “song has this overall selected-slice profile” separate from “song contains a
matching passage.” The latter ranks the best matching slice and can surface a
brief style change that averaging hides. Song-profile-to-slice-profile searches
can then reuse the same EffNet space with scope shown in the output.

Done when: genre-constrained sound results pass the requested filters, cross-genre
results have known classification, and song results show their contributing slices.
Do not add a weighted MuQ/EffNet blended score at this stage.

## Phase 3 — Genre-aware region and slice classification

Persist the region boundaries and normalized energy context already computed by
the slicer. Also store each slice's parent region and selection role (`start` or
`peak`); the current database does not retain those relationships.

Use the current song genre profile as context for a small adaptation of Metta's
sequence labeler. It scores energy, position within the song, and relative duration
against genre-specific section templates. This is a heuristic, not a trained
drop/verse detector. Its fit score must not be presented as calibrated confidence.

Required order: region detection → selected-slice classification → song genre
context → region labels. Label existing regions; do not merge, move, or regenerate
slice boundaries as a side effect. A slice inherits a region label only when its
provenance supports that assignment; arbitrary analysis windows may remain unknown.

For existing records, retain IDs and predictions. Recover region associations only
when rerunning the recorded slicer version reproduces the boundaries; otherwise
leave provenance unknown instead of silently attaching a different region plan.

Port only the profiles useful for the collection initially (electronic and generic),
with fallback and profile provenance visible. Review repeated section names in
Metta's templates carefully: its decoder uses template positions but some later
scoring uses names as dictionary keys, which can collapse distinct occurrences.

Done when: auditioned annotations across several songs plausibly identify sections,
first-run and repeat labeling use the same evidence, unknowns remain visible, and
classification does not invalidate stored MuQ/EffNet results. Only then expose
section-based search filters. Beat-grid alignment remains a separate later task.

## Phase 4 — Semantic loop search

Use Metta's text/audio retrieval concept and SQL shape. First run a bounded
feasibility check of the official CLaMP3 inference path on the M3 Pro with 16 GB:
one text prompt and several existing slices, measuring dependencies, memory,
latency, and whether CPU/MPS execution actually works. Do not assume CUDA-oriented
Metta code transfers unchanged.

Validate audio preprocessing against upstream before transferring the wrapper;
matching the final embedding dimension is insufficient proof of compatibility.
Do not copy Metta's hand-tuned pooling/fusion policy without comparison against
the official reference and identical test audio.

Specific discrepancies found in the
[old encoder wrapper](../metta/packages/metta-worker/metta_worker/extractors/clamp3_extractor.py):
the loop path passes the whole slice through MERT once, whereas the official audio
path uses five-second windows and its quickstart requests averaging hidden layers.
Metta retains layer vectors as sequence positions. It also uses mean/max fusion
where upstream uses a length-weighted segment mean. Verify these differences on
identical audio before selecting a preprocessing policy. Do not copy the old
exception-to-zero-vector fallback.

The [official CLaMP3 project](https://github.com/sanderwood/clamp3) documents a
shared audio/text space, but its setup does not establish macOS/MPS support.
Its [dependency pins](https://github.com/sanderwood/clamp3/blob/main/requirements.txt)
include NumPy 1.26.4 and Transformers 4.40.0; do not apply them wholesale to this
POC. Use the official
[audio preprocessing](https://github.com/sanderwood/clamp3/blob/main/preprocessing/audio/extract_mert.py)
and [embedding extraction](https://github.com/sanderwood/clamp3/blob/main/code/extract_clamp3.py)
as reference contracts. Decode original slice audio directly at the required
24 kHz, avoiding Metta's intermediate 16 kHz WAV and subsequent upsampling.

If feasible, store its audio embeddings separately, tagged with checkpoint and
preprocessing version. Encode text with the matching text path and execute cosine
search in PostgreSQL. Reuse genre and section filters from earlier phases.
Existing MuQ/EffNet data stays intact; semantic search requires re-reading audio,
not replacing those representations.

If CLaMP3 is impractical on this machine, pause semantic implementation and compare
a smaller matched audio/text model as a separate measured choice. Do not pretend
MuQ plus an unrelated text encoder provides semantic retrieval.

Done when: a fixed set of descriptions retrieves auditionable slices, stored and
query embeddings follow one verified contract, and genre/section filters operate
before limiting. Text embeddings expand the original POC scope specifically for
this feature; no caption generation or general chat interface is planned.

## Evaluation and delivery order

Use the existing VA and album as the first listening set. They are enough to test
mechanics and discover useful queries, but not to establish broad musical quality.

- Choose about ten varied seed slices and ten concrete text prompts when semantic
  retrieval becomes available. Include percussion, melodic, sparse, and dense audio.
- Compare MuQ layers using the same candidates and source-song exclusion. Do not
  declare layer 12 best merely because it appears in current command examples.
- Audition top-five results; record relevant / interesting / unrelated with a short
  reason. Compare genre-only, sound-only, and constrained retrieval separately.
- Count eligible slices and unique songs. Record missing-model coverage, latency,
  duplicate-source leakage, and relevant results among the auditioned five.
- Include silence/very quiet audio, short slices, missing genres, repeated analyses,
  absent source files, and queries with no eligible candidates.
- Add SQL integration checks for filters-before-limit, grouping-before-song-limit,
  consistent model/version/layer, and self/same-song exclusion.

Deliver Phase 1 first, listen, then Phase 2. Region classification and semantic
search can follow independently once their respective data and runtime checks pass.
No worker redesign, ANN indexes, combined score tuning, MAEST replacement, grid
snapping, or old frontend port is required to start discovering useful matches.
