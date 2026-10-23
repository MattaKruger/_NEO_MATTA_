# Discogs classifier transfer

Exploration, 2026-09-08. No runtime or database changes made.

## What the old project actually runs

| Path | Model and input | Output and use |
| --- | --- | --- |
| [Discogs extractor](../metta/packages/metta-worker/metta_worker/discogs_genre_extractor.py) | `discogs-effnet-bs64-1.pb`, mono 16 kHz, full song | Mean frame predictions; top five labels split on `---`; full 400-score vector. |
| [Semantic extractor](../metta/packages/metta-worker/metta_worker/essentia_semantic_extractor.py) | `discogs-maest-10s-pw-2.pb`, mono 16 kHz, complete 10-second chunks | Mean chunk logits followed by sigmoid; 400-score genre vector per slice. Also runs unrelated VGGish heads. |
| [Standalone classifier](../metta/packages/metta-worker/metta_worker/core/genre_classifier.py) | EffNet, normally seconds 45–75 | Top six structured labels; differs from the worker's whole-song path. |

The [worker](../metta/packages/metta-worker/metta_worker/worker.py) runs EffNet first
to decide whether to perform expensive per-slice processing. MAEST supplies loop
genre vectors. These are two different classifiers despite sharing the Discogs
taxonomy. Neither is a Discogs API lookup.

The checked-out MAEST graph exists locally (328 MB), alongside classifier metadata.
EffNet model files were not found in the searched checkout paths. Obtain the graph
and its matching label metadata from the official distribution; do not reuse a
different model's label order merely because both have 400 outputs.

## Proposed smallest transfer

Start with EffNet, the classifier explicitly named Discogs in the old pipeline.
Classify each stored analysis window, rather than silently assigning a whole-song
prediction to every slice. This deliberately changes the old worker's scope.
MAEST is the alternative if reproducing the old per-loop genre results is the goal.

1. Add a separate `classify --track-id ID` command, then `classify --all` for missing
   results. These are proposed commands, not available commands.
2. Read the existing path, offset and duration from `tracks`. Load that window as
   mono float32 at the classifier's 16 kHz. MuQ continues using 24 kHz.
3. Reuse one classifier across a sequential batch. MuQ need not load or run.
4. Validate output shape, finite scores and exact label count. Pool only the
   classifier's predictions; print the top five model scores and labels.
5. Add a Core `genre_predictions` table with `track_id` foreign key, model/version
   identifier, `scores vector(400)`, and timestamp; unique on track/model. Keep the
   matching ordered labels in versioned model metadata. Insert/update each result
   transactionally. New-table creation works with the current `create_all` setup;
   no destructive reset or migration framework is necessary.
6. Join predictions onto MuQ search results for display first. Add filtering or a
   separate genre-vector search only when wanted; do not combine distance scales
   into an arbitrary score.

Existing MuQ rows and vectors remain useful. Their 1024-dimensional embedding
column is not a destination for 400-dimensional classifier scores. A separate
table also lets already-analyzed albums be enriched without duplicate track rows.

The original representation-only scope would expand to optional genre prediction.
No worker registry, singleton classes, CUDA setup, web service, VGGish heads,
electronic-only gating, or genre-based changes to slicing are needed.

## Runtime gate and checks

See [platform research](DISCOGS-COMPATIBILITY.md). Check the pinned Essentia
TensorFlow wheel with Python 3.11 and this project's NumPy before integrating it.
Do not assume installing TensorFlow separately enables Essentia's inference APIs,
or that the classifier uses MPS merely because MuQ does.

First prove one real 30-second FLAC prediction: available inference algorithm,
correct graph output, matching labels, 400 finite scores, runtime and memory.
Then check an actual short slice against model requirements and choose an explicit
skip/error policy; never represent an unclassified slice with invented zero scores.
If choosing MAEST, preserve its complete-10-second chunk policy, report uncovered
tails, and skip windows shorter than 10 seconds instead of copying the old all-zero
fallback. Averaging logits and averaging probabilities are different operations.

Finally verify database persistence and rerun behavior with an existing slice ID,
and compare a few audible examples against the old pipeline using identical audio
windows. No classifier inference, dependency installation, or database write was
performed during this exploration.
