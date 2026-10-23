"""Export fixed MuQ listening batches and summarize Label Studio JSON reviews."""

import argparse
import hashlib
import json
import math
import random
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4

from sqlalchemy.engine import Engine, RowMapping
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from main import SAMPLE_RATE, load_audio
from mir.database import create_engine, find_similar_tracks, get_track

QA_ROOT = Path(__file__).resolve().parent / "qa"
LABELS = ("Unrelated", "Somewhat similar", "Strongly similar", "Unsure")


def export_batch(
    engine: Engine, source_ids: list[int], layer: int, limit: int, seed: int, output: Path
) -> None:
    """Freeze database search results and audio; publish only complete batches."""
    if not source_ids or any(identifier <= 0 for identifier in source_ids):
        raise ValueError("Supply at least one positive --track-id.")
    if layer < 0 or not 1 <= limit <= 20:
        raise ValueError("--layer must be nonnegative and --limit must be between 1 and 20.")
    root = QA_ROOT.resolve()
    output = output.resolve()
    if not output.is_relative_to(root) or output == root:
        raise ValueError(f"--output must be a new batch directory inside {root}.")
    relative = output.relative_to(root)
    if relative.parts[0] == "label-studio":
        raise ValueError("qa/label-studio is reserved for Label Studio's database.")
    if output.exists():
        raise ValueError(f"Output already exists: {output}. Choose a new batch name.")
    source_ids = list(dict.fromkeys(source_ids))
    batch_id = str(uuid4())
    tasks: list[dict[str, Any]] = []
    clips: dict[int, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".listening-", dir=output.parent) as temporary:
        staging = Path(temporary)
        (staging / "clips").mkdir()

        def snapshot(row: Mapping[str, Any] | RowMapping) -> dict[str, Any]:
            import numpy as np
            import soundfile as sf

            identifier = int(row["track_id"] if "track_id" in row else row["id"])
            if identifier in clips:
                return clips[identifier]
            path = Path(row["file_path"])
            offset, duration = float(row["analysis_offset"]), float(row["analysis_duration"])
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(f"Track {identifier}: invalid stored duration {duration}.")
            audio, _ = load_audio(path, offset, duration)
            if abs(audio.size / SAMPLE_RATE - duration) > 1 / SAMPLE_RATE + 1e-9:
                raise ValueError(f"Track {identifier}: audio is truncated or changed: {path}")
            if not np.isfinite(audio).all() or np.any(np.abs(audio) > 1):
                raise ValueError(f"Track {identifier}: non-finite or out-of-range audio: {path}")
            clip = Path("clips") / f"{uuid4()}.wav"
            sf.write(staging / clip, audio, SAMPLE_RATE, subtype="PCM_24")
            result = {
                "track_id": identifier,
                "file_path": str(path.resolve()),
                "analysis_offset": offset,
                "analysis_duration": duration,
                "model_name": row["model_name"],
                "clip": clip.as_posix(),
                "sha256": hashlib.sha256((staging / clip).read_bytes()).hexdigest(),
            }
            clips[identifier] = result
            return result

        for identifier in source_ids:
            source = get_track(engine, identifier)
            candidates = find_similar_tracks(engine, identifier, layer, limit, lab_mode=True)
            counts[str(identifier)] = len(candidates)
            for rank, candidate in enumerate(candidates, 1):
                distance = float(candidate["distance"])
                if not math.isfinite(distance):
                    raise ValueError(f"Track {candidate['track_id']}: non-finite cosine distance.")
                tasks.append(
                    {
                        "task_key": str(uuid4()),
                        "source": snapshot(source),
                        "candidate": snapshot(candidate),
                        "rank": rank,
                        "distance": distance,
                    }
                )
        if not tasks:
            raise ValueError(f"No candidates for source IDs {source_ids}; no batch exported.")
        random.Random(seed).shuffle(tasks)
        manifest = {
            "schema_version": 1,
            "rubric_version": "overall-similarity-v1",
            "batch_id": batch_id,
            "created_at": datetime.now(UTC).isoformat(),
            "source_ids": source_ids,
            "layer": layer,
            "limit": limit,
            "seed": seed,
            "exclude_same_file": True,
            "candidate_counts": counts,
            "playback_policy": "muq-input-v1",
            "sample_rate": SAMPLE_RATE,
            "tasks": tasks,
        }

        def audio_url(clip: str) -> str:
            return "/data/local-files/?d=" + quote((relative / clip).as_posix(), safe="/")

        imported = [
            {
                "data": {
                    "batch_id": batch_id,
                    "task_key": task["task_key"],
                    "reference_audio": audio_url(task["source"]["clip"]),
                    "candidate_audio": audio_url(task["candidate"]["clip"]),
                }
            }
            for task in tasks
        ]
        for name, value in (("manifest.json", manifest), ("tasks.json", imported)):
            (staging / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        if output.exists():
            raise ValueError(f"Output already exists: {output}.")
        staging.rename(output)
    print(f"Batch: {output}\nTasks: {len(tasks)}\nLayer: {layer}")
    for identifier, count in counts.items():
        print(f"Source {identifier}: {count}/{limit} candidates")
    print(f"Import: {output / 'tasks.json'}\nLabeling configuration: listening-labels.xml")


def review_time(annotation: dict[str, Any]) -> datetime:
    value = annotation.get("updated_at") or annotation.get("created_at")
    if not isinstance(value, str):
        raise ValueError("Completed annotation is missing its timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"Invalid annotation timestamp: {value}") from error
    if parsed.tzinfo is None:
        raise ValueError("Annotation timestamps must include a timezone.")
    return parsed


def summarize(manifest: dict[str, Any], exported: list[dict[str, Any]]) -> str:
    """Reconcile a full Label Studio export against its immutable batch manifest."""
    try:
        return _summarize(manifest, exported)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"Malformed manifest or full Label Studio JSON export: {error}") from error


def _summarize(manifest: dict[str, Any], exported: list[dict[str, Any]]) -> str:
    if manifest["schema_version"] != 1 or not isinstance(exported, list):
        raise ValueError("Expected a v1 manifest and full JSON task-list export.")
    tasks = {task["task_key"]: task for task in manifest["tasks"]}
    if len(tasks) != len(manifest["tasks"]) or not tasks:
        raise ValueError("Manifest must contain unique, nonempty tasks.")
    limit = manifest["limit"]
    if not isinstance(limit, int) or not 1 <= limit <= 20:
        raise ValueError("Invalid manifest search limit.")
    sources = manifest["source_ids"]
    if (
        not isinstance(sources, list)
        or not sources
        or any(type(identifier) is not int or identifier <= 0 for identifier in sources)
        or len(set(sources)) != len(sources)
    ):
        raise ValueError("Manifest source IDs must be unique positive integers.")
    if any(task["source"]["track_id"] not in sources for task in tasks.values()):
        raise ValueError("Manifest task references an unknown source.")
    counts_by_source = manifest["candidate_counts"]
    if set(counts_by_source) != {str(identifier) for identifier in sources}:
        raise ValueError("Manifest candidate counts do not match its sources.")
    for identifier in sources:
        entries = [t for t in tasks.values() if t["source"]["track_id"] == identifier]
        ranks = [t["rank"] for t in entries]
        if (
            len(entries) > limit
            or counts_by_source[str(identifier)] != len(entries)
            or any(type(rank) is not int for rank in ranks)
            or sorted(ranks) != list(range(1, len(entries) + 1))
            or len({t["candidate"]["track_id"] for t in entries}) != len(entries)
        ):
            raise ValueError(f"Source {identifier}: inconsistent candidate count or ranks.")
    ratings: dict[str, tuple[str, str]] = {}
    reviewers: set[int] = set()
    seen: set[str] = set()
    skipped: set[str] = set()
    events = 0
    cancelled_events = 0
    for task in exported:
        data = task["data"]
        key = data["task_key"]
        if data["batch_id"] != manifest["batch_id"] or key not in tasks:
            raise ValueError(f"Foreign batch or unknown task key: {key}")
        if key in seen:
            raise ValueError(f"Duplicate exported task: {key}")
        seen.add(key)
        annotations = task.get("annotations", [])
        if not isinstance(annotations, list):
            raise ValueError(f"Task {key}: annotations must be a list.")
        completed: list[tuple[datetime, str, str]] = []
        for annotation in annotations:
            events += 1
            reviewer = annotation.get("completed_by")
            if isinstance(reviewer, dict):
                reviewer = reviewer.get("id")
            if not isinstance(reviewer, int) or isinstance(reviewer, bool):
                raise ValueError(f"Task {key}: annotation is missing its reviewer ID.")
            reviewers.add(reviewer)
            if len(reviewers) > 1:
                raise ValueError("Multiple reviewers found; v1 requires a single-reviewer export.")
            cancelled = annotation.get("was_cancelled", False)
            if not isinstance(cancelled, bool):
                raise ValueError(f"Task {key}: invalid cancellation state.")
            if cancelled:
                cancelled_events += 1
                skipped.add(key)
                continue
            result = annotation["result"]
            if not isinstance(result, list):
                raise ValueError(f"Task {key}: result must be a list.")
            choices = [entry for entry in result if entry.get("from_name") == "similarity"]
            if len(choices) != 1:
                raise ValueError(f"Task {key}: expected exactly one similarity judgment.")
            choice = choices[0]
            labels = choice["value"]["choices"]
            if (
                choice.get("type") != "choices"
                or choice.get("to_name") != "candidate"
                or not isinstance(labels, list)
                or len(labels) != 1
                or labels[0] not in LABELS
            ):
                raise ValueError(f"Task {key}: invalid similarity label.")
            notes = [entry for entry in result if entry.get("from_name") == "note"]
            if len(notes) > 1:
                raise ValueError(f"Task {key}: duplicate notes.")
            texts = notes[0]["value"]["text"] if notes else []
            if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
                raise ValueError(f"Task {key}: invalid note.")
            completed.append((review_time(annotation), labels[0], " ".join(texts)))
        if completed:
            completed.sort(key=lambda item: item[0])
            if len(completed) > 1 and completed[-1][0] == completed[-2][0]:
                raise ValueError(f"Task {key}: ambiguous duplicate completed annotations.")
            _, label, note = completed[-1]
            ratings[key] = (label, note)
            skipped.discard(key)
    counts = Counter(label for label, _ in ratings.values())
    decisive = len(ratings) - counts["Unsure"]
    relevant = counts["Somewhat similar"] + counts["Strongly similar"]
    lines = [
        f"Batch: {manifest['batch_id']} | Layer: {manifest['layer']}",
        f"Completed: {len(ratings)}/{len(tasks)}",
        f"Unsure: {counts['Unsure']}",
        f"Skipped/cancelled: {len(skipped)}",
        f"Unreviewed: {len(tasks) - len(ratings) - len(skipped)}",
        f"Annotation events: {events} (cancelled: {cancelled_events})",
        *(f"{label}: {counts[label]}" for label in LABELS[:3]),
        f"Judged relevance: {relevant}/{decisive}"
        + (f" ({relevant / decisive:.0%})" if decisive else " (no decisive judgments)"),
    ]
    for identifier in manifest["source_ids"]:
        source_tasks = sorted(
            (task for task in tasks.values() if task["source"]["track_id"] == identifier),
            key=lambda task: task["rank"],
        )
        judged = [
            ratings[t["task_key"]][0]
            for t in source_tasks
            if t["task_key"] in ratings and ratings[t["task_key"]][0] != "Unsure"
        ]
        files = {t["candidate"]["file_path"] for t in source_tasks}
        lines.append(f"\nSource {identifier}: {len(source_tasks)} candidates, {len(files)} files")
        if len(source_tasks) == limit and len(judged) == limit:
            matches = sum(label != "Unrelated" for label in judged)
            lines.append(f"precision@{limit}: {matches}/{limit} ({matches / limit:.0%})")
        else:
            lines.append(f"precision@{limit}: unavailable ({len(judged)} decisive judgments)")
        for task in source_tasks:
            key = task["task_key"]
            label, note = ratings.get(key, ("Skipped" if key in skipped else "Unreviewed", ""))
            lines.append(
                f"  {task['rank']}. track_id={task['candidate']['track_id']} {label}"
                + (f" — {note}" if note else "")
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Create a Label Studio listening batch")
    export.add_argument("--track-id", type=int, action="append", required=True)
    export.add_argument("--layer", type=int, required=True)
    export.add_argument("--limit", type=int, default=5)
    export.add_argument("--seed", type=int, default=0)
    export.add_argument("--output", type=Path, required=True)
    report = commands.add_parser("report", help="Summarize full JSON annotations")
    report.add_argument("manifest", type=Path)
    report.add_argument("annotations", type=Path)
    args = parser.parse_args(argv)
    engine = None
    try:
        if args.command == "export":
            engine = create_engine()
            export_batch(engine, args.track_id, args.layer, args.limit, args.seed, args.output)
        else:
            manifest = json.loads(args.manifest.read_text())
            annotations = json.loads(args.annotations.read_text())
            print(summarize(manifest, annotations))
    except OperationalError as error:
        print(
            f"Database unavailable: {error.orig}\nStart PostgreSQL: docker compose up -d",
            file=sys.stderr,
        )
        return 1
    except SQLAlchemyError as error:
        print(f"Database error: {getattr(error, 'orig', error)}", file=sys.stderr)
        return 1
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(main())
