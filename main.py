"""Analyze a short FLAC window with MuQ, or search stored representations."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch
    from numpy.typing import NDArray

MODEL_NAME = "OpenMuQ/MuQ-large-msd-iter"
SAMPLE_RATE = 24_000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze", help="Store pooled MuQ embeddings from a local FLAC")
    analyze.add_argument("path", type=Path)
    analyze.add_argument(
        "--offset", type=float, default=0, help="Start time in seconds (default: 0)"
    )
    analyze.add_argument("--duration", type=float, default=30, help="Seconds to load (default: 30)")
    analyze.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    analyze.add_argument("--save-npz", action="store_true", help="Also export pooled vectors")
    analyze.add_argument("--output", type=Path, help="NPZ path; requires --save-npz")
    slices = commands.add_parser("slices", help="Select energy regions and embed start/peak slices")
    slices.add_argument("path", type=Path, help="FLAC file or directory (not recursive)")
    slices.add_argument("--slice-duration", type=float, default=30)
    slices.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    slices.add_argument(
        "--preview", action="store_true", help="Print slices without MuQ or database"
    )
    search = commands.add_parser("search", help="Find nearest tracks using one hidden-state layer")
    search.add_argument("--track-id", type=int, required=True)
    search.add_argument("--layer", type=int, required=True)
    search.add_argument("--limit", type=int, default=5)
    classify = commands.add_parser("classify", help="Add EffNet genres to stored slices")
    target = classify.add_mutually_exclusive_group(required=True)
    target.add_argument("--track-id", type=int)
    target.add_argument("--all", action="store_true", help="Classify slices missing EffNet results")
    args = parser.parse_args(argv)
    if args.command == "analyze":
        if not math.isfinite(args.offset) or args.offset < 0:
            parser.error("--offset must be finite and zero or greater")
        if not math.isfinite(args.duration) or args.duration <= 0:
            parser.error("--duration must be finite and positive")
        if args.output is not None and not args.save_npz:
            parser.error("--output requires --save-npz")
    elif args.command == "slices":
        if not math.isfinite(args.slice_duration) or args.slice_duration < 1:
            parser.error("--slice-duration must be finite and at least 1 second")
    elif args.command == "classify":
        if args.track_id is not None and args.track_id <= 0:
            parser.error("--track-id must be positive")
    else:
        if args.track_id <= 0:
            parser.error("--track-id must be positive")
        if args.layer < 0:
            parser.error("--layer must be zero or greater")
        if args.limit <= 0:
            parser.error("--limit must be positive")
    return args


def select_device(requested: str) -> torch.device:
    import torch

    if requested == "cpu":
        return torch.device("cpu")
    available = torch.backends.mps.is_available()
    if requested == "mps" and not available:
        raise ValueError(
            "MPS is unavailable. Try --device cpu, or verify MPS in your normal Terminal."
        )
    return torch.device("mps" if available else "cpu")


def load_audio(
    path: Path, offset: float, duration: float | None, *, sample_rate: int = SAMPLE_RATE
) -> tuple[NDArray[np.float32], float]:
    if not path.exists():
        raise ValueError(f"File does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    if path.suffix.lower() != ".flac":
        raise ValueError(f"Unsupported audio: expected a .flac file: {path}")
    if not math.isfinite(offset) or offset < 0:
        raise ValueError("Offset must be finite and zero or greater.")
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("Duration must be finite and positive.")
    import librosa
    import numpy as np
    import soundfile as sf

    try:
        # Inspect the header without decoding the full track. Passing the open file
        # to librosa also avoids an unrelated decoder fallback for corrupt FLACs.
        with sf.SoundFile(path) as source:
            if source.format != "FLAC":
                raise ValueError(f"Unsupported audio: file contents are not FLAC: {path}")
            original_duration = source.frames / source.samplerate
            if offset >= original_duration:
                raise ValueError(
                    f"Offset {offset:g} s is at or beyond the end ({original_duration:g} s)."
                )
            audio, _ = librosa.load(
                source,
                sr=sample_rate,
                mono=True,
                offset=offset,
                duration=min(duration, original_duration - offset)
                if duration is not None
                else None,
                dtype=np.float32,
            )
    except (sf.LibsndfileError, OSError) as error:
        raise ValueError(f"Cannot decode FLAC {path}: {error}") from error
    if audio.size == 0:
        raise ValueError("The selected audio region is empty.")
    return audio, original_duration


def load_model(device: torch.device) -> torch.nn.Module:
    # Search and ordinary checks do not need to import MuQ or download weights.
    import torch
    from muq import MuQ

    return MuQ.from_pretrained(MODEL_NAME).to(device=device, dtype=torch.float32).eval()


def extract_embeddings(
    hidden_states: tuple[torch.Tensor, ...], last_hidden_state: torch.Tensor
) -> dict[str, NDArray[np.float32]]:
    import torch

    from mir.database import validate_vector

    if not hidden_states:
        raise ValueError("MuQ returned no hidden states; output_hidden_states=True is required.")
    states = {f"hidden_{index:02d}": state for index, state in enumerate(hidden_states)}
    states["last_hidden_state"] = last_hidden_state
    pooled: dict[str, NDArray[np.float32]] = {}
    for name, state in states.items():
        if state.ndim != 3 or state.shape[0] != 1 or state.shape[1] == 0:
            raise ValueError(
                f"Unexpected {name} shape {tuple(state.shape)}; expected [1, time, dim]."
            )
        vector = state.mean(dim=1).squeeze(0).detach().to(device="cpu", dtype=torch.float32).numpy()
        validate_vector(vector)
        pooled[name] = vector
    return pooled


def save_npz(path: Path, pooled: dict[str, NDArray[np.float32]]) -> None:
    import numpy as np

    # A file handle preserves the exact --output path (NumPy otherwise adds .npz).
    with path.open("wb") as output:
        np.savez(output, allow_pickle=False, **pooled)


def slice_paths(path: Path) -> list[Path]:
    """Return one input file or sorted immediate FLAC children of a directory."""
    try:
        if not path.is_dir():
            return [path]
        paths = sorted(
            (
                child
                for child in path.iterdir()
                if child.is_file() and child.suffix.lower() == ".flac"
            ),
            key=lambda child: child.name,
        )
    except OSError as error:
        raise ValueError(f"Cannot read directory {path}: {error}") from error
    if not paths:
        raise ValueError(f"No FLAC files found in directory: {path}")
    return paths


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from sqlalchemy.exc import OperationalError, SQLAlchemyError

    from mir.database import (
        create_engine,
        find_similar_tracks,
        initialize_database,
        insert_analysis,
    )

    engine = None
    try:
        if args.command == "classify":
            from mir.genres import classify_tracks

            engine = create_engine()
            initialize_database(engine)
            classify_tracks(engine, args.track_id)
            return 0
        if args.command == "search":
            engine = create_engine()
            initialize_database(engine)
            results = find_similar_tracks(engine, args.track_id, args.layer, args.limit)
            print(f"Source track: {args.track_id}\nEmbedding: hidden_state layer {args.layer}")
            print("\nNearest:")
            for rank, row in enumerate(results, 1):
                print(
                    f"{rank}. track_id={row['track_id']}   distance={row['distance']:.3f}   "
                    f"{row['file_path']}  "
                    f"[{row['analysis_offset']:.2f}–"
                    f"{row['analysis_offset'] + row['analysis_duration']:.2f} s]"
                )
                if row["top_labels"]:
                    print(
                        "   Genres: "
                        + ", ".join(
                            f"{item['genre']} ({item['score']:.3f})" for item in row["top_labels"]
                        )
                    )
            if not results:
                print("No other tracks have this layer.")
            return 0

        root = args.path.expanduser().resolve()
        paths = slice_paths(root) if args.command == "slices" else [root]
        model = None
        for file_index, path in enumerate(paths, 1):
            print(f"File {file_index}/{len(paths)}: {path}", flush=True)
            if args.command == "analyze":
                device = select_device(args.device)
                audio, original_duration = load_audio(path, args.offset, args.duration)
                output_path = (args.output or path.with_suffix(".muq.npz")).expanduser().resolve()
                if args.save_npz and (
                    output_path == path or (output_path.exists() and output_path.samefile(path))
                ):
                    raise ValueError("NPZ output must not overwrite the input FLAC.")
            elif args.command == "slices":
                from mir.slicing import energy_regions, select_slices

                audio, original_duration = load_audio(path, 0, None)
                regions = energy_regions(audio, SAMPLE_RATE)
                windows = select_slices(audio, SAMPLE_RATE, regions, args.slice_duration)
                print(f"Input: {path}\nOriginal duration: {original_duration:.1f} s")
                print("Grid: unaligned seconds (beat/downbeat alignment deferred)")
                for index, (start, end) in enumerate(regions, 1):
                    print(f"Region {index}: {start:.2f}–{end:.2f} s")
                for index, (start, end, role) in enumerate(windows, 1):
                    print(f"Slice {index}: {start:.2f}–{end:.2f} s ({role})")
                if not windows:
                    raise ValueError("No slices of at least 1 second were selected.")
                if args.preview:
                    del audio
                    continue
                device = select_device(args.device)

            if engine is None:
                engine = create_engine()
                initialize_database(engine)
            if model is None:
                model = load_model(device)
            import torch

            if args.command == "analyze":
                windows = [(0.0, len(audio) / SAMPLE_RATE, "window")]
            for index, (start, end, role) in enumerate(windows, 1):
                segment = audio[round(start * SAMPLE_RATE) : round(end * SAMPLE_RATE)]
                tensor = (
                    torch.from_numpy(segment).unsqueeze(0).to(device=device, dtype=torch.float32)
                )
                with torch.inference_mode():
                    output = model(tensor, output_hidden_states=True)
                    last_shape = tuple(output.last_hidden_state.shape)
                    hidden_count = len(output.hidden_states)
                    pooled = extract_embeddings(output.hidden_states, output.last_hidden_state)
                del output, tensor
                actual_duration = len(segment) / SAMPLE_RATE
                offset = args.offset if args.command == "analyze" else start
                track_id = insert_analysis(
                    engine,
                    {
                        "file_path": str(path),
                        "file_name": path.name,
                        "sample_rate": SAMPLE_RATE,
                        "original_duration": original_duration,
                        "analysis_offset": offset,
                        "analysis_duration": actual_duration,
                        "model_name": MODEL_NAME,
                    },
                    pooled,
                )
                print(f"Input: {path}\nOriginal duration: {original_duration:.1f} s")
                print(f"Analysis region: {offset:.2f}–{offset + actual_duration:.2f} s")
                print(f"Sample rate: {SAMPLE_RATE} Hz\nDevice: {device}\nModel: {MODEL_NAME}")
                print(f"\nHidden states: {hidden_count}\nLast hidden state shape: {last_shape}")
                print(f"Pooled embedding dimension: {pooled['last_hidden_state'].size}")
                print(
                    f"\nDatabase: PostgreSQL\nTrack ID: {track_id}"
                    f"\nEmbeddings stored: {len(pooled)}"
                )
                if args.command == "slices":
                    print(f"Slice {index}/{len(windows)} ({role}) committed.\n", flush=True)
                elif args.save_npz:
                    try:
                        save_npz(output_path, pooled)
                    except OSError as error:
                        print(
                            f"Error: Track {track_id} was committed, "
                            f"but NPZ export failed: {error}",
                            file=sys.stderr,
                        )
                        return 1
                    print(f"NPZ: {output_path}")
            del segment, audio
    except OperationalError as error:
        print(
            f"Database unavailable: {error.orig}\nStart PostgreSQL: docker compose up -d",
            file=sys.stderr,
        )
        return 1
    except SQLAlchemyError as error:
        # Preserve the database's explanation without dumping all vector bind parameters.
        print(f"Database error: {getattr(error, 'orig', error)}", file=sys.stderr)
        return 1
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(main())
