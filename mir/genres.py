"""Optional EffNet classification of already-stored audio windows."""

import importlib
import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
from numpy.typing import NDArray
from sqlalchemy.engine import Engine

from mir.database import GENRE_DIMENSION, GENRE_MODEL, classification_targets, save_genres

MODEL_DIRECTORY = Path(__file__).resolve().parent.parent / ".cache" / "effnet"
MODEL_URL = "https://essentia.upf.edu/models/music-style-classification/discogs-effnet"
SAMPLE_RATE = 16_000


def model_file(suffix: str) -> Path:
    """Cache official model assets; incomplete downloads never become cache hits."""
    path = MODEL_DIRECTORY / f"{GENRE_MODEL}.{suffix}"
    if path.is_file():
        return path
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {path.name}...", flush=True)
        with urlopen(f"{MODEL_URL}/{path.name}", timeout=60) as response:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
                temporary = Path(output.name)
                shutil.copyfileobj(response, output)
        temporary.replace(path)
    except (OSError, URLError) as error:
        raise ValueError(f"Cannot download EffNet asset {path.name}: {error}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def load_classifier() -> tuple[Callable[[NDArray[np.float32]], NDArray[np.float32]], list[str]]:
    try:
        es = importlib.import_module("essentia.standard")
    except ImportError as error:
        raise ValueError(
            f"EffNet requires: uv sync --extra effnet. Essentia import failed: {error}"
        ) from error
    if not hasattr(es, "TensorflowPredictEffnetDiscogs"):
        raise ValueError("Essentia lacks TensorFlow support; run uv sync --extra effnet.")
    try:
        labels = json.loads(model_file("json").read_text())["classes"]
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read EffNet label metadata: {error}") from error
    if (
        not isinstance(labels, list)
        or len(labels) != GENRE_DIMENSION
        or not all(isinstance(label, str) and label for label in labels)
        or len(set(labels)) != GENRE_DIMENSION
    ):
        raise ValueError(f"EffNet metadata must contain {GENRE_DIMENSION} unique labels.")
    # The graph has a fixed batch of 64; 'same' removes padded batch predictions.
    # Repeat incomplete final patches, including short slices, rather than dropping audio.
    model = es.TensorflowPredictEffnetDiscogs(
        graphFilename=str(model_file("pb")),
        output="PartitionedCall:0",
        batchSize=64,
        lastBatchMode="same",
        lastPatchMode="repeat",
    )
    return model, labels


def pool_predictions(predictions: NDArray[np.float32]) -> NDArray[np.float32]:
    values = np.asarray(predictions, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != GENRE_DIMENSION:
        raise ValueError(f"Unexpected EffNet prediction shape: {values.shape}.")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("EffNet predictions must be finite sigmoid scores between 0 and 1.")
    # This output already has sigmoid applied. No extra sigmoid or normalization.
    return values.mean(axis=0, dtype=np.float32)


def classify_tracks(engine: Engine, track_id: int | None) -> None:
    from main import load_audio

    rows = classification_targets(engine, track_id)
    if not rows:
        print("No slices missing EffNet results.")
        return
    model, labels = load_classifier()
    for index, row in enumerate(rows, 1):
        print(
            f"[{index}/{len(rows)}] Track {row['id']}: {row['file_name']} "
            f"[{row['analysis_offset']:.2f}–"
            f"{row['analysis_offset'] + row['analysis_duration']:.2f} s]",
            flush=True,
        )
        audio, _ = load_audio(
            Path(row["file_path"]),
            row["analysis_offset"],
            row["analysis_duration"],
            sample_rate=SAMPLE_RATE,
        )
        if abs(len(audio) / SAMPLE_RATE - row["analysis_duration"]) > 1 / SAMPLE_RATE:
            raise ValueError("Source file no longer contains the entire stored slice.")
        try:
            scores = pool_predictions(model(audio))
        except RuntimeError as error:
            raise ValueError(
                f"Track {row['id']}: EffNet inference failed for the stored "
                f"{row['analysis_duration']:g} second window: {error}"
            ) from error
        top_labels = [
            {"genre": labels[int(i)], "score": float(scores[i])}
            for i in np.argsort(-scores, kind="stable")[:5]
        ]
        save_genres(engine, row["id"], scores, top_labels)
        print(
            "Saved: " + ", ".join(f"{item['genre']} ({item['score']:.3f})" for item in top_labels),
            flush=True,
        )
