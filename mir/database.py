"""SQLAlchemy Core tables and exact pgvector cosine search."""

import os
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    Double,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
    func,
    insert,
    literal,
    select,
    text,
)
from sqlalchemy import (
    create_engine as sqlalchemy_create_engine,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine, RowMapping
from sqlalchemy.exc import DBAPIError, OperationalError

# Checkpoint config: OpenMuQ/MuQ-large-msd-iter, encoder_dim / hidden_size.
# https://huggingface.co/OpenMuQ/MuQ-large-msd-iter/blob/main/config.json
VECTOR_DIMENSION = 1024
GENRE_DIMENSION = 400
GENRE_MODEL = "discogs-effnet-bs64-1"

metadata = MetaData()
tracks = Table(
    "tracks",
    metadata,
    Column("id", BigInteger, primary_key=True),
    Column("file_path", Text, nullable=False),
    Column("file_name", Text, nullable=False),
    Column("sample_rate", Integer, nullable=False),
    Column("original_duration", Double, nullable=False),
    Column("analysis_offset", Double, nullable=False),
    Column("analysis_duration", Double, nullable=False),
    Column("model_name", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)
embeddings = Table(
    "embeddings",
    metadata,
    Column("id", BigInteger, primary_key=True),
    Column("track_id", BigInteger, ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False),
    Column("embedding_type", Text, nullable=False),
    Column("layer_index", Integer),
    Column("dimension", Integer, nullable=False),
    Column("embedding", Vector(VECTOR_DIMENSION), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    UniqueConstraint("track_id", "embedding_type", "layer_index", name="uq_hidden_representation"),
    CheckConstraint(
        "(embedding_type = 'hidden_state' AND layer_index IS NOT NULL AND layer_index >= 0)"
        " OR (embedding_type = 'last_hidden_state' AND layer_index IS NULL)",
        name="valid_representation",
    ),
    CheckConstraint(f"dimension = {VECTOR_DIMENSION}", name="valid_dimension"),
)
Index(
    "uq_final_representation",
    embeddings.c.track_id,
    embeddings.c.embedding_type,
    unique=True,
    postgresql_where=embeddings.c.layer_index.is_(None),
)

genre_predictions = Table(
    "genre_predictions",
    metadata,
    Column("track_id", BigInteger, ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True),
    Column("model_name", Text, primary_key=True),
    Column("scores", Vector(GENRE_DIMENSION), nullable=False),
    Column("top_labels", JSONB, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)


def classification_targets(engine: Engine, track_id: int | None) -> list[RowMapping]:
    statement = select(tracks).order_by(tracks.c.id)
    if track_id is not None:
        statement = statement.where(tracks.c.id == track_id)
    else:
        exists = (
            select(genre_predictions.c.track_id)
            .where(
                genre_predictions.c.track_id == tracks.c.id,
                genre_predictions.c.model_name == GENRE_MODEL,
            )
            .exists()
        )
        statement = statement.where(~exists)
    with engine.connect() as connection:
        rows = list(connection.execute(statement).mappings())
    if track_id is not None and not rows:
        raise ValueError(f"Track {track_id} does not exist.")
    return rows


def save_genres(
    engine: Engine,
    track_id: int,
    scores: NDArray[np.float32],
    top_labels: list[dict[str, str | float]],
) -> None:
    if scores.shape != (GENRE_DIMENSION,) or not np.isfinite(scores).all():
        raise ValueError(f"Expected {GENRE_DIMENSION} finite EffNet scores.")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("EffNet scores must be between 0 and 1.")
    statement = pg_insert(genre_predictions).values(
        track_id=track_id,
        model_name=GENRE_MODEL,
        scores=scores,
        top_labels=top_labels,
    )
    statement = statement.on_conflict_do_update(
        index_elements=[genre_predictions.c.track_id, genre_predictions.c.model_name],
        set_={
            "scores": statement.excluded.scores,
            "top_labels": statement.excluded.top_labels,
            "created_at": func.now(),
        },
    )
    with engine.begin() as connection:
        connection.execute(statement)


def create_engine() -> Engine:
    url = os.getenv("DATABASE_URL", "postgresql+psycopg://mir:mir@localhost:5432/mir")
    if not url.startswith("postgresql+psycopg://"):
        raise ValueError("DATABASE_URL must start with postgresql+psycopg:// (Psycopg 3).")
    return sqlalchemy_create_engine(url, connect_args={"connect_timeout": 5})


def initialize_database(engine: Engine) -> None:
    with engine.begin() as connection:
        try:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except OperationalError:
            raise
        except DBAPIError as error:
            raise ValueError(
                f"Cannot initialize pgvector: {error.orig}. "
                "Use the pgvector Compose image and a database user allowed to create extensions."
            ) from error
        metadata.create_all(connection)
        # create_all skips indexes when their table already exists.
        for index in embeddings.indexes:
            index.create(connection, checkfirst=True)


def validate_vector(vector: NDArray[np.float32]) -> None:
    if vector.shape != (VECTOR_DIMENSION,):
        raise ValueError(
            f"MuQ vector dimension mismatch: got shape {vector.shape}, "
            f"expected ({VECTOR_DIMENSION},). Check the checkpoint and schema."
        )
    if not np.isfinite(vector).all():
        raise ValueError(
            "MuQ produced non-finite embeddings (NaN or infinity); use fp32 inference."
        )


def insert_analysis(
    engine: Engine,
    track_values: Mapping[str, object],
    pooled: Mapping[str, NDArray[np.float32]],
) -> int:
    for vector in pooled.values():
        validate_vector(vector)
    with engine.begin() as connection:
        track_id = connection.execute(insert(tracks).values(**track_values).returning(tracks.c.id))
        identifier = int(track_id.scalar_one())
        connection.execute(
            insert(embeddings),
            [
                {
                    "track_id": identifier,
                    "embedding_type": "last_hidden_state"
                    if name == "last_hidden_state"
                    else "hidden_state",
                    "layer_index": None if name == "last_hidden_state" else int(name[7:]),
                    "dimension": vector.size,
                    "embedding": vector,
                }
                for name, vector in pooled.items()
            ],
        )
    return identifier


def get_track(engine: Engine, track_id: int) -> RowMapping:
    with engine.connect() as connection:
        row = connection.execute(select(tracks).where(tracks.c.id == track_id)).mappings().first()
    if row is None:
        raise ValueError(f"Track {track_id} does not exist.")
    return row


def find_similar_tracks(
    engine: Engine, track_id: int, layer: int, limit: int, *, lab_mode: bool = False
) -> list[RowMapping]:
    if limit <= 0:
        raise ValueError("Search limit must be positive.")
    if layer < 0:
        raise ValueError("Layer must be zero or greater.")
    with engine.begin() as connection:
        source_track = (
            connection.execute(select(tracks).where(tracks.c.id == track_id)).mappings().first()
        )
        if source_track is None:
            raise ValueError(f"Track {track_id} does not exist.")
        source = embeddings.alias("source")
        selected = select(source.c.embedding).where(
            source.c.track_id == track_id,
            source.c.embedding_type == "hidden_state",
            source.c.layer_index == layer,
        )
        if not connection.scalar(select(selected.exists())):
            raise ValueError(f"Hidden-state layer {layer} does not exist for track {track_id}.")
        if (
            lab_mode
            and connection.scalar(select(func.vector_norm(selected.scalar_subquery()))) == 0
        ):
            raise ValueError(
                f"Track {track_id} layer {layer} has a zero-norm vector; "
                "cosine distance is undefined."
            )
        # The source vector and all distance calculations stay in PostgreSQL.
        distance = embeddings.c.embedding.cosine_distance(selected.scalar_subquery()).label(
            "distance"
        )
        statement = (
            select(
                tracks.c.id.label("track_id"),
                tracks.c.file_path,
                tracks.c.analysis_offset,
                tracks.c.analysis_duration,
                tracks.c.model_name,
                literal(source_track["model_name"]).label("source_model_name"),
                genre_predictions.c.top_labels,
                distance,
            )
            .join(embeddings, embeddings.c.track_id == tracks.c.id)
            .outerjoin(
                genre_predictions,
                (genre_predictions.c.track_id == tracks.c.id)
                & (genre_predictions.c.model_name == GENRE_MODEL),
            )
            .where(
                embeddings.c.embedding_type == "hidden_state",
                embeddings.c.layer_index == layer,
                tracks.c.id != track_id,
            )
            .order_by(distance, tracks.c.id)
            .limit(limit)
        )
        if lab_mode:
            statement = statement.where(
                tracks.c.file_path != source_track["file_path"],
                tracks.c.model_name == source_track["model_name"],
                func.vector_norm(embeddings.c.embedding) > 0,
            )
        return list(connection.execute(statement).mappings())
