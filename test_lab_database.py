"""Real pgvector checks: RUN_DATABASE_TESTS=1 uv run python -m unittest test_lab_database."""

import os
import unittest
from uuid import uuid4

import numpy as np
from sqlalchemy import delete

from mir.database import (
    VECTOR_DIMENSION,
    create_engine,
    find_similar_tracks,
    get_track,
    initialize_database,
    insert_analysis,
    tracks,
)


class LabDatabaseChecks(unittest.TestCase):
    @unittest.skipUnless(os.getenv("RUN_DATABASE_TESTS") == "1", "set RUN_DATABASE_TESTS=1")
    def test_lab_search_filters_before_limit(self) -> None:
        engine = create_engine()
        marker = f"lab-check-{uuid4()}"
        identifiers: list[int] = []
        base = np.zeros(VECTOR_DIMENSION, dtype=np.float32)
        base[0] = 1

        def add(path: str, vector: np.ndarray, model: str = marker) -> int:
            identifier = insert_analysis(
                engine,
                {
                    "file_path": f"/{marker}/{path}.flac",
                    "file_name": f"{path}.flac",
                    "sample_rate": 24000,
                    "original_duration": 60.0,
                    "analysis_offset": 0.0,
                    "analysis_duration": 30.0,
                    "model_name": model,
                },
                {"hidden_00": vector},
            )
            identifiers.append(identifier)
            return identifier

        try:
            initialize_database(engine)
            source = add("source", base)
            same_file = add("source", base)
            wrong_model = add("other-checkpoint", base, marker + "-other")
            zero = add("zero", np.zeros_like(base))
            near = base.copy()
            near[1] = 0.1
            neighbor = add("neighbor", near)
            farther = base.copy()
            farther[1] = 0.2
            second = add("second", farther)

            results = find_similar_tracks(engine, source, 0, 2, lab_mode=True)
            self.assertEqual([row["track_id"] for row in results], [neighbor, second])
            self.assertAlmostEqual(results[0]["distance"], 1 - 1 / np.sqrt(1.01), places=5)
            self.assertTrue(all(row["model_name"] == marker for row in results))
            self.assertTrue(all(row["source_model_name"] == marker for row in results))
            self.assertEqual(get_track(engine, source)["model_name"], marker)
            with self.assertRaisesRegex(ValueError, "zero-norm"):
                find_similar_tracks(engine, zero, 0, 2, lab_mode=True)
            with self.assertRaisesRegex(ValueError, "layer"):
                find_similar_tracks(engine, source, 1, 2, lab_mode=True)
            with self.assertRaisesRegex(ValueError, "does not exist"):
                get_track(engine, -1)

            # Ordinary search retains its existing same-file and cross-model behavior.
            ordinary = find_similar_tracks(engine, source, 0, 1_000_000)
            returned = {row["track_id"] for row in ordinary}
            self.assertTrue({same_file, wrong_model}.issubset(returned))
            self.assertNotIn(source, returned)
        finally:
            try:
                if identifiers:
                    with engine.begin() as connection:
                        connection.execute(delete(tracks).where(tracks.c.id.in_(identifiers)))
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
