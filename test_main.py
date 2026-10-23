"""Small runnable checks: uv run python -m unittest -v test_main."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4


class AnalyzerChecks(unittest.TestCase):
    def test_effnet_classification(self) -> None:
        from unittest.mock import MagicMock

        import numpy as np
        import soundfile as sf

        import main
        from mir import genres
        from mir.database import GENRE_DIMENSION

        predictions = np.zeros((2, GENRE_DIMENSION), dtype=np.float32)
        predictions[:, 7] = [0.4, 0.8]
        pooled = genres.pool_predictions(predictions)
        self.assertAlmostEqual(float(pooled[7]), 0.6)
        for invalid in (
            np.zeros((0, GENRE_DIMENSION), dtype=np.float32),
            np.zeros((2, 1280), dtype=np.float32),
            np.full((1, GENRE_DIMENSION), np.nan, dtype=np.float32),
            np.full((1, GENRE_DIMENSION), 2, dtype=np.float32),
        ):
            with self.assertRaises(ValueError):
                genres.pool_predictions(invalid)
        with contextlib.redirect_stderr(io.StringIO()):
            for argv in (
                ["classify"],
                ["classify", "--track-id", "0"],
                ["classify", "--all", "--track-id", "1"],
            ):
                with self.assertRaises(SystemExit):
                    main.parse_args(argv)
        self.assertTrue(main.parse_args(["classify", "--all"]).all)
        # Real FLAC decoding, mocked inference/storage: this is not a runtime or DB test.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "song.flac"
            sf.write(path, np.zeros((48000 * 3, 2), dtype=np.float32), 48000)
            row = {
                "id": 42,
                "file_name": path.name,
                "file_path": str(path),
                "analysis_offset": 1.0,
                "analysis_duration": 1.0,
            }
            model = MagicMock(return_value=predictions)
            labels = [f"Genre---{i}" for i in range(GENRE_DIMENSION)]
            with (
                patch.object(genres, "classification_targets", return_value=[row, row]),
                patch.object(genres, "load_classifier", return_value=(model, labels)) as load,
                patch.object(genres, "save_genres") as save,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                genres.classify_tracks(MagicMock(), None)
                load.assert_called_once()
                self.assertEqual(save.call_count, 2)
                audio = model.call_args.args[0]
                self.assertEqual((audio.shape, audio.dtype), ((16000,), np.float32))
                self.assertEqual(save.call_args.args[3][0]["genre"], "Genre---7")
            with (
                patch.object(genres, "classification_targets", return_value=[]),
                patch.object(genres, "load_classifier") as load,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                genres.classify_tracks(MagicMock(), None)
                load.assert_not_called()

    def test_muq_transformers_compatibility(self) -> None:
        import torch
        from muq import MuQ, MuQConfig
        from transformers import Wav2Vec2ConformerConfig

        # Real MuQ forward path, with a small random model and no weight download.
        config = Wav2Vec2ConformerConfig(
            hidden_size=16,
            num_attention_heads=2,
            intermediate_size=32,
            num_conv_pos_embedding_groups=2,
            num_conv_pos_embeddings=4,
        )
        model = MuQ(
            MuQConfig(
                encoder_dim=16,
                encoder_depth=1,
                conv_dim=4,
                codebook_size=8,
                stat={"melspec_2048_mean": 0.0, "melspec_2048_std": 1.0},
                w2v2_config=json.loads(config.to_json_string()),
            )
        ).eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 24000), output_hidden_states=True)
        self.assertTrue(output.hidden_states)
        self.assertEqual(output.last_hidden_state.shape[0], 1)
        self.assertEqual(output.last_hidden_state.shape[-1], 16)
        self.assertTrue(torch.isfinite(output.last_hidden_state).all())

    def test_cli_validation(self) -> None:
        import main

        args = main.parse_args(["analyze", "song.flac"])
        self.assertEqual((args.offset, args.duration, args.device), (0, 30, "auto"))
        for argv in (
            ["analyze", "song.flac", "--offset", "-1"],
            ["analyze", "song.flac", "--duration", "0"],
            ["analyze", "song.flac", "--duration", "nan"],
            ["analyze", "song.flac", "--offset", "inf"],
            ["analyze", "song.flac", "--output", "out.npz"],
            ["search", "--track-id", "1", "--layer", "0", "--limit", "0"],
            ["search", "--track-id", "1", "--layer", "-1"],
            ["slices", "song.flac", "--slice-duration", "0"],
            ["slices", "song.flac", "--slice-duration", "nan"],
        ):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main.parse_args(argv)
                self.assertEqual(error.exception.code, 2)

    def test_audio_pooling_and_npz(self) -> None:
        import numpy as np
        import soundfile as sf
        import torch

        import main
        from mir.database import VECTOR_DIMENSION

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "song.flac"
            # Stereo input at a different sample rate, with a known mono mean.
            stereo = np.tile(np.array([0.25, 0.75], dtype=np.float32), (48000, 1))
            sf.write(path, stereo, 48000)
            audio, original = main.load_audio(path, 0.75, 30)
            self.assertEqual(original, 1)
            self.assertEqual(audio.shape, (6000,))
            self.assertEqual(audio.dtype, np.float32)
            np.testing.assert_allclose(audio[100:-100], 0.5, atol=1e-4)
            with self.assertRaisesRegex(ValueError, "end"):
                main.load_audio(path, 1, 30)
            with self.assertRaisesRegex(ValueError, "file"):
                main.load_audio(Path(directory), 0, 30)
            path.write_text("not a FLAC")
            with self.assertRaisesRegex(ValueError, "decode"):
                main.load_audio(path, 0, 30)

            hidden = torch.stack(
                [torch.zeros(VECTOR_DIMENSION), torch.full((VECTOR_DIMENSION,), 2.0)]
            ).unsqueeze(0)
            pooled = main.extract_embeddings((hidden, hidden + 2), hidden + 4)
            self.assertEqual(set(pooled), {"hidden_00", "hidden_01", "last_hidden_state"})
            for name, value in zip(pooled, (1, 3, 5), strict=True):
                self.assertEqual(pooled[name].dtype, np.float32)
                np.testing.assert_array_equal(pooled[name], np.full(VECTOR_DIMENSION, value))
            output = Path(directory) / "custom.npz"
            main.save_npz(output, pooled)
            with np.load(output) as saved:
                for name in pooled:
                    np.testing.assert_array_equal(saved[name], pooled[name])
            with self.assertRaisesRegex(ValueError, "dimension"):
                main.extract_embeddings((hidden[:, :, :-1],), hidden)

    def test_slicing_boundaries_and_peak_selection(self) -> None:
        import numpy as np

        from mir.slicing import energy_regions, select_slices

        sr = 100
        audio = np.full(100 * sr, 0.1, dtype=np.float32)
        audio[70 * sr : 71 * sr] = 0.9
        slices = select_slices(audio, sr, [(0, 100)], 20)
        self.assertEqual(slices, [(0, 20, "start"), (60.5, 80.5, "peak")])
        self.assertEqual(select_slices(audio, sr, [(0, 5)], 20), [(0, 5, "start")])
        self.assertEqual(select_slices(audio, sr, [(0, 0.5)], 20), [])
        flat = np.zeros(70 * sr + 25, dtype=np.float32)
        self.assertEqual(energy_regions(flat, sr), [(0, 70.25)])
        self.assertEqual(select_slices(flat, sr, [(0, 70.25)], 20), [(0, 20, "start")])
        flat[32 * sr :] = 0.8
        regions = energy_regions(flat, sr)
        self.assertEqual(len(regions), 2)
        self.assertEqual(regions[0][0], 0)
        self.assertTrue(28 <= regions[0][1] <= 32)
        self.assertEqual(regions[0][1], regions[1][0])
        self.assertEqual(regions[-1][1], 70.25)
        for start, end, _ in select_slices(flat, sr, regions, 20):
            self.assertTrue(any(a <= start < end <= b for a, b in regions))
            self.assertLessEqual(end - start, 20)
        for duration in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                select_slices(audio, sr, [(0, 100)], duration)
        with self.assertRaisesRegex(ValueError, "beyond"):
            select_slices(audio, sr, [(0, 101)])

    def test_slices_cli_preview_and_model_reuse(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        import numpy as np
        import torch

        import main
        from mir.database import VECTOR_DIMENSION

        audio = np.zeros(60 * main.SAMPLE_RATE, dtype=np.float32)
        windows = [(0.0, 20.0, "start"), (40.0, 60.0, "peak")]
        output = SimpleNamespace(
            hidden_states=(torch.ones(1, 2, VECTOR_DIMENSION),),
            last_hidden_state=torch.ones(1, 2, VECTOR_DIMENSION),
        )
        # Controlled model/database stand-ins verify orchestration, not real inference/storage.
        model = MagicMock(return_value=output)
        with (
            patch("main.load_audio", return_value=(audio, 60)),
            patch("mir.slicing.select_slices", return_value=windows),
            patch("main.load_model", return_value=model) as load_model,
            patch("mir.database.create_engine") as create_engine,
            patch("mir.database.initialize_database"),
            patch("mir.database.insert_analysis", side_effect=[101, 102]) as insert_analysis,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main.main(["slices", "song.flac", "--preview"]), 0)
            load_model.assert_not_called()
            create_engine.assert_not_called()
            self.assertEqual(main.main(["slices", "song.flac", "--device", "cpu"]), 0)
            load_model.assert_called_once()
            self.assertEqual(model.call_count, 2)
            for call in model.call_args_list:
                self.assertTrue(call.kwargs["output_hidden_states"])
                self.assertEqual(call.args[0].shape, (1, 20 * main.SAMPLE_RATE))
                self.assertEqual(call.args[0].dtype, torch.float32)
            self.assertEqual(
                [call.args[1]["analysis_offset"] for call in insert_analysis.call_args_list],
                [0, 40],
            )
            self.assertTrue(
                all(
                    call.args[1]["analysis_duration"] == 20
                    for call in insert_analysis.call_args_list
                )
            )
            create_engine.return_value.dispose.assert_called_once()

            with tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                with self.assertRaisesRegex(ValueError, "No FLAC"):
                    main.slice_paths(folder)
                for name in ("b.FLAC", "a song.flac", "notes.txt"):
                    (folder / name).touch()
                (folder / "nested").mkdir()
                (folder / "nested" / "skip.flac").touch()
                files = main.slice_paths(folder)
                self.assertEqual([path.name for path in files], ["a song.flac", "b.FLAC"])
                load_model.reset_mock()
                create_engine.reset_mock()
                model.reset_mock()
                insert_analysis.reset_mock(side_effect=True)
                insert_analysis.return_value = 103
                self.assertEqual(main.main(["slices", directory, "--preview"]), 0)
                load_model.assert_not_called()
                create_engine.assert_not_called()
                self.assertEqual(main.main(["slices", directory, "--device", "cpu"]), 0)
                load_model.assert_called_once()
                create_engine.assert_called_once()
                self.assertEqual(model.call_count, 4)
                self.assertEqual(
                    [call.args[1]["file_name"] for call in insert_analysis.call_args_list],
                    ["a song.flac", "a song.flac", "b.FLAC", "b.FLAC"],
                )
                create_engine.return_value.dispose.assert_called_once()
                # Stop on the second file's decode error; keep the first file's inserts.
                insert_analysis.reset_mock()
                with (
                    patch(
                        "main.load_audio", side_effect=[(audio, 60), ValueError("bad FLAC")]
                    ) as loader,
                    contextlib.redirect_stderr(io.StringIO()) as errors,
                ):
                    self.assertEqual(main.main(["slices", directory, "--device", "cpu"]), 1)
                    self.assertEqual(loader.call_count, 2)
                    self.assertEqual(insert_analysis.call_count, 2)
                    self.assertIn("bad FLAC", errors.getvalue())

    def test_device_fallback(self) -> None:
        import main

        with patch("torch.backends.mps.is_available", return_value=False):
            self.assertEqual(str(main.select_device("auto")), "cpu")
            self.assertEqual(str(main.select_device("cpu")), "cpu")
            with self.assertRaisesRegex(ValueError, "MPS"):
                main.select_device("mps")
        with patch("torch.backends.mps.is_available", return_value=True):
            self.assertEqual(str(main.select_device("auto")), "mps")

    @unittest.skipUnless(os.getenv("RUN_DATABASE_TESTS") == "1", "set RUN_DATABASE_TESTS=1")
    def test_database_roundtrip(self) -> None:
        import numpy as np
        from sqlalchemy import delete, func, insert, select
        from sqlalchemy.exc import IntegrityError

        from mir.database import (
            VECTOR_DIMENSION,
            create_engine,
            embeddings,
            find_similar_tracks,
            initialize_database,
            insert_analysis,
            tracks,
        )

        engine = create_engine()
        marker = f"muq-check-{uuid4()}"
        values = {
            "file_path": marker,
            "file_name": "check.flac",
            "sample_rate": 24000,
            "original_duration": 1.0,
            "analysis_offset": 0.0,
            "analysis_duration": 1.0,
            "model_name": marker,
        }
        base = np.zeros(VECTOR_DIMENSION, dtype=np.float32)
        base[0] = 1
        near = base.copy()
        near[1] = 0.1
        far = np.zeros_like(base)
        far[1] = 1
        try:
            initialize_database(engine)
            initialize_database(engine)
            source, neighbor, distant = (
                insert_analysis(engine, values, {"hidden_00": vector, "last_hidden_state": vector})
                for vector in (base, near, far)
            )
            from mir.database import (
                GENRE_DIMENSION,
                classification_targets,
                genre_predictions,
                save_genres,
            )

            scores = np.full(GENRE_DIMENSION, 0.25, dtype=np.float32)
            labels = [{"genre": "Electronic---Techno", "score": 0.25}]
            save_genres(engine, neighbor, scores, labels)
            save_genres(engine, neighbor, scores, labels)
            self.assertNotIn(neighbor, [row["id"] for row in classification_targets(engine, None)])
            self.assertEqual(classification_targets(engine, neighbor)[0]["id"], neighbor)
            with engine.connect() as connection:
                saved = (
                    connection.execute(
                        select(genre_predictions).where(genre_predictions.c.track_id == neighbor)
                    )
                    .mappings()
                    .one()
                )
                np.testing.assert_array_equal(saved["scores"], scores)
            with engine.begin() as connection:
                self.assertEqual(
                    connection.scalar(select(func.count()).where(embeddings.c.track_id == source)),
                    2,
                )
                stored = connection.scalar(
                    select(embeddings.c.embedding).where(
                        embeddings.c.track_id == source, embeddings.c.layer_index == 0
                    )
                )
                np.testing.assert_array_equal(stored, base)
            # Both ordinary uniqueness and PostgreSQL NULL uniqueness must hold.
            for kind, layer in (("hidden_state", 0), ("last_hidden_state", None)):
                with self.assertRaises(IntegrityError), engine.begin() as connection:
                    connection.execute(
                        insert(embeddings).values(
                            track_id=source,
                            embedding_type=kind,
                            layer_index=layer,
                            dimension=VECTOR_DIMENSION,
                            embedding=base,
                        )
                    )
            # Invalid embedding metadata fails after the track INSERT, proving rollback.
            with self.assertRaises(IntegrityError):
                insert_analysis(engine, values, {"hidden_-1": base})
            with engine.begin() as connection:
                self.assertEqual(
                    connection.scalar(select(func.count()).where(tracks.c.model_name == marker)), 3
                )
            # Include existing rows without assuming an empty hobby database.
            with engine.begin() as connection:
                count = int(connection.scalar(select(func.count()).select_from(tracks)) or 0)
            results = find_similar_tracks(engine, source, 0, count)
            self.assertNotIn(source, [row["track_id"] for row in results])
            own = [row for row in results if row["track_id"] in (neighbor, distant)]
            self.assertEqual([row["track_id"] for row in own], [neighbor, distant])
            self.assertEqual(own[0]["top_labels"], labels)
            self.assertIsNone(own[1]["top_labels"])
            self.assertAlmostEqual(own[0]["distance"], 1 - 1 / np.sqrt(1.01), places=5)
            self.assertAlmostEqual(own[1]["distance"], 1, places=5)
            with self.assertRaisesRegex(ValueError, "layer"):
                find_similar_tracks(engine, source, 1, 5)
            with self.assertRaisesRegex(ValueError, "does not exist"):
                find_similar_tracks(engine, -1, 0, 5)
            with engine.begin() as connection:
                connection.execute(delete(tracks).where(tracks.c.id == source))
                self.assertEqual(
                    connection.scalar(select(func.count()).where(embeddings.c.track_id == source)),
                    0,
                )
        finally:
            with engine.begin() as connection:
                connection.execute(delete(tracks).where(tracks.c.model_name == marker))
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
