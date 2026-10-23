"""Listening batch checks; database behavior is covered separately."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf


class ListeningLabChecks(unittest.TestCase):
    def test_export_audio_identity_and_atomic_failure(self) -> None:
        import lab

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "song.flac"
            audio = np.concatenate([np.full(48000, x) for x in (0.1, 0.2, 0.3)])
            sf.write(path, audio, 48000, subtype="PCM_24")
            source = {
                "id": 1,
                "file_path": str(path),
                "model_name": "test-model",
                "analysis_offset": 1.0,
                "analysis_duration": 1.0,
            }
            candidate = dict(source, track_id=2, analysis_offset=2.0, distance=0.1)
            with (
                patch.object(lab, "QA_ROOT", root / "qa"),
                patch.object(lab, "get_track", side_effect=lambda _, key: dict(source, id=key)),
                patch.object(lab, "find_similar_tracks", return_value=[candidate]),
            ):
                output = root / "qa" / "space batch"
                lab.export_batch(MagicMock(), [1, 1], 12, 5, 0, output)
                manifest = json.loads((output / "manifest.json").read_text())
                tasks = json.loads((output / "tasks.json").read_text())
                self.assertEqual(manifest["source_ids"], [1])
                self.assertEqual(len(tasks), 1)
                self.assertEqual(len(list((output / "clips").glob("*.wav"))), 2)
                task = manifest["tasks"][0]
                self.assertEqual(task["rank"], 1)
                self.assertEqual(tasks[0]["data"]["task_key"], task["task_key"])
                self.assertIn("space%20batch", tasks[0]["data"]["reference_audio"])
                for role, expected in (("source", 0.2), ("candidate", 0.3)):
                    clip, rate = sf.read(output / task[role]["clip"])
                    self.assertEqual((len(clip), rate), (24000, 24000))
                    self.assertAlmostEqual(float(clip[12000]), expected, places=5)
                    self.assertEqual(len(task[role]["sha256"]), 64)
                # Two source tasks reuse a candidate clip; the same seed retains pair ordering.
                outputs = []
                for name in ("repeat1", "repeat2"):
                    target = root / "qa" / name
                    lab.export_batch(MagicMock(), [1, 3], 12, 5, 7, target)
                    saved = json.loads((target / "manifest.json").read_text())
                    self.assertEqual(len(list((target / "clips").glob("*.wav"))), 3)
                    outputs.append([(t["source"]["track_id"], t["rank"]) for t in saved["tasks"]])
                self.assertEqual(outputs[0], outputs[1])
                with self.assertRaisesRegex(ValueError, "exists"):
                    lab.export_batch(MagicMock(), [1], 12, 5, 0, output)
                source["file_path"] = str(root / "missing.flac")
                with self.assertRaisesRegex(ValueError, "does not exist"):
                    lab.export_batch(MagicMock(), [1], 12, 5, 0, root / "qa" / "broken")
                self.assertEqual(
                    sorted(p.name for p in (root / "qa").iterdir()),
                    ["repeat1", "repeat2", "space batch"],
                )

    def test_report_review_states_and_denominators(self) -> None:
        import lab

        # Synthetic full-export shape from official docs, not a live UI export.
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "batch_id": "batch",
            "limit": 5,
            "source_ids": [1],
            "candidate_counts": {"1": 5},
            "layer": 12,
            "tasks": [
                {
                    "task_key": str(i),
                    "source": {"track_id": 1},
                    "candidate": {"track_id": i + 10, "file_path": "song.flac"},
                    "rank": i,
                }
                for i in range(1, 6)
            ],
        }

        def annotation(label: str, time: str = "2026-09-08T10:00:00Z") -> dict:
            return {
                "completed_by": 7,
                "was_cancelled": False,
                "updated_at": time,
                "result": [
                    {
                        "from_name": "similarity",
                        "to_name": "candidate",
                        "type": "choices",
                        "value": {"choices": [label]},
                    }
                ],
            }

        exports: list[dict[str, Any]] = [
            {
                "data": {"batch_id": "batch", "task_key": "1"},
                "annotations": [
                    annotation("Unrelated"),
                    annotation("Strongly similar", "2026-09-08T11:00:00Z"),
                ],
            },
            {"data": {"batch_id": "batch", "task_key": "2"}, "annotations": [annotation("Unsure")]},
            {
                "data": {"batch_id": "batch", "task_key": "3"},
                "annotations": [{"completed_by": 7, "was_cancelled": True, "result": []}],
            },
            {
                "data": {"batch_id": "batch", "task_key": "4"},
                "annotations": [],
                "drafts": [{"result": []}],
            },
        ]
        report = lab.summarize(manifest, exports)
        self.assertIn("Completed: 2/5", report)
        self.assertIn("Unsure: 1", report)
        self.assertIn("Skipped/cancelled: 1", report)
        self.assertIn("Unreviewed: 2", report)
        self.assertIn("Judged relevance: 1/1", report)
        self.assertIn("precision@5: unavailable", report)
        self.assertEqual(report, lab.summarize(manifest, exports))
        for invalid in (dict(manifest, source_ids=[]), dict(manifest, candidate_counts={"1": 3})):
            with self.assertRaises(ValueError):
                lab.summarize(invalid, exports)
        duplicate_rank = copy.deepcopy(manifest)
        duplicate_rank["tasks"][1]["rank"] = 1
        with self.assertRaises(ValueError):
            lab.summarize(duplicate_rank, exports)
        for bad in ("Unknown", "", "strongly similar"):
            changed = copy.deepcopy(exports)
            changed[0]["annotations"] = [annotation(bad)]
            with self.assertRaises(ValueError):
                lab.summarize(manifest, changed)
        changed = copy.deepcopy(exports)
        changed[0]["annotations"][1]["completed_by"] = 8
        with self.assertRaisesRegex(ValueError, "reviewer"):
            lab.summarize(manifest, changed)
        changed = copy.deepcopy(exports)
        changed[0]["data"]["task_key"] = "foreign"
        with self.assertRaisesRegex(ValueError, "task"):
            lab.summarize(manifest, changed)
        complete = [
            {
                "data": {"batch_id": "batch", "task_key": str(i)},
                "annotations": [annotation("Somewhat similar")],
            }
            for i in range(1, 6)
        ]
        self.assertIn("precision@5: 5/5", lab.summarize(manifest, complete))
        self.assertIn("no decisive judgments", lab.summarize(manifest, []))
        malformed_exports: list[Any] = [{}, None, [None], exports + [exports[0]]]
        for malformed in malformed_exports:
            with self.assertRaises(ValueError):
                lab.summarize(manifest, malformed)


if __name__ == "__main__":
    unittest.main()
