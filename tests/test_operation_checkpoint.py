import tempfile
import unittest
from pathlib import Path

from src.experiments.operation_checkpoint import (
    CheckpointSignatureError,
    OperationCheckpoint,
)


class OperationCheckpointTests(unittest.TestCase):
    def test_retry_accepts_only_valid_normalized_result(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = OperationCheckpoint(Path(directory) / "checkpoint.json", "sig")
            calls = {"count": 0}

            def operation():
                calls["count"] += 1
                return {"ok": calls["count"] >= 2}

            def validate(value):
                if not value["ok"]:
                    raise ValueError("not ready")
                return {"ok": True, "normalized": True}

            result = checkpoint.execute("operation", operation, validate, 3)
            self.assertEqual(result, {"ok": True, "normalized": True})
            self.assertEqual(calls["count"], 2)

    def test_completed_operation_and_result_are_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            checkpoint = OperationCheckpoint(path, "sig")
            calls = {"count": 0}

            def operation():
                calls["count"] += 1
                return {"ok": True}

            checkpoint.execute("operation", operation, lambda value: value, 3)
            checkpoint.store_result("sample", {"value": 1})
            checkpoint.store_result("sample", {"value": 2})
            resumed = OperationCheckpoint(path, "sig")
            self.assertEqual(
                resumed.execute("operation", operation, lambda value: value, 3),
                {"ok": True},
            )
            self.assertEqual(calls["count"], 1)
            self.assertEqual(resumed.result_values(), [{"value": 2}])

    def test_signature_mismatch_fails_instead_of_reusing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            OperationCheckpoint(path, "old").save()
            with self.assertRaises(CheckpointSignatureError):
                OperationCheckpoint(path, "new")
