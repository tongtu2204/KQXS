import unittest

import pandas as pd
import numpy as np

from src.config import FINAL_TEST, TRAIN_DEVELOPMENT, VALIDATION, phase_for_date
from src.data import load_data, validate_data
from src.probability import validate_prediction_frame


class ProtocolTest(unittest.TestCase):
    def test_period_boundaries_are_fixed(self):
        self.assertEqual(TRAIN_DEVELOPMENT.end, pd.Timestamp("2022-12-31"))
        self.assertEqual(VALIDATION.start, pd.Timestamp("2023-01-01"))
        self.assertEqual(VALIDATION.end, pd.Timestamp("2024-12-31"))
        self.assertEqual(FINAL_TEST.start, pd.Timestamp("2025-01-01"))
        self.assertEqual(FINAL_TEST.end, pd.Timestamp("2026-12-31"))

    def test_phase_assignment(self):
        self.assertEqual(phase_for_date("2022-12-31"), "train_development")
        self.assertEqual(phase_for_date("2023-01-01"), "validation")
        self.assertEqual(phase_for_date("2025-01-01"), "final_test")

    def test_tracked_raw_data_covers_all_periods(self):
        summary = validate_data(load_data(rebuild=True))
        self.assertGreater(summary["period_counts"]["train_development"], 0)
        self.assertGreater(summary["period_counts"]["validation"], 0)
        self.assertGreater(summary["period_counts"]["final_test"], 0)

    def test_probability_validation_rejects_lookahead(self):
        probabilities = np.full((1, 100), 0.01)
        frame = pd.DataFrame(probabilities, columns=[f"p_{i:02d}" for i in range(100)])
        frame.insert(0, "history_end", pd.Timestamp("2025-01-01"))
        frame.insert(0, "actual", 0)
        frame.insert(0, "fold", "final_test")
        frame.insert(0, "date", pd.Timestamp("2025-01-01"))

        with self.assertRaisesRegex(ValueError, "history_end"):
            validate_prediction_frame(frame, source="test")


if __name__ == "__main__":
    unittest.main()
