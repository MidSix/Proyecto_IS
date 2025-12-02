import unittest
import pandas as pd

from src.backend.data_split import DataSplitter, DataSplitError, MIN_ROWS


class TestDataSplitter(unittest.TestCase):
    def setUp(self):
        # Create a dummy dataframe with enough rows (> MIN_ROWS which is 5)
        # I'm making it deterministic so tests don't flake out on me
        data = {
            "feature1": range(10),  # 0 to 9
            "feature2": [x * 2 for x in range(10)],
            "target": [0, 1] * 5
        }
        self.df = pd.DataFrame(data)
        self.splitter = DataSplitter(self.df)

    def test_split_basic_success(self):
        # Testing the happy path: Splitting with default parameters
        # We have 10 rows, default test_size is 0.2 -> 2 rows test, 8 rows train
        train, test_df = self.splitter.split(self.df)

        self.assertTrue(self.splitter.has_split())  # Internal state should update
        self.assertEqual(len(train), 8)
        self.assertEqual(len(test_df), 2)
        self.assertEqual(len(train) + len(test_df), 10)  # No data lost in the void

        # Also check if metadata was saved correctly
        meta = self.splitter.get_meta()
        self.assertEqual(meta["n_rows_total"], 10)
        self.assertEqual(meta["test_size"], 0.2)

    def test_split_custom_size(self):
        # Does it respect the test_size parameter?
        # Let's try 0.5 (50/50 split)
        train, test_df = self.splitter.split(self.df, test_size=0.5)
        self.assertEqual(len(train), 5)
        self.assertEqual(len(test_df), 5)

    def test_split_reproducibility(self):
        # This is CRITICAL for machine learning. Same seed MUST produce same split.
        # First split
        train1, test1 = self.splitter.split(self.df, random_seed=42)

        # Second split (create new instance to be sure)
        splitter2 = DataSplitter(self.df)
        train2, test2 = splitter2.split(self.df, random_seed=42)

        # Pandas testing utility is great here.
        # It checks values, types, and indices.
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(test1, test2)

    def test_error_not_enough_rows(self):
        # Our global MIN_ROWS is 5. Let's try to break it with a small dataframe.
        tiny_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})  # Only 3 rows
        splitter = DataSplitter(tiny_df)

        # Expecting DataSplitError
        with self.assertRaises(DataSplitError):
            splitter.split(tiny_df)

    def test_error_invalid_test_size(self):
        # test_size must be between 0 and 1.
        # Let's try a negative number and a number > 1
        with self.assertRaises(DataSplitError):
            self.splitter.split(self.df, test_size=-0.1)

        with self.assertRaises(DataSplitError):
            self.splitter.split(self.df, test_size=1.5)

    def test_error_no_dataframe(self):
        # Verify it complains if we provide nothing
        empty_splitter = DataSplitter(None)
        with self.assertRaises(DataSplitError):
            empty_splitter.split(None)

    def test_get_summary_without_split(self):
        # If we ask for a summary before splitting,
        # it should raise an error
        new_splitter = DataSplitter(self.df)  # Fresh instance
        with self.assertRaises(DataSplitError):
            new_splitter.get_split_summary()

    def test_main_flow_integration(self):
        # Simulating a typical workflow: Load -> Split -> Get Summary
        self.splitter.split(self.df, test_size=0.3, random_seed=123)
        summary = self.splitter.get_split_summary()

        self.assertIsInstance(summary, str)
        self.assertIn("Total df: 10", summary)
        self.assertIn("Seed used: 123", summary)


if __name__ == "__main__":
    unittest.main()