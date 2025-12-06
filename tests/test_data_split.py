"""Unit tests for the DataSplitter backend component.

Tests train/test splitting functionality with various configurations,
reproducibility with seeds, error handling, and integration workflows.
"""
import unittest
import pandas as pd

from src.backend.data_split import (
    DataSplitter,
    DataSplitError,
    MIN_ROWS
)


class TestDataSplitter(unittest.TestCase):
    """Test suite for DataSplitter class.

    Tests train/test splitting with different sizes, reproducibility
    with random seeds, error conditions, and summary generation.

    Methods
    -------
    setUp()
        Create deterministic DataFrame for testing.
    test_split_basic_success()
        Test default split parameters (80/20).
    test_split_custom_size()
        Test custom test_size parameter.
    test_split_reproducibility()
        Test same seed produces identical splits.
    test_error_not_enough_rows()
        Test error on DataFrame below MIN_ROWS.
    test_error_invalid_test_size()
        Test error on invalid test_size values.
    test_error_no_dataframe()
        Test error when DataFrame is None.
    test_get_summary_without_split()
        Test error on summary without prior split.
    test_main_flow_integration()
        Test typical workflow: split then get summary.
    """
    def setUp(self) -> None:
        """Create deterministic test DataFrame.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Create a dummy dataframe with
        # enough rows (> MIN_ROWS which is 5)
        # I'm making it deterministic so tests don't flake out on me
        num_rows = MIN_ROWS + 5
        data = {
            "feature1": range(num_rows),  # 0 to 9
            "feature2": [x * 2 for x in range(10)],
            "target": [0, 1] * 5
        }
        self.df = pd.DataFrame(data)
        self.splitter = DataSplitter(self.df)

    def test_split_basic_success(self) -> None:
        """Test default split parameters produce correct split.

        Verifies 80/20 default split on 10-row DataFrame yields 8
        training and 2 test rows with correct metadata.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Testing the happy path: Splitting with default parameters
        # We have 10 rows, default test_size
        # is 0.2 -> 2 rows test, 8 rows train
        train, test_df = self.splitter.split(self.df)
        self.assertTrue(self.splitter.has_split())  # Internal state
                                                        #should update
        self.assertEqual(len(train), 8)
        self.assertEqual(len(test_df), 2)
        self.assertEqual(len(train) + len(test_df), 10)  # No data lost
                                                            #in the void

        # Also check if metadata was saved correctly
        meta = self.splitter.get_meta()
        self.assertEqual(meta["n_rows_total"], 10)
        self.assertEqual(meta["test_size"], 0.2)

    def test_split_custom_size(self) -> None:
        """Test custom test_size parameter is respected.

        Verifies 50/50 split (test_size=0.5) produces equal-sized
        train and test sets.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Does it respect the test_size parameter?
        # Let's try 0.5 (50/50 split)
        train, test_df = self.splitter.split(self.df, test_size=0.5)
        self.assertEqual(len(train), 5)
        self.assertEqual(len(test_df), 5)

    def test_split_reproducibility(self) -> None:
        """Test same random seed produces identical splits.

        Critical for ML reproducibility. Verifies two independent splits
        with seed=42 produce identical train and test DataFrames.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # This is CRITICAL for machine learning.
        # Same seed MUST produce same split.
        # First split
        train1, test1 = self.splitter.split(self.df, random_seed=42)

        # Second split (create new instance to be sure)
        splitter2 = DataSplitter(self.df)
        train2, test2 = splitter2.split(self.df, random_seed=42)

        # Pandas testing utility is great here.
        # It checks values, types, and indices.
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(test1, test2)

    def test_error_not_enough_rows(self) -> None:
        """Test DataSplitError on DataFrame below MIN_ROWS.

        Verifies split() raises error when DataFrame has fewer rows
        than the minimum required.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Our global MIN_ROWS is 5.
        # Let's try to break it with a small dataframe.
        # Only 3 rows:
        tiny_df = pd.DataFrame(
                {
                "a": list(range(MIN_ROWS - 1)),
                "b": list(range(MIN_ROWS - 1))
                }
            )

        splitter = DataSplitter(tiny_df)

        # Expecting DataSplitError
        with self.assertRaises(DataSplitError):
            splitter.split(tiny_df)

    def test_error_invalid_test_size(self) -> None:
        """Test DataSplitError on invalid test_size values.

        Verifies error raised for negative and >1 test_size values.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # test_size must be between 0 and 1.
        # Let's try a negative number and a number > 1
        with self.assertRaises(DataSplitError):
            self.splitter.split(self.df, test_size=-0.1)

        with self.assertRaises(DataSplitError):
            self.splitter.split(self.df, test_size=1.5)

    def test_error_no_dataframe(self) -> None:
        """Test DataSplitError when DataFrame is None.

        Verifies split() raises error on None input.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Verify it complains if we provide nothing
        empty_splitter = DataSplitter(None)
        with self.assertRaises(DataSplitError):
            empty_splitter.split(None)

    def test_get_summary_without_split(self) -> None:
        """Test error when requesting summary before split.

        Verifies get_split_summary() raises error if called before
        split() is executed.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # If we ask for a summary before splitting,
        # it should raise an error
        new_splitter = DataSplitter(self.df)  # Fresh instance
        with self.assertRaises(DataSplitError):
            new_splitter.get_split_summary()

    def test_main_flow_integration(self) -> None:
        """Test typical workflow: split and retrieve summary.

        Simulates real usage: perform split with custom parameters,
        verify summary contains expected values.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Simulating a typical workflow: Load -> Split -> Get Summary
        self.splitter.split(self.df, test_size=0.3, random_seed=123)
        summary = self.splitter.get_split_summary()

        self.assertIsInstance(summary, str)
        self.assertIn("Total df: 10", summary)
        self.assertIn("Seed used: 123", summary)


if __name__ == "__main__":
    unittest.main()