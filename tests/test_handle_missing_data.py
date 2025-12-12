"""Unit tests for missing data handling backend component.

Tests all data preprocessing strategies for handling NaN values:
deletion, mean/median filling, and constant filling. Verifies error
handling for invalid strategies and edge cases.
"""
import unittest
import pandas as pd
import numpy as np

from src.backend.handle_missing_data import (
    handle_missing_data,
    MissingDataError,
)

class TestHandleMissingData(unittest.TestCase):
    """Test suite for handle_missing_data function.

    Tests all missing data strategies, error conditions, and edge cases
    like missing values that cannot be handled numerically.

    Methods
    -------
    setUp()
        Create DataFrame with NaN values for testing.
    test_raise_error_if_no_missing_values()
        Test error raised when DataFrame has no missing values.
    test_strategy_delete_rows()
        Test deletion strategy removes rows with NaN.
    test_strategy_fill_mean()
        Test mean filling strategy.
    test_strategy_fill_median()
        Test median filling strategy.
    test_strategy_fill_constant()
        Test constant value filling strategy.
    test_strategy_fill_constant_empty_error()
        Test error on empty constant with constant strategy.
    """
    def setUp(self) -> None:
        """Create DataFrame with NaN values for testing.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # We create a dataframe with holes
        # (NaNs) to test the strategies.
        # It gets recreated before EACH test
        # function to ensure a clean state.
        self.df = pd.DataFrame({
            "A": [1, 2, np.nan, 4, 5],     # Numeric with 1 missing
            "B": [10, np.nan, 30, 40, 50], # Numeric with 1 missing
            "C": ["a", "b", "c", "d", "e"] # Text column
                            #(should be ignored by numeric strategies)
        })
        self.cols = ["A", "B"]  # We usually operate on specific columns

    def test_raise_error_if_no_missing_values(self) -> None:
        """Test error raised when DataFrame has no missing values.

        Verifies handle_missing_data raises MissingDataError when
        called on clean data without NaN values.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Code explicitly checks
        # "if total_missing == 0". Let's verify that.
        clean_df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        with self.assertRaises(MissingDataError):
            # Should scream "No missing values found"
            handle_missing_data(clean_df, ["x", "y"],
                                "Delete rows with NaN")

    def test_strategy_delete_rows(self) -> None:
        """Test delete rows strategy removes NaN rows correctly.

        Verifies deletion strategy removes all rows containing NaN
        in specified columns.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # We have NaNs in index 2 (col A) and index 1 (col B).
        # If we delete rows with
        # NaN in A or B, we should lose 2 rows total.
        df_res, summary, msg = handle_missing_data(
            self.df.copy(), self.cols, "Delete rows with NaN"
        )

        # Original was 5 rows. We expect 3 rows remaining.
        self.assertEqual(len(df_res), 3)
        self.assertIn("Rows removed: 2", msg)

    def test_strategy_fill_mean(self) -> None:
        """Test fill with mean strategy replaces NaN correctly.

        Verifies mean filling computes correct mean and replaces NaN
        values with computed mean.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Column A: [1, 2, NaN, 4, 5]. Mean of (1+2+4+5)/4 = 3.0
        df_res, _, msg = handle_missing_data(
            self.df.copy(), self.cols, "Fill with mean"
        )

        # Check if the NaN at index 2 in col A became 3.0
        self.assertEqual(df_res["A"][2], 3.0)
        self.assertIn("filled with column mean", msg)

    def test_strategy_fill_median(self) -> None:
        """Test fill with median strategy replaces NaN correctly.

        Verifies median filling computes correct median and replaces
        NaN values with computed median.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Column B: [10, NaN, 30, 40, 50].
        # Median of (10, 30, 40, 50) is 35.0
        df_res, _, msg = handle_missing_data(
            self.df.copy(), self.cols, "Fill with median"
        )

        self.assertEqual(df_res["B"][1], 35.0)
        self.assertIn("filled with column median", msg)

    def test_strategy_fill_constant(self) -> None:
        """Test fill with constant strategy replaces NaN correctly.

        Verifies constant value filling replaces all NaN with user-
        specified constant value.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Filling with a specific value, e.g., "0"
        constant_val = "99"
        df_res, _, msg = handle_missing_data(
            self.df.copy(), self.cols, "Fill with constant",
            constant=constant_val
        )

        self.assertEqual(df_res["A"][2], 99.0)
        self.assertEqual(df_res["B"][1], 99.0)
        self.assertIn("filled with constant 99.0", msg)

    def test_strategy_fill_constant_empty_error(self) -> None:
        """Test error when constant strategy has empty constant value.

        Verifies MissingDataError raised when constant strategy is
        selected but no constant value provided.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # If user selects constant strategy but sends empty string
        with self.assertRaises(MissingDataError):
            handle_missing_data(
                self.df.copy(), self.cols,
                "Fill with constant", constant="   "
            )

    def test_unknown_strategy(self) -> None:
        """Raise MissingDataError for an unknown preprocessing strategy.

        Verifies that passing an unsupported strategy name causes the
        function to raise a MissingDataError rather than proceeding.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Just to cover the 'else' branch in our strategy function
        with self.assertRaises(MissingDataError):
            handle_missing_data(
                self.df.copy(),
                self.cols,
                "Destroy everything"
            )

    def test_summary_formatting(self) -> None:
        """Verify summary string reports total and per-column NaNs.

        Ensures the generated summary mentions the total number of
        NaN values and includes per-column counts for selected
        columns.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Check if the summary string is generated correctly
        _, summary, _ = handle_missing_data(
            self.df.copy(), self.cols, "Fill with mean"
        )
        # Should mention Total NaN values: 2
        self.assertIn("Total NaN values: 2", summary)
        self.assertIn("A: 1", summary)  # Detail part


if __name__ == "__main__":
    unittest.main()