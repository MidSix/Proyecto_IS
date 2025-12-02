import unittest
import pandas as pd
import numpy as np

from src.backend.handle_missing_data import (
    handle_missing_data,
    MissingDataError,
)

class TestHandleMissingData(unittest.TestCase):
    def setUp(self):
        # We create a dataframe with holes (NaNs) to test the strategies.
        # It gets recreated before EACH test function to ensure a clean state.
        self.df = pd.DataFrame({
            "A": [1, 2, np.nan, 4, 5],     # Numeric with 1 missing
            "B": [10, np.nan, 30, 40, 50], # Numeric with 1 missing
            "C": ["a", "b", "c", "d", "e"] # Text column (should be ignored by numeric strategies)
        })
        self.cols = ["A", "B"]  # We usually operate on specific columns

    def test_raise_error_if_no_missing_values(self):
        # Your code explicitly checks "if total_missing == 0". Let's verify that.
        clean_df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        with self.assertRaises(MissingDataError):
            # Should scream "No missing values found"
            handle_missing_data(clean_df, ["x", "y"],
                                "Delete rows with NaN")

    def test_strategy_delete_rows(self):
        # We have NaNs in index 2 (col A) and index 1 (col B).
        # If we delete rows with NaN in A or B, we should lose 2 rows total.
        df_res, summary, msg = handle_missing_data(
            self.df.copy(), self.cols, "Delete rows with NaN"
        )

        # Original was 5 rows. We expect 3 rows remaining.
        self.assertEqual(len(df_res), 3)
        self.assertIn("Rows removed: 2", msg)

    def test_strategy_fill_mean(self):
        # Column A: [1, 2, NaN, 4, 5]. Mean of (1+2+4+5)/4 = 3.0
        df_res, _, msg = handle_missing_data(
            self.df.copy(), self.cols, "Fill with mean"
        )

        # Check if the NaN at index 2 in col A became 3.0
        self.assertEqual(df_res["A"][2], 3.0)
        self.assertIn("filled with column mean", msg)

    def test_strategy_fill_median(self):
        # Column B: [10, NaN, 30, 40, 50]. Median of (10, 30, 40, 50) is 35.0
        df_res, _, msg = handle_missing_data(
            self.df.copy(), self.cols, "Fill with median"
        )

        self.assertEqual(df_res["B"][1], 35.0)
        self.assertIn("filled with column median", msg)

    def test_strategy_fill_constant(self):
        # Filling with a specific value, e.g., "0"
        # Note: Your code expects 'constant' as a string input from GUI, then floats it.
        constant_val = "99"
        df_res, _, msg = handle_missing_data(
            self.df.copy(), self.cols, "Fill with constant",
            constant=constant_val
        )

        self.assertEqual(df_res["A"][2], 99.0)
        self.assertEqual(df_res["B"][1], 99.0)
        self.assertIn("filled with constant 99.0", msg)

    def test_strategy_fill_constant_empty_error(self):
        # If user selects constant strategy but sends empty string
        with self.assertRaises(MissingDataError):
            handle_missing_data(
                self.df.copy(), self.cols,
                "Fill with constant", constant="   "
            )

    def test_unknown_strategy(self):
        # Just to cover the 'else' branch in your strategy function
        with self.assertRaises(MissingDataError):
            handle_missing_data(self.df.copy(), self.cols, "Destroy everything")

    def test_summary_formatting(self):
        # Check if the summary string is generated correctly
        _, summary, _ = handle_missing_data(
            self.df.copy(), self.cols, "Fill with mean"
        )
        # Should mention Total NaN values: 2
        self.assertIn("Total NaN values: 2", summary)
        self.assertIn("A: 1", summary)  # Detail part


if __name__ == "__main__":
    unittest.main()