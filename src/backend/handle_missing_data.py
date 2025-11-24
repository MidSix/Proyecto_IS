import pandas as pd
import unittest # For automatic testing purposes
import numpy as np # Needed to create NaN values for testing
# Don't need to use a class. We don't need to store states.
# I mean, you could store states but they are meaningless here.
# A class is only useful when you need to store meaningful states(attributes)
# and use methods associated to those states. We don't need that here.
class MissingDataError(Exception):
    # personal exception for missing data handling.
    # Look that this class inherits from Exception so we get all its methods
    # and class attributes, because __init__ is a method and here
    # we are not redefining it, we just inherit it.
    pass

def handle_missing_data(df, cols, strategy, constant=None):
    missing_counts = df[cols].isna().sum()
    total_missing = int(missing_counts.sum())
    detail = "\n".join(
        [f"{col}: {int(cnt)}" for col, cnt in missing_counts.items() if cnt > 0]
        )
    msg_NaN_summary = f"Total NaN values: {total_missing}\n\n{detail}"
    msg_preprocess_complete = ""
    if total_missing == 0:
        raise MissingDataError("No missing values found "
                                "in the selected columns.")
    result = strategy_handle_missing_data(df, cols, strategy, constant)
    df_processed, msg_preprocess_complete = result
    return df_processed, msg_NaN_summary, msg_preprocess_complete

def strategy_handle_missing_data(df, cols, strategy, constant=None):
        # this don't modify the original df, it returns a new one
        # so we need to reassign it in the GUI afterwards.
        # you can set this inplace too, but better not to.
        # is it better for testing this module.
    if strategy == "Delete rows with NaN":
        before = len(df)
        df = df.dropna(subset=cols)
        removed = before - len(df)
        msg_preprocess_complete = f"Rows removed: {removed}"

    elif strategy == "Fill with mean":
        for col in cols:
            mean = round(df[col].mean(), 4)
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(mean)
        msg_preprocess_complete = "Missing values filled with column mean."

    elif strategy == "Fill with median":
        for col in cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
        msg_preprocess_complete = "Missing values filled with column median."

    elif strategy == "Fill with constant":
        if len(constant.strip()) == 0:
            raise MissingDataError("Please provide a constant value")
        constant = float(constant)
        for col in cols:
            df[col] = df[col].fillna(constant)
        msg_preprocess_complete = (f"Missing values filled "
                                    f"with constant {constant}.")
    else:
        raise MissingDataError("Unknown strategy.")
    return df, msg_preprocess_complete
class TestHandleMissingData(unittest.TestCase):
    def setUp(self):
        # We create a dataframe with holes (NaNs) to test the strategies.
        # It gets recreated before EACH test function to ensure a clean state.
        self.df = pd.DataFrame({
            "A": [1, 2, np.nan, 4, 5],     # Numeric with 1 missing
            "B": [10, np.nan, 30, 40, 50], # Numeric with 1 missing
            "C": ["a", "b", "c", "d", "e"] # Text column (should be ignored by numeric strategies)
        })
        self.cols = ["A", "B"] # We usually operate on specific columns

    def test_raise_error_if_no_missing_values(self):
        # Your code explicitly checks "if total_missing == 0". Let's verify that.
        clean_df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        with self.assertRaises(MissingDataError):
            # Should scream "No missing values found"
            handle_missing_data(clean_df, ["x", "y"], "Delete rows with NaN")

    def test_strategy_delete_rows(self):
        # We have NaNs in index 2 (col A) and index 1 (col B).
        # If we delete rows with NaN in A or B, we should lose 2 rows total.
        df_res, summary, msg = handle_missing_data(self.df.copy(), self.cols, "Delete rows with NaN")
        
        # Original was 5 rows. We expect 3 rows remaining.
        self.assertEqual(len(df_res), 3)
        self.assertIn("Rows removed: 2", msg)

    def test_strategy_fill_mean(self):
        # Column A: [1, 2, NaN, 4, 5]. Mean of (1+2+4+5)/4 = 3.0
        df_res, _, msg = handle_missing_data(self.df.copy(), self.cols, "Fill with mean")
        
        # Check if the NaN at index 2 in col A became 3.0
        self.assertEqual(df_res["A"][2], 3.0)
        self.assertIn("filled with column mean", msg)

    def test_strategy_fill_median(self):
        # Column B: [10, NaN, 30, 40, 50]. Median of (10, 30, 40, 50) is 35.0
        df_res, _, msg = handle_missing_data(self.df.copy(), self.cols, "Fill with median")
        
        self.assertEqual(df_res["B"][1], 35.0)
        self.assertIn("filled with column median", msg)