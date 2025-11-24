from typing import Tuple, Optional, Dict
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import unittest
# Minimum number of rows required to perform a split
# global variable
MIN_ROWS = 5

class DataSplitError(Exception):
    pass

class DataSplitter:
    def __init__(self, df: pd.DataFrame = None):
        # Internal storage so other parts of the program can use it
        self.df = df
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.last_meta: Dict = {}

    def split(
        self,
        df: pd.DataFrame = None,
        test_size: float = 0.2,
        random_seed: Optional[int] = 42,
        shuffle: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        #handlers, should'nt be neccesary but well, never know.
        if df is None:
            raise DataSplitError("No dataframe loaded.")
        if not isinstance(df, pd.DataFrame):
            raise DataSplitError("The object isn't pandas.DataFrame.")

        n_rows = len(df)

        if n_rows < MIN_ROWS:
            raise DataSplitError(f"Not enough rows to split" \
                                f"(it has {n_rows}, minimum {MIN_ROWS}).")

        if not (0.0 < test_size < 1.0):
            raise DataSplitError("test_size must be a float between 0 and 1.")

        # sklearn makes a reproducible split with random_state
        train_df, test_df = train_test_split(df, test_size=test_size,
                                             random_state=random_seed,
                                             shuffle=shuffle)

        # Store internally for later use
        self.train_df = train_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)
        self.last_meta = {
            "n_rows_total": n_rows,
            "n_train": len(self.train_df),
            "n_test": len(self.test_df),
            "test_size": test_size,
            "random_seed": random_seed,
            "shuffle": shuffle
        }
        return self.train_df, self.test_df

    def get_meta(self) -> Dict:
        return self.last_meta.copy()

    def has_split(self) -> bool:
        return self.train_df is not None and self.test_df is not None

    def get_split_summary(self) -> str:
        if not self.has_split():
            raise DataSplitError("No split has been performed yet.")
        summary = self.get_meta()
        msg_summary = (
            f"Total df: {summary['n_rows_total']} rows\n"
            f"Training df: {summary['n_train']} rows\n"
            f"Test df: {summary['n_test']} rows\n"
            f"Seed used: {summary['random_seed']}"
        )
        return msg_summary

    def test_class(self):
        print(df.shape)
        objeto = DataSplitter(df)
        a,b = objeto.split()
        print("Training set")
        print(a)
        print()
        print("Test set")
        print(b)
class TestDataSplitter(unittest.TestCase):
    def setUp(self):
        # Create a dummy dataframe with enough rows (> MIN_ROWS which is 5)
        # I'm making it deterministic so tests don't flake out on me
        data = {
            "feature1": range(10), # 0 to 9
            "feature2": [x * 2 for x in range(10)],
            "target": [0, 1] * 5
        }
        self.df = pd.DataFrame(data)
        self.splitter = DataSplitter(self.df)

    def test_split_basic_success(self):
        # Testing the happy path: Splitting with default parameters
        # We have 10 rows, default test_size is 0.2 -> 2 rows test, 8 rows train
        train, test = self.splitter.split(self.df)
        
        self.assertTrue(self.splitter.has_split()) # Internal state should update
        self.assertEqual(len(train), 8)
        self.assertEqual(len(test), 2)
        self.assertEqual(len(train) + len(test), 10) # No data lost in the void
        
        # Also check if metadata was saved correctly
        meta = self.splitter.get_meta()
        self.assertEqual(meta["n_rows_total"], 10)
        self.assertEqual(meta["test_size"], 0.2)

    def test_split_custom_size(self):
        # Does it respect the test_size parameter?
        # Let's try 0.5 (50/50 split)
        train, test = self.splitter.split(self.df, test_size=0.5)
        self.assertEqual(len(train), 5)
        self.assertEqual(len(test), 5)

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
        tiny_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}) # Only 3 rows
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
        # If we ask for a summary before splitting, it should raise an error
        new_splitter = DataSplitter(self.df) # Fresh instance
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