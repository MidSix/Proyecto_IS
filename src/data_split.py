from typing import Tuple, Optional, Dict
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

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
            raise DataSplitError(f"Not enough rows to split (it has {n_rows}, minimum {MIN_ROWS}).")

        if not (0.0 < test_size < 1.0):
            raise DataSplitError("test_size must be a float between 0 and 1.")

        # sklearn makes a reproducible split with random_state
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_seed, shuffle=shuffle)

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


    def test_class(self):
        print(df.shape)
        objeto = DataSplitter(df)
        a,b = objeto.split()
        print("Training set")
        print(a)
        print()
        print("Test set")
        print(b)

if __name__ == "__main__":
    df = pd.DataFrame({
    "Edad": [25, 30, np.nan, 40, 35],
    "Altura": [1.75, 1.80, 1.65, np.nan, 1.70],
    "Peso": [70, 80, 60, 90, np.nan],
    "Genero": ["M", "M", "F", "M", "F"],
    "Salario": [30000, 40000, np.nan, 50000, 35000]
    })
    objeto = DataSplitter(df)
    #objeto.test_class()
    msg_summary = (
    "Division was correctly done.\n\n"
    f"Total df: rows\n"
    f"Training df: rows\n"
    f"Test df:  rows\n"
    f"Seed used:"
    )
    print(msg_summary)
    print(type(msg_summary))