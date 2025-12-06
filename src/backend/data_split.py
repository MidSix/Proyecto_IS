from typing import Tuple, Optional, Dict
import pandas as pd
from sklearn.model_selection import train_test_split
# Minimum number of rows required to perform a split (global
# variable).
MIN_ROWS = 5

class DataSplitError(Exception):
    """Exception raised for errors in data splitting operations.

    This exception is raised when an error occurs during the train-test
    split process, such as invalid parameters or insufficient data.
    """
    pass

class DataSplitter:
    """Splits a DataFrame into training and test subsets.

    This class encapsulates the functionality to split a pandas
    DataFrame into train and test sets using scikit-learn's
    train_test_split, while tracking metadata about the split.

    Attributes
    ----------
    df : pd.DataFrame or None
        Internal storage of the original DataFrame.
    train_df : pd.DataFrame or None
        Training subset after split, or None if not split yet.
    test_df : pd.DataFrame or None
        Test subset after split, or None if not split yet.
    last_meta : dict
        Metadata from the last split operation.

    Methods
    -------
    split(df, test_size, random_seed, shuffle)
        Split a DataFrame into train and test subsets.
    get_meta()
        Return metadata from the last split.
    has_split()
        Check if a split has been performed.
    get_split_summary()
        Return a formatted summary of the last split.
    """
    def __init__(self, df: pd.DataFrame = None) -> None:
        """Initialize the DataSplitter.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Initial DataFrame to store. Default is None.

        Returns
        -------
        None
        """
        # Internal storage so other parts of the program can reuse it.
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
        """Split a DataFrame into training and test subsets.

        This method uses scikit-learn's train_test_split to divide the
        provided DataFrame. Both subsets are stored internally and also
        returned. Metadata about the split is retained for later query.

        Parameters
        ----------
        df : pd.DataFrame, optional
            DataFrame to split. If None, raises DataSplitError.
        test_size : float, optional
            Fraction of data for test set. Must be between 0 and 1.
            Default is 0.2.
        random_seed : int or None, optional
            Seed for reproducibility. Default is 42. If None, splits
            will be different each time.
        shuffle : bool, optional
            Whether to shuffle data before split. Default is True.

        Returns
        -------
        tuple
            Tuple of (train_df, test_df), both as pd.DataFrame.

        Raises
        ------
        DataSplitError
            If df is None, not a DataFrame, has fewer than MIN_ROWS,
            or test_size is not between 0 and 1.
        """

        # Basic handlers; they should not be necessary, but we keep them
        # as a safety check.
        if df is None:
            raise DataSplitError("No dataframe loaded.")
        if not isinstance(df, pd.DataFrame):
            raise DataSplitError("The object isn't pandas.DataFrame.")

        n_rows = len(df)

        if n_rows < MIN_ROWS:
            raise DataSplitError(f"Not enough rows to split\n"
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
        """Return a copy of metadata from the last split operation.

        Returns
        -------
        dict
            Dictionary containing split metadata (n_rows_total, n_train,
            n_test, test_size, random_seed, shuffle).
        """
        return self.last_meta.copy()

    def has_split(self) -> bool:
        """Check if a split has been performed.

        Returns
        -------
        bool
            True if both train_df and test_df are set, False otherwise.
        """
        return self.train_df is not None and self.test_df is not None

    def get_split_summary(self) -> str:
        """Return a formatted summary of the last split operation.

        Returns
        -------
        str
            Multi-line string containing row counts for total, training,
            test data and the random seed used.

        Raises
        ------
        DataSplitError
            If no split has been performed yet.
        """
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
