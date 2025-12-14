import pandas as pd
from typing import Optional
# We do not need a class here because we do not store state.
# A class is only useful when attributes (state) are meaningful and
# methods operate on that state; in this module simple functions are
# enough.
class MissingDataError(Exception):
    """Custom exception for missing data handling.

    This exception is raised when issues occur during missing data
    processing. It inherits from Exception, reusing its methods and
    behavior.
    """
    # Custom exception for missing data handling. It inherits from
    # Exception, so it reuses its methods and behavior.
    pass

def handle_missing_data(
        df: pd.DataFrame,
        cols:list,
        strategy:str,
        constant:Optional[int] = None
        ):
    """Handle missing data in specified DataFrame columns.

    This function processes missing values according to the specified
    strategy and returns the processed DataFrame along with summary
    messages.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing missing values.
    cols : list
        Column names to process for missing values.
    strategy : str
        Strategy to apply: 'Delete rows with NaN', 'Fill with mean',
        'Fill with median', or 'Fill with constant'.
    constant : str or numeric, optional
        Constant value used to fill missing entries when strategy is
        'Fill with constant'. The function accepts a string or a
        numeric value; strings are stripped and converted to float.
        Default is None.

    Returns
    -------
    tuple
        Processed DataFrame, missing values summary message, and
        preprocessing completion message.

    Raises
    ------
    MissingDataError
        If no missing values found or invalid strategy provided.
    """
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

def strategy_handle_missing_data(
        df: pd.DataFrame,
        cols: list,
        strategy: str,
        constant: Optional[int]=None
        ):
    """Apply specified missing data handling strategy.

    This function executes the selected strategy to handle missing
    values in specified columns. It does not modify the original
    DataFrame but returns a new one, which is better for testing and
    for the GUI where the caller can reassign the processed DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with missing values.
    cols : list
        Column names to process.
    strategy : str
        Strategy name: 'Delete rows with NaN', 'Fill with mean',
        'Fill with median', or 'Fill with constant'.
    constant : str or numeric, optional
        Value used to fill missing entries when strategy is
        'Fill with constant'. Strings will be stripped and converted
        to float. Default is None.

    Returns
    -------
    tuple
        Processed DataFrame and completion status message.

    Raises
    ------
    MissingDataError
        If constant is empty or strategy is unknown.
    """
        # This function does not modify the original DataFrame. It returns
        # a new one, which is better for testing and for the GUI, where
        # the caller can reassign the processed DataFrame.
    if strategy == "Delete rows with NaN":
        before = len(df)
        df = df.dropna(subset=cols)
        removed = before - len(df)
        msg_preprocess_complete = f"Rows removed: {removed}"

    elif strategy == "Fill with mean":
        for col in cols:
            # Only if all elements in the column are numeric it does the
            # mean. The problem before was that it was trying to do
            # the mean without cheking out first if the column is
            # numeric.
            if pd.api.types.is_numeric_dtype(df[col]):
                mean = round(df[col].mean(), 4)
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
        constant = float(constant) #If not float it raises an error
        for col in cols:
            df[col] = df[col].fillna(constant)
        msg_preprocess_complete = (f"Missing values filled "
                                    f"with constant {constant}.")
    else:
        raise MissingDataError("Unknown strategy.")
    return df, msg_preprocess_complete