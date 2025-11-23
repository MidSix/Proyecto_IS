import pandas as pd
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