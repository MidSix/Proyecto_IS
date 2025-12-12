import sqlite3
import pandas as pd
import os  # Handle file paths and extensions that may vary
           # by OS (Linux, Windows, macOS)

# We decided to use pandas for the general dataframe handling
# because it provides a matrix-like structure and easy data
# manipulation.
class DataModule:
    """Container for loading and providing dataset access.

    This class encapsulates loading data from several file types and
    exposes helper methods for accessing features, target, and basic
    summaries. It is designed for use by the GUI and tests.

    Attributes
    ----------
    file_lists : dict
        Mapping of filename to absolute path added by the user.
    current_file_name : str or None
        Currently selected file name.
    current_file_type : str or None
        File extension (without dot) of the current file.
    current_dataframe : pd.DataFrame or None
        Loaded pandas DataFrame or None if not loaded.
    current_connection : sqlite3.Connection or None
        SQLite connection when a sqlite file is used.
    error_message : str or None
        Last error message, or None when last operation succeeded.

    Methods
    -------
    add_file_path(file_path, file_name)
        Add a file path under a given display name.
    set_current_file(file_name)
        Select a stored file name as current.
    load_data()
        Load the currently selected file into a DataFrame.
    is_empty()
        Return True if no DataFrame is loaded or it is empty.
    get_features()
        Return all columns except the last as features.
    get_target()
        Return the last column as target variable.
    direct_data_load(data)
        Load a pandas DataFrame directly (for tests).
    """
    # When initializing, the user can idle in the window waiting to
    # add file paths.
    def __init__(self):
        # This will store the paths of the files added by the user.
        # Key is the name, value is the path.
        self.file_lists = {}
        self.current_file_name = None
        self.current_file_type = None
        # Remember this is a pandas dataframe, it is a matrix made
        # internally as a 2D array.
        self.current_dataframe = None
        # This is a special SQLite object that handles the
        # connection to the database.
        self.current_connection = None
        # To store error messages and show them in the GUI
        # this variable stores a string with the current error message
        # and is None when the last operation was successful.
        self.error_message = None

    @property
    def get_file_names(self):
        # For the GUI: show user the files added and let them choose.
        return list(self.file_lists.keys())

    def add_file_path(self, file_path: str, file_name: str) -> bool:
        """Add a file path under a display name.

        Parameters
        ----------
        file_path : str
            Absolute or relative path to the file.
        file_name : str
            Display name for the file (used as a key).

        Returns
        -------
        bool
            True if the path was accepted and stored, False otherwise.
        """
        file_type = (
            os.path.splitext(file_name)[-1][1:].lower()
            if file_name
            else None
        )
        if (
            not os.path.isfile(file_path)
            or file_type not in ['sqlite', 'db', 'csv', 'xlsx', 'xls']
        ):
            self.error_message = (
                "File does not exist or is incompatible."
                "Please provide a "
                "valid file path."
            )
            return False
        self.error_message = None
        # This approach overwrites the path for each name
        # without checking if it already exists. It solves the issue
        # of having two files with the same name in different
        # directories: opening another file with
        # the same name just updates the stored path.
        self.file_lists[file_name] = file_path
        return True

    def set_current_file(self, file_name: str) -> bool:
        """Set the current file by name.

        Parameters
        ----------
        file_name : str
            Display name of the file to select as current.

        Returns
        -------
        bool
            True if the file name exists in file_lists, False otherwise.
        """
        # For the file selector in the GUI.
        if file_name not in self.file_lists:
            self.error_message = (
                "File name not found. Please provide a valid file name."
            )
            return False
        self.error_message = None
        self.current_file_path = self.file_lists[file_name]
        self.current_file_name = file_name
        # Get the file extension without the dot and in lowercase.
        self.current_file_type = os.path.splitext(
            self.current_file_name
        )[-1][1:].lower()
        return True

    def load_data(self) -> bool:
        """Load the currently selected file into a pandas DataFrame.

        This method supports sqlite/db, csv, xlsx and xls file types.
        It sets `self.current_dataframe` on success and returns True.
        On failure it sets `self.error_message` and returns False.

        Returns
        -------
        bool
            True if loading succeeded, False otherwise.
        """
        # Returns True if data loads successfully, False otherwise.
        try:
            if self.current_file_type in ['sqlite', 'db']:
                self.current_connection = sqlite3.connect(
                    self.current_file_path
                )
                # Fetch the first table name.
                cursor = self.current_connection.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table';"
                )
                tables = cursor.fetchall()

                # We think it's ok to raise error here because it's an
                # exceptional case.
                if not tables:
                    self.error_message = (
                        "No table found in the sqlite database."
                    )
                    raise ValueError(
                        "No table found in the sqlite database."
                    )

                table_name = tables[0][0]
                self.current_dataframe = pd.read_sql_query(
                    f"SELECT * FROM {table_name}",
                    self.current_connection,
                )
                # Good practice: explicitly close the connection
                # instead of relying on the garbage collector.
                # If we only reassign the connection, the object may
                # stay open and keep the SQLite file locked in read-only
                # mode until it is collected.
                self.current_connection.close()

            elif self.current_file_type == 'csv':
                self.current_dataframe = pd.read_csv(
                    self.current_file_path,
                    header=0,
                )
                # Pandas readers do not return None.
                # If the file is totally empty (no metadata or headers),
                # pandas raises its own error. If the file has headers
                # but no data, it returns an empty DataFrame, so
                # we can safely use .empty and raise our own error.
                if self.current_dataframe.empty:
                    raise ValueError(
                        "The CSV file is empty or could not be read "
                        "correctly."
                    )

            elif self.current_file_type in ['xlsx', 'xls']:
                self.current_dataframe = pd.read_excel(
                    self.current_file_path,
                    header=0,
                )
                if self.current_dataframe.empty:
                    raise ValueError(
                        "The Excel file is empty or could not be read "
                        "correctly."
                    )
            else:
                # It should not reach this point
                # because of previous checks.
                raise ValueError(
                    "Unsupported file type. Supported types are: "
                    "sqlite, db, csv, xlsx, xls."
                )
        except Exception as e:
            # This is for other reasons like file not found, permission
            # issues, etc.
            error_message = f"Error loading data: {e}"
            self.current_dataframe = None
            self.error_message = error_message
            return False
        self.error_message = None
        return True

    def is_empty(self) -> bool:
        """Check if current DataFrame is empty or not loaded.

        Returns
        -------
        bool
            True if no DataFrame is loaded or it is empty, False
            otherwise.
        """
        # Returns True if the dataframe is empty, False if it has data.
        # Fundamental: if path or file are incorrect,
        # self.current_dataframe is None. A None is not a pandas type,
        # so we cannot use pandas .empty on it. Returning True here
        # avoids changing the current implementation.
        if self.current_dataframe is None:
            return True
        # We can use pandas built-in function to check if the
        # dataframe is empty.
        return self.current_dataframe.empty

    def get_summary(self) -> pd.DataFrame:
        """Return a basic summary description of the current DataFrame.

        Returns
        -------
        pd.DataFrame or None
            Summary produced by `DataFrame.describe(include='all')`, or
            None if no data is available.
        """
        if self.is_empty():
            self.error_message = "No data to summarize."
            return None
        self.error_message = None
        return self.current_dataframe.describe(include='all')

    def showcase_data(self) -> bool:
        """Print the current DataFrame to the console.

        Returns
        -------
        bool
            True if the DataFrame was printed, False if no data.
        """
        # Just shows the data in the console. Returns True if
        # successful, False if not.
        if self.is_empty():
            self.error_message = "Dataframe is empty. Load data first."
            return False
        else:
            print(self.current_dataframe)
            self.error_message = None
            return True

    def main(self, file_path: str):
        """Wrapper that adds, selects and loads a file.

        This method adds the provided `file_path`, selects it as the
        current file, attempts to load it and returns the loaded
        DataFrame together with an error message (None on success).

        Parameters
        ----------
        file_path : str
            Path to the file to add and load.

        Returns
        -------
        tuple
            `(pd.DataFrame or None, str or None)` representing the
            loaded DataFrame and an error message.
        """
        # To simplify things and encapsulate this module. GUI will call
        # this function that handles all the internal logic and returns
        # the dataframe associated with the given path.
        self.error_message = None
        file_name = os.path.basename(file_path) if file_path else None

        ok = self.add_file_path(file_path, file_name)
        # I simply love this if statement XD.
        if not ok:
            return None, self.error_message

        ok = self.set_current_file(file_name)
        if not ok:
            return None, self.error_message

        ok = self.load_data()
        if not ok:
            return None, self.error_message

        # Basically if it reaches this point everything went ok so we
        # return the dataframe and no error message.
        return self.current_dataframe, None

    # ------ Functions added for linear regression testing purposes ----
    # Returns all columns except the last one as features. This follows
    # a common convention in machine learning: all columns except the
    # last are input features and the last column is the target.
    def get_features(self) -> pd.DataFrame:
        """Return all columns except the last as the feature set.

        Returns
        -------
        pd.DataFrame or None
            DataFrame containing feature columns, or None if no data is
            loaded.
        """
        if self.is_empty():
            self.error_message = "Dataframe is empty. Load data first."
            return None
        # All rows, all columns except the last one.
        return self.current_dataframe.iloc[:, :-1]

    # Returns the target variable, which is the last column.
    def get_target(self) -> pd.Series:
        """Return the target variable (the last column).

        Returns
        -------
        pd.Series or None
            Series containing the target column, or None if no data is
            loaded.
        """
        if self.is_empty():
            self.error_message = "Dataframe is empty. Load data first."
            return None
        self.error_message = None
        # All rows, only the last column.
        return self.current_dataframe.iloc[:, -1]

    # Loads data directly from a pandas DataFrame, mainly for testing
    # the linear regression module.
    def direct_data_load(self, data: pd.DataFrame) -> bool:
        """Load a pandas DataFrame directly into the module.

        This is primarily used by unit tests to inject a DataFrame
        without reading from disk.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame to load into the module.

        Returns
        -------
        bool
            True if the provided object was a DataFrame and was loaded
            successfully, False otherwise.
        """
        if not isinstance(data, pd.DataFrame):
            self.error_message = (
                "Provided data is not a pandas DataFrame."
            )
            return False
        self.current_dataframe = data
        return True