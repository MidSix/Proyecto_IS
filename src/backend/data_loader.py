import sqlite3
import pandas as pd
import os  # Handle file paths and extensions that may vary
           # by OS (Linux, Windows, macOS)
import unittest  # For automatic testing purposes
import tempfile  # Create temporary files for testing without using
                 # actual files on disk

# I decided to use pandas for the general dataframe handling, because
# it provides a matrix-like structure and easy data manipulation.
class DataModule:
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
        # This is an object special for SQLite that handles the
        # connection to the database.
        self.current_connection = None
        # To store error messages and show them in the GUI, this
        # variable will store a string with the current error message
        # and be None when the last operation was successful.
        self.error_message = None

    @property
    def get_file_names(self):
        # For the GUI: show user the files added and let them choose.
        return list(self.file_lists.keys())

    def add_file_path(self, file_path: str, file_name: str) -> bool:
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
                "File does not exist or is incompatible. Please provide a "
                "valid file path."
            )
            return False
        self.error_message = None
        # I think this is a better way to handle it, without testing if
        # the name already has a path. It has to write in the dict
        # every time but solves the problem of having two files with
        # the same name in different directories. If that happens and
        # we open another file with the same name, it will open
        # normally because every time we replace the value (path)
        # associated with that key (name).
        self.file_lists[file_name] = file_path
        return True

    def set_current_file(self, file_name: str) -> bool:
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

                # I think it's ok to raise error here because it's an
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
                # This is a good practice. Names are just pointers to
                # objects. If we just reassign it, the object still
                # exists, therefore the connection is still open with
                # that file until Python garbage collector closes it.
                # Until then, the SQL file is locked and if you try to
                # open it, it would open in read-only to avoid
                # corrupting the file. Basically it's better to close
                # the connection.
                self.current_connection.close()

            elif self.current_file_type == 'csv':
                self.current_dataframe = pd.read_csv(
                    self.current_file_path,
                    header=0,
                )
                # Just a curiosity. These pandas functions never return
                # None. If the file is completely empty, without
                # metadata or headers, it raises its own error. So we
                # don't use .empty with a None type object in any
                # case. This is why this is not causing issues, and if
                # the file doesn't have user data but has headers and
                # metadata, it returns an empty dataframe object. Then
                # we can use .empty safely and raise our own error.
                # You can test this by creating an empty file and
                # trying to load it. Pandas raises its own error
                # message, different from ours here.
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
                # It shouldn't reach this point because of previous
                # checks.
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
        # Returns True if the dataframe is empty, False if it has data.
        if self.current_dataframe is None:
            # This line is fundamental. If path or file are incorrect
            # then self.current_dataframe is None. A None type is not
            # a pandas type, so we can't use the pandas built-in to
            # check if it's empty with our current implementation. To
            # avoid changing it I just add this return xd.
            return True
        # We can use pandas built-in function to check if the
        # dataframe is empty.
        return self.current_dataframe.empty

    def get_summary(self):
        if self.is_empty():
            self.error_message = "No data to summarize."
            return None
        self.error_message = None
        return self.current_dataframe.describe(include='all')

    def showcase_data(self) -> bool:
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
        # To simplify things and encapsulate this module. GUI will call
        # this function that handles all the internal logic and returns
        # the dataframe associated with the given path.
        """Wrapper that adds, selects and loads a file.

        Returns (DataFrame or None, error_message or None).
        """
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
    def get_features(self) -> pd.DataFrame:
        # This function returns all columns except the last one. This
        # aims to get the features for linear models. To summarize, all
        # columns except the last one are features; last column is
        # target. This is a common convention in datasets for machine
        # learning. Features are the input variables that the model
        # will use to predict the target variable, which is the output
        # variable.
        if self.is_empty():
            self.error_message = "Dataframe is empty. Load data first."
            return None
        # All rows, all columns except the last one.
        return self.current_dataframe.iloc[:, :-1]

    def get_target(self) -> pd.Series:
        # As previously explained, this function gets the target
        # variable, which is the last column.
        if self.is_empty():
            self.error_message = "Dataframe is empty. Load data first."
            return None
        self.error_message = None
        # All rows, only the last column.
        return self.current_dataframe.iloc[:, -1]

    def direct_data_load(self, data: pd.DataFrame) -> bool:
        # This function was added mainly for testing purposes, to load
        # data directly from a pandas dataframe. Primarily for the
        # linear regression testing module.
        if not isinstance(data, pd.DataFrame):
            self.error_message = (
                "Provided data is not a pandas DataFrame."
            )
            return False
        self.current_dataframe = data
        return True
    # ------ Functions added for linear regression testing purposes ----
