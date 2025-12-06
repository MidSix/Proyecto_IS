"""Unit tests for the DataModule backend component.

Tests data loading functionality across multiple file formats (CSV, SQLite,
Excel). Verifies file path validation, current file selection, data loading,
and error handling.
"""
import os
import pandas as pd
import unittest  # For automatic testing purposes
import tempfile  # Create temporary files for testing without using
                 # actual files on disk

from src.backend.data_loader import DataModule

class TestDataModule(unittest.TestCase):
    """Test suite for DataModule class.

    Tests file path management, file selection, data loading from various
    formats, and error conditions.

    Methods
    -------
    setUp()
        Initialize DataModule instance before each test.
    test_add_file_path_invalid()
        Test adding non-existent file path returns False.
    test_set_current_file_invalid()
        Test setting non-existent file returns False.
    test_load_data_csv_success()
        Test successful CSV file loading and validation.
    test_main_returns_tuple_and_error_on_bad_path()
        Test main() returns None DataFrame and error on bad path.
    """
    def setUp(self) -> None:
        """Initialize DataModule instance for testing.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.data_module = DataModule()

    def test_add_file_path_invalid(self) -> None:
        """Test adding an invalid file path returns False.

        Verifies that attempting to add a non-existent file path
        returns False and sets an error message.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Test adding an invalid file path (easy one).
        result = self.data_module.add_file_path(
            "non_existent_file.csv",
            "non_existent_file.csv",
        )
        # As we set in the function, it should return False.
        self.assertFalse(result)
        # There should be an error message when trying to add an
        # invalid file path.
        self.assertIsNotNone(self.data_module.error_message)

    def test_set_current_file_invalid(self) -> None:
        """Test setting non-existent file as current fails.

        Creates temporary CSV file, adds it successfully, then attempts
        to set non-existent file as current and verifies failure.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # I mean, read the name, it's self-explanatory.
        # Add a valid temporary file first to populate file_lists.
        # Here the tempfile module comes in handy.
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.csv',
            delete=False,
        ) as tmp:
            # Mode w to write, suffix to set extension, delete False
            # to keep it after closing.
            tmp_path = tmp.name
            # Write some minimal content to make it a valid csv, not an
            # empty file.
            tmp.write("a,b\n1,2\n")
        try:
            add_ok = self.data_module.add_file_path(
                tmp_path,
                os.path.basename(tmp_path),
            )
            # It should be added successfully at first.
            self.assertTrue(add_ok)
            # Attempt to set a non-existent name.
            result = self.data_module.set_current_file(
                "non_existent_file.csv"
            )
            self.assertFalse(result)
            # There should be an error message when trying to set a
            # non-existent current file.
            self.assertIsNotNone(self.data_module.error_message)
        finally:
            # Is this structure necessary? No, but it's a good practice
            # to avoid leaving temp files behind in case of errors.
            try:
                # Clean up the temporary file after the test. Because
                # we set delete=False to keep it after closing.
                os.remove(tmp_path)
            except Exception:
                pass

    def test_load_data_csv_success(self) -> None:
        """Test successful CSV loading and verification.

        Creates temporary CSV, loads via DataModule, and verifies
        loaded DataFrame matches original using pandas testing utility.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Test loading a valid CSV file and then checking if the
        # dataframe is loaded correctly.
        # Create temporary csv and load it via the module.
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.csv',
            delete=False,
        ) as tmp:
            # Explained above.
            tmp_path = tmp.name
            df.to_csv(tmp_path, index=False)
        try:
            # These variables seem odd, but as stated before these
            # functions return booleans to help with error handling.
            add_ok = self.data_module.add_file_path(
                tmp_path,
                os.path.basename(tmp_path),
            )
            self.assertTrue(add_ok)
            set_ok = self.data_module.set_current_file(
                os.path.basename(tmp_path)
            )
            self.assertTrue(set_ok)
            load_ok = self.data_module.load_data()
            self.assertTrue(load_ok)
            # Dataframe should not be None after loading, if that
            # happens we are cooked.
            self.assertIsNotNone(self.data_module.current_dataframe)
            # What is this you may ask? Well, sometimes pandas adds an
            # index column when reading from csv, so to avoid false
            # negatives in the test we reset the index on both
            # dataframes before comparing. (Not an idea of mine, I
            # double checked the code with ChatGPT to be sure and this
            # was his only suggestion to avoid index issues).
            pd.testing.assert_frame_equal(
                self.data_module.current_dataframe.reset_index(
                    drop=True
                ),
                df.reset_index(drop=True),
            )
        finally:
            # I did not know of this "finally" structure until
            # recently. It's a way to ensure cleanup code runs even if
            # an error occurs in the try block.
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    def test_main_returns_tuple_and_error_on_bad_path(self) -> None:
        """Test main() returns None DataFrame and error on bad path.

        Verifies main() method returns (None, error_message) tuple
        when loading non-existent file.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # This is a good way to test the main function directly. It
        # checks if it returns the expected tuple structure and error
        # message on bad path.
        # Both variables to capture the returned tuple.
        df, err = self.data_module.main("non_existing_file.csv")
        # Dataframe should be None when loading fails.
        self.assertIsNone(df)
        # There should be an error message when loading fails.
        self.assertIsNotNone(err)


if __name__ == "__main__":
    unittest.main()

# Dude this archive has commentaries. I swear to god I did not intend
# to write an essay but it just happened. Sometimes I can't help
# myself.