import sqlite3
import pandas as pd
import os # To handle file paths and extensions that may vary by OS(linux, windows, macOS)
import unittest  #For automatic testing purposes
import tempfile #This will allow us to create temporary files for testing without needing actual files on disk
#I decided to use pandas for the general dataframe handling, because it provides a matrix-like structure and easy data manipulation
class DataModule:
    #Now when initializing, user can idle in the window waiting to add file paths
    def __init__(self):
        self.file_lists = {} #This will store the paths of the files added by the user, key is the name, value is the path
        self.current_file_name = None
        self.current_file_type = None
        self.current_dataframe = None #Remember this is a pandas dataframe, it is a matrix made internally as a 2D array
        self.current_connection = None #This is an object special for SQLite that handles the connection to the database
        self.error_message = None #To store error messages and show them in the GUI

    @property
    def get_file_names(self):#This is for the implementation of the GUI, to show the user the files he added, and let him choose one
        return list(self.file_lists.keys()) #Return the list of file names added by the user

    def add_file_path(self, file_path: str, file_name: str) -> bool:
        file_type = os.path.splitext(file_name)[-1][1:].lower() if file_name else None
        if not os.path.isfile(file_path) or file_type not in ['sqlite', 'db', 'csv', 'xlsx', 'xls']:
            print("File does not exist or is incompatible. Please provide a valid file path.")
            return False
        #I think this is a better way to handle it, without testing if the name has already a path
        #It has to write in the dictionary everytime but solves the problem of having two files
        #with same name in different directories. If that happens and open another file with same name
        #It would open normaly because everytime replaces the value(path) associated to that key(name).
        self.file_lists[file_name] = file_path # Store the path in the dictionary with the file name as key
        return True

    def set_current_file(self, file_name: str) -> bool:#This is for the file selector in the GUI
        if file_name not in self.file_lists:
            print("File name not found. Please provide a valid file name.")
            return False
        self.current_file_path = self.file_lists[file_name]
        self.current_file_name = file_name
        self.current_file_type = os.path.splitext(self.current_file_name)[-1][1:].lower() # Get the file extension without the dot and in lowercase
        return True

    def load_data(self) -> bool: #returns True if data load succesfully, false otherwise.
        try:
            if self.current_file_type in ['sqlite', 'db']:
                self.current_connection = sqlite3.connect(self.current_file_path)
                # Fetch the first table name
                cursor = self.current_connection.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()

                if not tables:
                    raise ValueError("No table found in the sqlite database.")

                table_name = tables[0][0]
                self.current_dataframe = pd.read_sql_query(f"SELECT * FROM {table_name}", self.current_connection)
                self.current_connection.close() #This is a good practice. Names are just pointers to objects.
                #If we just reasign it the object still exists, therefore the
                #connection is still opened with that file until python garbage
                #Collector closses it. But until that the sql file is locked and if you
                #try to open it, it would open in read-only to avoid corrupting the file. Basically it's
                #better to close connection in summary

            elif self.current_file_type == 'csv':
                self.current_dataframe = pd.read_csv(self.current_file_path, header=0)
                if self.current_dataframe.empty: #Just a curiosity.These pandas functions never returns None
                    #if the file is completely empty, without metadata or headers it raises its own error.
                    #so we don't use .empty with a None type object in any case, this is why this is not
                    #causing issues, and if the file don't have user data but has headers and metadata
                    #it returns an empty dataframe object so we can use .empty safely and raise our own error.
                    #You can test this behavior by creating an empty file in any of these functions and trying to load it.
                    #You will see pandas raises its own error message, different from the one we have here.
                    raise ValueError("The CSV file is empty or could not be read correctly.")

            elif self.current_file_type in ['xlsx', 'xls']:
                self.current_dataframe = pd.read_excel(self.current_file_path, header=0)
                if self.current_dataframe.empty:
                    raise ValueError("The Excel file is empty or could not be read correctly.")
            else:
                raise ValueError("Unsupported file type. Supported types are: sqlite, db, csv, xlsx, xls.")#It shouldn't reach this point because of previous checks
        except Exception as e:
            error_message = f"Error loading data: {e}"#This is for other reasons like file not found, permission issues, etc.
            print(error_message)
            self.current_dataframe = None
            self.error_message = error_message
            return False
        error_message = None
        return True

    def is_empty(self)->bool: #Returns True if the dataframe is empty, False if it has data
        if self.current_dataframe is None:
            print("Dataframe has not been loaded. Load data first.")
            return True #This line is fundamental. If path or file are incorrect then self.current_dataframe is None.
        #and a None type is not a pandas type, this means we can't use the pandas built-in to check if it's empty
        #with our current implementation. To not change it I just add this return xd.
        return self.current_dataframe.empty #We can use pandas built-in function to check if the dataframe is empty

    def get_summary(self):
        if self.is_empty():
            print("No data to summarize.")
            return None
        return self.current_dataframe.describe(include='all')

    def showcase_data(self)->bool: #Just shows the data in the console, returns True if successful, False if not
        if self.is_empty():
            print("Dataframe is empty. Load data first.")
            return False
        else:
            print(self.current_dataframe)
            return True

    def main(self, file_path:str) -> pd.DataFrame: #To simplify things and encapsulate this module. GUI will
        #call this function that handles all the internal logic and returns the dataframe associated
        #to the given path.
        file_name = os.path.basename(file_path) if file_path else None
        self.add_file_path(file_path, file_name)
        self.set_current_file(file_name)
        self.load_data()
        return self.current_dataframe, self.error_message
    #------ Functions added for linear regression testing purposes ------
    def get_features(self) -> pd.DataFrame: #This function returns all columns except the last one, this aims to get
        #the features for linear models, to summarize, all columns except the last one are features, last column is target
        #This is a common convention in datasets for machine learning, to explain what that means basically features are the input variables
        #that the model will use to predict the target variable, which is the output variable.
        if self.is_empty():
            print("Dataframe is empty. Load data first.")
            return None
        return self.current_dataframe.iloc[:, :-1] #All rows, all columns except the last one
    
    def get_target(self) -> pd.Series:#As previously explained, this function is to get the target variable, which is the last column
        if self.is_empty():
            print("Dataframe is empty. Load data first.")
            return None
        return self.current_dataframe.iloc[:, -1] #All rows, only the last column
    
    def direct_data_load(self, data:pd.DataFrame)->bool:
         #This function was added for testing purposes mainly, to load data directly from a pandas dataframe
         #Primarily for the linear regression testing module
        if not isinstance(data, pd.DataFrame):
            print("Provided data is not a pandas DataFrame.")
            return False
        self.current_dataframe = data
        return True
    #------ Functions added for linear regression testing purposes ------

class TestDataModule(unittest.TestCase):
    def setUp(self):
        self.data_module = DataModule()

    def test_add_file_path_invalid(self): #Test adding an invalid file path(easy one)
        result = self.data_module.add_file_path("non_existent_file.csv", "non_existent_file.csv")
        self.assertFalse(result)#As we set on the function, it should return False
        self.assertIsNotNone(self.data_module.error_message)#There should be an error message when trying to add an invalid file path

    def test_set_current_file_invalid(self):#I mean, read the name, it's self-explanatory
        # add a valid temporary file first to populate file_lists, here comes handy the tempfile module
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:#Mode w to write, suffix to set extension, delete false to keep it after closing
            tmp_path = tmp.name
            tmp.write("a,b\n1,2\n")#Write some minimal content to make it a valid csv, not an empty file
        try:
            add_ok = self.data_module.add_file_path(tmp_path, os.path.basename(tmp_path))#Add the temp file
            self.assertTrue(add_ok)#It should be added successfully at first
            # attempt to set a non-existent name
            result = self.data_module.set_current_file("non_existent_file.csv")
            self.assertFalse(result)
            self.assertIsNotNone(self.data_module.error_message)#There should be an error message when trying to set a non-existent current file
        finally:
            try:#Is this structure necessary? No, but it's a good practice to avoid leaving temp files behind in case of errors
                os.remove(tmp_path)#Clean up the temporary file after the test. Because we set delete=False to keep it after closing
            except Exception:
                pass

    def test_load_data_csv_success(self):#Test loading a valid CSV file and then checking if the dataframe is loaded correctly
        # create temporary csv and load it via the module
        df = pd.DataFrame({"x":[1,2,3], "y":[4,5,6]})#Sample dataframe to write to csv
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:#Explained above
            tmp_path = tmp.name
            df.to_csv(tmp_path, index=False)
        try:
            add_ok = self.data_module.add_file_path(tmp_path, os.path.basename(tmp_path))#This variables seem odd, but as stated before this functions return booleans to help with error handling
            self.assertTrue(add_ok)
            set_ok = self.data_module.set_current_file(os.path.basename(tmp_path))
            self.assertTrue(set_ok)
            load_ok = self.data_module.load_data()
            self.assertTrue(load_ok)
            self.assertIsNotNone(self.data_module.current_dataframe)#Dataframe should not be None after loading, if that happens we are cooked
            pd.testing.assert_frame_equal(self.data_module.current_dataframe.reset_index(drop=True), df.reset_index(drop=True))#What is this you may ask? Well, sometimes pandas adds an index column when reading 
                                                                                                                                #from csv, so to avoid false negatives in the test we reset the index on both dataframes before comparing
                                                                                                                                #(Not an idea of mine, I double checked the code with ChatGPT to be sure and this was his only suggestion to avoid index issues)
        finally:#I did not know of this "finally" structure until recently, it's a manner to ensure cleanup code runs even if an error occurs in the try block
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    def test_main_returns_tuple_and_error_on_bad_path(self):#This one is a good way to test the main function directly, it checks if it returns the expected tuple structure and error message on bad path
        df, err = self.data_module.main("non_existing_file.csv") #both vaiables to capture the returned tuple
        self.assertIsNone(df)#Dataframe should be None when loading fails
        self.assertIsNotNone(err)  #There should be an error message when loading fails

if __name__ == "__main__":
    unittest.main()

#Dude this archive has commentaries. I swear to god I did not intend to write an essay but it just happened. Sometimes I can't help myself.   