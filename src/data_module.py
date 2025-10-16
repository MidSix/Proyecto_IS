import sqlite3
import pandas as pd
import os # To handle file paths and extensions that may vary by OS(linux, windows, macOS)

#This data module must support SQlite, CSV, and Excel formats
#I decided to use pandas for the general dataframe handling, because it provides a matrix-like structure and easy data manipulation
class DataModule:
    #Now when initializing, user can idle in the window waiting to add file paths
    def __init__(self):
        self.file_lists = {} #This will store the paths of the files added by the user, key is the name, value is the path
        self.current_file_name = None
        self.current_file_type = None
        self.current_dataframe = None #Remember this is a pandas dataframe, it is a matrix made internally as a 2D array
        self.current_connection = None #This is an object special for SQLite that handles the connection to the database

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
                if self.current_dataframe.empty:
                    raise ValueError("The CSV file is empty or could not be read correctly.")

            elif self.current_file_type in ['xlsx', 'xls']:
                self.current_dataframe = pd.read_excel(self.current_file_path, header=0)
                if self.current_dataframe.empty:
                    raise ValueError("The Excel file is empty or could not be read correctly.")
            else:
                raise ValueError("Unsupported file type. Supported types are: sqlite, db, csv, xlsx, xls.")#It shouldn't reach this point because of previous checks
        except Exception as e:
            print(f"Error loading data: {e}")#This is for other reasons like file not found, permission issues, etc.
            self.current_dataframe = None
            return False
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
        if self.load_data():
            self.showcase_data()
        return self.current_dataframe

# I didn't figure it out how to use this properly.
    # def test_data_module(self): #A simple to test should go here, add later
    #     print("Testing DataModule...")
    # Ans=input("Do you want to test the data module? (y/n): ")
    # if Ans.lower() == 'y':
    #     test_data_module(None)
    # else:
    #     print("Ok, no test for you...")# Ni....")
    #     #if you read that last comment and you are the project reviewer, please ignore it, it's just a joke.
    #     #else, I was talking to you