import sqlite3
import pandas as pd
import os # To handle file paths and extensions that may vary by OS(linux, windows, macOS)

#This data module must support SQlite, CSV, and Excel formats
#I decided to use pandas for the general dataframe handling, because it provides a matrix-like structure and easy data manipulation
class DataModule:
    #When initializing, user must provide the file path to the data source
    def __init__(self, file_path):
        self.file_path = file_path if file_path else None #just for consistency. If file_path is void then it stores None
        self.file_name = os.path.basename(file_path) if file_path else None
        self.file_type = os.path.splitext(self.file_name)[-1][1:].lower() if self.file_name else None
        self.dataframe = None #Remember this is a pandas dataframe, it is a matrix made internally as a 2D array
        self.connection = None #This is an object special for SQLite that handles the connection to the database

    def load_data(self) -> bool:
        try:
            if self.file_type in ['sqlite', 'db']:
                self.connection = sqlite3.connect(self.file_path)
                # Fetch the first table name
                cursor = self.connection.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()

                if not tables:
                    raise ValueError("No se encontró ninguna tabla en la base de datos sqlite.")

                table_name = tables[0][0]
                self.dataframe = pd.read_sql_query(f"SELECT * FROM {table_name}", self.connection)

            elif self.file_type == 'csv':
                self.dataframe = pd.read_csv(self.file_path, header=0)
                if self.dataframe.empty:
                    raise ValueError("El archivo CSV está vacío o no se pudo leer correctamente.")

            elif self.file_type in ['xlsx', 'xls']:
                self.dataframe = pd.read_excel(self.file_path, header=0)
                if self.dataframe.empty:
                    raise ValueError("El archivo Excel está vacío o no se pudo leer correctamente.")
            else:
                raise ValueError("Unsupported file type. Supported types are: sqlite, db, csv, xlsx, xls.")
        except Exception as e:
            print(f"Error loading data: {e}")#This is for other reasons like file not found, permission issues, etc.
            self.dataframe = None
            return self.dataframe
        return True

    def is_empty(self)->bool: #Returns True if the dataframe is empty, False if it has data
        if self.dataframe is None:
            print("Dataframe has not been loaded. Load data first.")
            return True #This line is fundamental. If path or file are incorrect then self.dataframe is None.
        #and a None type is not a pandas type, this means we can't use the pandas built-in to check if it's empty
        #with our current implementation. To not change it I just add this return xd.
        return self.dataframe.empty #We can use pandas built-in function to check if the dataframe is empty

    def get_summary(self):
        if self.is_empty():
            print("No data to summarize.")
            return None
        return self.dataframe.describe(include='all')

    def showcase_data(self)->bool: #Just shows the data in the console, returns True if successful, False if not
        if self.is_empty():
            print("Dataframe is empty. Load data first.")
            return False
        else:
            print(self.dataframe)
            return True

def test_data_module(self):# Is it correct to
    if self.load_data(): #This conditional just avoid executing the next lines of code if the data didn't load
    #succesfully because it return the dataframe. If it's not empty, this conditional is True, if it's None then False.
        print(self.get_summary())
        self.showcase_data()

if __name__ == "__main__":

    Ans=input("Do you want to test the data module? (y/n): ")
    if Ans.lower() == 'y':
        test_data_module(None)
    else:
        print("Ok, no test for you.")

