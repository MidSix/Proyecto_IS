import sqlite3
import pandas as pd
import os # To handle file paths and extensions that may vary by OS(linux, windows, macOS)

#This data module must support SQlite, CSV, and Excel formats
#I decided to use pandas for the general dataframe handling, because it provides a matrix-like structure and easy data manipulation
class DataModule:
    #When initializing, user must provide the file path to the data source
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path) if file_path else None
        self.file_type = os.path.splitext(self.file_name)[-1][1:].lower() if self.file_name else None
        self.dataframe = None #Remember this is a pandas dataframe, it is a matrix made internally as a 2D array
        self.connection = None #This is an object special for SQLite that handles the connection to the database
    
    def load_data(self):
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
    
    def is_empty(self)->bool: #Returns True if the dataframe is empty, False if it has data
        if self.dataframe is None:
            print("Dataframe is has not been loaded. Load data first.")
        return self.dataframe.empty #We can use pandas built-in function to check if the dataframe is empty
    
    def internal_data_conversion(self) -> bool:
        if self.is_empty():
            return False
        last_dtypes = self.dataframe.dtypes
        for col in self.dataframe.columns:
            try:
                self.dataframe[col] = pd.to_numeric(self.dataframe[col])
                continue
            except Exception:
                pass
            try:
                self.dataframe[col] = pd.to_datetime(self.dataframe[col])
                continue
            except Exception:
                pass
            self.dataframe[col] = self.dataframe[col].astype(str)
        print("Data types before conversion:")
        print(last_dtypes)
        print("Data types after conversion:")
        print(self.dataframe.dtypes)
        changes = sum(last_dtypes != self.dataframe.dtypes)# Count how many columns changed type
        print(f"Conversion completed, ({changes}) changes were made.")
        
        if changes == 0:
            print(f"No changes in data types.")
        if changes < 0:
            print(f"Error in conversion, negative number of changes ({changes}) detected.")#Should never happen, but you never know
            return False
        return True
    
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
    path=input("Enter the file path to load data (SQLite, CSV, Excel): ")
    data_module1 = DataModule(path)
    data_module1.load_data()
    data_module1.internal_data_conversion()
    print(data_module1.get_summary())
    data_module1.showcase_data()

if __name__ == "__main__":
    Ans=input("Do you want to test the data module? (y/n): ")
    if Ans.lower() == 'y':
        test_data_module(None)
    else:
        print("Ok, no test for you.")

