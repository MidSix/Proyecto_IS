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
        if self.dataframe is not None:
            return self.dataframe.empty
        else:
            print("Dataframe is empty. Load data first.")
            return True    
    
    def internal_data_conversion(self)->bool:
        """
        Intenta convertir automáticamente las columnas a tipos apropiados:
        - Números a int/float
        - Fechas a datetime
        - Otros a string
        Opera sobre el DataFrame cargado intentanto primero números, luego fechas, y finalmente strings.
        1. Si la conversión a numérico falla, intenta convertir a fecha.
        2. Si la conversión a fecha también falla, convierte a string.
        """
        if self.is_empty():
            return False
        else:
            last_dtypes = self.dataframe.dtypes
            for col in self.dataframe.columns:
                # Intenta convertir a numérico
                try:# tries to convert to numeric first
                    self.dataframe[col] = pd.to_numeric(self.dataframe[col])
                    continue
                except Exception:
                    pass
                try:#Then tries to convert to datetime
                    self.dataframe[col] = pd.to_datetime(self.dataframe[col])
                    continue
                except Exception:
                    pass
                new_detypes = str(self.dataframe[col].dtype)
                self.dataframe[col] = self.dataframe[col].astype(str)#Finally, converts to string if both previous conversions fail
            print(f"Data conversion completed: {last_dtypes}-->{new_detypes}, Conversions successful: {last_dtypes != self.dataframe.dtypes}")
            return True
    
    def get_summary(self): #Returns a summary of the data, like number of rows, columns, data types, etc.
        if self.is_empty():
            return False
        else:
            return self.dataframe.describe(include='all')
    
    def showcase_data(self)->bool: #Just shows the data in the console, returns True if successful, False if not
        if self.is_empty():
            return False
        else:
            print(self.dataframe)
            return True
