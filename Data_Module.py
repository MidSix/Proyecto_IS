import sqlite3
import pandas as pd#This is for excel support and CSV
#This data module must support SQlite, CSV, and Excel formats
class DataModule:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.file_name = file_path.split('/')[-1] if file_path else None
        self.file_type = self.file_name.split('.')[-1].lower() if self.file_name else None
        self.dataframe = None
    
    def load_data(self):
        try:
            if self.file_type == '.sqlite' or '.db':
                self.connection = sqlite3.connect(self.file_path)
            elif self.file_type == 'csv':
                self.dataframe = pd.read_csv(self.file_path)
            elif self.file_type == 'xlsx' or 'xls':
                self.dataframe = pd.read_excel(self.file_path)
            else:
                raise ValueError("Unsupported file type. Supported types are: sqlite, csv, excel.")
        except Exception as e:
            print(f"Error loading data: {e}")