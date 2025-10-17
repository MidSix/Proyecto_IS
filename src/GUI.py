import sys
from data_module import *
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QFileDialog,
    QTableWidget, QTableWidgetItem, QMessageBox, QHeaderView)
from PyQt5.QtGui import QIcon
# define main window class
class Window(QWidget):
    def __init__(self):
        super().__init__()
        # basic window
        self.setWindowTitle("Linear regression")
        self.setGeometry(200, 200, 300, 300)
        self.setWindowIcon(QIcon("icon.jpg"))
        # interface elements
        # Here the widgets are created but not showed on screen
        self.label  = QLabel("Selecciona un archivo para cargar los datos")
        self.path_display = QLineEdit()
        self.path_display.setPlaceholderText("Ruta del archivo cargado...")
        self.path_display.setReadOnly(True)
        self.button = QPushButton("Cargar archivo")

        # connect button to greet
        self.button.clicked.connect(self.choose_file)

        # Table to display the data
        self.table = QTableWidget()
        self.table.setColumnCount(0)
        self.table.setRowCount(0)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.path_display)
        layout.addWidget(self.button)
        layout.addWidget(self.table)
        self.setLayout(layout)

        # Data module
        self.data = DataModule()

    # the function for the clicked button
    def choose_file(self):
        #QFileDialog.getOpenFileName trigger the file explorer. It returns a tuple with
        #two values. ruta is obviusly the path of the file. "_" is a name convention to say
        #that we don't care about that variable, in this case is the used filter. In this
        #case is "Archivos csv,sqlite,xls (*.csv ...) ..." is pretty likely that we won't use that in future.
        #Sintaxis of the filter:
        #"name_of_the_filter_can_be_any (name.extension)".
        #(*.*)" ->any name and any extension basically all files.  ";;" other filter applied
        #in the next selectable box in explorer.

        ruta, _ = QFileDialog.getOpenFileName( self, "Select a file", "", "Archivos csv, sqlite, xls (*.csv *.sqlite *.db *.xlsx *.xls);" \
            "; csv (*.csv);; " \
            "sqlite (*.sqlite *.db);; excel (*.xlsx *.xls)" )

        if not ruta:
            return

        # Show route
        self.path_display.setText(ruta)

        # Try to load the data
        try:
            data_frame = self.data.main(ruta)
            if data_frame is None or data_frame.empty:
                QMessageBox.warning(self, "Warning", "The file is empty or could not be read correctly.")
                return

            # Show in table
            self.load_table(data_frame)
            QMessageBox.information(self, "Successfull upload", "File uploaded seccessfully.")

        except Exception as e:
            QMessageBox.critical(self, "Error uploading file", f"The file could not be loaded:\n{str(e)}")

    # Fill the table with the dataframe
    def load_table(self, df):
        self.table.clear()
        self.table.setRowCount(len(df.index))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(df.columns.astype(str).tolist())

        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                valor = str(df.iat[i, j])
                self.table.setItem(i, j, QTableWidgetItem(valor))

        # Size and scroll
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()

        # Activate scrolls
        self.table.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)
        self.table.setVerticalScrollMode(QTableWidget.ScrollPerPixel)


def main():
    app = QApplication([sys.argv]) #for now we don't need CLI options, so waiting for a list of parameters
    #is not necessary at the moment. Changable if needed.
    window = Window()
    window.show() # show the window
    sys.exit(app.exec_()) # run the loop

if __name__ == "__main__":
    #This block of code won't execute when this module its importorted from main, so is just for test.
    main()

