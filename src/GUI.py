import sys
from data_module import *
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QFileDialog,
    QTableWidget, QTableWidgetItem, QMessageBox, QHeaderView, QListWidget, QAbstractItemView, QHBoxLayout)
from PyQt5.QtGui import QIcon
import qdarkstyle

# define main window class
class Window(QWidget):
    def __init__(self):
        super().__init__()
        # basic window
        self.setWindowTitle("Linear regression")
        self.setWindowIcon(QIcon("icon.jpg"))
        # interface elements
        # Here the widgets are created but not showed on screen
        self.label  = QLabel("Select a file to dowload the data")
        self.path_display = QLineEdit()
        self.path_display.setPlaceholderText("Uploaded file path...")
        self.path_display.setReadOnly(True)
        self.button = QPushButton("Upload file")

        # connect button to greet
        self.button.clicked.connect(self.choose_file)

        # Table to display the data
        self.table = QTableWidget()
        self.table.setColumnCount(0)
        self.table.setRowCount(0)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        # Column selector
        self.input_label = QLabel("Selecciona columnas de entrada (features):")
        self.input_selector = QListWidget()
        self.input_selector.setSelectionMode(QAbstractItemView.MultiSelection)
        self.output_label = QLabel("Selecciona la columna de salida (target):")
        self.output_selector = QListWidget()
        self.output_selector.setSelectionMode(QAbstractItemView.SingleSelection)
        self.confirm_button = QPushButton("Confirmar selección")
        self.confirm_button.clicked.connect(self.confirm_selection)

        # Hide initially
        for i in [self.input_label, self.input_selector,
                  self.output_label, self.output_selector,
                  self.confirm_button]: i.setVisible(False)

        # Layout to upload files
        top_controls = QHBoxLayout()
        top_controls.addWidget(self.label)
        top_controls.addWidget(self.path_display)
        top_controls.addWidget(self.button)

        # Layout selectors
        bottom_panel = QVBoxLayout()
        bottom_panel.addWidget(self.input_label)
        bottom_panel.addWidget(self.input_selector)
        bottom_panel.addWidget(self.output_label)
        bottom_panel.addWidget(self.output_selector)
        bottom_panel.addWidget(self.confirm_button)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(top_controls)
        main_layout.addWidget(self.table)
        main_layout.addLayout(bottom_panel)

        # Adjust table
        main_layout.setStretch(0, 0)
        main_layout.setStretch(1, 8)
        main_layout.setStretch(2, 2)

        self.setLayout(main_layout)

        # Data module
        self.data = DataModule()
        self.current_df = None
        self.selected_inputs = []
        self.selected_output = None

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

        rute, _ = QFileDialog.getOpenFileName( self, "Select a file", "", "Files csv, sqlite, xls (*.csv *.sqlite *.db *.xlsx *.xls);" \
            "; csv (*.csv);; " \
            "sqlite (*.sqlite *.db);; excel (*.xlsx *.xls)" )

        if not rute:
            return

        # Show route
        self.path_display.setText(rute)

        # Try to load the data
        try:
            data_frame, error_message = self.data.main(rute)
            if data_frame is None: #wether the file has header/metadata or its
                #completely empty, both situations raise an error which is
                #handled in data module, returning None as dataframe.
                #We will never use the second comprobation so we can relly just
                #on comprobating if its None or not.
                QMessageBox.warning(self, "Warning", error_message)
                return

            # Show in table
            self.load_table(data_frame)
            QMessageBox.information(self, "Successfull upload", "File uploaded seccessfully.")

            # Show column selector
            self.show_column_selectors(data_frame)

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
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        # Activate scrolls
        self.table.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)
        self.table.setVerticalScrollMode(QTableWidget.ScrollPerPixel)

    # Show selectors
    def show_column_selectors(self, df):
        columns = df.columns.astype(str).tolist()

        # Clear previous selectors
        self.input_selector.clear()
        self.output_selector.clear()

        # Fill with column names
        self.input_selector.addItems(columns)
        self.output_selector.addItems(columns)

        # Make visible
        for i in [self.input_label, self.input_selector, self.output_label, self.output_selector, self.confirm_button]: i.setVisible(True)

    # Confirm selection
    def confirm_selection(self):
        self.selected_inputs = [i.text() for i in self.input_selector.selectedItems()]
        selected_output_items = self.output_selector.selectedItems()
        self.selected_output = selected_output_items[0].text() if selected_output_items else None

        if not self.selected_inputs and not self.selected_output:
            QMessageBox.warning(self, "Error", "You must select at least one input column and one output column.")
            return
        if not self.selected_inputs:
            QMessageBox.warning(self, "Error", "You must select at least one input column.")
            return
        if not self.selected_output:
            QMessageBox.warning(self, "Error", "You must select one output column.")
            return

        QMessageBox.information(self, "Selection confirmed",
            f"Inputs: {', '.join(self.selected_inputs)}\nOutput: {self.selected_output}")

def main():
    app = QApplication(sys.argv) #for now we don't need CLI options, so waiting for a list of parameters
    #is not necessary at the moment. Changable if needed.
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window = Window()
    window.showMaximized() # show the window
    sys.exit(app.exec_()) # run the loop

if __name__ == "__main__":
    #This block of code won't execute when this module its importorted from main, so is just for test.
    main()

