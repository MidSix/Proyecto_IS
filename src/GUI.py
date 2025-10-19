# -*- coding: utf-8 -*-
import sys  # Support for command-line arguments and sys.exit
from data_module import *

# Base for Qt's model–view pattern
from PyQt5.QtCore import Qt, QAbstractTableModel, QVariant

# Qt Widgets
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QFileDialog,
    QTableView, QMessageBox, QHeaderView, QListWidget, QAbstractItemView, QHBoxLayout
)

from PyQt5.QtGui import QIcon  # Window icon
import qdarkstyle  # Dark theme; minimal cost compared to model–view performance gain


# -----------------------------
# Qt Model exposing a pandas.DataFrame to the view
# (functionally equivalent to “Table to display the data” from the old code,
# but using the model–view approach: no QTableWidgetItem created per cell)
# -----------------------------
class PandasModel(QAbstractTableModel):
    """Lightweight model: does not create QTableWidgetItem per cell; the view requests data on demand."""

    def __init__(self, df, parent=None):
        super().__init__(parent)        # Initialize QAbstractTableModel
        self._df = df                   # Keep a reference to the DataFrame

    # Number of rows
    def rowCount(self, parent=None):
        # If parent is valid, it means the model is hierarchical (not our case)
        if parent and parent.isValid():
            return 0
        return len(self._df.index)

    # Number of columns
    def columnCount(self, parent=None):
        if parent and parent.isValid():
            return 0
        return len(self._df.columns)

    # Return the data for a specific cell and "role" (DisplayRole = visible text)
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return QVariant()
        if role in (Qt.DisplayRole, Qt.EditRole):
            # Fast access via iat; cast to str to avoid expensive formatting
            val = self._df.iat[index.row(), index.column()]
            return "" if val is None else str(val)
        # Optional: right-align numbers if desired
        # if role == Qt.TextAlignmentRole:
        #     return Qt.AlignRight | Qt.AlignVCenter
        return QVariant()

    # Column and row headers
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return QVariant()
        if orientation == Qt.Horizontal:
            # Column names from the DataFrame
            return str(self._df.columns[section])
        else:
            # Show row index (or "" for slightly faster rendering)
            # return str(self._df.index[section])
            return str(section)

    # Optional: sorting when clicking on headers
    def sort(self, column, order):
        # Temporarily block signals to avoid multiple redraws
        self.layoutAboutToBeChanged.emit()
        ascending = (order == Qt.AscendingOrder)
        # Sort by selected column; inplace to avoid duplicating memory
        self._df.sort_values(by=self._df.columns[column], ascending=ascending, inplace=True, kind="mergesort")
        self._df.reset_index(drop=True, inplace=True)
        # Notify the view that the layout has changed
        self.layoutChanged.emit()


# define main window class
class Window(QWidget):
    def __init__(self):
        super().__init__()
        # Basic window setup
        self.setWindowTitle("Linear regression")
        self.setWindowIcon(QIcon("icon.jpg"))

        # Interface elements
        # Here widgets are created but not yet displayed
        self.label  = QLabel("Select a file to download the data")
        self.path_display = QLineEdit()
        self.path_display.setPlaceholderText("Uploaded file path...")
        self.path_display.setReadOnly(True)
        self.button = QPushButton("Upload file")

        # Connect button to file selection
        self.button.clicked.connect(self.choose_file)

        # Table to display the data (now as QTableView using model–view, much more efficient)
        self.table = QTableView()
        self.table.setSortingEnabled(True)            # Allow sorting by clicking headers
        self.table.setAlternatingRowColors(True)      # Better readability

        # Header configuration (replaces old ResizeToContents behavior)
        hh = self.table.horizontalHeader()            # Horizontal header
        vh = self.table.verticalHeader()              # Vertical header

        # >>> NEW: all columns have equal width, very low-cost
        hh.setStretchLastSection(False)
        hh.setSectionResizeMode(QHeaderView.Stretch)  # Distribute visible space equally among all columns

        # Fixed row height (equivalent to old ResizeToContents for vertical, but without the overhead)
        vh.setDefaultSectionSize(24)
        vh.setMinimumSectionSize(20)
        # If you don't need to show the index, this saves some width and repainting
        # vh.setVisible(False)

        # Column selectors
        self.input_label = QLabel("Selecciona columnas de entrada (features):")
        self.input_selector = QListWidget()
        self.input_selector.setSelectionMode(QAbstractItemView.MultiSelection)
        self.output_label = QLabel("Selecciona la columna de salida (target):")
        self.output_selector = QListWidget()
        self.output_selector.setSelectionMode(QAbstractItemView.SingleSelection)
        self.confirm_button = QPushButton("Confirmar selección")
        self.confirm_button.clicked.connect(self.confirm_selection)

        # Hide selectors until a DataFrame is loaded
        for i in [self.input_label, self.input_selector,
                  self.output_label, self.output_selector,
                  self.confirm_button]:
            i.setVisible(False)

        # Layout for file upload controls
        top_controls = QHBoxLayout()
        top_controls.addWidget(self.label)
        top_controls.addWidget(self.path_display)
        top_controls.addWidget(self.button)

        # Layout for selectors
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

        # Adjust table stretch ratios
        main_layout.setStretch(0, 0)
        main_layout.setStretch(1, 8)
        main_layout.setStretch(2, 2)

        self.setLayout(main_layout)

        # Data module
        self.data = DataModule()
        self.current_df = None
        self.selected_inputs = []
        self.selected_output = None

    # Function called when clicking the upload button
    def choose_file(self):
        # QFileDialog.getOpenFileName triggers the OS file explorer.
        # It returns a tuple with two values:
        # - ruta: the selected file path
        # - "_": a throwaway variable for the selected filter (we ignore it)
        #
        # Filter syntax:
        # "filter_name (name.extension)"
        # Example: (*.*) means all files.
        # Filters can be combined using ";;" to make multiple filter options.
        ruta, _ = QFileDialog.getOpenFileName(
            self, "Select a file", "",
            "Files csv, sqlite, xls (*.csv *.sqlite *.db *.xlsx *.xls);; "
            "csv (*.csv);; sqlite (*.sqlite *.db);; excel (*.xlsx *.xls)"
        )

        if not ruta:
            return  # User canceled

        # Show file path in the QLineEdit
        self.path_display.setText(ruta)

        # Try to load the data
        try:
            # data_module.main(ruta) returns (data_frame, error_message)
            data_frame, error_message = self.data.main(ruta)
            if data_frame is None:
                # If file has missing header/metadata or is completely empty,
                # DataModule handles the error internally and returns None as DataFrame.
                QMessageBox.warning(self, "Warning", error_message)
                return

            # Show data in the table (equivalent to “Fill the table with the dataframe” in old code)
            self.load_table(data_frame)
            QMessageBox.information(self, "Successful upload", "File uploaded successfully.")

            # Show column selectors
            self.show_column_selectors(data_frame)

        except Exception as e:
            QMessageBox.critical(self, "Error uploading file", f"The file could not be loaded:\n{str(e)}")

    # Fill the table with the dataframe (now connects DF to the view via model)
    def load_table(self, df):
        # In the old code: clear + setRow/ColCount + setHorizontalHeaderLabels + nested loops with setItem
        # In the new version: assign a model that exposes the DF; the view requests data lazily.
        self.current_df = df

        # Temporarily disable updates to avoid redraws while setting the model
        self.table.setUpdatesEnabled(False)

        # Assign the model; from now on, the view requests data on demand
        model = PandasModel(df, self)
        self.table.setModel(model)

        # (Optional) Adjust columns to fit content once; costly if many columns
        # self.table.resizeColumnsToContents()

        # Re-enable updates
        self.table.setUpdatesEnabled(True)

        # Note: QTableView already manages scroll performance efficiently, so we don't need ScrollPerPixel here.

    # Show column selectors
    def show_column_selectors(self, df):
        columns = df.columns.astype(str).tolist()

        # Clear previous selectors
        self.input_selector.clear()
        self.output_selector.clear()

        # Fill with column names
        self.input_selector.addItems(columns)
        self.output_selector.addItems(columns)

        # Make visible
        for i in [self.input_label, self.input_selector, self.output_label, self.output_selector, self.confirm_button]:
            i.setVisible(True)

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

        QMessageBox.information(
            self, "Selection confirmed",
            f"Inputs: {', '.join(self.selected_inputs)}\nOutput: {self.selected_output}"
        )


# -----------------------------
# Entry point
# -----------------------------
def main():
    app = QApplication(sys.argv)  # For now we don't need CLI options, so waiting for a list of parameters
    # Not necessary at the moment. Changeable if needed.

    # Apply dark theme (as in the old code)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    window = Window()
    window.showMaximized()  # Show the main window
    sys.exit(app.exec_())   # Run the event loop


if __name__ == "__main__":
    # This block will not execute when the module is imported; it's only for standalone testing.
    main()
