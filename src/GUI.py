# -*- coding: utf-8 -*-
import sys
import pandas as pd
from data_module import *

from PyQt5.QtCore import Qt, QAbstractTableModel, QVariant
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QFileDialog,
    QTableView, QMessageBox, QHeaderView, QListWidget, QAbstractItemView, QHBoxLayout,
    QComboBox
)
from PyQt5.QtGui import QIcon, QBrush, QColor
import qdarkstyle
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
from data_split import DataSplitter,DataSplitError


# Lightweight Qt model exposing a pandas.DataFrame to QTableView
class PandasModel(QAbstractTableModel):
    """Lightweight model: the view requests data lazily; no per-cell QTableWidgetItem."""

    def __init__(self, df, parent=None):
        super().__init__(parent)
        self._df = df
        self.highlight_cols = set()            # Here we are going to store
        #our highlighted columns, a set to avoid duplicates.
        self.highlight_color = QColor("#290908")  # just the color

    def rowCount(self, parent=None):
        if parent and parent.isValid():
            return 0
        return len(self._df.index)

    def columnCount(self, parent=None):
        if parent and parent.isValid():
            return 0
        return len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return QVariant()

        row = index.row()
        col = index.column()
        val = self._df.iat[row, col]
        col_name = self._df.columns[col]

        if role in (Qt.DisplayRole, Qt.EditRole):
            return "" if pd.isna(val) else str(val)

        if role == Qt.BackgroundRole:
            # Si la columna está marcada, pintamos TODA la columna
            if col_name in self.highlight_cols:
                return QBrush(self.highlight_color)

        return QVariant()

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._df.columns[section])
            else:
                return str(section)

        if role == Qt.BackgroundRole and orientation == Qt.Horizontal:
            col_name = self._df.columns[section]
            if col_name in self.highlight_cols:
                return QBrush(self.highlight_color)

        return QVariant()

    def set_highlight_by_missing(self, columns):
        """Highlight columns with at least one NaN value"""
        columns = columns or []
        # Looks complicated at first glance but it's not a big deal
        #First of all is a set the avoid selecting duplicates columns(it could
        #happen when you select the same column in input-output, it's a trivial
        #regression but you never know what the user does). This is a set
        #comprehension. first we iterate over columns. Each c is a column
        #then in self._df[c] we select the column, .isna() returns a dataframe
        #replacing each element of the column(not in-place obviusly
        #not replacing the elements of the original column of the dataframe)
        #with boolean values, True if it's nan, False otherwise. .any() returns
        #a boolean if the new pandas has at least one True, that means, if has
        #at least one NaN element in that column. So we add c, that column to
        #the set.
        self.highlight_cols = {c for c in columns if c in self._df.columns and self._df[c].isna().any()}

        # Repintar celdas y encabezados
        if self.rowCount() and self.columnCount():
            top_left = self.index(0, 0)
            bottom_right = self.index(self.rowCount() - 1, self.columnCount() - 1)
            self.dataChanged.emit(top_left, bottom_right, [Qt.BackgroundRole])
            self.headerDataChanged.emit(Qt.Horizontal, 0, self.columnCount() - 1)

    def sort(self, column, order):
        """Called by the view when header sorting is enabled."""
        self.layoutAboutToBeChanged.emit()
        ascending = (order == Qt.AscendingOrder)
        # Stable sort keeps relative order of ties (nicer UX when toggling)
        self._df.sort_values(
            by=self._df.columns[column],
            ascending=ascending,
            inplace=True,
            kind="mergesort"
        )
        self._df.reset_index(drop=True, inplace=True)
        self.layoutChanged.emit()

    def set_dataframe(self, df):
        """Refresh the entire model after preprocessing."""
        self.beginResetModel()
        self._df = df
        self.endResetModel()
# Main Window
class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Linear Regression - Data Preprocessing")
        self.setWindowIcon(QIcon("icon.jpg"))

        # ----------------- Bottom Panel Setup -----------------
        self.bottom_panel_widget = QWidget()

        # ----------------- File Load Section -----------------
        self.label = QLabel("Path")
        self.path_display = QLineEdit()
        self.path_display.setPlaceholderText("Select a file to load the data")
        self.path_display.setReadOnly(True)
        self.btn_open_file = QPushButton("Open File")
        self.btn_open_file.clicked.connect(self.choose_file)

        # ----------------- Table Section -----------------
        self.table = QTableView()
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)

        hh = self.table.horizontalHeader()
        vh = self.table.verticalHeader()
        hh.setStretchLastSection(False)
        hh.setSectionResizeMode(QHeaderView.Stretch)
        hh.setSortIndicatorShown(True)
        vh.setDefaultSectionSize(24)
        vh.setMinimumSectionSize(20)

        # ----------------- Column Selectors -----------------
        self.container_selector_widget = QWidget()
        self.container_preprocess_widget = QWidget()
        self.input_label = QLabel("Select input columns (features)")
        self.input_selector = QListWidget()
        self.input_selector.setSelectionMode(QAbstractItemView.MultiSelection)
        # Make the list compact and scrollable instead of expanding and stealing space
        self.input_selector.setUniformItemSizes(True)
        self.input_selector.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.output_label = QLabel("Select output column (target)")
        self.output_selector = QComboBox()

        self.confirm_button = QPushButton("Confirm selection")
        self.confirm_button.clicked.connect(self.confirm_selection)

        # ----------------- Preprocessing Controls -----------------
        self.preprocess_label = QLabel("Handle missing data")
        self.strategy_box = QComboBox()
        self.strategy_box.addItems([
            "Delete rows with NaN",
            "Fill with mean",
            "Fill with median",
            "Fill with constant"
        ])
        self.strategy_box.currentTextChanged.connect(self.strategy_box_changed)
        self.apply_button = QPushButton("Apply preprocessing")
        self.apply_button.clicked.connect(self.handle_missing_data)
        self.constant_name_edit = QLineEdit()
        self.constant_name_edit.setPlaceholderText("Constant name")

        # ----------------- Split Button -----------------
        self.split_button = QPushButton("Split data into training/test")
        self.split_button.clicked.connect(self.open_split_window)
        self.split_button.setVisible(False)

        # Initially hidden
        for w in [
            self.input_label, self.input_selector, self.output_label,
            self.output_selector, self.confirm_button, self.preprocess_label,
            self.strategy_box, self.apply_button, self.constant_name_edit,
            self.split_button
        ]:
            w.setVisible(False)

        # ----------------- Layout setup -----------------
        top_controls = QHBoxLayout()
        top_controls.addWidget(self.label)
        top_controls.addWidget(self.path_display)
        top_controls.addWidget(self.btn_open_file)

        #Bottom_layout:
        bottom_panel = QHBoxLayout()

        # Creation of vertical views and stack the widgets on it.
        input_col = QVBoxLayout()
        input_col.addWidget(self.input_label)
        input_col.addWidget(self.input_selector)

        output_col = QVBoxLayout()
        output_col.addWidget(self.output_label)
        output_col.addWidget(self.output_selector)
        output_col.addWidget(self.confirm_button)

        preprocess_col = QVBoxLayout()
        preprocess_col.addWidget(self.preprocess_label)
        preprocess_col.addWidget(self.strategy_box)
        preprocess_col.addWidget(self.constant_name_edit)
        preprocess_col.addWidget(self.apply_button)
        preprocess_col.addWidget(self.split_button)

        # Group the layouts into another layout but this time a horizontal one
        container_selector_layout = QHBoxLayout()
        container_selector_layout.addLayout(input_col)
        container_selector_layout.addLayout(output_col)

        #Envolpe the layout into a widget, this is for setting maximum width
        self.container_selector_widget.setLayout(container_selector_layout)
        self.container_preprocess_widget.setLayout(preprocess_col)

        bottom_panel.addWidget(self.container_selector_widget, alignment=Qt.AlignLeft)
        bottom_panel.addWidget(self.container_preprocess_widget, alignment=Qt.AlignRight)

        self.bottom_panel_widget.setLayout(bottom_panel)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_controls)
        main_layout.addWidget(self.table)
        main_layout.addWidget(self.bottom_panel_widget)
        main_layout.setStretch(1, 8)
        self.setLayout(main_layout)

        # ----------------- Data State -----------------
        self.data = DataModule()
        self.current_df = None
        self.selected_inputs = []
        self.selected_output = None

    # ------------------- Methods ------------------------------------------------
    def choose_file(self):
        ruta, _ = QFileDialog.getOpenFileName(
            self, "Select a file", "",
            "Files csv, sqlite, xls (*.csv *.sqlite *.db *.xlsx *.xls);; "
            "csv (.csv);; sqlite (.sqlite .db);; excel (.xlsx *.xls)"
        )
        if not ruta:
            return
        self.path_display.setText(ruta)

        try:
            df, error_message = self.data.main(ruta)
            if df is None:
                QMessageBox.warning(self, "Warning", error_message)
                return

            self.load_table(df)
            QMessageBox.information(self, "Success", "File loaded successfully.")
            self.container_preprocess_widget.hide()
            self.show_column_selectors(df)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"The file could not be loaded:\n{str(e)}")

    def load_table(self, df):
        self.current_df = df
        self.table.setUpdatesEnabled(False)
        self.table.setModel(PandasModel(df, self))
        # IMPORTANT: re-enable sorting after setting the model
        self.table.setSortingEnabled(True)
        # Optional: show sort indicator default (e.g., first column ascending)
        # self.table.horizontalHeader().setSortIndicator(0, Qt.AscendingOrder)
        self.table.setUpdatesEnabled(True)

    # Column selectors
    def show_column_selectors(self, df):
        columns = df.columns.astype(str).tolist()
        self.input_selector.clear()
        self.output_selector.clear()
        self.input_selector.addItems(columns)
        self.output_selector.addItems(columns)

        for w in [
            self.input_label, self.input_selector, self.output_label,
            self.output_selector, self.confirm_button
        ]:
            w.setVisible(True)

    def confirm_selection(self):
        self.selected_inputs = [i.text() for i in self.input_selector.selectedItems()]
        self.selected_output = self.output_selector.currentText()

        if not self.selected_inputs or not self.selected_output:
            QMessageBox.warning(self, "Error", "Please select both input and output columns.")
            return

        QMessageBox.information(
            self, "Selection confirmed",
            f"Inputs: {', '.join(self.selected_inputs)}\nOutput: {self.selected_output}"
        )

        # Show preprocessing controls once columns are selected
        for w in [self.preprocess_label, self.strategy_box, self.apply_button, self.container_preprocess_widget]:
            w.setVisible(True)

        # --- Call the set_highlight_by_missing() function with the selected
        #cols
        cols = self.selected_inputs + [self.selected_output]
        model = self.table.model()
        if hasattr(model, "set_highlight_by_missing"):
            model.set_highlight_by_missing(cols)

    # Missing data detection and preprocessing
    def handle_missing_data(self):
        if self.current_df is None:
            QMessageBox.warning(self, "Error", "No dataset loaded.")
            return

        cols = self.selected_inputs + [self.selected_output]
        df = self.current_df
        missing_counts = df[cols].isna().sum()
        total_missing = int(missing_counts.sum())

        if total_missing == 0:
            QMessageBox.information(self, "No Missing Data", "No missing values found in the selected columns.")
            return

        detail = "\n".join([f"{col}: {int(cnt)}" for col, cnt in missing_counts.items() if cnt > 0])
        QMessageBox.warning(self, "Missing Data Detected", f"Total NaN values: {total_missing}\n\n{detail}")

        strategy = self.strategy_box.currentText()

        try:
            if strategy == "Delete rows with NaN":
                before = len(df)
                df.dropna(subset=cols, inplace=True)
                removed = before - len(df)
                msg = f"Rows removed: {removed}"

            elif strategy == "Fill with mean":
                for col in cols:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col].fillna(df[col].mean(), inplace=True)
                msg = "Missing values filled with column mean."

            elif strategy == "Fill with median":
                for col in cols:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col].fillna(df[col].median(), inplace=True)
                msg = "Missing values filled with column median."

            elif strategy == "Fill with constant":
                written_cte = self.constant_name_edit.text()
                if written_cte:
                    for col in cols:
                        df[col].fillna(written_cte, inplace=True)
                    msg = f"Missing values filled with constant: {written_cte}"
                else:
                    QMessageBox.warning(self, "Error", "Please provide a constant value.")
                    return
            else:
                QMessageBox.warning(self, "Error", "Unknown preprocessing strategy.")
                return

            # Refresh table
            model = self.table.model()
            if hasattr(model, "set_dataframe"):
                model.set_dataframe(df)
            else:
                self.load_table(df)

            if hasattr(model, "set_highlight_by_missing"):
                model.set_highlight_by_missing(cols)

            QMessageBox.information(self, "Preprocessing Completed", msg)

            # Show split button after successful preprocessing
            self.split_button.setVisible(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during preprocessing:\n{str(e)}")

    # ----------------- TRAIN/TEST -----------------
    def open_split_window(self):
        if self.current_df is None or self.current_df.empty:
            QMessageBox.warning(self, "Error", "There isn't any data avaliable to split.")
            return

        self.split_window = SplitWidget(self.current_df)
        self.split_window.show()


    def dynamic_size(self):
        #Ok, this is a function that establish the height of the widgets
        #shown below, this solves all the inconsistencies with height that we
        #discussed previously. The problem was effectively fixed height using
        #pixels, this naturally changes how the program looks with different
        #resolutions but this is not a problem anymore, the height is measured
        #using a percentage of your window, so with 4k resolution, HD resolution
        #potato resolution, all take the same ammount of screen.

        #In summary, these numbers are the percentage of the window
        selector_container_size_width = 0.75
        preprocess_container_size = 0.15
        bottom_panel_widget_mainimum_height = 0.2
        bottom_panel_widget_minimum_height = 0.2
        h = max(self.height(), 1) #When resizing could be less than 1 in some
        w = max(self.width(), 1)
        #cases, to avoid issues we establish this handler.
        self.container_selector_widget.setFixedWidth(int(w * selector_container_size_width))

        self.apply_button.setFixedWidth(int(w * preprocess_container_size))

        self.bottom_panel_widget.setMinimumHeight(int(h * bottom_panel_widget_minimum_height))
        self.bottom_panel_widget.setMaximumHeight(int(h * bottom_panel_widget_mainimum_height))

    def strategy_box_changed(self, option_selected) -> None:
        is_cte = option_selected == "Fill with constant"
        self.constant_name_edit.setVisible(is_cte)
        if is_cte:
            self.constant_name_edit.setFocus()
        else:
            self.constant_name_edit.clear()
        return None
    #--------------------------------------------------------
    #ShowEvent, resizeEvent: are methods from Qwidget, here we are applying
    #Polymorphism, changing the behaviour of the superclass methods in our
    #subclass Window. By calling our own
    #dynamic_size when they execute. So when we resize and show windows
    #we automatically are applying dynamic_size
    def showEvent(self, event):
        super().showEvent(event)
        self.dynamic_size()

    def resizeEvent(self, event):
        self.dynamic_size()
        return super().resizeEvent(event)
    #--------------------------------------------------------
class SplitWidget(QWidget):
    def __init__(self, df_preprocessed, parent=None):
        super().__init__(parent)
        self.df = df_preprocessed
        self.splitter = DataSplitter()
        self.train_df = None
        self.test_df = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Entradas
        self.test_edit = QLineEdit("0.2")
        self.seed_edit = QLineEdit("42")
        btn = QPushButton("Split data")
        btn.clicked.connect(self.on_split)

        # Tablas para mostrar los datos
        self.train_label = QLabel("Training set:")
        self.train_table = QTableView()
        self.test_label = QLabel("Test set:")
        self.test_table = QTableView()

        # Layout
        layout.addWidget(QLabel("Test fraction (e.g. 0.2):"))
        layout.addWidget(self.test_edit)
        layout.addWidget(QLabel("Seed (reproducibility):"))
        layout.addWidget(self.seed_edit)
        layout.addWidget(btn)
        layout.addWidget(self.train_label)
        layout.addWidget(self.train_table)
        layout.addWidget(self.test_label)
        layout.addWidget(self.test_table)

    def on_split(self):
        try:
            test_frac = float(self.test_edit.text())
            seed = int(self.seed_edit.text())

            self.train_df, self.test_df = self.splitter.split(
                self.df, test_size=test_frac, random_seed=seed
            )
            meta = self.splitter.get_meta()

            # Mostrar tablas
            self.train_table.setModel(PandasModel(self.train_df))
            self.test_table.setModel(PandasModel(self.test_df))

            # Mensaje
            QMessageBox.information(
                self,
                "Split successfully completed",
                f"Division was correctly done.\n\n"
                f"Training set: {meta['n_train']} rows\n"
                f"Test set: {meta['n_test']} rows"
            )

        except (ValueError, DataSplitError) as e:
            QMessageBox.critical(self, "Error", str(e))

# Entry point
def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    window = Window()
    window.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
