# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
from data_module import *

from PyQt5.QtCore import Qt, QAbstractTableModel, QVariant
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QFileDialog,
    QTableView, QMessageBox, QHeaderView, QListWidget, QAbstractItemView, QHBoxLayout,
    QInputDialog, QComboBox
)
from PyQt5.QtGui import QIcon
import qdarkstyle


# ------------------------------------------------------------
# Lightweight Qt model exposing a pandas.DataFrame to QTableView
# ------------------------------------------------------------
class PandasModel(QAbstractTableModel):
    """Lightweight model: the view requests data lazily; no per-cell QTableWidgetItem."""

    def __init__(self, df, parent=None):
        super().__init__(parent)
        self._df = df

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
        if role in (Qt.DisplayRole, Qt.EditRole):
            val = self._df.iat[index.row(), index.column()]
            return "" if pd.isna(val) else str(val)
        return QVariant()

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return QVariant()
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        else:
            return str(section)

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


# ------------------------------------------------------------
# Main Window
# ------------------------------------------------------------
class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Linear Regression - Data Preprocessing")
        self.setWindowIcon(QIcon("icon.jpg"))

        # ----------------- File Upload Section -----------------
        self.label = QLabel("Select a file to download the data")
        self.path_display = QLineEdit()
        self.path_display.setPlaceholderText("Uploaded file path...")
        self.path_display.setReadOnly(True)
        self.button = QPushButton("Upload file")
        self.button.clicked.connect(self.choose_file)

        # ----------------- Table Section -----------------
        self.table = QTableView()
        self.table.setSortingEnabled(True)            # enable now; we will re-enable after setModel too
        self.table.setAlternatingRowColors(True)

        hh = self.table.horizontalHeader()
        vh = self.table.verticalHeader()
        hh.setStretchLastSection(False)
        hh.setSectionResizeMode(QHeaderView.Stretch)  # equal width columns, auto on resize
        hh.setSortIndicatorShown(True)                # show the sort arrow on the header
        vh.setDefaultSectionSize(24)
        vh.setMinimumSectionSize(20)

        # ----------------- Column Selectors -----------------
        self.input_label = QLabel("Select input columns (features):")
        self.input_selector = QListWidget()
        self.input_selector.setSelectionMode(QAbstractItemView.MultiSelection)
        # Make the list compact and scrollable instead of expanding and stealing space
        self.input_selector.setUniformItemSizes(True)
        self.input_selector.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.input_selector.setMaximumHeight(160)     # limit height so the table keeps the focus area
        self.input_selector.setMinimumHeight(120)

        self.output_label = QLabel("Select output column (target):")
        self.output_selector = QListWidget()
        self.output_selector.setSelectionMode(QAbstractItemView.SingleSelection)
        self.output_selector.setUniformItemSizes(True)
        self.output_selector.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.output_selector.setMaximumHeight(160)
        self.output_selector.setMinimumHeight(120)

        self.confirm_button = QPushButton("Confirm selection")
        self.confirm_button.clicked.connect(self.confirm_selection)

        # ----------------- Preprocessing Controls -----------------
        self.preprocess_label = QLabel("Handle missing data:")
        self.strategy_box = QComboBox()
        self.strategy_box.addItems([
            "Delete rows with NaN",
            "Fill with mean",
            "Fill with median",
            "Fill with constant"
        ])
        self.apply_button = QPushButton("Apply preprocessing")
        self.apply_button.clicked.connect(self.handle_missing_data)

        # Initially hidden
        for w in [
            self.input_label, self.input_selector, self.output_label,
            self.output_selector, self.confirm_button, self.preprocess_label,
            self.strategy_box, self.apply_button
        ]:
            w.setVisible(False)

        # Layout setup
        top_controls = QHBoxLayout()
        top_controls.addWidget(self.label)
        top_controls.addWidget(self.path_display)
        top_controls.addWidget(self.button)

        bottom_panel = QVBoxLayout()
        bottom_panel.addWidget(self.input_label)
        bottom_panel.addWidget(self.input_selector)
        bottom_panel.addWidget(self.output_label)
        bottom_panel.addWidget(self.output_selector)
        bottom_panel.addWidget(self.confirm_button)
        bottom_panel.addWidget(self.preprocess_label)
        bottom_panel.addWidget(self.strategy_box)
        bottom_panel.addWidget(self.apply_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_controls)
        main_layout.addWidget(self.table)
        main_layout.addLayout(bottom_panel)
        main_layout.setStretch(1, 8)  # keep the table as the main focus
        self.setLayout(main_layout)

        # Data state
        self.data = DataModule()
        self.current_df = None
        self.selected_inputs = []
        self.selected_output = None

    # ------------------------------------------------------------
    # File selection and loading
    # ------------------------------------------------------------
    def choose_file(self):
        ruta, _ = QFileDialog.getOpenFileName(
            self, "Select a file", "",
            "Files csv, sqlite, xls (*.csv *.sqlite *.db *.xlsx *.xls);; "
            "csv (*.csv);; sqlite (*.sqlite *.db);; excel (*.xlsx *.xls)"
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
            QMessageBox.information(self, "Success", "File uploaded successfully.")
            self.show_column_selectors(df)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"The file could not be loaded:\n{str(e)}")

    # ------------------------------------------------------------
    # Load DataFrame into the model-view table
    # ------------------------------------------------------------
    def load_table(self, df):
        self.current_df = df
        self.table.setUpdatesEnabled(False)
        self.table.setModel(PandasModel(df, self))
        # IMPORTANT: re-enable sorting *after* setting the model
        self.table.setSortingEnabled(True)
        # Optional: show sort indicator default (e.g., first column ascending)
        # self.table.horizontalHeader().setSortIndicator(0, Qt.AscendingOrder)
        self.table.setUpdatesEnabled(True)

    # ------------------------------------------------------------
    # Column selectors
    # ------------------------------------------------------------
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
        selected_output_items = self.output_selector.selectedItems()
        self.selected_output = selected_output_items[0].text() if selected_output_items else None

        if not self.selected_inputs or not self.selected_output:
            QMessageBox.warning(self, "Error", "Please select both input and output columns.")
            return

        QMessageBox.information(
            self, "Selection confirmed",
            f"Inputs: {', '.join(self.selected_inputs)}\nOutput: {self.selected_output}"
        )

        # Show preprocessing controls once columns are selected
        for w in [self.preprocess_label, self.strategy_box, self.apply_button]:
            w.setVisible(True)

    # ------------------------------------------------------------
    # Missing data detection and preprocessing
    # ------------------------------------------------------------
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
                msg = "Missing values filled with column mean (numeric columns only)."

            elif strategy == "Fill with median":
                for col in cols:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col].fillna(df[col].median(), inplace=True)
                msg = "Missing values filled with column median (numeric columns only)."

            elif strategy == "Fill with constant":
                val, ok = QInputDialog.getText(self, "Fill Constant", "Enter constant value:")
                if not ok:
                    return
                for col in cols:
                    df[col].fillna(val, inplace=True)
                msg = f"Missing values filled with constant: {val}"

            else:
                QMessageBox.warning(self, "Error", "Unknown preprocessing strategy.")
                return

            # Refresh table
            model = self.table.model()
            if hasattr(model, "set_dataframe"):
                model.set_dataframe(df)
            else:
                self.load_table(df)

            QMessageBox.information(self, "Preprocessing Completed", msg)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during preprocessing:\n{str(e)}")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    window = Window()
    window.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
