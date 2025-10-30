# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd
import qdarkstyle
from data_module import *

from PyQt5.QtCore import Qt, QAbstractTableModel, QVariant
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QFileDialog,
    QTableView, QMessageBox, QHeaderView, QListWidget, QAbstractItemView, QHBoxLayout,
    QComboBox, QSizePolicy, QStackedWidget
)
from PyQt5.QtGui import QIcon, QBrush, QColor
from data_split import DataSplitter,DataSplitError
#from AppKit import NSApplication, NSImage

#All persistense .py files(stored in the computer) have a global
#variable __file__ that you can see printing the globals() namespace.
#This variable contains the path to the file. os.path.dirname(__file__)
#gets the directory path in which is stored the file. Because the icon
#is in the same directory(folder) we just apply a join and get the path
#to our ICON no matter where our workspace is.
ICON = os.path.join(os.path.dirname(__file__), "icon.jpg") #Global inmutable cte

# Lightweight Qt model exposing a pandas.DataFrame to QTableView
#This follows the MVC design pattern: model, view, controller.
#So this class works as the model for the view in the table.
#That means, here, inside this class lives the data shown in the table.
#remember that an abstract class can't be instanciated directly
#because it doesn't have the methods with the tag
#@abstractmethod implemented. Just defines
#some sort of contract in which the subclass of the abstractclass must
#implement those methods to be instanciated. Those methods are the ones
#we are doing polymorphism here: rowCount,columnCount,data
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
        #First of all is a set that avoid selecting duplicates columns(it could
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
        if self.highlight_cols:
            return True
        return False

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

# Main Window - controler in the MVC design pattern
# manages the interaction between user(view) and data(model).
class SetupWindow(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.setWindowTitle("Linear Regression - Setup")
        self.stacked_widget = stacked_widget
        # ----------------- Panel Setup -----------------
        self.top_panel_widget = QWidget()
        self.bottom_panel_widget = QWidget()
        # ----------------- Top setup -----------------
        self.label = QLabel("Path")
        self.path_display = QLineEdit()
        self.path_display.setPlaceholderText("Select a file to load the data")
        self.path_display.setReadOnly(True)
        self.btn_open_file = QPushButton("Open File")
        self.btn_open_file.clicked.connect(self.choose_file)
        # ----------------- Table setup -----------------
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

        # ----------------- bottom setup - input -----------------
        self.container_selector_widget = QWidget()
        self.container_preprocess_widget = QWidget()
        self.container_splitter_widget = QWidget()

        self.input_label = QLabel("Input columns (features)")
        self.input_selector = QListWidget()
        self.input_selector.setSelectionMode(QAbstractItemView.MultiSelection)
        self.input_selector.itemClicked.connect(self.selected_item_changed)
        self.input_selector.setUniformItemSizes(True)
        self.input_selector.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.output_label = QLabel("Output column (target)")
        self.output_selector = QComboBox()
        self.output_selector.currentIndexChanged.connect(self.selected_item_changed)

        self.confirm_button = QPushButton("Confirm selection")
        self.confirm_button.clicked.connect(self.confirm_selection)

        # ----------------- bottom setup - output -----------------
        self.preprocess_label = QLabel("Handle missing data:\nRed columns have at least\none NaN value")
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

        # ----------------- bottom setup - splitter -----------------
        self.splitter = DataSplitter()
        self.test_edit_label = QLabel("Test fraction (e.g. 0.2):")
        self.test_edit = QLineEdit("0.2")
        self.seed_edit_label = QLabel("Seed (reproducibility):")
        self.seed_edit = QLineEdit("42")
        self.split_button = QPushButton("Split data into training/test")
        self.split_button.clicked.connect(self.splitting_dataframe)
        # ----------------- bottom setup - splitter -----------------
        self.summary_split_label = QLabel()
        # ----------------- Set visibility -----------------
        widgets = [
            self.input_label, self.input_selector, self.output_label,
            self.output_selector, self.confirm_button, self.preprocess_label,
            self.strategy_box, self.apply_button, self.constant_name_edit,
            self.split_button, self.test_edit, self.seed_edit,
            self.test_edit_label, self.seed_edit_label,
            self.summary_split_label
            ]
        def hide_widgets():
            for w in widgets:
                w.setVisible(False)

        # ----------------- Layout setup -----------------
        top_panel_layout = QHBoxLayout()
        top_panel_layout.addWidget(self.label)
        top_panel_layout.addWidget(self.path_display)
        top_panel_layout.addWidget(self.btn_open_file)

        #Bottom_layout:
        bottom_panel_layout = QHBoxLayout()

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

        splitter_col = QVBoxLayout()
        splitter_col.addWidget(self.test_edit_label)
        splitter_col.addWidget(self.test_edit)
        splitter_col.addWidget(self.seed_edit_label)
        splitter_col.addWidget(self.seed_edit)
        splitter_col.addWidget(self.split_button)

        splitter_confirmation_col = QVBoxLayout()
        splitter_confirmation_col.addWidget(self.summary_split_label)
        # Group the layouts into another layout but this time a horizontal one
        container_selector_layout = QHBoxLayout()
        container_selector_layout.addLayout(input_col)
        container_selector_layout.addLayout(output_col)

        container_splitter_layout = QHBoxLayout()
        container_splitter_layout.addLayout(splitter_col)
        container_splitter_layout.addLayout(splitter_confirmation_col)

        #Envolpe the layout into a widget, this is for setting maximum width
        self.container_selector_widget.setLayout(container_selector_layout)
        self.container_preprocess_widget.setLayout(preprocess_col)
        self.container_splitter_widget.setLayout(container_splitter_layout)

        bottom_panel_layout.addWidget(self.container_selector_widget, alignment=Qt.AlignLeft)
        bottom_panel_layout.addWidget(self.container_preprocess_widget, alignment=Qt.AlignLeft)
        bottom_panel_layout.addWidget(self.container_splitter_widget, alignment=Qt.AlignLeft)

        #set layouts on our widgets:
        self.top_panel_widget.setLayout(top_panel_layout)
        self.bottom_panel_widget.setLayout(bottom_panel_layout)
        #This size policy tells the inmediate superior layout how to set width
        #and height to this widget using its sizeHint[It's the recommended size
        #a widget should have to displayed everything inside it correctly]
        #So who is the inmediate superior layout? main_layout. This SizePolicy
        #is telling main_layout how to treat bottom_panel. It's saying:
        #Don't let it have a width bigger than its sizeHint(minimum size
        #in which everything looks good) and height don't care(second argument)
        #In order to bottom_panel_widget has the minimum width all its children
        #must be next to each other which is what we want, so with a single line
        #we solve the problem xd.
        self.bottom_panel_widget.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        #Important! here we are setting the min and max height
        #the bottom_panel_widget must have in order to show correctly
        #in the smalest resolutions and biggest resolutions.
        #for mid resolutions we established setStretch.
        c_p_w, c_s_w, b_p_w = self.container_preprocess_widget,self.container_selector_widget, self.bottom_panel_widget
        b_p_w.setMinimumHeight(c_p_w.layout().sizeHint().height()+ 18)
        b_p_w.setMaximumHeight(c_s_w.layout().sizeHint().height())
        hide_widgets()

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.top_panel_widget)
        main_layout.addWidget(self.table)
        main_layout.addWidget(self.bottom_panel_widget)
        #The sintax of this is simple. setStretch.(widget_indice,ammount of partitions)
        #we sum all the ammount of partitions, 1 + 16 + 3 = 20. That means we
        #divide the window into 20 pieces and assign 16 pieces to the table, 3
        #pieces to the bottom_panel_widget and 1 piece to the top_pane_widget
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 20)
        main_layout.setStretch(2, 4)

        self.setLayout(main_layout)
        # ----------------- Data State -----------------
        self.data = DataModule()
        self.current_df = None
        self.selected_inputs = []
        self.selected_output = None

    # ------------------- Methods ------------------------------------------------
    def selected_item_changed(self, text):
        elements = [self.container_preprocess_widget, self.container_splitter_widget]
        for element in elements:
            if element.isVisible():
                element.setVisible(False)

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
            QMessageBox.information(self, "Success", "File loaded successfully.")
            self.container_preprocess_widget.hide()
            self.container_splitter_widget.hide()
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
        self.constant_name_edit.clear() #To clear the constant_name when we
        #confirm selection.
        # Show preprocessing controls once columns are selected
        preprocess_widgets = [self.preprocess_label, self.strategy_box,
        self.apply_button, self.container_preprocess_widget]

        splitter_widgets = [self.test_edit_label,self.test_edit,
        self.seed_edit_label,self.seed_edit, self.split_button]

        for w in preprocess_widgets + splitter_widgets:
            w.setVisible(True)

        self.container_preprocess_widget.setVisible(False)
        self.container_splitter_widget.setVisible(False)
        # --- Call the set_highlight_by_missing() function with the selected

        # self.selected_inputs + [self.selected_output] -> Just a concatenation
        # dict.fromkeys() dict is a class. This call the method fromkeys() of
        # class dict that basically take every element from an iterable, in
        # this case our list, and converted it to a key with default value None.
        # Why this? Because the user can have -2 IQ and select the same input
        # and output column, if this happens we are gonna work with a duplicated
        # column, if we select a column "A" with NaN in input and output with
        # 207 NaN for example, it will count twice and says we have 414
        # NaN values, and when splitting we'll have duplicated columns
        cols = list(dict.fromkeys([self.selected_output] + self.selected_inputs))
        model = self.table.model()
        # model.set_highlight_by_missing(cols) is not empty if
        # there are al least One NaN values in the selected columns
        # and it's empty when there are no selected columns with NaN values
        # So it's a conditional to prove when it's empty or not
        if hasattr(model, "set_highlight_by_missing"):
            if not model.set_highlight_by_missing(cols):
                self.container_preprocess_widget.setVisible(False)
                self.container_splitter_widget.setVisible(True)
                if self.summary_split_label.isVisible():
                    self.summary_split_label.setVisible(False)
            else:
                self.container_splitter_widget.setVisible(False)
                self.container_preprocess_widget.setVisible(True)
    # Missing data detection and preprocessing
    def handle_missing_data(self):
        if self.current_df is None:
            QMessageBox.warning(self, "Error", "No dataset loaded.")
            return

        cols = list(dict.fromkeys([self.selected_output] + self.selected_inputs))
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
            self.container_preprocess_widget.hide()
            self.container_splitter_widget.show()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during preprocessing:\n{str(e)}")

    # ----------------- TRAIN/TEST -----------------
    def splitting_dataframe(self):
        widgets = [self.summary_split_label]
        model = self.table.model()
        cols = self.selected_inputs + [self.selected_output]
        if self.current_df is None or self.current_df.empty:
            QMessageBox.warning(self, "Error", "There isn't any data avaliable to split.")
            return
        if model.highlight_cols:
            QMessageBox.warning(self, "Error", "There are missing data, please preprocess it first.")
            return
        test_size = float(self.test_edit.text())
        seed = int(self.seed_edit.text())
        self.train_df, self.test_df = self.splitter.split(self.current_df[cols], test_size, seed)
        summary = self.splitter.get_meta()
        # Mensaje
        msg_summary = (
        "Division was correctly done.\n\n"
        f"Total df: {summary["n_rows_total"]} rows\n"
        f"Training df: {summary['n_train']} rows\n"
        f"Test df: {summary['n_test']} rows\n"
        f"Seed used: {summary['random_seed']}"
        )
        print(msg_summary)
        QMessageBox.information(self,"succesful",msg_summary)

        self.summary_split_label.setText(msg_summary)
        for w in widgets:
            w.setVisible(True)

    def strategy_box_changed(self, option_selected) -> None:
        is_cte = option_selected == "Fill with constant"
        self.constant_name_edit.setVisible(is_cte)
        if is_cte:
            self.constant_name_edit.setFocus()
        else:
            self.constant_name_edit.clear()
        return None

    def go_to_result_window(self):
        self.stacked_widget.setCurrentIndex(1)  # cambia a la segunda ventana

class ResultWindow(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.placeholder_text = QLabel(
        "Remember that to access this feature\n"
        "you first need to load the data,\n"
        "preprocess if needed and split\n"
        "the data into training and test sets.\n"
        )

        self.placeholder_text.setAlignment(Qt.AlignCenter)
        self.placeholder_text.setStyleSheet("color: gray; font-size: 16px;")

        layout = QVBoxLayout()
        #Con layout.addWidget(widget, stretch) le asignas el factor de stretch al añadir el widget.
        #Con layout.setStretch(indice, stretch) ajustas el factor de stretch después, refiriéndote al índice del ítem dentro del layout.
        layout.addWidget(self.placeholder_text, 1)
        self.setLayout(layout)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Linear Regression - Main window")
        #Stack windows------------------------------------
        self.stacked_widget = QStackedWidget()
        self.setup_window = SetupWindow(self.stacked_widget)
        self.result_window = ResultWindow(self.stacked_widget)

        self.stacked_widget.addWidget(self.setup_window)  # índice 0
        self.stacked_widget.addWidget(self.result_window) # indice 1
        #----------------------------------------------------
        self.setup_window_button = QPushButton("Setup Window")
        self.setup_window_button.clicked.connect(self.change_to_setup_window)
        self.result_window_button = QPushButton("Result Window")
        self.result_window_button.clicked.connect(self.change_to_result_window)
        widgets = [self.setup_window_button,self.result_window_button]

        def hide_widgets():
            for widget in widgets:
                widget.hide()

        #layouts:

        #top layout - main bar:
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.setup_window_button)
        top_layout.addWidget(self.result_window_button)
        #Container of main bar - this for the border:------------------
        top_panel_widget = QWidget()
        top_panel_widget.setLayout(top_layout)
        top_panel_widget.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        #---------------------------------------------------------------
        main_layout = QVBoxLayout()
        main_layout.addWidget(top_panel_widget)
        main_layout.addWidget(self.stacked_widget)
        self.setLayout(main_layout)
    #Methods:
    def change_to_setup_window(self):
        self.stacked_widget.setCurrentIndex(0)
    def change_to_result_window(self):
        self.stacked_widget.setCurrentIndex(1)


# Just a function to set the icon.jpg as the app icon and as the docker icon
# at the moment just compatible with MacOS.
def set_app_icon(app):
    #ICON is a global inmutable constant. No need to declare global
    #cause we are not going to redefine it, just read the value. So global
    #Is unnecesary
    app.setWindowIcon(QIcon(ICON))
    # darwin is like the kernel of Mac. So when you ask which the sys its name
    # just returns "darwin" in MacOS
    if sys.platform == "darwin":
        try:
            img = NSImage.alloc().initWithContentsOfFile_(ICON)
            if img is not None:
                NSApplication.sharedApplication().setApplicationIconImage_(img)
        except Exception:
            pass

# Entry point
def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    set_app_icon(app)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()