import pandas as pd
from backend.data_loader import (
    DataModule
)

from backend.data_split import (
    DataSplitter,
    DataSplitError
)
from backend.handle_missing_data import (
    handle_missing_data,
    MissingDataError
)
from frontend.table_model import PandasModel
from PyQt5.QtCore import (
    Qt,
    pyqtSlot,
    pyqtSignal
)

from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout,
    QFileDialog, QTableView, QMessageBox, QHeaderView, QListWidget,
    QAbstractItemView, QHBoxLayout, QComboBox, QSizePolicy
)
class SetupWindow(QWidget):
    """Data setup and preprocessing window for linear regression.

    Manages the complete workflow: file loading, column selection,
    missing data preprocessing, and train/test splitting. Emits signals
    to communicate with other windows. Uses PandasModel for table display
    with NaN highlighting.

    Attributes
    ----------
    stacked_widget : QStackedWidget
        Reference to parent stacked widget for navigation.
    data : DataModule
        Data loader for handling multiple file formats.
    current_df : pd.DataFrame
        Currently loaded DataFrame.
    selected_inputs : list
        List of selected input (feature) column names.
    selected_output : str
        Selected output (target) column name.
    model_description : str
        User-provided description for saved models.
    was_succesfully_plotted : bool
        Flag indicating if model was successfully plotted.

    Methods
    -------
    choose_file()
        Open file dialog and load selected data file.
    load_table(df)
        Display DataFrame in table view with PandasModel.
    show_column_selectors(df)
        Populate input/output column selectors from DataFrame.
    confirm_selection()
        Validate and confirm selected input/output columns.
    handle_missing_data_GUI()
        Apply missing data preprocessing strategy.
    split_dataframe()
        Split data into train/test sets.
    create_model(msg_summary)
        Emit signal with split data for model creation.
    strategy_box_changed(option_selected)
        Toggle constant input field when strategy changes.
    cant_be_plotted(res)
        Handle signal when model cannot be plotted.
    reset_to_initial_state()
        Clear all selections and reset UI to initial state.
    """
    # Signals to communicate with ResultWindow
    # Signals that are sent
    train_test_df_ready = pyqtSignal(object)
    another_file_opened = pyqtSignal()
    def __init__(self, stacked_widget) -> None:
        """Initialize the setup window.

        Sets up all UI components for data loading, column selection,
        preprocessing, and train/test splitting.

        Parameters
        ----------
        stacked_widget : QStackedWidget
            Reference to the parent stacked widget for navigation.

        Returns
        -------
        None
        """
        super().__init__()
        self.setWindowTitle("Linear Regression - Setup")
        self.stacked_widget = stacked_widget
        # ----------------- Panel Setup --------------------------------
        self.top_panel_widget = QWidget()
        self.bottom_panel_widget = QWidget()
        # ----------------- Top setup ----------------------------------
        self.label = QLabel("Path")
        self.path_display = QLineEdit()
        self.path_display.setPlaceholderText("Select a dataset file to load")
        self.path_display.setReadOnly(True)
        self.btn_open_file = QPushButton("Open File")
        self.btn_open_file.clicked.connect(self.choose_file)
        # ----------------- Table setup --------------------------------
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
        # ----------------- Bottom containers --------------------------
        self.container_selector_widget = QWidget()
        self.container_preprocess_widget = QWidget()
        self.container_splitter_widget = QWidget()
        # ----------------- bottom setup - features --------------------
        self.input_label = QLabel("Input columns (features)")
        self.input_selector = QListWidget()
        self.input_selector.setSelectionMode(QAbstractItemView.MultiSelection)
        self.input_selector.itemClicked.connect(self.selected_item_changed)
        self.input_selector.setUniformItemSizes(True)
        self.input_selector.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.output_label = QLabel("Output column (target)")
        self.output_selector = QComboBox()
        self.output_selector.currentIndexChanged.connect(
            self.selected_item_changed
            )
        self.confirm_button = QPushButton("Confirm selection")
        self.confirm_button.clicked.connect(self.confirm_selection)
        # ----------------- bottom setup - target-----------------------
        self.preprocess_label = QLabel("Handle missing data:\n"
                                        "Red columns have at least\n"
                                        "one NaN value")
        self.strategy_box = QComboBox()
        self.strategy_box.addItems([
            "Delete rows with NaN",
            "Fill with mean",
            "Fill with median",
            "Fill with constant"
        ])
        self.strategy_box.currentTextChanged.connect(self.strategy_box_changed)
        self.apply_button = QPushButton("Apply preprocessing")
        self.apply_button.clicked.connect(self.handle_missing_data_GUI)
        self.constant_name_edit = QLineEdit()
        self.constant_name_edit.setPlaceholderText("Constant name")
        # Hide constant input by default
        self.constant_name_edit.setVisible(False)

        # ----------------- bottom setup - splitter --------------------
        self.splitter = DataSplitter()
        self.split_test_label = QLabel("Split data into training/test")
        self.test_edit_label = QLabel("Test fraction (e.g. 0.2):")
        self.test_edit = QLineEdit("0.2")
        self.seed_edit_label = QLabel("Seed (reproducibility):")
        self.seed_edit = QLineEdit("42")
        self.split_button = QPushButton("Create model")
        self.split_button.clicked.connect(self.split_dataframe)

        # ----------------- bottom model_creation_message --------------
        self.container_summary_model = QWidget()
        self.container_summary_model_vlayout= QVBoxLayout()
        self.summary_model_label = QLabel("Model creation summary:")
        self.summary_model_creation_label = QLabel()
        self.summary_model_creation_label.setStyleSheet("""
                                    QLabel {
                                    font-family: 'Consolas';
                                    font-size: 13px;
                                    color: #E0E0E0;
                                    }
                                    """)
        self.container_summary_model_vlayout.addWidget(
            self.summary_model_label
            )
        self.container_summary_model_vlayout.addStretch(1)
        self.container_summary_model_vlayout.addWidget(
            self.summary_model_creation_label
            )
        self.container_summary_model.setLayout(
            self.container_summary_model_vlayout
            )

        # ----------------- Set visibility -----------------------------
        containers = [
            self.container_selector_widget,
            self.container_preprocess_widget,
            self.container_splitter_widget,
            self.container_summary_model
            ]
        def hide_widgets():
            for w in containers:
                w.setVisible(False)

        # ----------------- Layout setup -------------------------------
        top_panel_layout = QHBoxLayout()
        top_panel_layout.addWidget(self.label)
        top_panel_layout.addWidget(self.path_display)
        top_panel_layout.addWidget(self.btn_open_file)

        # Bottom_layout:
        bottom_panel_layout = QHBoxLayout()

        # Creation of vertical views and stack the widgets on it.
        input_col = QVBoxLayout()
        input_col.addWidget(self.input_label)
        input_col.addWidget(self.input_selector)

        output_col = QVBoxLayout()
        output_col.addWidget(self.output_label)
        output_col.addWidget(self.output_selector)
        #To add some space between label and selector
        output_col.addStretch(1)
        output_col.addWidget(self.confirm_button)

        preprocess_col = QVBoxLayout()
        preprocess_col.addWidget(self.preprocess_label)
        preprocess_col.addWidget(self.strategy_box)
        preprocess_col.addWidget(self.constant_name_edit)
        preprocess_col.addStretch(1)
        preprocess_col.addWidget(self.apply_button)

        splitter_col = QVBoxLayout()
        splitter_col.addWidget(self.split_test_label)
        splitter_col.addWidget(self.test_edit_label)
        splitter_col.addWidget(self.test_edit)
        splitter_col.addWidget(self.seed_edit_label)
        splitter_col.addWidget(self.seed_edit)
        splitter_col.addWidget(self.split_button)

        # Group the layouts into another
        # layout but this time a horizontal one
        container_selector_layout = QHBoxLayout()
        container_selector_layout.addLayout(input_col)
        container_selector_layout.addLayout(output_col)

        container_splitter_layout = QHBoxLayout()
        container_splitter_layout.addLayout(splitter_col)

        # Envolpe the layout into a widget,
        # this is for setting maximum width
        self.container_selector_widget.setLayout(container_selector_layout)
        self.container_preprocess_widget.setLayout(preprocess_col)
        self.container_splitter_widget.setLayout(container_splitter_layout)

        bottom_panel_layout.addWidget(self.container_selector_widget,
                                      alignment=Qt.AlignLeft)
        bottom_panel_layout.addWidget(self.container_preprocess_widget,
                                      alignment=Qt.AlignLeft)
        bottom_panel_layout.addWidget(self.container_splitter_widget,
                                      alignment=Qt.AlignLeft)
        bottom_panel_layout.addWidget(self.container_summary_model,
                                      alignment=Qt.AlignLeft)

        #set layouts on our widgets:
        self.top_panel_widget.setLayout(top_panel_layout)
        self.bottom_panel_widget.setLayout(bottom_panel_layout)
        #This size policy tells the inmediate superior layout
        #how to set width and height to this widget using its
        #sizeHint[It's the recommended size a widget should have to
        #displayed everything inside it correctly] So who is the
        #inmediate superior layout? main_layout. This SizePolicy
        #is telling main_layout how to treat bottom_panel. It's saying:
        #Don't let it have a width bigger than its sizeHint(minimum size
        #in which everything looks good) and height
        #don't care(second argument) In order to bottom_panel_widget
        #has the minimum width all its children must be next to each
        #other which is what we want, so with a single line
        #we solve the problem.
        self.bottom_panel_widget.setSizePolicy(QSizePolicy.Maximum,
                                               QSizePolicy.Preferred)
        #Important! here we are setting the min and max height
        #the bottom_panel_widget must have in order to show correctly
        #in the smalest resolutions and biggest resolutions.
        #for mid resolutions we established setStretch.
        c_split_w, c_s_w, b_p_w = (self.container_splitter_widget,
                                    self.container_selector_widget,
                                    self.bottom_panel_widget)
        b_p_w.setMinimumHeight(c_split_w.layout().sizeHint().height()+ 15)
        b_p_w.setMaximumHeight(c_s_w.layout().sizeHint().height())
        hide_widgets()

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.top_panel_widget)
        main_layout.addWidget(self.table)
        main_layout.addWidget(self.bottom_panel_widget)
        # The sintax of this is simple. setStretch.
        # (widget_indice,ammount of partitions) we sum all the ammount
        # of partitions, 1 + 16 + 3 = 20. That means we divide the
        # window into 20 pieces and assign 16 pieces to the table, 3
        # pieces to the bottom_panel_widget and 1 piece to
        # the top_pane_widget
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 20)
        main_layout.setStretch(2, 4)

        self.setLayout(main_layout)
        # ----------------- Data State ---------------------------------
        self.data = DataModule()
        self.current_df = None
        self.selected_inputs = []
        self.selected_output = None
        self.model_description = ""
        self.was_succesfully_plotted = True

    # ------------------- Methods --------------------------------------
    def hide_containers(self, hide_container_selector_widget: bool = False
                        ) -> None:
        """Hide preprocessing and splitting containers.

        Optionally hides the column selector as well. Used to reset
        UI state between operations.

        Parameters
        ----------
        hide_container_selector_widget : bool, optional
            If True, also hide the column selector. Default is False.

        Returns
        -------
        None
        """
        def hide_them():
            for container in containers:
                if container.isVisible():
                    container.setVisible(False)

        containers = [self.container_preprocess_widget,
                    self.container_splitter_widget,
                    self.container_summary_model]
        if hide_container_selector_widget:
            containers.append(self.container_selector_widget)
            hide_them()
            return
        hide_them()

    def selected_item_changed(self, text: str) -> None:
        """Hide containers when column selection changes.

        Called when user modifies selected input/output columns.

        Parameters
        ----------
        text : str
            Text of the clicked item (unused, signal parameter).

        Returns
        -------
        None
        """
        self.hide_containers()

    def choose_file(self) -> None:
        """Open file dialog and load selected data file.

        Supports CSV, SQLite, Excel formats. Populates table,
        shows column selectors, and emits another_file_opened signal.
        Displays error messages if file loading fails.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
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
            self.hide_containers()
            self.show_column_selectors(df)
            self.container_selector_widget.setVisible(True)
            QMessageBox.information(self, "Success", "File "
                                            "loaded successfully.")
            # Hide containers when opening a new file 
            self.another_file_opened.emit()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"The file could not "
                                f"be loaded:\n{str(e)}")

    def load_table(self, df: pd.DataFrame) -> None:
        """Display DataFrame in table view with PandasModel.

        Creates a PandasModel for the given DataFrame and displays
        it in the QTableView with sorting enabled.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to display in the table.

        Returns
        -------
        None
        """
        self.current_df = df
        self.table.setUpdatesEnabled(False)
        self.table.setModel(PandasModel(df, self))
        self.table.setSortingEnabled(True)
        self.table.setUpdatesEnabled(True)

    # Column selectors
    def show_column_selectors(self, df: pd.DataFrame) -> None:
        """Populate and display column selector widgets.

        Populates input (multiselect) and output (combobox) selectors
        with DataFrame column names. Makes selector widgets visible.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame whose columns populate the selectors.

        Returns
        -------
        None
        """
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

    def confirm_selection(self) -> None:
        """Validate and confirm selected input/output columns.

        Checks that at least one input and one output are selected.
        Removes duplicate columns by using dict.fromkeys(). Highlights
        columns with NaN values or shows splitter if no NaNs exist.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.selected_inputs = [
            i.text() for i in self.input_selector.selectedItems()
            ]
        self.selected_output = self.output_selector.currentText()
        if not self.selected_inputs or not self.selected_output:
            QMessageBox.warning(self, "Error", "Please select both "
                                        "input and output columns.")
            return
        QMessageBox.information(
            self, "Selection confirmed",
            f"Inputs: {', '.join(self.selected_inputs)}\n"
            f"Output: {self.selected_output}"
        )
        # To clear the constant_name field
        self.constant_name_edit.clear()
        self.hide_containers()
        # --- Call the set_highlight_by_missing() function with
        # the selected dict.fromkeys() dict is a class. This call the
        # method fromkeys() of class dict that basically take every
        # element from an iterable, in this case our list, and converted
        # it to a key with default value None.
        # Why this? Because the user can have -2 IQ and select
        # the same input and output column, if this happens we are gonna
        # work with a duplicated column, if we select a column "A" with
        # NaN in input and output with 207 NaN for example,
        # it will count twice and says we have 414 NaN values, and when
        # splitting we'll have duplicated columns
        cols = list(dict.fromkeys([self.selected_output]
                                  + self.selected_inputs))
        table_model = self.table.model()
        # model.set_highlight_by_missing(cols) is not empty if
        # there are al least One NaN values in the selected columns
        # and it's empty when there are no selected columns with
        # NaN values So it's a conditional to prove when it's
        # empty or not
        if hasattr(table_model, "set_highlight_by_missing"):
            if not table_model.set_highlight_by_missing(cols):
                self.container_preprocess_widget.setVisible(False)
                self.container_splitter_widget.setVisible(True)
            else:
                self.container_splitter_widget.setVisible(False)
                self.container_preprocess_widget.setVisible(True)

    # Missing data detection and preprocessing
    def handle_missing_data_GUI(self) -> None:
        """Apply missing data preprocessing strategy.

        Retrieves user-selected strategy (delete/mean/median/constant),
        applies handle_missing_data backend function, updates table model,
        and shows preprocessing summary. Displays error if strategy fails.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # This function only executes when user presses the apply_button
        if self.current_df is None:
            QMessageBox.warning(self, "Error", "No dataset loaded.")
            return
        cols = list(dict.fromkeys([self.selected_output]
                                  + self.selected_inputs))
        strategy = self.strategy_box.currentText()
        written_cte = self.constant_name_edit.text()
        try:
            result=handle_missing_data(self.current_df,cols,strategy,
                                       written_cte)
            df_clean, msg_process_completed, msg_nan_summary = result
            if strategy == "Delete rows with NaN":
                self.current_df = df_clean
        except MissingDataError as e:
            QMessageBox.warning(self, "Error", str(e))
            return
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during "
                                                f"preprocessing:\n{str(e)}")
            return
        QMessageBox.information(self, "Missing Data Detected", msg_nan_summary)
        try:
            self.refresh_table_model()
            table_model = self.table.model()
            if hasattr(table_model, "set_highlight_by_missing"):
                table_model.set_highlight_by_missing(cols)

            QMessageBox.information(self,
                                    "Preprocessing Completed",
                                    msg_process_completed)

            # Show split button after successful preprocessing
            self.container_preprocess_widget.hide()
            self.container_splitter_widget.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during "
                                                f"preprocessing:\n{str(e)}")

    def refresh_table_model(self) -> None:
        """Refresh table view with current DataFrame.

        Updates the table model with the current DataFrame by calling
        set_dataframe() if available, otherwise reloads the entire model.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.current_df is None:
            return
        table_model = self.table.model()
        if hasattr(table_model, "set_dataframe"):
            table_model.set_dataframe(self.current_df)
        else:
            self.load_table(self.current_df)
        return None
    # ----------------- TRAIN/TEST -------------------------------------
    def split_dataframe(self) -> None:
        """Split data into train/test sets and prepare for modeling.

        Validates that data is loaded and preprocessed, extracts test
        size and seed parameters, calls DataSplitter.split(), displays
        split summary, and emits train_test_df_ready signal with data.
        Shows error messages for invalid inputs or missing data.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # We assume it's True, if not ResultWindow emit a signal
        # to change this attribute self.was_succesfully_plotted.
        # This is for showing or not the message saying the plot was
        # succesfull.
        self.was_succesfully_plotted = True
        model = self.table.model()
        cols = [self.selected_output] + self.selected_inputs #This order
        # is used to select x_train/test and y_train/test.
        #  self.selected_output can only be one, so if we index 0 we
        # are getting y_train/test and the rammaining
        # is x_train/test.
        if self.current_df is None or self.current_df.empty:
            QMessageBox.warning(self, "Error", "There isn't any "
                                "data avaliable to split.")
            return
        if model.highlight_cols:
            QMessageBox.warning(self, "Error", "There are missing data, "
                                "please preprocess it first.")
            return
        test_size = float(self.test_edit.text())
        seed = int(self.seed_edit.text())
        try:
            self.train_df, self.test_df = self.splitter.split(
                self.current_df[cols], test_size, seed
                )
            msg_summary = self.splitter.get_split_summary()
        except DataSplitError as e:
            QMessageBox.warning(self, "Error", str(e))
            return
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during"
                                f"data splitting:\n{str(e)}")
            return
        # Display successful split message with summary
        QMessageBox.information(self, "Split successful",
        f"Division was correctly done.\n\n{msg_summary}")
        # Prepare interface for model creation
        self.container_summary_model.hide()
        self.summary_model_creation_label.clear()
        self.create_model(msg_summary)

    def create_model(self, msg_summary: str) -> None:
        """Emit signal with train/test data for model creation.

        Prepares data payload and emits train_test_df_ready signal.
        Handles plotting errors from ResultWindow and displays
        appropriate success/failure messages.

        Parameters
        ----------
        msg_summary : str
            Summary message of the train/test split operation.

        Returns
        -------
        None
        """
        payload = [(self.train_df, self.test_df), msg_summary]
        self.train_test_df_ready.emit(payload)

        # Maintain plotting/error logic: if ResultWindow
        # reported that it could not be plotted
        if not self.was_succesfully_plotted:
            QMessageBox.warning(
                self,
                "Failure",
                f"Failure when creating the model:\n"
                f" {str(self.plotted_error)}"
                )
            # Do not display summary_model_creation_label if there was an error
            return

        # Success notifications (those that were previously after the Split)
        if len(self.selected_inputs) > 1:
            QMessageBox.information(self, "Model sucessfully created",
                                    "multiple regression succesfully done\n\n"
                                    "can't be plotted\n\n")
            self.summary_model_creation_label.setText("multiple regression "
                                                      "succesfully done\n"
                                                      "can't be plotted\n\n"
                                                        f"{msg_summary}")
        else:
            QMessageBox.information(self, "Model sucessfully created",
                                    "Simple regression succesfully done\n\n"
                                    "plotted on Model Management\n\n")
            self.summary_model_creation_label.setText("Simple regression "
                                                      "succesfully done\n"
                                                      "plotted on "
                                                      "Model Management\n\n"
                                                        f"{msg_summary}")
        self.container_summary_model.show()

    def strategy_box_changed(self, option_selected: str) -> None:
        """Toggle constant input field based on selected strategy.

        Shows the constant value input field only when the user
        selects "Fill with constant" strategy.

        Parameters
        ----------
        option_selected : str
            The strategy option selected in the dropdown.

        Returns
        -------
        None
        """
        is_cte = option_selected == "Fill with constant"
        self.constant_name_edit.setVisible(is_cte)
        if is_cte:
            self.constant_name_edit.setFocus()
        else:
            self.constant_name_edit.clear()
        return None

    #-------------------------Connections:------------------------------
    # Signals that are received
    @pyqtSlot(object)
    def cant_be_plotted(self, res: object) -> None:
        """Receive signal that model cannot be plotted.

        Sets flag and stores error for later display. Called when
        ResultWindow fails to plot multiple regression model.

        Parameters
        ----------
        res : object
            Error information from ResultWindow.

        Returns
        -------
        None
        """
        self.was_succesfully_plotted = False
        self.plotted_error = res
    #-------------------------------------------------------------------

    def reset_to_initial_state(self) -> None:
        """Reset all UI components to initial state.

        Clears all selections, file paths, table model, preprocessing
        state, and test/seed parameters. Used when loading new files
        or resetting the workflow.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Reset path
        self.path_display.clear()
        # Clear table
        self.table.setModel(None)
        self.current_df = None
        # Reset selections
        self.selected_inputs = []
        self.selected_output = None
        # Clear column selectors
        self.input_selector.clear()
        self.output_selector.clear()
        # Hide all bottom containers
        self.container_selector_widget.hide()
        self.container_preprocess_widget.hide()
        self.container_splitter_widget.hide()
        self.container_summary_model.hide()
        # Reset summary panel
        self.summary_model_creation_label.clear()
        # Hide constant input
        self.constant_name_edit.clear()
        self.constant_name_edit.setVisible(False)
        # Reset highlight in the table
        # (only if a model was previously loaded)
        if hasattr(self.table.model(), "clear_highlight"):
            try:
                self.table.model().clear_highlight()
            except Exception:
                pass
        # Reset test/seed boxes
        self.test_edit.setText("0.2")
        self.seed_edit.setText("42")
