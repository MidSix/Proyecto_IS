# -*- coding: utf-8 -*-
import joblib
import numpy as np
from backend.linear_regression_io import (
    load_model_data,
    save_model_data
)
from backend.linear_regression_creation import LinearRegressionModel

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
from PyQt5.QtCore import (
    Qt,
    pyqtSlot,
    pyqtSignal
)
from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout,
    QFileDialog, QMessageBox, QHBoxLayout, QSizePolicy, QTextEdit,
    QScrollArea
)

class ResultWindow(QWidget):
    """Model visualization and prediction window.

    Displays trained linear regression models with metrics, parity plots,
    regression plots (for simple regression), and prediction interface.
    Allows saving trained models and loading previously saved models.
    Handles both simple and multiple linear regression.

    Attributes
    ----------
    model : LinearRegressionModel
        The trained linear regression model instance.
    train_df : pd.DataFrame
        Training DataFrame (first column is target).
    test_df : pd.DataFrame
        Test DataFrame (first column is target).
    metrics : list
        Model metrics computed from fit_and_evaluate().
    toolbar : NavigationToolbar
        Matplotlib toolbar for simple regression plot.
    graph : FigureCanvas
        Matplotlib canvas for simple regression plot.
    parity_toolbar : NavigationToolbar
        Matplotlib toolbar for parity plot.
    parity_graph : FigureCanvas
        Matplotlib canvas for parity plot.
    prediction_fields : dict
        Dictionary mapping feature names to QLineEdit widgets.

    Methods
    -------
    clear_result_window()
        Clear all graphs and reset prediction UI.
    load_model_data_GUI(model_data)
        Display loaded model data and prediction interface.
    train_test_df_res(data)
        Receive train/test data, fit model, and display results.
    simple_linear_regression()
        Display results for simple (1 feature) regression.
    multiple_linear_regression()
        Display results for multiple feature regression.
    create_parity_figure()
        Create and display parity plot (actual vs predicted).
    save_model_dialog()
        Save trained model with optional description.
    load_model_data_dialog()
        Load previously saved model file.
    perform_prediction()
        Make prediction using user-entered feature values.
    build_prediction_inputs()
        Create input fields for each model feature.
    """
    cant_be_plotted = pyqtSignal(object)
    model_loaded = pyqtSignal()
    def __init__(self, stacked_widget) -> None:
        """Initialize the result window.

        Sets up the UI for model display, graphs, predictions, and
        model saving/loading. Initializes with placeholder text.

        Parameters
        ----------
        stacked_widget : QStackedWidget
            Reference to the parent stacked widget for navigation.

        Returns
        -------
        None
        """
        super().__init__()
        self.stacked_widget = stacked_widget
        self.toolbar = None
        self.parity_toolbar = None
        self.graph = None
        self.parity_graph = None
        self.model = LinearRegressionModel()
        #------------------------widgets--------------------------------
        self.placeholder_text = QLabel(
        "Remember that to access this feature\n"
        "you first need to load the model\n"
        "or load the data. And then preprocess\n"
        "if needed and split the data\n"
        "into training and test sets.\n"
        )
        self.placeholder_text.setAlignment(Qt.AlignCenter)
        self.placeholder_text.setStyleSheet("color: gray; font-size: 16px;")
        self.summary_model_widget = QLabel()
        self.summary_model_widget.setAlignment(Qt.AlignLeft)
        #Just some QSS to make the self.summary_model_widget
        #looks better.
        self.summary_model_widget.setStyleSheet("""
                                    QLabel {
                                    font-family: 'Consolas';
                                    font-size: 14pt;
                                    color: #E0E0E0;
                                    }
                                    """)
        self.model_description_edit = QTextEdit()
        self.model_description_edit.setPlaceholderText("description "
                                                        "(optional)")
        self.save_button = QPushButton("Save model")
        self.save_button.clicked.connect(self.save_model_dialog)

        # --- LOAD MODEL (new) ---
        self.model_path_label = QLabel("Path")
        self.model_path_label.setStyleSheet("color: gray;")
        self.model_path_display = QLineEdit()
        self.model_path_display.setReadOnly(True)
        self.model_path_display.setPlaceholderText("Select a"
                                                    " model file to load")
        self.load_model_button = QPushButton("Load model")
        self.load_model_button.clicked.connect(self.load_model_data_dialog)
        self.load_model_top_layout = QHBoxLayout()
        self.load_model_top_layout.addWidget(self.model_path_label)
        self.load_model_top_layout.addWidget(self.model_path_display)
        self.load_model_top_layout.addWidget(self.load_model_button)

        # ------------------ Prediction Panel -------------------
        self.container_prediction_widget = QWidget()
        self.prediction_layout = QVBoxLayout()
        self.prediction_title = QLabel("Make a Prediction")
        self.prediction_title.setStyleSheet("font-size: 16pt;"
                                            " font-weight: bold;")
        self.prediction_inputs_widget = QWidget()
        self.prediction_inputs_layout = QVBoxLayout()
        self.prediction_inputs_widget.setLayout(self.prediction_inputs_layout)
        self.btn_predict = QPushButton("Make prediction")
        self.btn_predict.clicked.connect(self.perform_prediction)
        self.prediction_result = QLabel("")
        self.prediction_result.setStyleSheet("font-size: 14pt; color: cyan;")
        self.prediction_layout.addWidget(self.prediction_title)
        self.prediction_layout.addWidget(self.prediction_inputs_widget)
        self.prediction_layout.addWidget(
            self.btn_predict,
            alignment=Qt.AlignCenter
            )
        self.btn_predict.setSizePolicy(
            QSizePolicy.Maximum,
            QSizePolicy.Preferred
            )
        self.prediction_layout.addWidget(self.prediction_result)
        self.container_prediction_widget.setLayout(self.prediction_layout)
        self.container_prediction_widget.hide()

        #------------------------containers-----------------------------
        self.container_scroll_area = QWidget()
        self.container_model_widget = QWidget()
        self.container_description_widget = QWidget()
        self.container_graph_widget = QWidget()
        self.container_model_widget.setSizePolicy(
            QSizePolicy.Maximum,
            QSizePolicy.Preferred
            )
        self.container_simple_regression_graph_widget = QWidget()
        self.container_multiple_regression_graph_widget =QWidget()
        #-------------------------Layouts-------------------------------
        self.container_description_layout = QVBoxLayout()
        self.container_model_layout = QVBoxLayout()
        self.container_graph_layout = QVBoxLayout()
        self.container_simple_regression_graph_layout = QHBoxLayout()
        self.container_multiple_regression_graph_layout = QHBoxLayout()
        self.container_scroll_area_layout = QHBoxLayout()
        self.container_of_scroll_area_layout = QVBoxLayout()
        self.main_layout = QVBoxLayout()
        #-------------------------Set Layouts---------------------------
        #-------------------container_scroll_area_layout----------------
        self.container_description_layout.addWidget(
            self.model_description_edit
            )
        self.container_description_layout.addWidget(
            self.save_button,
            alignment=Qt.AlignCenter
            )
        self.save_button.setSizePolicy(
            QSizePolicy.Maximum,
            QSizePolicy.Preferred
            )
        self.container_description_widget.setLayout(
            self.container_description_layout
            )
        self.container_model_layout.addWidget(self.summary_model_widget)
        self.container_model_layout.addWidget(self.container_prediction_widget)
        self.container_model_layout.addWidget(
            self.container_description_widget
            )
        self.container_model_layout.setStretch(0, 5)
        self.container_model_layout.setStretch(1, 3)
        self.container_model_widget.setLayout(self.container_model_layout)
        # Allow the model widget to expand horizontally
        # so path lineedit can stretch across window
        self.container_model_widget.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred
            )
        self.container_scroll_area_layout.addWidget(
            self.container_model_widget, 0
            )
        self.container_scroll_area_layout.addWidget(
            self.container_graph_widget, 1
            )
        self.container_scroll_area.setLayout(self.container_scroll_area_layout)
        # ---- Wrap container_scroll_area in
        # scroll area for vertical scrolling ---------------------------
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setWidget(self.container_scroll_area)

        # ----------------- TOP PANEL WIDGET ---------------------------
        self.top_panel_widget = QWidget()
        self.top_panel_widget.setLayout(self.load_model_top_layout)
        # Ensure the top panel doesn't get squashed vertically
        self.top_panel_widget.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed
            )
        # ----------------- MAIN LAYOUT -------------
        # Build the main_layout
        self.main_layout.addWidget(self.top_panel_widget)
        self.main_layout.addWidget(self.placeholder_text, 1)
        self.main_layout.addWidget(self.scroll_area)
        self.show_all_containers(True)
        self.setLayout(self.main_layout)
        self.container_description_widget.hide()
        self.scroll_area.hide()

    # Methods:
    def clear_result_window(self) -> None:
        """Clear all graphs and reset prediction UI.

        Removes matplotlib canvases and toolbars, clears prediction
        results and model description. Safe to call even if no graphs
        exist (catches exceptions).

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        try:
            if self.toolbar is not None:
                self.container_graph_layout.removeWidget(
                    self.container_simple_regression_graph_widget
                    )
                self.container_simple_regression_graph_widget.hide()
                self.toolbar.deleteLater()
                self.graph.deleteLater()
                self.toolbar = None
                self.graph = None
            if self.parity_toolbar is not None:
                self.container_graph_layout.removeWidget(
                    self.container_multiple_regression_graph_widget
                    )
                self.container_multiple_regression_graph_widget.hide()
                self.parity_toolbar.deleteLater()
                self.parity_graph.deleteLater()
                self.parity_toolbar = None
                self.parity_graph = None
            self.container_prediction_widget.hide()
            self.prediction_result.clear()
            self.model_description_edit.clear()
        except Exception:
                pass

    def load_model_data_GUI(self, model_data: dict) -> None:
        """Display loaded model data and update prediction interface.

        Clears previous results, extracts model metadata, rebuilds
        prediction inputs, and shows model summary with description.
        Displays error if loading fails.

        Parameters
        ----------
        model_data : dict
            Dictionary containing model metadata and description.

        Returns
        -------
        None
        """
        try:
            self.clear_result_window()
            self.placeholder_text.hide()
            summary_lines, description = load_model_data(model_data)
            self.summary_model_widget.setText("\n".join(summary_lines))

            # Rebuild prediction UI using model_data structure
            self.model.feature_names = model_data["input_columns"]
            self.model.target_name = model_data["output_column"]
            self.build_prediction_inputs()
            self.container_prediction_widget.show()
            self.container_description_widget.show()

            # Show suitable containers
            self.model_description_edit.setPlainText(description or "")
            self.show_all_containers(True)
            self.scroll_area.show()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to"
                                 f"show loaded model:\n{str(e)}")

    def multiple_linear_regression(self) -> None:
        """Display results for multiple feature regression.

        Sets model summary text and creates parity plot for multiple
        regression case. Hides simple regression graph container.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.summary_model_widget.setText(self.metrics[2])
        self.show_all_containers(True)
        self.create_parity_figure()
        self.container_graph_widget.setLayout(self.container_graph_layout)
        self.container_multiple_regression_graph_widget.setVisible(True)
        self.scroll_area.show()

    def simple_linear_regression(self) -> None:
        """Display results for simple (single feature) regression.

        Displays model summary, creates regression and parity plots
        with navigation toolbars, and manages graph containers.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.summary_model_widget.setText(self.metrics[2])
        fig = self.model.get_plot_figure()
        self.graph = FigureCanvas(fig)
        self.toolbar = NavigationToolbar(self.graph, self)
        self.toolbar.setOrientation(Qt.Vertical)
        self.toolbar.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self.toolbar.setMaximumWidth(self.toolbar.sizeHint().width())
        self.container_simple_regression_graph_layout.addWidget(self.toolbar)
        self.container_simple_regression_graph_layout.addWidget(self.graph)
        self.container_simple_regression_graph_widget.setLayout(
            self.container_simple_regression_graph_layout
            )
        self.container_graph_layout.addWidget(
            self.container_simple_regression_graph_widget
            )
        self.create_parity_figure()
        self.container_graph_widget.setLayout(self.container_graph_layout)
        self.show_all_containers(True)
        self.container_simple_regression_graph_widget.setVisible(True)
        self.container_multiple_regression_graph_widget.setVisible(True)
        self.scroll_area.show()
    #---------------------------Connections-----------------------------
    @pyqtSlot(object)
    def another_file_opened(self) -> None:
        """Clear display when new file is loaded in SetupWindow.

        Clears graphs and results, shows placeholder text to indicate
        that user needs to load/create a model first.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.clear_result_window()
        self.model_path_display.clear()
        self.summary_model_widget.hide()
        self.scroll_area.hide()
        self.placeholder_text.show()
    @pyqtSlot(object)
    def train_test_df_res(self, data: list) -> None:
        """Receive train/test data, fit model, and display results.

        Receives signal from SetupWindow with training and test data,
        fits the linear regression model, computes metrics, and displays
        appropriate visualization (simple or multiple regression).
        Emits cant_be_plotted signal if model cannot be visualized.

        Parameters
        ----------
        data : list
            List containing train/test DataFrames and split summary.
            Format: [(train_df, test_df), summary_dict]

        Returns
        -------
        None
        """
        #IMPORTANT to understand, trust me, worth to undertand.
        #So we ran our own event from class SetupWindow.
        #That event is sent to MainWindow alongside the data obtained
        # from method 'split_dataframe' of SetupWindow (So the event is
        #activated when we succesfully split the data and called
        # inside that method) MainWindow works as intermediary and
        #connects the event received with a method that calls this
        #class ResultWindow and then its method
        #train_test_df_res, the one we are now.
        #So the argument data have all we get from splitting method of
        #SetupWindow. Well this was not that easy.
        #data[tuple,dict]. tuple: (train_df,test_df). Inside those:
        #train_df : first column is the output column, the target.
        #The other columns are the input/features.
        #So just take the first element from train_df and you
        #have the output, then take the rest columns and have the input.
        #test_df: same logic.
        #dict: some summary of the operation to show it on screen maybe:
        #{'n_rows_total': 20640, 'n_train': 16512, 'n_test': 4128,
        #'test_size': 0.2, 'random_seed': 42, 'shuffle': True}
        #that's an example of an output the dict had.

        self.clear_result_window()

        self.train_df = data[0][0]
        self.test_df  = data[0][1]
        self.model.set_df(self.train_df, self.test_df) #This extract
        #input-output columns from train,test df and store it
        #in attributes inside linear_regression module. This help us
        #to not pass this same df over and over again for each method
        #we call inside model.
        self.metrics = self.model.fit_and_evaluate()
        error = self.metrics[3]
        if error is not None:
            self.cant_be_plotted.emit(error)
            return
        self.placeholder_text.hide() #self explanatory xd
        self.build_prediction_inputs()
        self.container_prediction_widget.show()
        self.container_description_widget.show()
        if len(np.ravel(self.model.coef_)) != 1:
            self.multiple_linear_regression()
            return
        else:
            self.simple_linear_regression()
            return

    def create_parity_figure(self) -> None:
        """Create and display parity plot (actual vs predicted).

        Generates y vs y_hat plot from model, creates matplotlib canvas
        and toolbar, and adds to graph layout.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        parity_fig = self.model.get_y_vs_yhat_figure()
        self.parity_graph = FigureCanvas(parity_fig)
        self.parity_toolbar = NavigationToolbar(self.parity_graph, self)
        self.parity_toolbar.setOrientation(Qt.Vertical)
        self.parity_toolbar.setSizePolicy(
            QSizePolicy.Maximum, QSizePolicy.Preferred
            )
        self.parity_toolbar.setMaximumWidth(
            self.parity_toolbar.sizeHint().width()
            )
        self.container_multiple_regression_graph_layout.addWidget(
            self.parity_toolbar
            )
        self.container_multiple_regression_graph_layout.addWidget(
            self.parity_graph
            )
        self.container_multiple_regression_graph_widget.setLayout(
            self.container_multiple_regression_graph_layout
            )
        self.container_graph_layout.addWidget(
            self.container_multiple_regression_graph_widget
            )

    def save_model_dialog(self) -> None:
        """Save trained model with optional description.

        Opens save file dialog, prompts user for model description,
        and saves model using save_model_data backend function.
        Displays success or error messages accordingly.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Save user-written description as attribute
        self.model_description = self.model_description_edit.toPlainText()
        if not self.model_description.strip():
            # Give notice but continue with the creation of the model
            QMessageBox.information(self, "Info", "No model description"
                                                    " was added.\n"
            "The model will be created without a description.")
        else:
            QMessageBox.information(self, "Info", "Model description"
                                                    " captured.\n"
            "It will be attached to the model.")
        try:
            # Dialogue to choose a route
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save model",
                "",
                "Models (*.joblib)"
            )
            save_model_data(file_path, self.model, self.model_description)
            QMessageBox.information(
                self,
                "Success",
                f"The model has been saved successfully in:\n{file_path}"
            )
            self.model_description_edit.clear()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error saving",
                f"An error occurred while saving the model:\n{str(e)}"
            )
    def show_all_containers(self, value: bool) -> None:
        """Show or hide all major UI containers.

        Controls visibility of model summary, graphs, description,
        and scroll area containers.

        Parameters
        ----------
        value : bool
            True to show containers, False to hide.

        Returns
        -------
        None
        """
        widgets = [
            self.container_model_widget,
            self.container_description_widget,
            self.summary_model_widget,
            self.container_graph_widget,
            self.model_description_edit,
            self.scroll_area
            ]
        for widget in widgets:
            widget.setVisible(value)

    def load_model_data_dialog(self) -> None:
        """Load previously saved model file.

        Opens file dialog for .joblib model files, validates model
        structure, loads model object and metadata, and displays results.
        Emits model_loaded signal and shows error messages for invalid
        files.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load model",
            "",
            "Models (*.joblib);;All files (*.*)"
        )
        if not file_path:
            return

        try:
            model_data = joblib.load(file_path)
        except Exception as e:
            QMessageBox.critical(self, "Error loading", f"Could not load"
                                                    f" model file:\n{str(e)}")
            return

        # Validate model structure
        required = {"formula", "input_columns", "output_column",
                    "metrics", "description"}
        if (not isinstance(model_data, dict) or not
            required.issubset(model_data.keys())):
            QMessageBox.critical(self, "Invalid model", "The selected file"
                                        " does not contain a valid model.")
            return

        # Check for model object in loaded data
        if "model" not in model_data or model_data["model"] is None:
            QMessageBox.critical(
                self,
                "Invalid model object",
                "The selected file does not contain"
                "a valid trained model object. "
                "It was likely created with an older"
                "version of the application."
            )
            return
        # Set self.model to loaded model object
        self.model = model_data["model"]
        self.model_path_display.setText(file_path)
        self.load_model_data_GUI(model_data)
        self.model_loaded.emit()
        QMessageBox.information(self, "Model loaded", "Model loaded\n"
                                                        "successfully.")

    def build_prediction_inputs(self) -> None:
        """Create input fields for each model feature.

        Dynamically generates QLineEdit widgets for each feature in the
        model, stores references in prediction_fields dictionary for
        later retrieval of user values.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Clear previous inputs
        for i in reversed(range(self.prediction_inputs_layout.count())):
            item = self.prediction_inputs_layout.takeAt(i)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            del item
        self.prediction_fields = {}
        for feature in self.model.feature_names:
            row = QHBoxLayout()
            label = QLabel(feature + ":")
            edit = QLineEdit()
            edit.setPlaceholderText("Enter numeric value")
            row.addWidget(label)
            row.addWidget(edit)
            container = QWidget()
            container.setLayout(row)
            self.prediction_inputs_layout.addWidget(container)
            self.prediction_fields[feature] = edit

    def perform_prediction(self) -> None:
        """Make prediction using user-entered feature values.

        Retrieves values from prediction input fields, validates all
        fields are filled with numeric values, calls model.predict(),
        and displays result. Shows error messages for invalid inputs.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        try:
            values = []
            for feature, edit in self.prediction_fields.items():
                text = edit.text().strip()
                if text == "":
                    QMessageBox.warning(
                        self,
                        "Error",
                        f"Missing value for: {feature}"
                        )
                    return
                try:
                    values.append(float(text))
                except Exception:
                    QMessageBox.warning(
                        self,
                        "Error",
                        f"Invalid numeric value for: {feature}"
                        )
                    return
            X = np.array(values).reshape(1, -1)
            y_pred = self.model.predict(X)
            pred_val = float(y_pred[0])
            self.prediction_result.setText(
                f"{self.model.target_name}: {pred_val:.4f}"
                )
        except Exception as e:
            QMessageBox.critical(self, "Prediction error", str(e))