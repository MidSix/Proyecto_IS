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
    QFileDialog, QMessageBox, QHBoxLayout, QSizePolicy, QTextEdit
)

class ResultWindow(QWidget):
    cant_be_plotted = pyqtSignal(object)
    def __init__(self, stacked_widget):
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
        "you first need to load the data,\n"
        "preprocess if needed and split\n"
        "the data into training and test sets.\n"
        )
        self.placeholder_text.setAlignment(Qt.AlignCenter)
        self.placeholder_text.setStyleSheet("color: gray; font-size: 16px;")
        self.summary = QLabel()
        self.summary.setAlignment(Qt.AlignLeft)
        #Just some QSS to make the self.summary looks better.
        self.summary.setStyleSheet("""
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
        self.model_path_display.setPlaceholderText("Select a model to load")
        self.load_model_button = QPushButton("Load model")
        self.load_model_button.clicked.connect(self.load_model_dialog_from_result)
        self.load_model_top_layout = QHBoxLayout()
        self.load_model_top_layout.addWidget(self.model_path_label)
        self.load_model_top_layout.addWidget(self.model_path_display)
        self.load_model_top_layout.addWidget(self.load_model_button)

        #------------------------containers-----------------------------
        self.main_container = QWidget()
        self.container_model_widget = QWidget()
        self.container_description_widget = QWidget()
        self.container_graph_widget = QWidget()
        self.container_model_widget.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self.container_simple_regression_graph_widget = QWidget()
        self.container_multiple_regression_graph_widget =QWidget()
        #-------------------------Layouts-------------------------------
        self.container_description_layout = QVBoxLayout()
        self.container_model_layout = QVBoxLayout()
        self.container_graph_layout = QVBoxLayout()
        self.container_simple_regression_graph_layout = QHBoxLayout()
        self.container_multiple_regression_graph_layout = QHBoxLayout()
        self.main_container_layout = QHBoxLayout()
        self.main_layout = QVBoxLayout()
        #-------------------------Set Layouts---------------------------
        self.container_description_layout.addWidget(self.model_description_edit)
        self.container_description_layout.addWidget(self.save_button)
        self.container_description_widget.setLayout(self.container_description_layout)
        self.container_model_layout.addWidget(self.summary)
        self.container_model_layout.addWidget(self.container_description_widget)
        self.container_model_layout.setStretch(0, 5)
        self.container_model_layout.setStretch(1, 3)
        self.container_model_widget.setLayout(self.container_model_layout)
        # Allow the model widget to expand horizontally so path lineedit can stretch across window
        self.container_model_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.main_container_layout.addWidget(self.container_model_widget, 0)
        self.main_container_layout.addWidget(self.container_graph_widget, 1)
        self.main_container.setLayout(self.main_container_layout)

        # ----------------- TOP PANEL WIDGET -------------
        self.top_panel_widget = QWidget()
        self.top_panel_widget.setLayout(self.load_model_top_layout)
        # Ensure the top panel doesn't get squashed vertically
        self.top_panel_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # Build the main_layout
        self.main_layout.addWidget(self.top_panel_widget)
        self.main_layout.addWidget(self.placeholder_text, 1)
        self.main_layout.addWidget(self.main_container)
        self.show_all_containers(True)
        self.setLayout(self.main_layout)
        self.container_description_widget.hide()

    # Methods:
    def clear_result_window(self):
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
            self.model_description_edit.clear()
        except Exception:
                pass

    def load_model_data_GUI(self, model_data: dict):
        try:
            summary_lines, description = load_model_data(model_data)
            self.summary.setText("\n".join(summary_lines))

            self.container_description_widget.show()

            # Add the description if it exists
            self.model_description_edit.setPlainText(description or "")

            # Show suitable containers
            self.clear_result_window()
            self.placeholder_text.hide()
            self.show_all_containers(True)
            self.main_container.show()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to"
                                 f"show loaded model:\n{str(e)}")

    def multiple_linear_regression(self):
        self.clear_result_window()
        self.summary.setText(self.metrics[2])
        self.show_all_containers(True)
        self.create_parity_figure()
        self.container_graph_widget.setLayout(self.container_graph_layout)
        self.container_multiple_regression_graph_widget.setVisible(True)
        self.main_container.show()

    def simple_linear_regression(self):
        self.clear_result_window()
        self.summary.setText(self.metrics[2])
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
        self.main_container.show()
    #---------------------------Connections-----------------------------
    @pyqtSlot(object)
    def another_file_opened(self):
        self.clear_result_window()
        self.summary.hide()
        self.main_container.hide()
        self.placeholder_text.show()
    @pyqtSlot(object)
    def train_test_df_res(self, data:list):
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
        self.container_description_widget.show()
        if len(np.ravel(self.model.coef_)) != 1:
            self.multiple_linear_regression()
            return
        else:
            self.simple_linear_regression()
            return

    def create_parity_figure(self):
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

    def save_model_dialog(self):
        """
        It allows you to save the trained model
        along with its essential information.
        Data and graphics are excluded.
        """
        # Save user-written description as attribute
        self.model_description = self.model_description_edit.toPlainText()
        if not self.model_description.strip():
            # Give notice but continue with the creation of the model
            QMessageBox.information(self, "Info", "No model description"
                                                    "was added.\n"
            "The model will be created without a description.")
        else:
            QMessageBox.information(self, "Info", "Model description"
                                                    "captured.\n"
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
    def show_all_containers(self, value):
        widgets = [
            self.container_model_widget,
            self.container_description_widget,
            self.summary,
            self.container_graph_widget,
            self.model_description_edit,
            self.main_container
            ]
        for widget in widgets:
            widget.setVisible(value)

    def load_model_dialog_from_result(self):
        """Load model but from ResultWindow (Model Management tab)."""
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
                                                    f"model file:\n{str(e)}")
            return

        # Validate model structure
        required = {"formula", "input_columns", "output_column",
                    "metrics", "description"}
        if (not isinstance(model_data, dict) or not
            required.issubset(model_data.keys())):
            QMessageBox.critical(self, "Invalid model", "The selected file"
                                        "does not contain a valid model.")
            return

        # Update UI
        self.model_path_display.setText(file_path)
        self.load_model_data_GUI(model_data)
        QMessageBox.information(self, "Model loaded", "Model loaded"
                                                        "successfully.")

if __name__ == "__main__":
    print("prueba")