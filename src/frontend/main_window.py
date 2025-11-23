# -*- coding: utf-8 -*-
from frontend.setup_window import SetupWindow
from frontend.result_window import ResultWindow
from frontend.welcome_window import WelcomeWindow
from PyQt5.QtCore import (
    pyqtSlot,
)
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout,
    QHBoxLayout, QSizePolicy, QStackedWidget,
)
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Linear Regression - Main window")
        #Stack windows--------------------------------------------------
        self.stacked_widget = QStackedWidget()
        self.welcome_window = WelcomeWindow(self.stacked_widget)
        self.setup_window = SetupWindow(self.stacked_widget)
        self.result_window = ResultWindow(self.stacked_widget)

        self.stacked_widget.addWidget(self.welcome_window) # índice 0
        self.stacked_widget.addWidget(self.setup_window)  # índice 1
        self.stacked_widget.addWidget(self.result_window) # indice 2
        #------------------Conections-----------------------------------
        self.setup_window.another_file_opened.connect(self.another_file_opened)
        self.setup_window.train_test_df_ready.connect(self.train_test_df_ready)
        self.result_window.cant_be_plotted.connect(self.cant_be_plotted)
        #---------------------------------------------------------------
        self.welcome_window_button = QPushButton("Quick Start")
        self.welcome_window_button.clicked.connect(
            self.change_to_welcome_window
            )
        self.setup_window_button = QPushButton("Data Management")
        self.setup_window_button.clicked.connect(
            self.change_to_setup_window
            )
        self.result_window_button = QPushButton("Model Managent")
        self.result_window_button.clicked.connect(
            self.change_to_result_window
            )
        widgets = [self.setup_window_button,self.result_window_button]

        def hide_widgets():
            for widget in widgets:
                widget.hide()

        #layouts:

        #top layout - main bar:
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.welcome_window_button)
        top_layout.addWidget(self.setup_window_button)
        top_layout.addWidget(self.result_window_button)

        #Container of main bar - this for the border:-------------------
        top_panel_widget = QWidget()
        top_panel_widget.setLayout(top_layout)
        top_panel_widget.setSizePolicy(
            QSizePolicy.Maximum, QSizePolicy.Preferred
            )
        #---------------------------------------------------------------
        main_layout = QVBoxLayout()
        main_layout.addWidget(top_panel_widget)
        main_layout.addWidget(self.stacked_widget)
        self.setLayout(main_layout)
    #Methods:
    def change_to_welcome_window(self):
        self.stacked_widget.setCurrentIndex(0)
    def change_to_setup_window(self):
        self.stacked_widget.setCurrentIndex(1)
    def change_to_result_window(self):
        self.stacked_widget.setCurrentIndex(2)

    #MainWindow as the orchestrator, the one
    #which handle the communication between these two classes.
    @pyqtSlot()
    def another_file_opened(self):
        self.result_window.another_file_opened()

    @pyqtSlot(object)
    def train_test_df_ready(self, res):
        self.result_window.train_test_df_res(res)

    @pyqtSlot(object)
    def cant_be_plotted(self, res):
        self.setup_window.cant_be_plotted(res)

if __name__ == "__main__":
    pass