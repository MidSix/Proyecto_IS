from src.frontend.setup_window import SetupWindow
from src.frontend.result_window import ResultWindow
from src.frontend.welcome_window import WelcomeWindow
from PyQt5.QtCore import (
    pyqtSlot,
)
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout,
    QHBoxLayout, QSizePolicy, QStackedWidget,
    QButtonGroup
)
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.quick_guide = "Quick Guide"
        self.data_management = "Data Management"
        self.model_management = "Model Management"
        self.setWindowTitle(f"Linear Regression - {self.quick_guide}")
        #Stack windows--------------------------------------------------
        self.stacked_widget = QStackedWidget()
        self.welcome_window = WelcomeWindow(self.stacked_widget)
        self.setup_window = SetupWindow(self.stacked_widget)
        self.result_window = ResultWindow(self.stacked_widget)

        self.stacked_widget.addWidget(self.welcome_window) # índice 0
        self.stacked_widget.addWidget(self.setup_window)  # índice 1
        self.stacked_widget.addWidget(self.result_window) # indice 2
        #------------------Conections-----------------------------------
        #here we connect the signals we got from the two windows
        self.setup_window.another_file_opened.connect(self.another_file_opened)
        self.setup_window.train_test_df_ready.connect(self.train_test_df_ready)
        self.result_window.cant_be_plotted.connect(self.cant_be_plotted)
        self.result_window.model_loaded.connect(self.reset_setup_window)
        #---------------------------------------------------------------
        self.welcome_window_button = QPushButton(self.quick_guide)
        self.welcome_window_button.clicked.connect(
            self.change_to_welcome_window
            )
        self.setup_window_button = QPushButton(self.data_management)
        self.setup_window_button.clicked.connect(
            self.change_to_setup_window
            )
        self.result_window_button = QPushButton(self.model_management)
        self.result_window_button.clicked.connect(
            self.change_to_result_window
            )

        # Make buttons checkable:
        for btn in (
            self.welcome_window_button,
            self.setup_window_button,
            self.result_window_button,
        ):
            btn.setCheckable(True)

        #Navigation button group - to make them exclusive:
        #In other words, only one can be selected at a time.
        #just manages the logic of exclusivity, no UI component.
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_group.addButton(self.welcome_window_button)
        self.nav_group.addButton(self.setup_window_button)
        self.nav_group.addButton(self.result_window_button)

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
        #Just some Qstyle for the buttons in the top panel:
        top_panel_widget.setStyleSheet("""
            QPushButton {
                padding: 6px 12px;
                border: none;
                background-color: #333;
                color: white;
            }
            QPushButton:hover {
                background-color: #444;
            }
            QPushButton:checked {
                background-color: #045394;
            }
        """)
        main_layout = QVBoxLayout()
        main_layout.addWidget(top_panel_widget)
        main_layout.addWidget(self.stacked_widget)
        self.setLayout(main_layout)
        self.welcome_window_button.setChecked(True)
    #Methods:
    def change_to_welcome_window(self):
        self.setWindowTitle(f"Linear Regression - {self.quick_guide}")
        self.stacked_widget.setCurrentIndex(0)
        self.welcome_window_button.setChecked(True)
    def change_to_setup_window(self):
        self.setWindowTitle(f"Linear Regression - {self.data_management}")
        self.stacked_widget.setCurrentIndex(1)
        self.setup_window_button.setChecked(True)
    def change_to_result_window(self):
        self.setWindowTitle(f"Linear Regression - {self.model_management}")
        self.stacked_widget.setCurrentIndex(2)
        self.result_window_button.setChecked(True)

    #MainWindow as the orchestrator, the one
    #which handle the communication between these two classes.
    #Signals received from SetupWindow and ResultWindow.
    #-------------------------Connections:------------------------------
    #Signals received from SetupWindow and sent to ResultWindow.
    @pyqtSlot()
    def another_file_opened(self):
        self.result_window.another_file_opened()

    @pyqtSlot(object)
    def train_test_df_ready(self, res):
        self.result_window.train_test_df_res(res)
    #Signals received from ResultWindow and sent to SetupWindow.
    @pyqtSlot(object)
    def cant_be_plotted(self, res):
        self.setup_window.cant_be_plotted(res)

    @pyqtSlot()
    def reset_setup_window(self):
        self.setup_window.reset_to_initial_state()