from frontend.setup_window import SetupWindow
from frontend.result_window import ResultWindow
from frontend.welcome_window import WelcomeWindow
from PyQt5.QtCore import (
    pyqtSlot,
)
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout,
    QHBoxLayout, QSizePolicy, QStackedWidget,
    QButtonGroup
)
class MainWindow(QWidget):
    """Main application window with tabbed interface.

    This class orchestrates the main user interface, managing three
    stacked windows: welcome, data setup, and results. It coordinates
    communication between windows via PyQt5 signals and slots.

    Attributes
    ----------
    quick_guide : str
        Label for the welcome window tab.
    data_management : str
        Label for the data setup window tab.
    model_management : str
        Label for the results window tab.
    stacked_widget : QStackedWidget
        Container for switching between windows.
    welcome_window : WelcomeWindow
        Welcome/guide window.
    setup_window : SetupWindow
        Data upload and split window.
    result_window : ResultWindow
        Model training and results window.
    nav_group : QButtonGroup
        Manages exclusive button selection.

    Methods
    -------
    change_to_welcome_window()
        Switch to the welcome window.
    change_to_setup_window()
        Switch to the data setup window.
    change_to_result_window()
        Switch to the results window.
    another_file_opened()
        Signal handler for file open events.
    train_test_df_ready(res)
        Signal handler for train-test split completion.
    cant_be_plotted(res)
        Signal handler for plot unavailability.
    reset_setup_window()
        Signal handler to reset the setup window.
    """
    def __init__(self) -> None:
        """Initialize the MainWindow with stacked windows and navigation."""
        super().__init__()
        self.quick_guide = "Quick Guide"
        self.data_management = "Data Management"
        self.model_management = "Model Management"
        self.setWindowTitle(f"Linear Regression - {self.quick_guide}")
        # Stack windows--------------------------------------------------
        self.stacked_widget = QStackedWidget()
        self.welcome_window = WelcomeWindow(self.stacked_widget)
        self.setup_window = SetupWindow(self.stacked_widget)
        self.result_window = ResultWindow(self.stacked_widget)

        self.stacked_widget.addWidget(self.welcome_window) # índice 0
        self.stacked_widget.addWidget(self.setup_window)  # índice 1
        self.stacked_widget.addWidget(self.result_window) # indice 2
        #------------------Conections-----------------------------------
        # Here we connect the signals we got from the two windows
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

        # Navigation button group - to make them exclusive:
        # In other words, only one can be selected at a time.
        # just manages the logic of exclusivity, no UI component.
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_group.addButton(self.welcome_window_button)
        self.nav_group.addButton(self.setup_window_button)
        self.nav_group.addButton(self.result_window_button)

        # Layouts:

        # Top layout - main bar:
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.welcome_window_button)
        top_layout.addWidget(self.setup_window_button)
        top_layout.addWidget(self.result_window_button)

        # Container of main bar - this for the border:-------------------
        top_panel_widget = QWidget()
        top_panel_widget.setLayout(top_layout)
        top_panel_widget.setSizePolicy(
            QSizePolicy.Maximum, QSizePolicy.Preferred
            )
        #---------------------------------------------------------------
        # Just some Qstyle for the buttons in the top panel:
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
    # Methods:
    def change_to_welcome_window(self) -> None:
        """Switch to the welcome/quick guide window.

        Updates the window title and stacks index to display the
        welcome window. Sets its button as checked.

        Returns
        -------
        None
        """
        self.setWindowTitle(f"Linear Regression - {self.quick_guide}")
        self.stacked_widget.setCurrentIndex(0)
        self.welcome_window_button.setChecked(True)

    def change_to_setup_window(self) -> None:
        """Switch to the data management window.

        Updates the window title and stacks index to display the
        setup window. Sets its button as checked.

        Returns
        -------
        None
        """
        self.setWindowTitle(f"Linear Regression - {self.data_management}")
        self.stacked_widget.setCurrentIndex(1)
        self.setup_window_button.setChecked(True)

    def change_to_result_window(self) -> None:
        """Switch to the model management window.

        Updates the window title and stacks index to display the
        results window. Sets its button as checked.

        Returns
        -------
        None
        """
        self.setWindowTitle(f"Linear Regression - {self.model_management}")
        self.stacked_widget.setCurrentIndex(2)
        self.result_window_button.setChecked(True)

    # MainWindow as the orchestrator, the one
    # which handle the communication between these two classes.
    # Signals received from SetupWindow and ResultWindow.
    #-------------------------Connections:------------------------------
    # Signals received from SetupWindow and sent to ResultWindow.
    @pyqtSlot()
    def another_file_opened(self) -> None:
        """Notify ResultWindow that a new file has been opened.

        This signal handler is triggered when SetupWindow opens a new
        data file. Propagates the signal to ResultWindow to reset any
        previous results.

        Returns
        -------
        None
        """
        self.result_window.another_file_opened()

    @pyqtSlot(object)
    def train_test_df_ready(self, res: object) -> None:
        """Pass train-test split results to ResultWindow.

        This signal handler receives the train-test split results from
        SetupWindow and forwards them to ResultWindow for model training.

        Parameters
        ----------
        res : object
            Train-test split result object from SetupWindow.

        Returns
        -------
        None
        """
        self.result_window.train_test_df_res(res)

    # Signals received from ResultWindow and sent to SetupWindow.
    @pyqtSlot(object)
    def cant_be_plotted(self, res: object) -> None:
        """Notify SetupWindow that a plot cannot be generated.

        This signal handler receives notification from ResultWindow when
        plotting is not possible (e.g., multiple features) and forwards
        it to SetupWindow to inform the user.

        Parameters
        ----------
        res : object
            Error or status information from ResultWindow.

        Returns
        -------
        None
        """
        self.setup_window.cant_be_plotted(res)

    @pyqtSlot()
    def reset_setup_window(self) -> None:
        """Reset SetupWindow to initial state after model loading.

        This signal handler is triggered when ResultWindow loads a
        saved model. Resets SetupWindow to allow a new workflow.

        Returns
        -------
        None
        """
        self.setup_window.reset_to_initial_state()