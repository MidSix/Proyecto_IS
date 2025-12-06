from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QScrollArea, QSizePolicy
)

from PyQt5.QtCore import (
    Qt
)
class WelcomeWindow(QWidget):
    """Welcome screen showing the app's quick start guide.

    Displays formatted HTML instructions in a scrollable area covering
    9 main steps: loading data, selecting features, preprocessing,
    splitting, training, visualization, prediction, saving, and
    loading models.

    Attributes
    ----------
    stacked_widget : QStackedWidget
        Reference to the parent stacked widget for window navigation.
    welcome_message : QLabel
        Label containing HTML-formatted welcome guide text.
    scroll_area : QScrollArea
        Scrollable container for the welcome message.
    layout : QVBoxLayout
        Main vertical layout for the window.

    Methods
    -------
    __init__(stacked_widget)
        Initialize the welcome window with formatted instructions.
    """
    def __init__(self, stacked_widget) -> None:
        """Initialize the welcome window.

        Sets up the formatted HTML welcome message, styling, and
        scrollable layout.

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
        self.setWindowTitle("Quick Guide")
        self.layout = QVBoxLayout()
#----------------------------set_label----------------------------------
        self.welcome_message = QLabel()
        self.welcome_message.setStyleSheet("""
                                QLabel {
                                font-family: 'Consolas';
                                font-size: 12pt;
                                color: #E0E0E0;
                                }
                                """)
        #this is just some HTML to format the welcome message
        self.welcome_message.setText(
            "<h1><b>Welcome to the Linear Regression App</b></h1><br>"

            "<b>1. Load data</b><br>"
                "• Use the button 'Open File'"
                "in Data Management to open a file<br>"

                "• Select between CSV, excel and MySQL"
                "data formats to load<br><br>"

            "<b>2. Select features and target</b><br>"
                "• Select input(features) columns<br>"
                "• Select output(target) column<br><br>"

            "<b>3. Confirm your selection</b><br>"
                "• Just press the 'confirm selection button'<br><br>"

            "<b>4. If you selected columns with NaN values</b><br>"
                "• Press the dropdown and select one of the four options<br>"
                "• Then press 'Apply preprocessing to"
                "handle your missing data'<br><br>"

            "<b>5. Split data into train and test sets</b><br>"
                "• Fill both input fields <br>"

                "• Use the 'create_model' button to split your data<br>"
                "  and create your linear regression model<br>"

                "• Now a summary message will appear"
                " besides where you pressed<br>"
                "  'create_model' there you'll find a "
                "summary of your model<br><br>"

            "<b>6. See your linear_regression</b><br>"
                "• Just hit the 'Model Management' button on the top bar<br>"

                "• There you'll find your parity graph "
                "and simple regression graph<br>"
                "  (if simple otherwise you'll only find the parity graph)<br>"

                "• Consult usefull metrics such as R² and "
                "MSE for both train and test<br><br>"

            "<b>7. Make a prediction</b><br>"
                "• Use the 'make a prediction' section "
                "by assigning different values<br>"
                "  ​​to the input variables to see the model's "
                "output after the prediction.<br><br>"

            "<b>8. Save your model</b><br>"
                "• You're able to add a description "
                "before saving your model<br>"
                "• To save your model just press "
                "the 'save model button'<br><br>"
            "<b>9. Load your model</b><br>"
            "• Just hit the 'load model' button "
            "and select a previosly saved model<br>"
        )
        # Label behavior
        self.welcome_message.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Expanding
        )
        self.welcome_message.setAlignment(Qt.AlignLeft)
        # ---------------------------- Container -----------------------
        self.container_label_widget = QWidget()
        self.container_label_layout = QVBoxLayout(self.container_label_widget)
        self.container_label_layout.addWidget(
            self.welcome_message,
            alignment=Qt.AlignCenter
            )

        # ---------------------------- Scroll Area ---------------------
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setWidget(self.container_label_widget)

        # Add scroll area to main layout
        self.layout.addWidget(self.scroll_area)

        self.setLayout(self.layout)