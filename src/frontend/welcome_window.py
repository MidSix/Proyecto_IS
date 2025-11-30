from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout
)

from PyQt5.QtCore import (
    Qt
)
class WelcomeWindow(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.setWindowTitle("Quick Start")
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
        self.layout.addWidget(self.welcome_message, alignment=Qt.AlignCenter)
        #this is just some HTML to format the welcome message
        self.welcome_message.setText(
            "<h1><b>Welcome to the Linear Regression App</b></h1><br>"
            "<b>1. Load data</b><br>"
            "• Use the button 'Open File' in Data Management to open a file<br>"
            "• Select between CSV, excel and MySQL data formats to load<br><br>"
            "<b>2. Select features and target</b><br>"
            "• Select input(features) columns<br>"
            "• Select output(target) column<br><br>"
            "<b>3. Confirm your selection</b><br>"
            "• Just press the 'confirm selection button'<br><br>"
            "<b>4. If you selected columns with NaN values</b><br>"
            "• Press the dropdown and select one of the four options<br>"
            "• Then press 'Apply preprocessing to handle your missing data'<br><br>"
            "<b>5. Split data into train and test sets</b><br>"
            "• Fill both input fields <br>"
            "• Use the 'create_model' button to split your data<br>"
            "  and create your linear regression model<br>"
            "• Now a summary message will appear besides where you pressed<br>"
            "  'create_model' there you'll find a summary of your model<br><br>"
            "<b>6. See your linear_regression</b><br>"
            "• Just hit the 'Model Management' button on the top bar<br>"
            "• There you'll find your parity graph and simple regression graph<br>"
            "  (if simple otherwise you'll only find the parity graph)<br>"
            "• Consult usefull metrics such as R² and MSE for both train and test<br><br>"
            "<b>7. Make a prediction</b><br>"
            "• Use the 'make a prediction' section by assigning different values<br>"
            "  ​​to the input variables to see the model's output after the prediction.<br><br>"
            "<b>8. Save your model</b><br>"
            "• You're able to add a description before saving your model<br>"
            "• To save your model just press the 'save model button'<br><br>"
            "<b>9. Load your model</b><br>"
            "• Just hit the 'load model' button and select a previosly saved model<br>"
        )

        self.setLayout(self.layout)