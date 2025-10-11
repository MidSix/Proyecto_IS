import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout

class window(QWidget):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Greeting")
        self.setGeometry(200, 200, 300, 300)
        
        self.label = QLabel("What's your name?")
        self.text_box = QLineEdit()
        self.button = QPushButton("Accept")
        self.greeting = QLabel("")
        
        self.button.clicked.connect(self.greet)
        
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.text_box)
        layout.addWidget(self.button)
        layout.addWidget(self.greeting)
        
        self.setLayout(layout)
    
    def greet(self):
        name = self.text_box.text().strip()
        if name:
            self.greeting.setText(f"Hello, {name}!")
        else:
            self.greeting.setText("Please, write your name.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = window()
    window.show()
    sys.exit(app.exec_())