import sys
import qdarkstyle
from PyQt5.QtWidgets import QApplication

from src.frontend.main_window import (
    MainWindow
)
from src.frontend.edit_combobox import (
    setup_global_combobox_behavior
)
from src.frontend.set_icon_app import (
    set_app_icon
)

# Entry point
def main():
    app = QApplication(sys.argv)
    # Apply global fix for all QComboBox popups
    setup_global_combobox_behavior()
    # Apply dark style to the entire application
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    set_app_icon(app)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()