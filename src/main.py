# -*- coding: utf-8 -*-
import os
import sys
from PyQt5.QtWidgets import QApplication
import qdarkstyle
from frontend.main_window import (
    MainWindow
)
from PyQt5.QtGui import QIcon

#All persistense .py files(stored in the computer) have a global
#variable __file__ that you can see printing the globals() namespace.
#This variable contains the path to the file. os.path.dirname(__file__)
#gets the directory path in which is stored the file. Because the icon
#is in the same directory(folder) we just apply a join and get the path
#to our ICON no matter where our wcorkspace is.
#Global inmutable cte
ICON = os.path.join(os.path.dirname(__file__), "icon.jpg")
# Just a function to set the icon.jpg as the app
#icon and as the docker icon at the moment just compatible with MacOS.
def set_app_icon(app):
    #ICON is a global inmutable constant. No need to declare global
    #cause we are not going to redefine it, just read the value.
    #So global Isn't unnecesary
    app.setWindowIcon(QIcon(ICON))

# Entry point
def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    set_app_icon(app)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()