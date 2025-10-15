import sys
from data_module import *
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QIcon
# define main window class
class Window(QWidget):
    def __init__(self):
        super().__init__()
        # basic window
        self.setWindowTitle("Linear regression")
        self.setGeometry(200, 200, 300, 300)
        self.setWindowIcon(QIcon("icon.jpg"))

        # interface elements
        # Here the widgets are created but not showed on screen
        self.label  = QLabel("Primero carga los datos")
        self.button = QPushButton("Cargar archivos")

        # connect button to greet
        self.button.clicked.connect(self.choose_file)

        # organize widgets
        #Here we show the previously created widgets on screen
        layout = QVBoxLayout() #A pre layout configurator.
        layout.addWidget(self.label)
        layout.addWidget(self.button)

        self.setLayout(layout)

    # the function for the clicked button
    def choose_file(self):
        #QFileDialog.getOpenFileName trigger the file explorer. It returns a tuple with
        #two values. ruta is obviusly the path of the file. "_" is a name convention to say
        #that we don't care about that variable, in this case is the used filter. In this
        #case is "Archivos csv,sqlite,xls (*.csv ...) ..." is pretty likely that we won't use that in future.
        #Sintaxis of the filter:
        #"name_of_the_filter_can_be_any (name.extension)".
        #(*.*)" ->any name and any extension basically all files.  ";;" other filter applied
        #in the next selectable box in explorer.

        ruta, _ = QFileDialog.getOpenFileName(
            self,
            "Select a file",
            "",
            "Archivos csv, sqlite, xls (*.csv *.sqlite *.db *.xlsx *.xls);; csv (*.csv);; " \
            "sqlite (*.sqlite *.db);; excel (*.xlsx .*xls)"
        )
        if ruta:
            self.label.setText(f"Archivo seleccionado:\n{ruta}")
            data = DataModule(ruta)
            data.load_data()
            data.showcase_data() #Shown in the terminal after uploading the file


def main():
    app = QApplication([]) #for now we don't need CLI options, so waiting for a list of parameters
    #is not necessary at the moment. Changable if needed.
    window = Window()
    window.show() # show the window
    sys.exit(app.exec_()) # run the loop

if __name__ == "__main__":
    #This block of code won't execute when this module its importorted from main, so is just for test.
    main()

