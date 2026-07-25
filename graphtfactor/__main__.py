import sys

from PySide6.QtWidgets import QApplication

from graphtfactor.gui.main_window import GraphTFactorWindow



def main():

    app = QApplication(sys.argv)

    window = GraphTFactorWindow()

    window.show()

    sys.exit(
        app.exec()
    )



if __name__ == "__main__":

    main()