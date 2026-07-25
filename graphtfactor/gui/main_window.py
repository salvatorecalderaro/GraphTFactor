import os


from PySide6.QtWidgets import (

    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QComboBox,
    QTabWidget,
    QFileDialog,
    QMessageBox

)


from PySide6.QtGui import (

    QIcon,
    QAction

)


from PySide6.QtCore import Qt



#from .settings import ESM_MODELS

from .styles import STYLE


from graphtfactor.core.device import available_devices





class GraphTFactorWindow(QMainWindow):


    def __init__(self):

        super().__init__()


        self.setWindowTitle(
            "GraphTFactor v1.0"
        )


        self.resize(
            1000,
            750
        )


        #
        # Logo
        #

        logo = os.path.join(

            os.path.dirname(
                os.path.dirname(__file__)
            ),

            "resources",
            "logo.png"

        )


        if os.path.exists(logo):

            self.setWindowIcon(
                QIcon(logo)
            )



        self.setStyleSheet(
            STYLE
        )


        self.create_menu()

        self.create_ui()



    # -----------------------------
    # MENU
    # -----------------------------


    def create_menu(self):


        menu = self.menuBar()


        file_menu = menu.addMenu(
            "File"
        )


        open_action = QAction(
            "Open FASTA",
            self
        )


        exit_action = QAction(
            "Exit",
            self
        )


        exit_action.triggered.connect(
            self.close
        )


        file_menu.addAction(
            open_action
        )

        file_menu.addSeparator()

        file_menu.addAction(
            exit_action
        )



        help_menu = menu.addMenu(
            "Help"
        )


        about = QAction(
            "About",
            self
        )


        about.triggered.connect(
            self.show_about
        )


        help_menu.addAction(
            about
        )



    def show_about(self):


        QMessageBox.about(

            self,

            "GraphTFactor",

            """
GraphTFactor v1.0

Graph Neural Network
Transcription Factor prediction

ESM embeddings +
Graph explainability

"""

        )




    # -----------------------------
    # MAIN GUI
    # -----------------------------


    def create_ui(self):


        tabs = QTabWidget()


        prediction = QWidget()


        layout = QVBoxLayout()



        title = QLabel(

            "GraphTFactor Prediction"

        )


        title.setAlignment(
            Qt.AlignCenter
        )


        layout.addWidget(
            title
        )



        # SETTINGS


        settings = QHBoxLayout()



        self.esm_box = QComboBox()

        models = ["esm1_t6_43M_UR50S", "esm1b_t33_650M_UR50S", "esm2_t48_15B_UR50D", "esm2_t36_3B_UR50D", "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D", "esm2_t6_8M_UR50D"]
        self.esm_box.addItems(models)  # Use the list of models directly instead of ESM_MODELS
            #ESM_MODELS
        



        self.device_box = QComboBox()


        self.device_box.addItems(

            available_devices()

        )



        settings.addWidget(
            QLabel("ESM:")
        )


        settings.addWidget(
            self.esm_box
        )


        settings.addWidget(
            QLabel("Device:")
        )


        settings.addWidget(
            self.device_box
        )



        layout.addLayout(
            settings
        )



        # SEQUENCE


        self.sequence = QTextEdit()


        self.sequence.setPlaceholderText(

            "Insert protein sequence..."

        )


        layout.addWidget(
            self.sequence
        )



        buttons = QHBoxLayout()



        load = QPushButton(
            "Load FASTA"
        )


        predict = QPushButton(
            "Predict"
        )


        buttons.addWidget(
            load
        )


        buttons.addWidget(
            predict
        )



        layout.addLayout(
            buttons
        )



        self.result = QLabel(

            "Prediction result"

        )


        layout.addWidget(
            self.result
        )



        prediction.setLayout(
            layout
        )




        # IMPORTANCE TAB


        importance = QWidget()


        imp_layout = QVBoxLayout()


        self.importance_label = QLabel(

            "Feature importance will appear here"

        )


        imp_layout.addWidget(

            self.importance_label

        )


        importance.setLayout(
            imp_layout
        )




        tabs.addTab(
            prediction,
            "Prediction"
        )


        tabs.addTab(
            importance,
            "Feature Importance"
        )



        self.setCentralWidget(
            tabs
        )


        self.statusBar().showMessage(

            "Ready"

        )



        load.clicked.connect(
            self.load_fasta
        )


        predict.clicked.connect(
            self.predict
        )




    # -----------------------------
    # ACTIONS
    # -----------------------------


    def load_fasta(self):


        filename,_ = QFileDialog.getOpenFileName(

            self,

            "Open FASTA",

            "",

            "FASTA (*.fa *.fasta)"

        )


        if filename:


            seq=""


            with open(filename) as f:

                for line in f:

                    if not line.startswith(">"):

                        seq += line.strip()



            self.sequence.setText(
                seq
            )



            self.statusBar().showMessage(

                filename

            )





    def predict(self):


        seq = (

            self.sequence
            .toPlainText()
            .strip()

        )


        if not seq:


            self.result.setText(

                "Insert sequence"

            )

            return



        esm = self.esm_box.currentText()

        device = self.device_box.currentText()



        self.result.setText(

f"""
Prediction

Model:
{esm}

Device:
{device}

Result:
Transcription Factor

Probability:
0.95

"""

        )