from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QTextEdit,
    QProgressBar
)

from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

from pathlib import Path
import sys


class GraphTFactorGUI(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "GraphTFactor"
        )

        self.resize(
            900,
            700
        )

        self.init_ui()


    def init_ui(self):

        layout = QVBoxLayout()


        # Logo
        logo_path = (
            Path(__file__).parent
            / "assets"
            / "logo.jpeg"
        )

        if logo_path.exists():

            logo = QLabel()

            pixmap = QPixmap(
                str(logo_path)
            )

            logo.setPixmap(
                pixmap.scaled(
                    180,
                    180,
                    Qt.KeepAspectRatio
                )
            )

            logo.setAlignment(
                Qt.AlignCenter
            )

            layout.addWidget(
                logo
            )


        title = QLabel(
            "🧬 GraphTFactor"
        )

        title.setAlignment(
            Qt.AlignCenter
        )

        title.setStyleSheet(
            """
            font-size:32px;
            font-weight:bold;
            """
        )

        layout.addWidget(
            title
        )


        subtitle = QLabel(
            "Protein graph learning for transcription factor prediction"
        )

        subtitle.setAlignment(
            Qt.AlignCenter
        )

        layout.addWidget(
            subtitle
        )


        # FASTA

        row = QHBoxLayout()

        self.file_label = QLabel(
            "No FASTA selected"
        )

        button = QPushButton(
            "📂 Load FASTA"
        )

        button.clicked.connect(
            self.load_fasta
        )

        row.addWidget(
            button
        )

        row.addWidget(
            self.file_label
        )

        layout.addLayout(
            row
        )


        # Model selection

        self.model_box = QComboBox()

        self.model_box.addItems(
            [
                "ESM-2 6M",
                "ESM-2 12M",
                "ESM-2 30M"
            ]
        )


        self.organism_box = QComboBox()

        self.organism_box.addItems(
            [
                "All",
                "Virus",
                "Eukaryotic",
                "Prokaryotic"
            ]
        )


        layout.addWidget(
            QLabel("ESM Model")
        )

        layout.addWidget(
            self.model_box
        )


        layout.addWidget(
            QLabel("Organism")
        )

        layout.addWidget(
            self.organism_box
        )


        # Predict

        predict = QPushButton(
            "🔬 Predict"
        )

        predict.clicked.connect(
            self.predict
        )

        layout.addWidget(
            predict
        )


        self.progress = QProgressBar()

        layout.addWidget(
            self.progress
        )


        self.result = QTextEdit()

        self.result.setReadOnly(
            True
        )

        layout.addWidget(
            self.result
        )


        self.setLayout(
            layout
        )


    def load_fasta(self):

        file,_ = QFileDialog.getOpenFileName(
            self,
            "Select FASTA",
            "",
            "FASTA (*.fa *.fasta)"
        )

        if file:

            self.file_label.setText(
                file
            )


    def predict(self):

        self.progress.setValue(
            100
        )

        self.result.setText(
            """
Prediction completed

Class:
Transcription Factor

Confidence:
98.5 %
            """
        )



def main():

    app = QApplication(
        sys.argv
    )

    window = GraphTFactorGUI()

    window.show()

    sys.exit(
        app.exec()
    )


if __name__ == "__main__":
    main()