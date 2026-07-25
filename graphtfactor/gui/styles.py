# graphtfactor/gui/styles.py


STYLE = """

/* Main window */

QMainWindow {

    background-color: #121212;

    color: #eeeeee;

}



/* Menu bar */

QMenuBar {

    background-color: #1e1e1e;

    color: #ffffff;

    padding: 4px;

}


QMenuBar::item:selected {

    background-color: #2d2d2d;

}



/* Menus */

QMenu {

    background-color: #1e1e1e;

    color: #ffffff;

    border: 1px solid #333333;

}


QMenu::item:selected {

    background-color: #2563eb;

}



/* Labels */

QLabel {

    color: #eeeeee;

    font-size: 14px;

}



/* Text input */

QTextEdit {

    background-color: #1e1e1e;

    color: #ffffff;

    border: 1px solid #444444;

    border-radius: 6px;

    padding: 8px;

    font-size: 14px;

}



/* Buttons */

QPushButton {

    background-color: #2563eb;

    color: white;

    border-radius: 6px;

    padding: 8px 16px;

    font-size: 14px;

}


QPushButton:hover {

    background-color: #3b82f6;

}


QPushButton:pressed {

    background-color: #1d4ed8;

}



/* Combo boxes */

QComboBox {

    background-color: #1e1e1e;

    color: white;

    border: 1px solid #444444;

    border-radius: 5px;

    padding: 6px;

}



QComboBox:hover {

    border: 1px solid #2563eb;

}



QComboBox QAbstractItemView {

    background-color: #1e1e1e;

    color:white;

    selection-background-color:#2563eb;

}



/* Tabs */

QTabWidget::pane {

    border: 1px solid #333333;

    background:#121212;

}


QTabBar::tab {

    background:#1e1e1e;

    color:#aaaaaa;

    padding:10px 20px;

    border-radius:5px;

}


QTabBar::tab:selected {

    background:#2563eb;

    color:white;

}



/* Status bar */

QStatusBar {

    background:#1e1e1e;

    color:#aaaaaa;

}



/* Group boxes */

QGroupBox {

    color:#ffffff;

    border:1px solid #333333;

    border-radius:8px;

    margin-top:10px;

}


QGroupBox::title {

    color:#60a5fa;

}

"""