from PySide6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout
from PySide6.QtCore import QTimer
from energy_skill import run_energy_skill

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.video = QLabel()
        self.btn = QPushButton("Start Energy Skill")
        self.btn.clicked.connect(self.start_skill)

        layout = QVBoxLayout()
        layout.addWidget(self.video)
        layout.addWidget(self.btn)
        self.setLayout(layout)

        self.timer = QTimer()

    def start_skill(self):
        update_fn = run_energy_skill(self.video, cam_index=4)
        self.timer.timeout.connect(update_fn)
        self.timer.start(16)

if __name__ == "__main__":
    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec()
