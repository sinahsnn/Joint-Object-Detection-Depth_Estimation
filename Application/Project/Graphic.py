import os
import sys

import numpy as np
from PIL import Image
from PyQt5.QtGui import QPixmap, QBrush, QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QInputDialog, QPushButton, QCheckBox, \
    QFileDialog, QTableWidget, QTableWidgetItem, QGridLayout

from model_utils import load_models, process


class TableWindow(QWidget):

    def __init__(self, data):
        super(TableWindow, self).__init__()
        self.data = data
        self.keys = ['color', 'name', 'confidence', 'depth']
        self.initUI()

    def initUI(self):

        self.setGeometry(1270, 300, 400, 200)
        self.setWindowTitle('Table')

        # Grid Layout
        grid = QGridLayout()
        self.setLayout(grid)

        # Create Empty 5x5 Table
        table = QTableWidget(self)
        table.setRowCount(len(self.data))
        table.setColumnCount(len(self.keys))

        # Enter data onto Table
        for m in range(len(self.data)):
            for n in range(1, len(self.keys)):
                item = self.data[m][self.keys[n]]
                if not isinstance(item, str):
                    item = f'{item:.2f}'
                new_item = QTableWidgetItem(item)
                table.setItem(m, n, new_item)

        for m in range(len(self.data)):
            item = self.data[m][self.keys[0]]
            new_item = QTableWidgetItem('█████')
            new_item.setForeground(QBrush(QColor(item)))
            table.setItem(m, 0, new_item)

        # Add Header
        horHeaders = ['', 'Name', 'Confidence', 'Depth']
        table.setHorizontalHeaderLabels(horHeaders)

        # Adjust size of Table
        table.resizeColumnsToContents()
        table.resizeRowsToContents()

        # Add Table to Grid
        grid.addWidget(table, 0, 0)


class Window(QMainWindow):

    def __init__(self, paths):
        super(Window, self).__init__()
        self.overFlag = False
        self.indoorFlag = False
        self.threshold = 0.
        self.image = None
        self.result_image = None
        self.result_dict = None
        self.table = None

        self.models = load_models(*paths)
        self.initUI()

    def initUI(self):
        self.setFixedSize(670, 580)
        self.setGeometry(600, 300, 670, 580)
        self.setWindowTitle('Joint Object Detection & Depth Estimation')

        btn1 = QPushButton('Load Image', self)
        btn1.resize(btn1.sizeHint())
        btn1.move(10, 500)
        btn1.clicked.connect(self.load)

        btn2 = QPushButton('Set Threshold', self)
        btn2.resize(btn2.sizeHint())
        btn2.move(120, 500)
        btn2.clicked.connect(self.setThreshold)

        btn3 = QPushButton('Feed', self)
        btn3.resize(btn3.sizeHint())
        btn3.move(230, 500)
        btn3.clicked.connect(self.feedforward)

        btn4 = QPushButton('Show Table', self)
        btn4.resize(btn4.sizeHint())
        btn4.move(340, 500)
        btn4.clicked.connect(self.showTable)

        btn5 = QPushButton('Save Results', self)
        btn5.resize(btn5.sizeHint())
        btn5.clicked.connect(self.save)
        btn5.move(450, 500)

        btn6 = QPushButton('Exit', self)
        btn6.resize(btn6.sizeHint())
        btn6.clicked.connect(sys.exit)
        btn6.move(560, 500)

        cb1 = QCheckBox('Over', self)
        cb1.resize(cb1.sizeHint())
        cb1.stateChanged.connect(self.changeOver)
        cb1.move(15, 545)

        cb2 = QCheckBox('Indoor', self)
        cb2.setChecked(True)
        cb2.resize(cb2.sizeHint())
        cb2.stateChanged.connect(self.changeIndoor)
        cb2.move(125, 545)
        self.indoorFlag = True

        self.threshold_label = QLabel(self)
        self.threshold_label.move(235, 540)
        self.threshold_label.resize(200, 30)
        self.threshold_label.setText(f'Depth Threshold = {self.threshold:.2f}')

        self.result = QLabel(self)
        self.result.move(450, 540)
        self.result.resize(200, 30)
        self.result.setText('')

        self.label = QLabel(self)
        self.label.move(15, 10)

        self.show()

    def showImage(self, img, new_w=640, new_h=480):

        width, height = img.size
        if height > new_h or width > new_w:
            interp = Image.BILINEAR
        else:
            interp = Image.BICUBIC

        min_scale = min(new_w / width, new_h / height)
        new_shape = (round(min_scale * width), round(min_scale * height))
        im = img.resize(new_shape, interp)

        path = os.getcwd() + '/__temp__.jpg'
        im.save(path)
        pixmap = QPixmap(path)
        os.remove(path)

        self.label.setPixmap(pixmap)
        self.label.move(15 + (new_w - new_shape[0]) // 2, 10 + (new_h - new_shape[1]) // 2)
        self.label.resize(pixmap.width(), pixmap.height())

    def load(self):
        filePath = QFileDialog.getOpenFileName(self, 'Load Image', os.getcwd() + '/..', '*.jpg')
        if not os.path.isfile(filePath[0]):
            return

        if self.table is not None and self.table.isVisible():
            self.table.hide()

        self.image = Image.open(filePath[0])
        self.result_image = None
        self.result_dict = None
        self.table = None

        self.showImage(self.image)

    def save(self):
        if self.result_image is None:
            self.result.setText('There is no processed image!')
            return

        filePath = QFileDialog.getSaveFileName(self, 'Select Directory: Enter the name of output', os.getcwd() + '/..')
        if not os.path.dirname(filePath[0]):
            return

        path = os.path.splitext(filePath[0])[0]
        self.result_image.save(path + '.jpg')
        with open(path + '.txt', 'w') as f:
            f.write('{:<20} {:<10} {:<10} \n'.format('Name', 'Depth', 'Confidence'))
            for line in self.result_dict:
                f.write('{:<20} {:<10} {:<10} \n'.format(line['name'], '{:.2f}'.format(line['depth']), '{:.2f}'.format(line['confidence'])))

        self.result.setText('Saved!')

    def setThreshold(self):
        text, ok = QInputDialog.getText(self, 'Dialog Box', 'Enter threshold:')
        if ok:
            try:
                self.threshold = float(text)
            except:
                self.threshold = 0.
            self.threshold_label.setText(f'Depth Threshold = {self.threshold:.2f}')

    def changeOver(self, state):
        if state:
            self.overFlag = True
        else:
            self.overFlag = False

    def changeIndoor(self, state):
        if state:
            self.indoorFlag = True
        else:
            self.indoorFlag = False

    def feedforward(self):

        if self.image is None:
            self.result.setText('Load an image!')
            return

        if self.table is not None and self.table.isVisible():
            self.table.hide()

        def check_func(depth):
            if self.threshold == 0.:
                return True
            if self.overFlag:
                return depth > self.threshold
            else:
                return depth < self.threshold

        self.result_image, self.result_dict = process(*self.models, np.array(self.image), check_func, self.indoorFlag)
        self.showImage(self.result_image)

        self.result.setText('Finished!')

    def showTable(self):
        if self.result_dict is None:
            self.result.setText('There is no processed image!')
            return

        if self.table is not None and self.table.isVisible():
            self.table.hide()

        self.table = TableWindow(self.result_dict)
        self.table.show()


def main():
    app = QApplication(sys.argv)
    paths = ['yolov5', 'Models/yolov5l.pt', 'Models/nyu.pt', 'Models/kitti.pt']
    w = Window(paths)
    app.exec_()


if __name__ == '__main__':
    main()
