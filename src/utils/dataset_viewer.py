#! /usr/bin/python2
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np

from PyQt4 import QtGui, QtCore
from qimage2ndarray import array2qimage

from data_manager import DataManager

class PedesGallery(QtGui.QWidget):

    def __init__(self, parent=None):
        super(PedesGallery, self).__init__(parent)

        self._create_ui()

    def show(self, pedes):

    def _create_ui(self):


class MainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.dm = DataManager(verbose=True)

        self._create_panels()
        self._create_menus()

        self.setWindowTitle("Person Re-id Dataset Viewer")
        self.setCentralWidget(self._pgal)
        self.showMaximized()
        self.show()

    def open(self):
        fpath = QtGui.QFileDialog.getOpenFileName(self, "Open File",
            QtCore.QDir.homePath(), tr("Matlab File (*.mat)"))

        # TODO: Handle errors
        self.dm.read(fpath)

    def _create_panels(self):
        self._pgal = PedesGallery(self)

    def _create_menus(self):
        # Actions
        open_act = QtGui.QAction("Open", self)
        open_act.setShortcut(QtGui.QKeySequence(QtGui.QKeySequence.Open))
        open_act.triggerd.connect(self.open)

        # Menu Bar
        menubar = self.menuBar()
        fileMenu = menubar.addMenu("&File")
        filemenu.addAction(open_act)

