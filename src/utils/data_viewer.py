#! /usr/bin/python2
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np

from PyQt4 import QtGui, QtCore
from qimage2ndarray import array2qimage

from data_manager import DataManager


class ImagesGallery(QtGui.QWidget):
    """Images Gallery (ImagesGallery)

    The ImagesGallery class is a widget that display images in a QHBoxLayout.
    """

    def __init__(self, parent=None):
        super(ImagesGallery, self).__init__(parent)

        self._create_ui()

    def show_images(self, images):
        """Show images in a QHBoxLayout

        Args:
            images: An array of images. Each image is a numpy matrix.
        """

        nimages = images.shape[0]
        cur_nwidgets = len(self.subwidgets)

        layout = self.layout()

        # Expand of shrink the sub widgets list

        if nimages > cur_nwidgets:
            for i in xrange(nimages - cur_nwidgets):
                x = QtGui.QLabel()
                self.subwidgets.append(x)
                layout.addWidget(x)
        else:
            for i in xrange(nimages, cur_nwidgets):
                layout.removeWidget(self.subwidgets[i])
                self.subwidgets[i].deleteLater()
            self.subwidgets = self.subwidgets[0:nimages]

        for i, x in enumerate(self.subwidgets):
            qimg = array2qimage(images[i])
            x.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def _create_ui(self):
        self.setLayout(QtGui.QHBoxLayout())
        self.subwidgets = []


class PedesGallery(QtGui.QWidget):
    """Pedestrian Images Gallery (PedesGallery)

    The PedesGallery class is a widget that display images of a pedestrian from 
    different views.
    """

    def __init__(self, parent=None):
        super(PedesGallery, self).__init__(parent)

        self._create_ui()

    def show_pedes(self, pedes):
        """Show images of a same pedestrian from different views 

        Args:
            pedes: An array of different views of the pedestrian. Each element 
            is itself an array of images in that view.
        """

        nviews = pedes.shape[0]
        cur_nwidgets = len(self.subwidgets)
        
        layout = self.layout()

        # Expand or shrink the sub widgets list

        if nviews > cur_nwidgets:
            for i in xrange(nviews - cur_nwidgets):
                x = ImagesGallery()
                self.subwidgets.append(x)
                layout.addWidget(x)
        else:
            for i in xrange(nviews, cur_nwidgets):
                layout.removeWidget(self.subwidgets[i])
                self.subwidgets[i].deleteLater()
            self.subwidgets = self.subwidgets[0:nviews]

        for i, x in enumerate(self.subwidgets):
            x.show_images(np.squeeze(pedes[i]))


    def _create_ui(self):
        self.setLayout(QtGui.QVBoxLayout())
        self.subwidgets = []


class MainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.dm = DataManager(verbose=True)

        self._create_panels()
        self._create_menus()

        self.setWindowTitle("Person Re-id Dataset Viewer")
        self.setCentralWidget(self.gallery)
        self.showMaximized()

    def open(self):
        fpath = QtGui.QFileDialog.getOpenFileName(self, "Open File",
            QtCore.QDir.homePath(), "Matlab File (*.mat)")

        # TODO: Handle errors
        self.dm.read(str(fpath))  # Convert QString into Python String

    def _create_panels(self):
        self.gallery = PedesGallery(self)

    def _create_menus(self):
        # Actions
        open_act = QtGui.QAction("Open", self)
        open_act.setShortcut(QtGui.QKeySequence(QtGui.QKeySequence.Open))
        open_act.triggered.connect(self.open)

        # Menu Bar
        menubar = self.menuBar()
        fileMenu = menubar.addMenu("&File")
        fileMenu.addAction(open_act)


def main():
    app = QtGui.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
