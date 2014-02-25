#!/usr/bin/python2
# -*- coding: utf-8 -*-

from PySide import QtGui

from reid.utils.gui_utils import ndarray2qimage
from reid.utils.gui_flow_layout import FlowLayout


class ImagesGallery(QtGui.QWidget):
    """Images Gallery (ImagesGallery)

    The ImagesGallery class is a widget that display images in a QHBoxLayout.
    """

    def __init__(self, layout='Flow', n_cols=None, parent=None):
        super(ImagesGallery, self).__init__(parent)

        self._create_ui(layout, n_cols)

    def show_images(self, images):
        """Show images in a flow layout

        Args:
            images: An array of images. Each image is a numpy matrix.
        """

        if type(images) is list:
            nimages = len(images)
        else:
            images = images.squeeze(axis=0)
            nimages = images.shape[0]

        cur_nwidgets = len(self.subwidgets)

        # Expand of shrink the sub widgets list
        if nimages > cur_nwidgets:
            for i in xrange(nimages - cur_nwidgets):
                x = QtGui.QLabel()

                if self.n_cols is None:
                    self.layout.addWidget(x)
                else:
                    r, c = len(self.subwidgets) // self.n_cols, \
                           len(self.subwidgets) % self.n_cols
                    self.layout.addWidget(x, r, c)

                self.subwidgets.append(x)
        else:
            for i in xrange(nimages, cur_nwidgets):
                self.layout.removeWidget(self.subwidgets[i])
                self.subwidgets[i].deleteLater()
            self.subwidgets = self.subwidgets[0:nimages]

        for i, x in enumerate(self.subwidgets):
            qimg = ndarray2qimage(images[i])
            x.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def _create_ui(self, layout, n_cols):
        self.n_cols = n_cols

        if layout == 'Flow':
            self.layout = FlowLayout()
        elif layout == 'HBox':
            self.layout = QtGui.QHBoxLayout()
        elif layout == 'VBox':
            self.layout = QtGui.QVBoxLayout()
        elif layout == 'Grid':
            self.layout = QtGui.QGridLayout()

        self.setLayout(self.layout)

        self.subwidgets = []
