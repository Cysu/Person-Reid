#!/usr/bin/python2
# -*- coding: utf-8 -*-

from PyQt4 import QtGui
from qimage2ndarray import array2qimage

from reid.utils.gui_flow_layout import FlowLayout


class ImagesGallery(QtGui.QWidget):
    """Images Gallery (ImagesGallery)

    The ImagesGallery class is a widget that display images in a QHBoxLayout.
    """

    def __init__(self, parent=None):
        super(ImagesGallery, self).__init__(parent)

        self._create_ui()

    def show_images(self, images):
        """Show images in a flow layout

        Args:
            images: An array of images. Each image is a numpy matrix.
        """

        nimages = images.shape[1]
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
            qimg = array2qimage(images[0, i])
            x.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def _create_ui(self):
        self.setLayout(FlowLayout())
        self.subwidgets = []