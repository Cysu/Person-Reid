#! /usr/bin/python2
# -*- coding: utf-8 -*-

import numpy as np

from PyQt4 import QtGui, QtCore


# Refer to http://pyqt.sourceforge.net/Docs/PyQt4/qabstractitemmodel.html
# for subclassing QAbstractItemModel


class DataTreeModel(QtGui.QAbstractItemModel):

    def __init__(self, data, parent=None):
        super(DataTreeModel, self).__init__(parent)

    def parent(self, index):
        pass

    def data(self, index, role):
        pass

    def index(self, row, column, parent=QtGui.QModelIndex()):
        pass

    def rowCount(self, parent=QtGui.QModelIndex()):
        pass

    def columnCount(self, parent=QtGui.QModelIndex()):
        pass

    def hasChildren(self, parent=QtGui.QModelIndex()):
        pass

        
