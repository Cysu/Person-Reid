#! /usr/bin/python2
# -*- coding: utf-8 -*-

import numpy as np

from PyQt4 import QtGui, QtCore

from data_tree_node import DataTreeNode


# Refer to http://pyqt.sourceforge.net/Docs/PyQt4/qabstractitemmodel.html
# for subclassing QAbstractItemModel


class DataTreeModel(QtCore.QAbstractItemModel):
    """The tree model of data (DataTreeModel)

    The DataTreeModel class is the model of tree view widget in data viewer.
    """

    def __init__(self, data_manager, parent=None):
        super(DataTreeModel, self).__init__(parent)

        # Setup model data
        self.root = DataTreeNode(QtCore.QVariant(["Name", "Size"]))

        for gid in xrange(data_manager.n_groups()):
            # Get the pedestrian Matrix
            P = data_manager.get_pedes(gid)

            # Group node
            gdata = QtCore.QVariant(["Group {0}".format(gid), 
                                     "{0}×{1}".format(P.shape[0], P.shape[1])])
            gnode = DataTreeNode(gdata, self.root)

            self.root.add_child(gnode)

            # Pedestrian nodes
            for pid in xrange(P.shape[0]):
                pdata = QtCore.QVariant(["Pedestrian {0}".format(pid), ""])
                pnode = DataTreeNode(pdata, gnode)

                gnode.add_child(pnode)


    def parent(self, index):
        """Get the parent index of a tree node

        Args:
            index: A QModelIndex of the queried tree node

        Returns:
            A QModelIndex of the corresponding parent node
        """

        if not index.isValid(): return QModelIndex()

        p = index.internalPointer().get_parent()

        if p == self.root: return QModelIndex()

        return self.createIndex(p.get_parent().get_child_index(p), 0, p)


    def data(self, index, role):
        if not index.isValid() or role != QtCore.DisplayRole: return QVariant()

        return index.internalPointer().get_data(index.column())

    def index(self, row, column, parent=QtCore.QModelIndex()):
        if not self.hasIndex(row, column, parent): return QtCore.QModelIndex()

        p = parent.internalPointer() if parent.isValid() else self.root

        c = p.get_child(row)

        return self.createIndex(row, column, c)

    def rowCount(self, parent=QtCore.QModelIndex()):
        if parent.column() > 0: return 0

        p = parent.internalPointer() if parent.isValid() else self.root

        return p.n_children()

    def columnCount(self, parent=QtCore.QModelIndex()):
        p = parent.internalPointer() if parent.isValid() else self.root

        return p.n_columns()      
