#! /usr/bin/python2
# -*- coding: utf-8 -*-

from PyQt4 import QtCore
from PyQt4.QtCore import QVariant, QModelIndex, Qt

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
        self._root = DataTreeNode([QVariant("Name"), QVariant("Size")])

        for gid in xrange(data_manager.n_groups()):
            # Get the pedestrian Matrix
            P = data_manager.get_pedes(gid)

            # Group node
            gdata = [QVariant("Group {0}".format(gid)), 
                     QVariant("{0}×{1}".format(P.shape[0], P.shape[1]))]
            gnode = DataTreeNode(gdata, self._root)

            self._root.add_child(gnode)

            # Pedestrian nodes
            for pid in xrange(P.shape[0]):
                pdata = [QVariant("Pedestrian {0}".format(pid)), 
                         QVariant("")]
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

        if p == self._root: return QModelIndex()

        return self.createIndex(p.get_parent().get_child_index(p), 0, p)

    def headerData(self, section, orientation, role):
        if orientation != Qt.Horizontal or role != Qt.DisplayRole: 
            return QVariant()

        return self._root.get_data(section)

    def data(self, index, role):
        if not index.isValid() or role != Qt.DisplayRole: return QVariant()

        return index.internalPointer().get_data(index.column())

    def index(self, row, column, parent=QModelIndex()):
        if not self.hasIndex(row, column, parent): return QModelIndex()

        p = parent.internalPointer() if parent.isValid() else self._root

        c = p.get_child(row)

        return self.createIndex(row, column, c)

    def rowCount(self, parent=QModelIndex()):
        if parent.column() > 0: return 0

        p = parent.internalPointer() if parent.isValid() else self._root

        return p.n_children()

    def columnCount(self, parent=QModelIndex()):
        p = parent.internalPointer() if parent.isValid() else self._root

        return p.n_columns()      
