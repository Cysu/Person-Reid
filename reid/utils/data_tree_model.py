#!/usr/bin/python2
# -*- coding: utf-8 -*-

from PySide import QtCore
from PySide.QtCore import QModelIndex, Qt

from reid.utils.data_tree_node import DataTreeNode


class DataTreeModel(QtCore.QAbstractItemModel):
    """The tree model of data (DataTreeModel)

    The DataTreeModel class is the model of tree view widget in data viewer.
    """

    def __init__(self, data_loader, parent=None):
        super(DataTreeModel, self).__init__(parent)

        # Setup model data
        self._root = DataTreeNode(["Name", "Size"])

        for gid in xrange(data_loader.get_n_groups()):
            # Get the pedestrian Matrix
            P = data_loader.get_pedes(gid)

            # Group node
            gdata = ["Group {0}".format(gid), 
                     "{0}×{1}".format(P.shape[0], P.shape[1])]
            gnode = DataTreeNode(gdata, self._root)

            self._root.add_child(gnode)

            # Pedestrian nodes
            for pid in xrange(P.shape[0]):
                n_images = data_loader.get_n_images(gid, pid)

                pdata = ["Pedestrian {0}".format(pid), 
                         ' + '.join(map(str, n_images))]
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
            return None

        return self._root.get_data(section)

    def data(self, index, role):
        if not index.isValid() or role != Qt.DisplayRole: return None

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
