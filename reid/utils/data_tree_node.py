#!/usr/bin/python2
# -*- coding: utf-8 -*-

class DataTreeNode(object):
    """Node of data in tree model (DataTreeNode)

    The instance of DataTreeNode class is the node in tree model. A node has a
    parent and some children nodes. The node iteself contains several columns 
    of data.
    """

    def __init__(self, data, parent=None):
        """Initialize the DataTreeNode

        Args:
            data: A list of QVariant variables
            parent: The parent DataTreeNode

        """

        self._data = data
        self._parent = parent
        self._children = []

    def add_child(self, node):
        """Add a child node

        Args:
            node: The child DataTreeNode to be appended
        """

        self._children.append(node)

    def get_child(self, index):
        """Get a child node

        Args:
            index: An integer index number

        Returns:
            The corresponding child DataTreeNode
        """

        return self._children[index]

    def get_child_index(self, child):
        """Get the index of a particular child node

        Args:
            child: Queried child DataTreeNode

        Returns:
            An integer indicates the index of the corresponding node
        """

        return self._children.index(child)

    def n_children(self):
        """Get the number of children"""

        return len(self._children)

    def get_data(self, column):
        """Get the data of a column

        Args:
            column: Queried column index

        Returns:
            The data stored in the column
        """

        return self._data[column]

    def n_columns(self):
        """Get the number of data columns"""

        return len(self._data)

    def get_parent(self):
        """Get the parent node"""

        return self._parent
