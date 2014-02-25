#!/usr/bin/python2
# -*- coding: utf-8 -*-

import sys
import os
import shelve
import numpy
import attrconf
from scipy.io import loadmat, savemat
from PySide import QtGui, QtCore
from reid.utils.gui_images_gallery import ImagesGallery


class LabellingWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(LabellingWindow, self).__init__(parent)

        self.cpath = os.path.join(QtCore.QDir.homePath(), '.labelattr.db')
        self.fpath = None
        self.is_dirty = False

        self.set_codec('UTF-8')

        self._create_menus()
        self._create_panels()

        self.setWindowTitle(self.tr("Attribute Labelling"))
        self.showMaximized()

    @staticmethod
    def show_message(message):
        msgbox = QtGui.QMessageBox()
        msgbox.setText(message)
        msgbox.exec_()

    @staticmethod
    def set_codec(codec_name):
        codec = QtCore.QTextCodec.codecForName(codec_name)
        QtCore.QTextCodec.setCodecForLocale(codec)
        QtCore.QTextCodec.setCodecForCStrings(codec)
        QtCore.QTextCodec.setCodecForTr(codec)

    @staticmethod
    def trim(attrname):
        prefix = ['gender', 'age', 'race', 'accessory', 'carrying', 'upperBody', 'lowerBody', 'hair']
        for p in prefix:
            if attrname.startswith(p):
                return attrname[len(p):]
        return attrname

    @property
    def home_path(self):
        d = shelve.open(self.cpath)
        ret = d['home_path'] if 'home_path' in d else QtCore.QDir.homePath()
        d.close()
        return ret

    @home_path.setter
    def home_path(self, hpath):
        d = shelve.open(self.cpath)
        d['home_path'] = hpath
        d.close()

    def closeEvent(self, event):
        if self.is_dirty:
            msgbox = QtGui.QMessageBox()
            msgbox.setIcon(QtGui.QMessageBox.Warning)
            msgbox.setText(self.tr("Save changes before closing?"))
            msgbox.setStandardButtons(QtGui.QMessageBox.Discard |
                                      QtGui.QMessageBox.Cancel |
                                      QtGui.QMessageBox.Save)
            msgbox.setDefaultButton(QtGui.QMessageBox.Cancel)

            ret = msgbox.exec_()
            if ret == QtGui.QMessageBox.Discard:
                event.accept()
            elif ret == QtGui.QMessageBox.Cancel:
                event.ignore()
            else:
                self.save()
                event.accept()
        else:
            event.accept()

    @QtCore.Slot()
    def open(self):
        fpath, __ = QtGui.QFileDialog.getOpenFileName(self, self.tr("Open"),
            self.home_path, self.tr("Matlab File (*.mat)"))

        if not fpath: return

        self.fpath = fpath
        self.home_path = QtCore.QFileInfo(fpath).absolutePath()

        data = loadmat(fpath)
        self.mat_images = data['images']
        self.mat_attributes = data['attributes']

        self.cur_pid = 0
        self.is_dirty = False
        self.show_pid(self.cur_pid)

        self._next_act.setEnabled(True)
        self._prev_act.setEnabled(True)
        self._next_unlabelled_act.setEnabled(True)

    @QtCore.Slot()
    def save(self):
        if self.fpath is None: return

        self.record()

        savemat(self.fpath, {
            'images': self.mat_images,
            'attributes': self.mat_attributes
        })

        self.is_dirty = False
        self.setWindowTitle(self.tr("Attribute Labelling"))

    @QtCore.Slot()
    def save_as(self):
        fpath, __ = QtGui.QFileDialog.getSaveFileName(self, self.tr("Save As"),
            QtCore.QDir.homePath(), self.tr("Matlab File (*.mat)"))

        if not fpath: return

        self.fpath = fpath
        self.save()

    @QtCore.Slot()
    def next(self):
        self.record()

        if self._check_labelling_act.isChecked() and \
            not self.check_pid(self.cur_pid): return

        if self.cur_pid + 1 < self.mat_images.shape[0]:
            self.cur_pid += 1
            self.show_pid(self.cur_pid)

    @QtCore.Slot()
    def prev(self):
        self.record()

        if self._check_labelling_act.isChecked() and \
            not self.check_pid(self.cur_pid): return

        if self.cur_pid - 1 >= 0:
            self.cur_pid -= 1
            self.show_pid(self.cur_pid)

    @QtCore.Slot()
    def next_unlabelled(self):
        self.record()

        if self._check_labelling_act.isChecked() and \
            not self.check_pid(self.cur_pid): return

        i = self.cur_pid
        while i < self.mat_attributes.shape[0]:
            if not self.check_pid(i, False): break
            i += 1

        if i == self.mat_attributes.shape[0]:
            self.show_message(self.tr("No Unlabelled Data"))
        else:
            self.cur_pid = i
            self.show_pid(i)

    def record(self):
        self.is_dirty = True
        self.setWindowTitle(self.tr("Attribute Labelling *"))

        attr = numpy.zeros_like(self.mat_attributes[self.cur_pid, 0])

        for i, grp in enumerate(self._unival_groups):
            j = grp.checkedId()
            if j >= 0:
                k = attrconf.names.index(attrconf.unival[i][j])
                attr[k, 0] = 1

        for i, grp in enumerate(self._multival_groups):
            for j, button in enumerate(grp.buttons()):
                if button.isChecked():
                    k = attrconf.names.index(attrconf.multival[i][j])
                    attr[k, 0] = 1

        self.mat_attributes[self.cur_pid, 0] = attr

    def show_pid(self, pid):
        imgs = []
        for v in xrange(self.mat_images.shape[1]):
            if self.mat_images[pid, v].size == 0: break
            for k in xrange(self.mat_images[pid, v].shape[1]):
                imgs.append(self.mat_images[pid, v][0, k])

        self._gallery.show_images(imgs)

        attr = self.mat_attributes[pid, 0]

        for i, grp in enumerate(self._unival_groups):
            grp.setExclusive(False)
            for button in grp.buttons():
                button.setChecked(False)
            grp.setExclusive(True)

            for j, name in enumerate(attrconf.unival[i]):
                k = attrconf.names.index(name)
                grp.button(j).setChecked(attr[k, 0])

        for i, grp in enumerate(self._multival_groups):
            for j, name in enumerate(attrconf.multival[i]):
                k = attrconf.names.index(name)
                grp.button(j).setChecked(attr[k, 0])

        self._status.setText('{0} / {1}'.format(pid, self.mat_images.shape[0]))

    def check_pid(self, pid, alert=True):
        attr = self.mat_attributes[pid, 0]

        badfields = []

        for i, grp in enumerate(attrconf.unival):
            subattr = attr[[attrconf.names.index(s) for s in grp]]
            if subattr.sum() != 1: badfields.append(attrconf.unival_titles[i])

        for i, grp in enumerate(attrconf.multival):
            subattr = attr[[attrconf.names.index(s) for s in grp]]
            if subattr.sum() == 0: badfields.append(attrconf.multival_titles[i])

        if badfields and alert:
            self.show_message(self.tr("Error in fields: {0}".format(
                ", ".join(badfields))))

        return not badfields

    def _create_menus(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu(self.tr("&File"))
        file_menu.addAction(self.tr("Open"),
                            self, QtCore.SLOT("open()"),
                            QtGui.QKeySequence.Open)
        file_menu.addAction(self.tr("Save"),
                            self, QtCore.SLOT("save()"),
                            QtGui.QKeySequence.Save)
        file_menu.addAction(self.tr("Save As"),
                            self, QtCore.SLOT("save_as()"),
                            QtGui.QKeySequence.SaveAs)

        self._next_act = QtGui.QAction(self.tr("Next"), self)
        self._next_act.setShortcut(QtGui.QKeySequence.Forward)
        self._next_act.triggered.connect(self.next)
        self._next_act.setEnabled(False)

        self._prev_act = QtGui.QAction(self.tr("Prev"), self)
        self._prev_act.setShortcut(QtGui.QKeySequence.Back)
        self._prev_act.triggered.connect(self.prev)
        self._prev_act.setEnabled(False)

        self._next_unlabelled_act = QtGui.QAction(self.tr("Next Unlabelled"), self)
        self._next_unlabelled_act.triggered.connect(self.next_unlabelled)
        self._next_unlabelled_act.setEnabled(False)

        self._check_labelling_act = QtGui.QAction(self.tr("Check Labelling"), self)
        self._check_labelling_act.setCheckable(True)
        self._check_labelling_act.setChecked(True)

        edit_menu = menu_bar.addMenu(self.tr("&Edit"))
        edit_menu.addAction(self._next_act)
        edit_menu.addAction(self._prev_act)
        edit_menu.addAction(self._next_unlabelled_act)
        edit_menu.addAction(self._check_labelling_act)

        self._status = QtGui.QLabel(self.tr("No dataset opened"))

        toolbar = self.addToolBar(self.tr("Tool Bar"))
        toolbar.addWidget(self._status)
        toolbar.addAction(self._next_act)
        toolbar.addAction(self._prev_act)
        toolbar.addAction(self._next_unlabelled_act)
        toolbar.addAction(self._check_labelling_act)

    def _create_panels(self):
        self._gallery = ImagesGallery(layout='Grid', n_cols=5)

        unival_gboxes = \
            [QtGui.QGroupBox(title) for title in attrconf.unival_titles]

        multival_gboxes = \
            [QtGui.QGroupBox(title) for title in attrconf.multival_titles]

        self._unival_groups = []
        self._multival_groups = []

        for i in xrange(len(attrconf.unival)):
            group = QtGui.QButtonGroup()
            group.setExclusive(True)
            layout = QtGui.QVBoxLayout()
            for j, a in enumerate(attrconf.unival[i]):
                button = QtGui.QRadioButton(self.tr(self.trim(a)))
                group.addButton(button, j)
                layout.addWidget(button)
            layout.addStretch(1)
            self._unival_groups.append(group)
            unival_gboxes[i].setLayout(layout)

        for i in xrange(len(attrconf.multival)):
            group = QtGui.QButtonGroup()
            group.setExclusive(False)
            layout = QtGui.QVBoxLayout()
            for j, a in enumerate(attrconf.multival[i]):
                button = QtGui.QCheckBox(self.tr(self.trim(a)))
                group.addButton(button, j)
                layout.addWidget(button)
            layout.addStretch(1)
            self._multival_groups.append(group)
            multival_gboxes[i].setLayout(layout)

        layout = QtGui.QGridLayout()
        for i, gbox in enumerate(unival_gboxes):
            r, c = i // 5, i % 5
            if r == 2:
                r = 1
                c = 5
            layout.addWidget(gbox, r, c)

        for i, gbox in enumerate(multival_gboxes):
            if i < 2:
                r = 1
                c = 6 + i
            else:
                r = 0
                c = 3 + i
            layout.addWidget(gbox, r, c)

        attrpanel = QtGui.QWidget()
        attrpanel.setLayout(layout)

        sa = QtGui.QScrollArea()
        sa.setBackgroundRole(QtGui.QPalette.Base)
        sa.setWidget(attrpanel)

        layout = QtGui.QHBoxLayout()
        layout.addWidget(self._gallery)
        layout.addWidget(sa)

        frame = QtGui.QWidget()
        frame.setLayout(layout)

        self.setCentralWidget(frame)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = LabellingWindow()
    window.show()
    sys.exit(app.exec_())
