#!/usr/bin/python2
# -*- coding: utf-8 -*-

import sys
import cPickle
import numpy
import theano
import theano.tensor as T

from reid.preproc import imageproc
from reid.datasets import loader
from reid.datasets.datasets import Datasets
from reid.models.mlp import MultiLayerPerceptron as Mlp
from reid.models.layer import Layer

import reid.costs as costs
import reid.optimization.sgd as sgd

from PyQt4 import QtGui
from reid.utils.gui_images_gallery import ImagesGallery


_cached_datasets = 'cache/foreground_datasets.pkl'
_cached_model = 'cache/foreground_model.pkl'


def _prepare_data(load_from_cache=False, save_to_cache=False):

    if load_from_cache:
        with open(_cached_datasets, 'rb') as f:
            datasets = cPickle.load(f)
    else:
        # Setup data files

        image_filename = 'data/parse/cuhk_large_labeled_subsampled.mat'
        parse_filename = 'data/parse/cuhk_large_labeled_subsampled_parse.mat'

        images = loader.get_images_list(image_filename)
        parses = loader.get_images_list(parse_filename)

        # Pre-processing

        for i, image in enumerate(images):
            image = imageproc.subtract_luminance(image)
            image = imageproc.scale_per_channel(image, [0, 1])
            images[i] = image

        images = imageproc.images2mat(images)

        for i, parse in enumerate(parses):
            parse = imageproc.binarize(parse, 0)
            parses[i] = parse

        parses = imageproc.images2mat(parses)

        # Prepare the datasets
        
        datasets = Datasets(images, parses)
        datasets.split(train_ratio=0.5, valid_ratio=0.3)

    if save_to_cache:
        with open(_cached_datasets, 'wb') as f:
            cPickle.dump(datasets, f, protocol=cPickle.HIGHEST_PROTOCOL)

    return datasets

def _train_model(datasets, load_from_cache=False, save_to_cache=False):

    if load_from_cache:
        with open(_cached_model, 'rb') as f:
            model = cPickle.load(f)
    else:
        # Build model

        numpy_rng = numpy.random.RandomState(999987)
        layers = [Layer(numpy_rng, 38400, 1024, T.nnet.sigmoid),
                  Layer(numpy_rng, 1024, 12800, T.nnet.sigmoid)]

        model = Mlp(layers,
                    cost_func=costs.MeanBinaryCrossEntropy,
                    error_func=costs.MeanBinaryCrossEntropy)

        sgd.train(model, datasets)

    if save_to_cache:
        with open(_cached_model, 'wb') as f:
            cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)

    return model

def _view_result(model, datasets):

    class Widget(QtGui.QWidget):

        def __init__(self, result_func, datasets, parent=None):
            super(Widget, self).__init__(parent)

            self._result_func = result_func
            self._datasets = datasets

            self._cur_index = -1

            self.setWindowTitle("Result Viewer")
            self.resize(800, 500)

            self._create_ui()
            self._move_to_center()
            
        def next_sample(self):
            if self._cur_index + 1 < datasets.test_y.get_value(borrow=True).shape[0]:
                self._cur_index += 1
                self._show_sample(self._cur_index)

        def prev_sample(self):
            if self._cur_index - 1 >= 0:
                self._cur_index -= 1
                self._show_sample(self._cur_index)

        def _show_sample(self, index):

            x = self._datasets.test_x.get_value(borrow=True)[index, :]
            y = self._result_func(x)
            target = self._datasets.test_y.get_value(borrow=True)[index, :]

            y = numpy.int8(y * 255).reshape(160, 80)
            target = numpy.int8(target * 255).reshape(160, 80)

            images = numpy.asarray([[y], [target]]).reshape(1,2,160,80)

            self._gallery.show_images(images)

        def _create_ui(self):

            # Tool Bar
            self._toolbar = QtGui.QToolBar()

            next_act = QtGui.QAction("Next", self)
            next_act.setShortcut(QtGui.QKeySequence(QtGui.QKeySequence.Forward))
            next_act.triggered.connect(self.next_sample)

            prev_act = QtGui.QAction("Prev", self)
            prev_act.setShortcut(QtGui.QKeySequence(QtGui.QKeySequence.Back))
            prev_act.triggered.connect(self.prev_sample)

            self._toolbar.addAction(next_act)
            self._toolbar.addAction(prev_act)

            # Images Gallery
            self._gallery = ImagesGallery()

            # Overall Layout
            layout = QtGui.QVBoxLayout()
            layout.addWidget(self._toolbar)
            layout.addWidget(self._gallery)

            self.setLayout(layout)

        def _move_to_center(self):

            fg = self.frameGeometry()
            cp = QtGui.QDesktopWidget().availableGeometry().center()
            fg.moveCenter(cp)
            self.move(fg.topLeft())


    x = T.vector('x')
    y = model.get_outputs(x)[-1]

    result_func = theano.function(inputs=[x], outputs=y)

    app = QtGui.QApplication(sys.argv)
    widget = Widget(result_func, datasets)
    widget.show()
    return app.exec_()
            

if __name__ == '__main__':

    datasets = _prepare_data(load_from_cache=True, save_to_cache=False)

    model = _train_model(datasets, load_from_cache=True, save_to_cache=False)

    _view_result(model, datasets)
