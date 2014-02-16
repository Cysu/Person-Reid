#!/usr/bin/python2
# -*- coding: utf-8 -*-

import os
import numpy
import attrconf
import bodyconf
from reid.utils.cache_manager import CacheManager


cachem = CacheManager(os.path.join('..', 'cache'), 'attr')


@cachem.save('raw')
def load_data(datasets):
    """Load images, corresponding body parts maps and attributes vectors

    Args:
        datasets: A list of datasets names
    
    Returns:
        A list of tuples. Each is for one pedestrian in the form of
        ``(image, body_parts_map, attribute)``.
    """

    data = cachem.load('raw')

    if data is None:
        from scipy.io import loadmat

        data = []

        for name in datasets:
            matfp = os.path.join('..', 'data', 'attributes', name + '_parse.mat')
            matdata = loadmat(matfp)

            m, n = matdata['images'].shape
            for i in xrange(m):
                for j in xrange(n):
                    if matdata['images'][i, j].size == 0: break
                    data.append((matdata['images'][i, j],
                                 matdata['bodyparts'][i, j],
                                 matdata['attributes'][i, 0].ravel()))

    return data


@cachem.save('decomp')
def decompose(rawdata, dilation_radius=3):
    """Decompose pedestrain into several body parts groups and 

    This function will generate an image containing only the region of
    particular body parts for each group.

    Args:
        rawdata: A list of pedestrian tuples returned by ``load_data``
        dilation_radius: The radius of dilation to be performed on group region

    Returns:
        A list of tuples with each tuple for one pedestrian in the form of
        ``(img, [img_0, img_1, ... , img_m], attribute)`` where ``img`` is the
        original image and ``m`` is the number of body parts groups.
    """

    data = cachem.load('decomp')

    if data is None:
        import skimage.morphology as morph
        selem = morph.square(2*dilation_radius)

        # Decompose pedestrian image into body parts
        def decomp_body(img, bpmap):
            imgs = [0] * len(bodyconf.groups)

            for i, grp in enumerate(bodyconf.groups):
                # mask = dilate(region_0 | region_1 | ... | region_k)
                regions = [(bpmap == pixval) for pixval in grp]
                mask = reduce(lambda x, y: x|y, regions)
                mask = morph.binary_dilation(mask, selem)
                imgs[i] = img * numpy.expand_dims(mask, axis=2)

            return imgs

        # Decompose attributes vector into uni- and multi- value groups
        def decomp_attr(attr):
            attrs = []

            for grp in attrconf.unival:
                subattr = attr[[attrconf.names.index(s) for s in grp]]
                if subattr.sum() != 1: return None
                attrs.append(numpy.where(subattr == 1)[0])

            for grp in attrconf.multival:
                subattr = attr[[attrconf.names.index(s) for s in grp]]
                if subattr.sum() == 0: return None
                attrs.append(subattr)

            return attrs

        data = []
        for img, bpmap, attr in rawdata:
            attr = decomp_attr(attr)
            if attr is None: continue
            body = decomp_body(img, bpmap)
            data.append((img, body, attr))

    return data


@cachem.save('dataset')
def create_dataset(data):
    """Create dataset for model training, validation and testing

    This function will pre-process the decomposed images and flatten each sample
    into a vector.

    Args:
        data: A list of pedestrian tuples returned by ``decomp_body``

    Returns:
        A Dataset object to be used for model training, validation and
        testing.
    """

    dataset = cachem.load('dataset')

    if dataset is None:
        from reid.utils.dataset import Dataset
        from reid.preproc import imageproc

        def imgprep(img):
            img = imageproc.imresize(img, (80, 30))
            img = imageproc.subtract_luminance(img)
            img = numpy.rollaxis(img, 2)
            return img

        m = len(data)
        X, Y = [0] * m, [0] * m

        for i, (__, imgs, attr) in enumerate(data):
            X[i] = [imgprep(img) for img in imgs]
            X[i] = numpy.asarray(X[i], dtype=numpy.float32).ravel()
            Y[i] = numpy.concatenate(attr).astype(numpy.float32)

        X = numpy.asarray(X)
        Y = numpy.asarray(Y)

        X = numpy.tanh(X - X.mean(axis=0))

        dataset = Dataset(X, Y)
        dataset.split(0.7, 0.2)

    return dataset


@cachem.save('model')
def train_model(dataset):
    """Train deep model

    This function will build up a deep neural network and train it using given
    dataset.

    Args:
        dataset: A Dataset object returned by ``create_dataset``

    Returns:
        The trained deep model.
    """

    model = cachem.load('model')

    if model is None:
        import reid.models.active_functions as actfuncs
        import reid.models.cost_functions as costfuncs
        from reid.models.layers import ConvPoolLayer, FullConnLayer
        from reid.models.layers import CompLayer, DecompLayer
        from reid.models.neural_net import NeuralNet, MultiwayNeuralNet
        from reid.models.evaluate import Evaluator
        from reid.optimization import sgd

        output_sizes = [len(grp) for grp in attrconf.unival + attrconf.multival]
        target_sizes = [1] * len(attrconf.unival) + [len(grp) for grp in attrconf.multival]

        # Build up model
        input_decomp = DecompLayer([(3,80,30)] * 4)

        columns = MultiwayNeuralNet([NeuralNet([
            ConvPoolLayer((64,3,3,3), (2,2), (3,80,30), actfuncs.tanh, False),
            ConvPoolLayer((64,64,3,3), (2,2), None, actfuncs.tanh, True)
        ]) for __ in xrange(len(bodyconf.groups))])

        feature_comp = CompLayer(strategy='Maxout')

        classify_1 = FullConnLayer(6912, 99)
        classify_2 = FullConnLayer(99, 99)

        attr_decomp = DecompLayer(
            [(sz,) for sz in output_sizes],
            [actfuncs.softmax] * len(attrconf.unival) + \
            [actfuncs.sigmoid] * len(attrconf.multival)
        )

        model = NeuralNet([input_decomp, columns, feature_comp, classify_1, classify_2, attr_decomp])

        # Build up adapter
        adapter = DecompLayer([(sz,) for sz in target_sizes])

        # Build up evaluator
        cost_functions = [costfuncs.mean_negative_loglikelihood] * len(attrconf.unival) + \
                         [costfuncs.mean_binary_cross_entropy] * len(attrconf.multival)

        error_functions = [costfuncs.mean_number_misclassified] * len(attrconf.unival) + \
                          [costfuncs.mean_zeroone_error_rate] * len(attrconf.multival)

        evaluator = Evaluator(model, cost_functions, error_functions, adapter,
                              regularize=1e-3)

        # Train the model
        sgd.train(evaluator, dataset,
                  learning_rate=5e-3, momentum=0.9,
                  batch_size=300, n_epoch=200,
                  learning_rate_decr=1.0, patience_incr=1.5)

    return model


@cachem.save('result')
def compute_result(model, dataset, data):
    """Compute output value of data samples by using the trained model

    Args:
        model: A NeuralNet model
        dataset: A Dataset object returned by ``create_dataset``
        data:  list of pedestrian tuples returned by ``decomp_body``

    Returns:
        A tuple (train, valid, test) where each is three matrices
        (image_tensor4D, output_matrix, target_matrix)
    """

    result = cachem.load('result')

    if result is None:
        import theano
        import theano.tensor as T
        from reid.models.layers import CompLayer
        from reid.models.neural_net import NeuralNet

        model = NeuralNet([model, CompLayer()])
        x = T.matrix()
        f = theano.function(inputs=[x], outputs=model.get_output(x))

        def compute_output(X):
            outputs = [f(X[i:i+1, :]).ravel() for i in xrange(X.shape[0])]
            return numpy.asarray(outputs)

        images = numpy.asarray([p[0] for p in data])

        train = (images[dataset.train_ind],
                 compute_output(dataset.train_x.get_value(borrow=True)),
                 dataset.train_y.get_value(borrow=True))

        valid = (images[dataset.valid_ind],
                 compute_output(dataset.valid_x.get_value(borrow=True)),
                 dataset.valid_y.get_value(borrow=True))

        test = (images[dataset.test_ind],
                compute_output(dataset.test_x.get_value(borrow=True)),
                dataset.test_y.get_value(borrow=True))

        result = (train, valid, test)

    return result


def show_stats(result):
    """Show the statistics of the result

    Args:
        result: Tuple returned by ``compute_result``
    """

    train, valid, test = result

    output_sizes = [len(grp) for grp in attrconf.unival + attrconf.multival]
    target_sizes = [1] * len(attrconf.unival) + [len(grp) for grp in attrconf.multival]

    output_seg = [0] + list(numpy.cumsum(output_sizes))
    target_seg = [0] + list(numpy.cumsum(target_sizes))

    def print_stats(title, outputs, targets):
        print "Statistics of {0}".format(title)
        print "=" * 80

        for i, (grptitle, grp) in \
                enumerate(zip(attrconf.unival_titles, attrconf.unival)):
            print "{0}, frequency, accuracy".format(grptitle)

            o = outputs[:, output_seg[i]:output_seg[i+1]]
            t = targets[:, target_seg[i]:target_seg[i+1]]
            p = o.argmax(axis=1).reshape(o.shape[0], 1)

            freqs, accs = [0] * len(grp), [0] * len(grp)
            for j, attrname in enumerate(grp):
                freqs[j] = (t == j).mean()
                accs[j] = ((t == j) & (p == j)).sum() * 1.0 / (t == j).sum()
                if numpy.isnan(accs[j]): accs[j] = 0
                print "{0},{1},{2}".format(attrname, freqs[j], accs[j])

            freqs = numpy.asarray(freqs)
            accs = numpy.asarray(accs)

            print "Overall accuracy = {0}".format((freqs * accs).sum())
            print ""

        for i, (grptitle, grp) in \
                enumerate(zip(attrconf.multival_titles, attrconf.multival)):
            print "{0}, frequency, TPR, FPR".format(grptitle)

            offset = len(attrconf.unival)

            o = outputs[:, output_seg[offset+i]:output_seg[offset+i+1]]
            t = targets[:, target_seg[offset+i]:target_seg[offset+i+1]]
            p = o.round()

            # Any multi-value group must have at least one attribute activated
            for k in xrange(p.shape[0]):
                if p[k, :].sum() == 0:
                    v = o[k, :].argmax()
                    p[k, v] = 1

            freqs, tprs, fprs = [0] * len(grp), [0] * len(grp), [0] * len(grp)
            for j, attrname in enumerate(grp):
                tj, pj = t[:, j], p[:, j]
                freqs[j] = (tj == 1).mean()
                tprs[j] = ((tj == 1) & (pj == 1)).sum() * 1.0 / (tj == 1).sum()
                fprs[j] = ((tj == 0) & (pj == 1)).sum() * 1.0 / (tj == 0).sum()
                if numpy.isnan(tprs[j]): tprs[j] = 0
                if numpy.isnan(fprs[j]): fprs[j] = 0
                print "{0},{1},{2},{3}".format(attrname, freqs[j], tprs[j], fprs[j])

            freqs = numpy.asarray(freqs)
            tprs = numpy.asarray(tprs)
            fprs = numpy.asarray(fprs)

            print "Overal TPR = {0}, FPR = {1}".format(
                (freqs * tprs).sum() / freqs.sum(),
                (freqs * fprs).sum() / freqs.sum()
            )
            print ""

    print_stats("Training Set", train[1], train[2])
    print_stats("Validation Set", valid[1], valid[2])
    print_stats("Testing Set", test[1], test[2])


def show_result(result):
    """Show the result in GUI

    Args:
        result: Tuples returned by ``compute_result``
    """

    import sys
    from PySide import QtGui, QtCore
    from reid.utils.gui_utils import ndarray2qimage

    output_sizes = [len(grp) for grp in attrconf.unival + attrconf.multival]
    target_sizes = [1] * len(attrconf.unival) + [len(grp) for grp in attrconf.multival]

    output_seg = [0] + list(numpy.cumsum(output_sizes))
    target_seg = [0] + list(numpy.cumsum(target_sizes))

    def compare_unival(table, output, target):
        cur_row = 0
        for i, (grptitle, grp) in \
                enumerate(zip(attrconf.unival_titles, attrconf.unival)):
            table.setItem(cur_row, 0, QtGui.QTableWidgetItem(grptitle))
            cur_row += 1

            o = output[output_seg[i]:output_seg[i+1]]
            t = target[target_seg[i]:target_seg[i+1]]
            p = o.argmax()

            for j, attrname in enumerate(grp):
                table.setItem(cur_row, 0, QtGui.QTableWidgetItem(attrname))
                table.setItem(cur_row, 1, QtGui.QTableWidgetItem('*' if t[0] == j else ''))
                table.setItem(cur_row, 2, QtGui.QTableWidgetItem('√' if p == j else ''))
                table.setItem(cur_row, 3, QtGui.QTableWidgetItem('{0:.5}'.format(o[j])))
                cur_row += 1

            cur_row += 1

    def compare_multival(table, output, target):
        cur_row = 0
        for i, (grptitle, grp) in \
                enumerate(zip(attrconf.multival_titles, attrconf.multival)):
            table.setItem(cur_row, 0, QtGui.QTableWidgetItem(grptitle))
            cur_row += 1

            offset = len(attrconf.unival)

            o = output[output_seg[offset+i]:output_seg[offset+i+1]]
            t = target[target_seg[offset+i]:target_seg[offset+i+1]]
            p = o.round()

            # Any multi-value group must have at least one attribute activated
            if p.sum() == 0:
                v = o.argmax()
                p[v] = 1

            for j, attrname in enumerate(grp):
                table.setItem(cur_row, 0, QtGui.QTableWidgetItem(attrname))
                table.setItem(cur_row, 1, QtGui.QTableWidgetItem('*' if t[j] == 1 else ''))
                table.setItem(cur_row, 2, QtGui.QTableWidgetItem('√' if p[j] == 1 else ''))
                table.setItem(cur_row, 3, QtGui.QTableWidgetItem('{0:.5}'.format(o[j])))
                cur_row += 1

            cur_row += 1


    class ResultViewer(QtGui.QMainWindow):
        def __init__(self, sets, parent=None):
            super(ResultViewer, self).__init__(parent)

            self.set_codec("UTF-8")

            self._sets = sets

            # Set default index to the first pedestrian in training set
            self._cur_sid = self._cur_pid = 0

            self._create_menus()
            self._create_panels()

            self.setWindowTitle(self.tr("Result Viewer"))
            self.showMaximized()

        def set_codec(self, codec_name):
            codec = QtCore.QTextCodec.codecForName(codec_name)
            QtCore.QTextCodec.setCodecForLocale(codec)
            QtCore.QTextCodec.setCodecForCStrings(codec)
            QtCore.QTextCodec.setCodecForTr(codec)

        @QtCore.Slot()
        def show_train_set(self):
            self._cur_sid = 0
            self._cur_pid = 0
            self._show_current()

        @QtCore.Slot()
        def show_valid_set(self):
            self._cur_sid = 1
            self._cur_pid = 0
            self._show_current()

        @QtCore.Slot()
        def show_test_set(self):
            self._cur_sid = 2
            self._cur_pid = 0
            self._show_current()

        @QtCore.Slot()
        def show_next_pedes(self):
            if self._cur_pid + 1 < len(self._sets[self._cur_sid][0]):
                self._cur_pid += 1
                self._show_current()
            else:
                msg = QtGui.QMessageBox()
                msg.setText(self.tr("Reach the end of the set"))
                msg.exec_()

        @QtCore.Slot()
        def show_prev_pedes(self):
            if self._cur_pid - 1 >= 0:
                self._cur_pid -= 1
                self._show_current()
            else:
                msg = QtGui.QMessageBox()
                msg.setText(self.tr("Reach the beginning of the set"))
                msg.exec_()

        def _show_current(self):
            data = self._sets[self._cur_sid]
            img = data[0][self._cur_pid]
            output = data[1][self._cur_pid]
            target = data[2][self._cur_pid]

            pixmap = QtGui.QPixmap.fromImage(ndarray2qimage(img))
            self.image_panel.setPixmap(pixmap)

            self.unival_table.hide()
            self.multival_table.hide()

            compare_unival(self.unival_table, output, target)
            compare_multival(self.multival_table, output, target)

            self.unival_table.show()
            self.multival_table.show()

        def _create_menus(self):
            menu_bar = self.menuBar()

            sets_menu = menu_bar.addMenu("&Sets")
            sets_menu.addAction(self.tr("Training Set"),
                                self, QtCore.SLOT("show_train_set()"),
                                QtGui.QKeySequence(self.tr("Ctrl+1")))
            sets_menu.addAction(self.tr("Validation Set"),
                                self, QtCore.SLOT("show_valid_set()"),
                                QtGui.QKeySequence(self.tr("Ctrl+2")))
            sets_menu.addAction(self.tr("Testing Set"),
                                self, QtCore.SLOT("show_test_set()"),
                                QtGui.QKeySequence(self.tr("Ctrl+3")))
 
            pedes_menu = menu_bar.addMenu("&Pedestrians")
            pedes_menu.addAction(self.tr("Next"),
                                 self, QtCore.SLOT("show_next_pedes()"),
                                 QtGui.QKeySequence.Forward)
            pedes_menu.addAction(self.tr("Prev"),
                                 self, QtCore.SLOT("show_prev_pedes()"),
                                 QtGui.QKeySequence.Back)

        def _create_panels(self):
            self.image_panel = QtGui.QLabel()
            self.unival_table = self._create_table(49)
            self.multival_table = self._create_table(72)

            layout = QtGui.QHBoxLayout()
            layout.addWidget(self.image_panel)
            layout.addWidget(self.unival_table)
            layout.addWidget(self.multival_table)

            frame = QtGui.QWidget()
            frame.setLayout(layout)

            self.setCentralWidget(frame)

        def _create_table(self, n_rows):
            header = QtGui.QHeaderView(QtCore.Qt.Orientation.Horizontal)
            header.setResizeMode(QtGui.QHeaderView.ResizeToContents)

            table = QtGui.QTableWidget(n_rows, 4)
            table.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
            table.setHorizontalHeader(header)
            table.setHorizontalHeaderLabels(
                ['Name', 'Target', 'Prediction', 'Output'])
            table.verticalHeader().setVisible(False)

            return table

    app = QtGui.QApplication(sys.argv)
    viewer = ResultViewer(result)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    data = load_data(attrconf.datasets)
    data = decompose(data)
    dataset = create_dataset(data)

    model = train_model(dataset)
    result = compute_result(model, dataset, data)

    show_stats(result)
    show_result(result)
