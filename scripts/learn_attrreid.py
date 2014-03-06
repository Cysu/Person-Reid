#!/usr/bin/python2
# -*- coding: utf-8 -*-

import os
import numpy
import attrconf
import bodyconf
from reid.utils.cache_manager import CacheManager


cachem = CacheManager(os.path.join('..', 'cache'), 'attrreid')


@cachem.save('raw')
def load_data(datasets):
    """Load images, corresponding body parts maps and attributes vectors

    Args:
        datasets: A list of datasets names

    Returns:
        Two lists. [(img, bpmap, attr)] and [(pid, vid)].
    """

    print "Loading ..."

    data = cachem.load('raw')

    if data is not None: return data

    from scipy.io import loadmat

    data, indices = [], []

    cumpid = 0

    for dname in datasets:
        matfp = os.path.join('..', 'data', 'attrreid', dname + '_parse.mat')
        matdata = loadmat(matfp)

        m, n = matdata['images'].shape
        for pid in xrange(m):
            for vid in xrange(n):
                if matdata['images'][pid, vid].size == 0: break
                for k in xrange(matdata['images'][pid, vid].shape[1]):
                    img = matdata['images'][pid, vid][0, k]
                    bpmap = matdata['bodyparts'][pid, vid][0, k]
                    attr = matdata['attributes'][pid, vid][0, k].ravel()
                    data.append((img, bpmap, attr))
                    indices.append((cumpid + pid, vid))
        cumpid += m

    return (data, indices)


@cachem.save('decomp')
def decompose(rawdata, dilation_radius=3):
    """Decompose body parts and attributes

    Args:
        rawdata: [(img, bpmap, attr)]
        dilation_radius: The radius of dilation structure element

    Returns:
        The decomposed data. [(img, [parts], [attrs])].
    """

    print "Decomposing ..."

    data = cachem.load('decomp')

    if data is not None: return data

    import skimage.morphology as morph
    selem = morph.square(2*dilation_radius)

    # Decompose pedestrian image into body parts
    def decomp_body(img, bpmap):
        parts = [0] * len(bodyconf.groups)

        for i, grp in enumerate(bodyconf.groups):
            # mask = dilate(region_0 | region_1 | ... | region_k)
            regions = [(bpmap == pixval) for pixval in grp]
            mask = reduce(lambda x, y: x|y, regions)
            mask = morph.binary_dilation(mask, selem)
            parts[i] = img * numpy.expand_dims(mask, axis=2)

        return parts

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
        attrs = decomp_attr(attr)
        assert attrs is not None
        parts = decomp_body(img, bpmap)
        data.append((img, parts, attrs))

    return data


@cachem.save('prep')
def preprocess(decdata):
    """Pre-process images and attributes to matrix

    Args:
        decdata: [(img, parts, attrs)]

    Returns:
        The pre-processed image. ([X], [A]).
    """

    print "Pre-processing ..."

    data = cachem.load('prep')

    if data is not None: return data

    from skimage.color import rgb2lab
    from reid.preproc import imageproc

    def imgprep(img, mean_luminance):
        img = imageproc.imresize(img, (80, 30))
        img = imageproc.subtract_luminance(img, mean_luminance)
        img = numpy.rollaxis(img, 2)
        return img

    m = len(decdata)
    X, A = [0] * m, [0] * m

    for i, (img, parts, attrs) in enumerate(decdata):
        mean_luminance = rgb2lab(img)[:,:,0].mean()
        X[i] = [imgprep(part, mean_luminance) for part in parts]
        X[i] = numpy.asarray(X[i], dtype=numpy.float32).ravel()
        A[i] = numpy.concatenate(attrs).astype(numpy.float32)

    X = numpy.asarray(X)
    A = numpy.asarray(A)

    X = numpy.tanh(X - X.mean(axis=0))

    return (X, A)


@cachem.save('sample')
def sample(indices, pos_downsample=1.0, neg_pos_ratio=1.0):
    """Sample positive and negative data

    Args:
        indices: [(pid, vid)]
        neg_pos_ratio: #neg / #pos

    Returns:
        The sampled data indices. (train, vaid, test) each is [(i, j, 0/1)].
    """

    print "Sampling ..."

    data = cachem.load('sample')

    if data is not None: return data

    import random
    from reid.utils.math_utils import numpy_rng

    n_imgs = len(indices)

    def gensamples(pids):
        samples = []

        # Positive samples
        for i in xrange(n_imgs):
            if indices[i][0] not in pids: continue
            j = i + 1
            while j < n_imgs and indices[i][0] == indices[j][0]:
                if numpy_rng.rand() < pos_downsample:
                    samples.append((i, j, True))
                j += 1

        # Negative samples
        n = int(len(samples) * neg_pos_ratio)
        for k in xrange(n):
            while True:
                i, j = numpy_rng.randint(0, n_imgs, 2)
                if indices[i][0] in pids and indices[j][0] in pids and \
                        indices[i][0] != indices[j][0] and \
                        (i,j,False) not in samples and (j,i,False) not in samples:
                    samples.append((i, j, False))
                    break

        random.shuffle(samples)
        return samples

    # Split by pid
    m = indices[-1][0] + 1
    m_train = int(m * 0.7)
    m_valid = int(m * 0.2)

    p = numpy_rng.permutation(m)

    train_pids = p[0 : m_train]
    valid_pids = p[m_train : m_train+m_valid]
    test_pids = p[m_train+m_valid : ]

    train = gensamples(train_pids)
    valid = gensamples(valid_pids)
    test = gensamples(test_pids)

    return (train, valid, test, train_pids, valid_pids, test_pids)


@cachem.save('dataset')
def create_dataset(X, A, samples):
    """Create dataset for model training, validation and testing

    Args:
        X: m×d_X matrix
        A: m×d_A matrix
        samples: (train, vaid, test) each is [(i, j, 0/1)]

    Returns:
        Dataset X = X1_X2, Y = A1_A2_(0/1)
    """

    print "Creating dataset ..."

    dataset = cachem.load('dataset')

    if dataset is not None: return dataset

    from reid.utils.dataset import Dataset

    def genset(s):
        I = numpy.asarray([i for i, __, __ in s])
        J = numpy.asarray([j for __, j, __ in s])
        L = numpy.asarray([l for __, __, l in s]).astype(numpy.float32)

        x = numpy.hstack((X[I], X[J]))
        y = numpy.hstack((A[I], A[J], L.reshape(L.shape[0], 1)))

        return (x, y)

    return Dataset(train_set=genset(samples[0]),
                   valid_set=genset(samples[1]),
                   test_set=genset(samples[2]))


@cachem.save('model')
def train_model(dataset):
    """Train deep model

    Args:
        dataset: Dataset X = X1_X2, Y = A1_A2_(0/1)

    Returns:
        The trained deep model
    """

    print "Training ..."

    model = cachem.load('model')

    if model is not None: return model

    import reid.models.active_functions as af
    import reid.models.cost_functions as cf
    from reid.models.layers import ConvPoolLayer, FullConnLayer, IdentityLayer, FilterParingLayer
    from reid.models.layers import CompLayer, DecompLayer, CloneLayer
    from reid.models.neural_net import NeuralNet, MultiwayNeuralNet
    from reid.models.evaluate import Evaluator
    from reid.optimization import sgd

    output_sizes = [len(grp) for grp in attrconf.unival + attrconf.multival]
    target_sizes = [1] * len(attrconf.unival) + [len(grp) for grp in attrconf.multival]

    # Feature extraction module
    def feature_extraction():
        decomp = DecompLayer([(3,80,30)] * len(bodyconf.groups))
        column = MultiwayNeuralNet([NeuralNet([
            ConvPoolLayer((64,3,3,3), (2,2), (3,80,30), af.tanh, False),
            ConvPoolLayer((64,64,3,3), (2,2), None, af.tanh, True)
        ]) for __ in xrange(len(bodyconf.groups))])
        comp = CompLayer(strategy='Maxout')
        return NeuralNet([decomp, column, comp])

    fe = feature_extraction()
    feat_module = NeuralNet([
        DecompLayer([(3*80*30*len(bodyconf.groups),)] * 2),
        MultiwayNeuralNet([fe, fe]),
        CompLayer()
    ])

    # Attribute classification module
    def attribute_classification():
        fcl_1 = FullConnLayer(6912, 1024, af.tanh)
        fcl_2 = FullConnLayer(1024, 104)
        decomp = DecompLayer(
            [(sz,) for sz in output_sizes],
            [af.softmax] * len(attrconf.unival) + \
            [af.sigmoid] * len(attrconf.multival)
        )
        return NeuralNet([fcl_1, fcl_2, decomp], through=True)

    ac = NeuralNet([attribute_classification(), CompLayer()])
    attr_module = NeuralNet([
        DecompLayer([(6912,)] * 2),
        MultiwayNeuralNet([ac, ac]),
        CompLayer()
    ])

    # Person re-identification module
    def person_reidentification():
        fp = FilterParingLayer((64,18,6), 4, (2,2), True)
        fcl_1 = FullConnLayer(2592, 256, af.tanh)
        return NeuralNet([fp, fcl_1])

    reid_module = person_reidentification()

    model = NeuralNet([
        feat_module,
        CloneLayer(2),
        MultiwayNeuralNet([
            attr_module,
            reid_module
        ]),
        CompLayer(),
        FullConnLayer(104+104+256, 256, af.tanh),
        FullConnLayer(256, 2, af.softmax)
    ])

    # Evaluator
    def reid_cost(output, target):
        k = sum(target_sizes)
        return k * cf.mean_negative_loglikelihood(output, target)

    def reid_error(output, target):
        k = (len(attrconf.unival) + len(attrconf.multival))
        return k * cf.mean_negative_loglikelihood(output, target)


    cost_func = [
        [cf.mean_negative_loglikelihood] * len(attrconf.unival) + \
            [cf.mean_binary_cross_entropy] * len(attrconf.multival),
        [cf.mean_negative_loglikelihood] * len(attrconf.unival) + \
            [cf.mean_binary_cross_entropy] * len(attrconf.multival),
        reid_cost
    ]
    error_func = [
        [cf.mean_number_misclassified] * len(attrconf.unival) + \
            [cf.mean_zeroone_error_rate] * len(attrconf.multival),
        [cf.mean_number_misclassified] * len(attrconf.unival) + \
            [cf.mean_zeroone_error_rate] * len(attrconf.multival),
        reid_error
    ]

    def target_adapter():
        d1 = DecompLayer([(sum(target_sizes),), (sum(target_sizes),), (1,)])
        d2 = DecompLayer([(sz,) for sz in target_sizes])
        return NeuralNet([
            d1,
            MultiwayNeuralNet([d2, d2, IdentityLayer()])
        ])

    adapter = target_adapter()

    evaluator = Evaluator(model, cost_func, error_func, adapter,
                          regularize=1e-3)

    # Training
    sgd.train(evaluator, dataset,
              learning_rate=1e-4, momentum=0.9,
              batch_size=300, n_epoch=200,
              learning_rate_decr=1.0, patience_incr=1.5)

    return model


@cachem.save('result')
def compute_result(model, dataset, images, samples):
    """Compute output for dataset

    Args:
        model: Deep model
        dataset: Dataset X = X1_X2, Y = A1_A2_(0/1)
        images: [img]
        samples: (train, vaid, test) each is [(i, j, 0/1)]

    Returns:
        (train, valid, test) where each is
        (img_list_1, img_list_2 output_matrix, target_matrix)
    """

    print "Computing result ..."

    result = cachem.load('result')

    if result is not None: return result

    import theano
    import theano.tensor as T
    from reid.models.layers import CompLayer

    x = T.matrix()
    y, thr = model.get_output(x)
    thr.append(y)
    comp = CompLayer()
    y, thr = comp.get_output(thr)

    f = theano.function(inputs=[x], outputs=y)

    def compute_output(X):
        outputs = [f(X[i:i+1, :]).ravel() for i in xrange(X.shape[0])]
        return numpy.asarray(outputs)

    train = ([images[i] for i, __, __ in samples[0]],
             [images[j] for __, j, __ in samples[0]],
             compute_output(dataset.train_x.get_value(borrow=True)),
             dataset.train_y.get_value(borrow=True))

    valid = ([images[i] for i, __, __ in samples[1]],
             [images[j] for __, j, __ in samples[1]],
             compute_output(dataset.valid_x.get_value(borrow=True)),
             dataset.valid_y.get_value(borrow=True))

    test = ([images[i] for i, __, __ in samples[2]],
            [images[j] for __, j, __ in samples[2]],
            compute_output(dataset.test_x.get_value(borrow=True)),
            dataset.test_y.get_value(borrow=True))

    return (train, valid, test)


def show_stats(result):
    """Show the statistics of attributes classification

    Args:
        result: (train, valid, test) where each is
            [(I, J), output_matrix, target_matrix]
    """

    train, valid, test = result

    output_sizes = [len(grp) for grp in attrconf.unival + attrconf.multival]
    target_sizes = [1] * len(attrconf.unival) + [len(grp) for grp in attrconf.multival]

    output_seg = [0] + list(numpy.cumsum(output_sizes))
    target_seg = [0] + list(numpy.cumsum(target_sizes))

    output_offset = sum(output_sizes)
    target_offset = sum(target_sizes)

    def print_stats(title, outputs, targets):
        print "Statistics of {0}".format(title)
        print "=" * 80

        for i, (grptitle, grp) in \
                enumerate(zip(attrconf.unival_titles, attrconf.unival)):
            print "{0}, frequency, accuracy".format(grptitle)

            o1 = outputs[:, output_seg[i]:output_seg[i+1]]
            t1 = targets[:, target_seg[i]:target_seg[i+1]]
            p1 = o1.argmax(axis=1).reshape(o1.shape[0], 1)

            o2 = outputs[:, output_offset+output_seg[i]:output_offset+output_seg[i+1]]
            t2 = targets[:, target_offset+target_seg[i]:target_offset+target_seg[i+1]]
            p2 = o2.argmax(axis=1).reshape(o2.shape[0], 1)

            freqs, accs = [0] * len(grp), [0] * len(grp)
            for j, attrname in enumerate(grp):
                freqs[j] = ((t1 == j).mean() + (t2 == j).mean()) / 2.0
                accs[j] = (((t1 == j) & (p1 == j)).sum() + ((t2 == j) & (p2 == j)).sum()) * 1.0 / \
                          ((t1 == j).sum() + (t2 == j).sum())
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

            o1 = outputs[:, output_seg[offset+i]:output_seg[offset+i+1]]
            t1 = targets[:, target_seg[offset+i]:target_seg[offset+i+1]]
            p1 = o1.round()

            o2 = outputs[:, output_offset+output_seg[offset+i]:output_offset+output_seg[offset+i+1]]
            t2 = targets[:, target_offset+target_seg[offset+i]:target_offset+target_seg[offset+i+1]]
            p2 = o2.round()

            # Any multi-value group must have at least one attribute activated
            for k in xrange(p1.shape[0]):
                if p1[k, :].sum() == 0:
                    v = o1[k, :].argmax()
                    p1[k, v] = 1
                if p2[k, :].sum() == 0:
                    v = o2[k, :].argmax()
                    p2[k, v] = 1

            freqs, tprs, fprs = [0] * len(grp), [0] * len(grp), [0] * len(grp)
            for j, attrname in enumerate(grp):
                t1j, p1j = t1[:, j], p1[:, j]
                t2j, p2j = t2[:, j], p2[:, j]
                freqs[j] = ((t1j == 1).mean() + (t2j == 1).mean()) / 2.0
                tprs[j] = (((t1j == 1) & (p1j == 1)).sum() + ((t2j == 1) & (p2j == 1)).sum()) * 1.0 / \
                          ((t1j == 1).sum() + (t2j == 1).sum())
                fprs[j] = (((t1j == 0) & (p1j == 1)).sum() + ((t2j == 0) & (p2j == 1)).sum()) * 1.0 / \
                          ((t1j == 0).sum() + (t2j == 0).sum())
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

    print_stats("Training Set", train[2], train[3])
    print_stats("Validation Set", valid[2], valid[3])
    print_stats("Testing Set", test[2], test[3])


@cachem.save('cmc')
def show_cmc(model, X, indices, samples):
    print "Computing cmc ..."

    result = cachem.load('cmc')

    if result is not None: return result

    import theano
    import theano.tensor as T
    from reid.models.layers import CompLayer
    from reid.utils import cmc

    x = T.matrix()
    y, thr = model.get_output(x)
    thr.append(y)
    comp = CompLayer()
    y, thr = comp.get_output(thr)

    f = theano.function(inputs=[x], outputs=y)

    test_pids = samples[-1]

    gX, gY, pX, pY = [], [], [], []
    for i, (pid, vid) in enumerate(indices):
        if pid not in test_pids: continue
        if vid == 0:
            gX.append(i)
            gY.append(pid)
        else:
            pX.append(i)
            pY.append(pid)

    def compute_distance(i, j):
        y = f(numpy.hstack((X[gX[i]:gX[i]+1, :], X[pX[j]:pX[j]+1, :]))).ravel()
        return -y[-1]

    return cmc.count_lazy(compute_distance, gY, pY, 100, 10)


def show_result(result):
    """Show the result in GUI

    Args:
        result: (train, valid, test) where each is
            (img_list_1, img_list_2, output_matrix, target_matrix)
    """

    import sys
    from PySide import QtGui, QtCore
    from reid.utils.gui_utils import ndarray2qimage

    output_sizes = [len(grp) for grp in attrconf.unival + attrconf.multival]
    target_sizes = [1] * len(attrconf.unival) + [len(grp) for grp in attrconf.multival]

    output_seg = [0] + list(numpy.cumsum(output_sizes))
    target_seg = [0] + list(numpy.cumsum(target_sizes))

    output_offset = sum(output_sizes)
    target_offset = sum(target_sizes)

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
            img0 = data[0][self._cur_pid]
            img1 = data[1][self._cur_pid]
            output = data[2][self._cur_pid]
            target = data[3][self._cur_pid]

            pixmap0 = QtGui.QPixmap.fromImage(ndarray2qimage(img0))
            pixmap1 = QtGui.QPixmap.fromImage(ndarray2qimage(img1))
            self.image_panel[0].setPixmap(pixmap0)
            self.image_panel[1].setPixmap(pixmap1)

            self.unival_table[0].hide()
            self.multival_table[0].hide()
            self.unival_table[1].hide()
            self.multival_table[1].hide()

            compare_unival(self.unival_table[0], output[:output_offset], target[:target_offset])
            compare_multival(self.multival_table[0], output[:output_offset], target[:target_offset])
            compare_unival(self.unival_table[1], output[output_offset:], target[target_offset:])
            compare_multival(self.multival_table[1], output[output_offset:], target[target_offset:])

            self.unival_table[0].show()
            self.multival_table[0].show()
            self.unival_table[1].show()
            self.multival_table[1].show()

            self.reid_panel.setText('Output: {0:.5f}\nTarget: {1}'.format(output[-1], target[-1]))

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
            uni_rows = sum([len(grp)+2 for grp in attrconf.unival]) - 1
            mul_rows = sum([len(grp)+2 for grp in attrconf.multival]) - 1

            self.image_panel = [QtGui.QLabel() for __ in xrange(2)]
            self.unival_table = [self._create_table(uni_rows) for __ in xrange(2)]
            self.multival_table = [self._create_table(mul_rows) for __ in xrange(2)]
            self.reid_panel = QtGui.QLabel()

            imglayout = QtGui.QVBoxLayout()
            imglayout.addWidget(self.image_panel[0])
            imglayout.addWidget(self.image_panel[1])
            imglayout.addWidget(self.reid_panel)

            tab = QtGui.QTabWidget()
            for i in xrange(2):
                tablayout = QtGui.QHBoxLayout()
                tablayout.addWidget(self.unival_table[i])
                tablayout.addWidget(self.multival_table[i])
                w = QtGui.QWidget()
                w.setLayout(tablayout)
                tab.addTab(w, 'person {0}'.format(i))

            layout = QtGui.QHBoxLayout()
            layout.addLayout(imglayout)
            layout.addWidget(tab)

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
    data, indices = load_data(attrconf.datasets)
    data = decompose(data)
    X, A = preprocess(data)
    s = sample(indices, pos_downsample=0.3, neg_pos_ratio=2.0)
    dataset = create_dataset(X, A, s)

    model = train_model(dataset)

    result = compute_result(model, dataset, [p for p, __, __ in data], s)
    show_stats(result)
    print show_cmc(model, X, indices, s)

    show_result(result)
