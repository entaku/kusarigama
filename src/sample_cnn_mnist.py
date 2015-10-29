__author__ = 'team-entaku'
import sys
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.cluster.vq
import scipy.spatial.distance as distance
from chainer import computational_graph as c
from matplotlib.pyplot import show
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import random

import argparse

import pickle
import six

import chainer
from chainer import computational_graph as c
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
from chainer.functions import caffe

if __name__ == "__main__":
    caffe_model = 'lenet_iter_10000.caffemodel'
    caffe_func = caffe.CaffeFunction(caffe_model)


    def predict(x):
        """

        :param x: numpy.array
        """
        y, = caffe_func(inputs={'data': x}, outputs=['ip2'], train=False)
        return F.softmax(y)


    file_name = sys.argv[1]
    orig = cv2.imread(file_name, 1)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY).astype('float32')
    gray_resized = cv2.resize(gray, (28, 28))
    gray28 = 1 - (gray_resized / 255.)
    gray28_expanded = np.expand_dims(gray28, axis=2)

    batch = np.zeros((1, 1, 28, 28), dtype='float32')
    batch[0] = gray28_expanded.transpose(2, 0, 1)

    x = chainer.Variable(batch, volatile=True)
    p_dist = predict(x)

    print np.argmax(p_dist.data)



    # gray28.resize((1, 784))
    # h0 = chainer.Variable(gray28)
    # model = pickle.load(open(model_name, 'rb'))
    # h1 = model.l1(h0)
    # h2 = model.l2(h1)
    # y = model.l3(h2)
    # print y.data
    # print F.softmax(y).data
