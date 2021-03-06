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

if __name__ == "__main__":
    file_name = sys.argv[1]
    orig = cv2.imread(file_name, 1)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    gray_float = gray.astype('float32')
    gray28 = 1 - (cv2.resize(gray_float, (28, 28)) / 255.)
    gray28.resize((1, 784))
    h0 = chainer.Variable(gray28)
    model = pickle.load(open('trained-mnist-model', 'rb'))
    h1 = model.l1(h0)
    h2 = model.l2(h1)
    y = model.l3(h2)
    print y.data
    print F.softmax(y).data
