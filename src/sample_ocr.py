__author__ = 'team-entaku'

import cv2
import numpy as np

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


def process_mser(orig, delta, min_area, max_area, max_variation, min_diversity, max_evolution, area_threshold,
                 min_margin, edge_blur_size):
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER(delta, min_area, max_area, max_variation, min_diversity, max_evolution, area_threshold,
                    min_margin, edge_blur_size)
    regions = mser.detect(gray, None)
    rects = [cv2.minAreaRect(r.reshape(-1, 1, 2)) for r in regions]
    boxes = [np.int0(cv2.cv.BoxPoints(rect)) for rect in rects]
    return boxes


def run_mser_video(
        model,
        delta,
        min_area,
        max_area,
        max_variation,
        min_diversity,
        max_evolution,
        area_threshold,
        min_margin,
        edge_blur_size):

    cap = cv2.VideoCapture(0)
    captured_image = None

    while True:

        key = cv2.waitKey(10)

        if captured_image is not None:
            continue

        if key == -1:
            res, orig = cap.read()
            boxes = process_mser(orig,
                                 delta,
                                 min_area,
                                 max_area,
                                 max_variation,
                                 min_diversity,
                                 max_evolution,
                                 area_threshold,
                                 min_margin, edge_blur_size)
            cv2.drawContours(orig, boxes, -1, (0, 255, 0), 1)
            cv2.imshow('img', orig)
        else:
            res, orig = cap.read()
            captured_image = orig
            boxes = process_mser(orig,
                                 delta,
                                 min_area,
                                 max_area,
                                 max_variation,
                                 min_diversity,
                                 max_evolution,
                                 area_threshold,
                                 min_margin, edge_blur_size)
            gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            dev = gray.copy()
            for box in boxes:
                (x, y), width, height = bounding_rect(box)
                if True:
                    cropped = gray[x:(x + width), y:(y + height)].astype('float32')
                    if cropped.shape[0] * cropped.shape[1] > 784:
                        cv2.rectangle(dev, (x, y), (x + width, y + height), (0), 1)
                        gray28 = 1 - (cv2.resize(cropped, (28, 28)) / 255.)
                        gray28.resize((1, 784))
                        h0 = chainer.Variable(gray28)
                        digit = scan(h0)
                        if digit != -1:
                            font_face = cv2.FONT_HERSHEY_PLAIN
                            font_scale = 1.0
                            color = (0, 255, 0)
                            cv2.putText(captured_image, str(digit), (x, y), font_face, font_scale, color)

            cv2.imshow('img', captured_image)
            cv2.imshow('dev', dev)

    cap.release()
    cv2.destroyAllWindows()


def scan(h0):
    h1 = model.l1(h0)
    h2 = model.l2(h1)
    y = model.l3(h2)
    f = F.softmax(y).data
    if f.max() > 0.6:
        return np.argmax(f)
    else:
        return -1


def bounding_rect(arr):
    t_arr = np.transpose(arr)
    x_min = t_arr[0].min()
    x_max = t_arr[0].max()
    y_min = t_arr[1].min()
    y_max = t_arr[1].max()
    width = x_max - x_min
    height = y_max - y_min
    return (x_min, y_min), width, height


def run_video(model):
    delta = 5
    min_area = 200
    max_area = 1000
    max_variation = 0.25
    min_diversity = 0.2
    max_evolution = 200
    area_threshold = 1.01
    min_margin = 0.003
    edge_blur_size = 5
    run_mser_video(model, delta, min_area, max_area, max_variation, min_diversity, max_evolution, area_threshold,
                   min_margin,
                   edge_blur_size)


if __name__ == "__main__":
    model = pickle.load(open('trained-mnist-model', 'rb'))
    run_video(model)
