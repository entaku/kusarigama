# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
import sys
import random

import cv2
import numpy as np
from PIL import Image

import chainer
from chainer import cuda
import chainer.functions as F
from chainer.functions import caffe

# Usage: python cnn_age_gender.py amil_khan.jpg caffenet gender_net.caffemodel --mean mean.npy
# Usage:  python cnn_age_gender.py sara.jpg caffenet age_net.caffemodel --mean mean.npy

parser = argparse.ArgumentParser(
    description='Evaluate a Caffe reference model on ILSVRC2012 dataset')
parser.add_argument('image', help='Path to inspection image file')
parser.add_argument('model_type', choices=('alexnet', 'caffenet', 'googlenet'),
                    help='Model type (alexnet, caffenet, googlenet)')
parser.add_argument('model', help='Path to the pretrained Caffe model')
parser.add_argument('--mean', '-m', default='ilsvrc_2012_mean.npy',
                    help='Path to the mean file')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='Zero-origin GPU ID (nevative value indicates CPU)')
args = parser.parse_args()

#Caffeモデルをロード
print('Loading Caffe model file %s...' % args.model, file=sys.stderr)
func = caffe.CaffeFunction(args.model)
print('Loaded', file=sys.stderr)
if args.gpu >= 0:
    cuda.init(args.gpu)
    func.to_gpu()

if args.model_type == 'alexnet' or args.model_type == 'caffenet':
    in_size = 227
    mean_image = np.load(args.mean)

    def forward(x, t):
        y, = func(inputs={'data': x}, outputs=['fc8'], train=False)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
    def predict(x): #推測関数
        y, = func(inputs={'data': x}, outputs=['fc8'], train=False)
        return F.softmax(y)
elif args.model_type == 'googlenet':
    in_size = 224
    # Constant mean over spatial pixels
    mean_image = np.ndarray((3, 256, 256), dtype=np.float32)
    mean_image[0] = 104
    mean_image[1] = 117
    mean_image[2] = 123

    def forward(x, t):
        y, = func(inputs={'data': x}, outputs=['loss3/classifier'],
                  disable=['loss1/ave_pool', 'loss2/ave_pool'],
                  train=False)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
    def predict(x): #推測関数
        y, = func(inputs={'data': x}, outputs=['loss3/classifier'],
                  disable=['loss1/ave_pool', 'loss2/ave_pool'],
                  train=False)
        return F.softmax(y)

cropwidth = 256 - in_size
start = cropwidth // 2
stop = start + in_size
mean_image = mean_image[:, start:stop, start:stop].copy()
target_shape = (256, 256)
output_side_length=256

#画像ファイルを読み込み
image = cv2.imread(args.image)

#比較可能なサイズにリサイズ&クロップ
height, width, depth = image.shape
new_height = output_side_length
new_width = output_side_length
if height > width:
  new_height = output_side_length * height / width
else:
  new_width = output_side_length * width / height
resized_img = cv2.resize(image, (new_width, new_height))
height_offset = (new_height - output_side_length) / 2
width_offset = (new_width - output_side_length) / 2
image= resized_img[height_offset:height_offset + output_side_length,
width_offset:width_offset + output_side_length]

#画像から平均画像を引く
image = image.transpose(2, 0, 1)
image = image[:, start:stop, start:stop].astype(np.float32)
image -= mean_image

x_batch = np.ndarray(
        (1, 3, in_size,in_size), dtype=np.float32)
x_batch[0]=image

if args.gpu >= 0:
  x_batch=cuda.to_gpu(x_batch)

#推測
x = chainer.Variable(x_batch, volatile=True)
score = predict(x)

if args.gpu >= 0:
  score=cuda.to_cpu(score.data)

age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
gender_list=['Male','Female']
predictIndex = np.argmax(score.data[0].tolist())

if  len(score.data[0].tolist()) == 2:
    print(gender_list[predictIndex])
else:
    print(age_list[predictIndex])
