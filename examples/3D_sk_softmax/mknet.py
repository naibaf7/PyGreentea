from __future__ import print_function
import sys, os, math
import numpy as np
from numpy import float32, int32, uint8, dtype

# Load PyGreentea
# Relative path to where PyGreentea resides
pygt_path = '../..'
sys.path.append(pygt_path)
import pygreentea.pygreentea as pygt

import caffe
from caffe import layers as L
from caffe import params as P
from caffe import to_proto
from pygreentea.pygreentea import metalayers as ML


net = caffe.NetSpec()
net.data = L.MemoryData(dim=[1, 1], ntop=1)
net.label = L.MemoryData(dim=[1, 1], ntop=1, include=[dict(phase=0)])

net.sknet = ML.SKNet(net.data,
                     fmap_start=64,
                     conv=[[1,8,8],[2,6,6],[2,4,4]],
                     pool=[[1,2,2],[1,2,2],[1,2,2]],
                     padding=[8,88,88])

net.out = L.Convolution(net.sknet, kernel_size=[1,1,1], num_output=9, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
net.prob = L.Softmax(net.out, ntop=1, in_place=False, include=[dict(phase=1)])
net.loss = L.SoftmaxWithLoss(net.out, net.label, ntop=0, loss_weight=1.0, include=[dict(phase=0)])

pygt.fix_input_dims(net,
                    [net.data, net.label],
                    max_shapes = [[9,250,250],[9,250,250]],
                    shape_coupled = [-1, -1, 1])

protonet = net.to_proto()
protonet.name = 'net';

# Store the network as prototxt
with open(protonet.name + '.prototxt', 'w') as f:
    print(protonet, file=f)
