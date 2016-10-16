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
                     ip_depth=0,
                     dropout=0,
                     fmap_inc_rule = lambda fmaps: 80,
                     fmap_dec_rule = lambda fmaps: 80,
                     fmap_bridge_rule = lambda fmaps: 3,
                     fmap_start=80,
                     conv=[[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],
                     pool=[[2,2,2],[2,2,2],[2,2,2],[1,1,1]],
                     padding=[85,85,85])

net.prob = L.Softmax(net.sknet, ntop=1, in_place=False, include=[dict(phase=1)])
net.loss = L.SoftmaxWithLoss(net.sknet, net.label, ntop=0, loss_weight=1.0, include=[dict(phase=0)])

pygt.fix_input_dims(net,
                    [net.data, net.label],
                    max_shapes = [[130,130,130],[130,130,130]],
                    shape_coupled = [-1, 0, 0])

protonet = net.to_proto()
protonet.name = 'net';

# Store the network as prototxt
with open(protonet.name + '.prototxt', 'w') as f:
    print(protonet, file=f)
