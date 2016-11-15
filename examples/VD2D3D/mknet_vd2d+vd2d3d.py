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

fmaps_vd2d = [24, 24, 36, 36, 48, 48, 60, 60]
net.sknet2d = ML.SKNet(net.data,
                     fmap_start=24,
                     conv=[[1,3,3],[1,3,3],[1,2,2],[1,3,3],[1,3,3],[1,3,3],[1,3,3],[1,3,3],[1,3,3]],
                     activation=['relu', 'relu', 'tanh', 'relu', 'tanh', 'relu', 'tanh', 'relu', 'tanh', 'relu'],
                     pool=[[1,1,1],[1,1,1],[1,2,2],[1,1,1],[1,2,2],[1,1,1],[1,2,2],[1,1,1],[1,2,2]],
                     padding=[0,108,108],
                     fmap_inc_rule = lambda x: fmaps_vd2d.pop(0),
                     fmap_bridge_rule = lambda x: 200,
                     fmap_dec_rule = lambda x: 0,
                     ip_depth = 1,
                     dropout = 0.0)

net.out2d = L.Convolution(net.sknet2d, kernel_size=[1,1,1], num_output=2, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
net.mergecrop = L.MergeCrop(net.out2d, net.data)

fmaps_vd2d3d = [24, 24, 36, 36, 48, 48, 60, 60]
net.sknet3d = ML.SKNet(net.mergecrop,
                     fmap_start=24,
                     conv=[[1,3,3],[1,3,3],[1,2,2],[1,3,3],[1,3,3],[1,3,3],[1,3,3],[2,3,3],[2,3,3]],
                     activation=['relu', 'relu', 'tanh', 'relu', 'tanh', 'relu', 'tanh', 'relu', 'tanh', 'relu'],
                     pool=[[1,1,1],[1,1,1],[1,2,2],[1,1,1],[1,2,2],[1,1,1],[2,2,2],[1,1,1],[1,1,1]],
                     padding=[4,84,84],
                     fmap_inc_rule = lambda x: fmaps_vd2d3d.pop(0),
                     fmap_bridge_rule = lambda x: 100,
                     fmap_dec_rule = lambda x: 0,
                     ip_depth = 1,
                     hybrid_dimensions = [0],
                     dropout = 0.0)

net.out3d = L.Convolution(net.sknet3d, kernel_size=[1,1,1], num_output=2, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
net.prob = L.Softmax(net.out3d, ntop=1, in_place=False, include=[dict(phase=1)])
net.loss = L.SoftmaxWithLoss(net.out3d, net.label, ntop=0, loss_weight=1.0, include=[dict(phase=0)])

pygt.fix_input_dims(net,
                    [net.data, net.label],
                    max_shapes = [[100,250,250],[100,250,250]],
                    shape_coupled = [-1, -1, 1])

protonet = net.to_proto()
protonet.name = 'net_vd2d+vd2d3d';

# Store the network as prototxt
with open(protonet.name + '.prototxt', 'w') as f:
    print(protonet, file=f)
