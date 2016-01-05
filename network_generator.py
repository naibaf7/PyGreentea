from __future__ import print_function
import os, sys, inspect
import h5py
import numpy as np
import matplotlib
import random
import math, copy
import multiprocessing
from Crypto.Random.random import randint
from functools import partial

# Import pycaffe
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2

# General variables
# Size of a float variable
fsize = 4

class NetworkGenerator:
    def __init__(self, netconf, mode):
        self.netconf = copy.deepcopy(netconf)
        self.mode = copy.deepcopy(mode)
    
    def compute_memory_weights(self, shape_arr):
        memory = 0
        for i in range(0,len(shape_arr)):
            memory += shape_arr[i][1]
        return memory
            
    def compute_memory_buffers(self, shape_arr):
        memory = 0
        for i in range(0,len(shape_arr)):
            memory = max(memory, shape_arr[i][0])
        return memory
            
    def compute_memory_blobs(self, shape_arr):
        memory = 0
        for i in range(0,len(shape_arr)):
            mem = fsize * shape_arr[i][2]
            for j in range(0,len(shape_arr[i][4])):
                mem *= shape_arr[i][4][j]
            memory += mem
        return memory
    
    def update_shape(self, shape_arr, update):
        last_shape = shape_arr[-1]
        new_shape = [update[0](last_shape[0]), update[1](last_shape[1]), update[2](last_shape[2]),
                     [update[3][min(i,len(update[3])-1)](last_shape[3][i]) for i in range(0,len(last_shape[3]))],
                     [update[4][min(i,len(update[4])-1)](last_shape[4][i]) for i in range(0,len(last_shape[4]))]]
        shape_arr += [new_shape]
        print ("TEST B: %s" % [update[4][min(i,len(update[4])-1)]([1,1,1][i]) for i in range(0,3)])
        
        return shape_arr
    
    def data_layer(self, shape):
        data, label = L.MemoryData(dim=shape, ntop=2)
        return data, label
    
    def conv_relu(self, run_shape, bottom, num_output, kernel_size=[3], stride=[1], pad=[0], dilation=[1], group=1, weight_std=0.01):
        # The convolution buffer and weight memory
        weight_mem = fsize * num_output * run_shape[-1][2]
        conv_buff = fsize * run_shape[-1][2]
        for i in range(0,len(run_shape[-1][4])):        
            conv_buff *= kernel_size[min(i,len(kernel_size)-1)]
            conv_buff *= run_shape[-1][4][i]
            weight_mem *= kernel_size[min(i,len(kernel_size)-1)]
        
        # Shape update rules
        update =  [lambda x: conv_buff, lambda x: weight_mem, lambda x: num_output]
        update += [[lambda x: x, lambda x: x, lambda x: x]]
        update += [[lambda x, i=i: x - (kernel_size[min(i,len(kernel_size)-1)] - 1) * (run_shape[-1][3][i]) for i in range(0,len(run_shape[-1][4]))]]
        self.update_shape(run_shape, update)
        
        conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride, dilation=dilation,
                                    num_output=num_output, pad=pad, group=group,
                                    param=[dict(lr_mult=1),dict(lr_mult=2)],
                                    weight_filler=dict(type='gaussian', std=weight_std),
                                    bias_filler=dict(type='constant'))
        
        relu = L.ReLU(conv, in_place=True, negative_slope=self.netconf.relu_slope)
        last = relu
        
        if (self.netconf.dropout > 0):
            drop = L.Dropout(last, in_place=True, dropout_ratio=self.netconf.dropout)
            last = drop
        
        if (self.netconf.use_batchnorm == True):
            bnl = L.BatchNorm(last, in_place=True,
                              param=[dict(lr_mult=0,decay_mult=0),dict(lr_mult=0,decay_mult=0),dict(lr_mult=0,decay_mult=0)],
                              batch_norm_param=dict(use_global_stats=(self.mode == caffe_pb2.TEST), moving_average_fraction=self.netconf.batchnorm_maf))
            last = bnl
            
        return conv, last
    
    def convolution(self, run_shape, bottom, num_output, kernel_size=[3], stride=[1], pad=[0], dilation=[1], group=1, weight_std=0.01):
        # The convolution buffer and weight memory
        weight_mem = fsize * num_output * run_shape[-1][2]
        conv_buff = fsize * run_shape[-1][2]
        for i in range(0,len(run_shape[-1][4])):        
            conv_buff *= kernel_size[min(i,len(kernel_size)-1)]
            conv_buff *= run_shape[-1][4][i]
            weight_mem *= kernel_size[min(i,len(kernel_size)-1)]
        
        # Shape update rules
        update =  [lambda x: conv_buff, lambda x: weight_mem, lambda x: num_output]
        update += [[lambda x: x, lambda x: x, lambda x: x]]
        update += [[lambda x, i=i: x - (kernel_size[min(i,len(kernel_size)-1)] - 1) * (run_shape[-1][3][i]) for i in range(0,len(run_shape[-1][4]))]]
        self.update_shape(run_shape, update)
        
        return L.Convolution(bottom, kernel_size=kernel_size, stride=stride, dilation=dilation,
                                    num_output=num_output, pad=pad, group=group,
                                    param=[dict(lr_mult=1),dict(lr_mult=2)],
                                    weight_filler=dict(type='gaussian', std=weight_std),
                                    bias_filler=dict(type='constant'))
    
    def max_pool(self, run_shape, bottom, kernel_size=[2], stride=[2], pad=[0], dilation=[1]): 
        # Shape update rules
        update =  [lambda x: 0, lambda x: 0, lambda x: x]
        update += [[lambda x, i=i: x * dilation[min(i,len(dilation)-1)] for i in range(0,len(run_shape[-1][4]))]]
        # Strictly speaking this update rule is not complete, but should be sufficient for USK
        if dilation[0] == 1 and kernel_size[0] == stride[0]:
            update += [[lambda x, i=i: x / (kernel_size[min(i,len(kernel_size)-1)]) for i in range(0,len(run_shape[-1][4]))]]
        else:
            update += [[lambda x, i=i: x - (kernel_size[min(i,len(kernel_size)-1)] - 1) * (run_shape[-1][3][i]) for i in range(0,len(run_shape[-1][4]))]]
        self.update_shape(run_shape, update)
    
        return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=kernel_size, stride=stride, pad=pad, dilation=dilation)
    
    def upconv(self, run_shape, bottom, num_output_dec, num_output_conv, weight_std=0.01, kernel_size=[2], stride=[2]):
        # Shape update rules
        update =  [lambda x: 0, lambda x: 0, lambda x: num_output_dec]
        update += [[lambda x: x, lambda x: x, lambda x: x]]
        update += [[lambda x, i=i: kernel_size[min(i,len(kernel_size)-1)] * x for i in range(0,len(run_shape[-1][4]))]]
        self.update_shape(run_shape, update)
        
        deconv = L.Deconvolution(bottom, convolution_param=dict(num_output=num_output_dec, kernel_size=kernel_size, stride=stride, pad=[0], dilation=[1], group=num_output_dec,
                                                                weight_filler=dict(type='constant', value=1), bias_term=False),
                                 param=dict(lr_mult=0, decay_mult=0))
    
        # The convolution buffer and weight memory
        weight_mem = fsize * num_output_conv * num_output_dec
        conv_buff = fsize * run_shape[-1][2]
        for i in range(0,len(run_shape[-1][4])):        
            conv_buff *= 2
            conv_buff *= run_shape[-1][4][i]
        
        # Shape update rules
        update =  [lambda x: conv_buff, lambda x: weight_mem, lambda x: num_output_conv]
        update += [[lambda x: x, lambda x: x, lambda x: x]]
        update += [[lambda x, i=i: x for i in range(0,len(run_shape[-1][4]))]]
        self.update_shape(run_shape, update)
    
        conv = L.Convolution(deconv, num_output=num_output_conv, kernel_size=[1], stride=[1], pad=[0], dilation=[1], group=1,
                                param=[dict(lr_mult=1),dict(lr_mult=2)],
                                weight_filler=dict(type='gaussian', std=weight_std),
                                bias_filler=dict(type='constant'))
        return deconv, conv
    
    def mergecrop(self, run_shape, bottom_a, bottom_b):
    
        # Shape update rules
        update =  [lambda x: 0, lambda x: 0, lambda x: 2*x]
        update += [[lambda x: x, lambda x: x, lambda x: x]]
        update += [[lambda x, i=i: x for i in range(0,len(run_shape[-1][4]))]]
        self.update_shape(run_shape, update)
    
        return L.MergeCrop(bottom_a, bottom_b, forward=[1,1], backward=[1,1])
    
    def implement_usknet(self, netconf, netmode, net, run_shape, fmaps_start, fmaps_end):
        # Chained blob list to construct the network (forward direction)
        blobs = []
    
        if netmode == caffe_pb2.TEST:
            self.netconf.dropout = 0
    
        # All networks start with data
        blobs = blobs + [net.data]
        
        fmaps = fmaps_start
    
        if netconf.unet_depth > 0:
            # U-Net downsampling; 2*Convolution+Pooling
            for i in range(0, netconf.unet_depth):
                conv, relu = self.conv_relu(run_shape, blobs[-1], fmaps, kernel_size=[3], weight_std=math.sqrt(2.0/float(run_shape[-1][2]*pow(3,len(run_shape[-1][4])))))
                blobs = blobs + [relu]
                conv, relu = self.conv_relu(run_shape, blobs[-1], fmaps, kernel_size=[3], weight_std=math.sqrt(2.0/float(run_shape[-1][2]*pow(3,len(run_shape[-1][4])))))
                blobs = blobs + [relu]  # This is the blob of interest for mergecrop (index 2 + 3 * i)
                pool = self.max_pool(run_shape, blobs[-1], kernel_size=netconf.unet_downsampling_strategy[i], stride=netconf.unet_downsampling_strategy[i])
                blobs = blobs + [pool]
                fmaps = netconf.unet_fmap_inc_rule(fmaps)
    
        
        # If there is no SK-Net component, fill with 2 convolutions
        if (netconf.unet_depth > 0 and netconf.sknet_conv_depth == 0):
            conv, relu = self.conv_relu(run_shape, blobs[-1], fmaps, kernel_size=[3], weight_std=math.sqrt(2.0/float(run_shape[-1][2]*pow(3,len(run_shape[-1][4])))))
            blobs = blobs + [relu]
            conv, relu = self.conv_relu(run_shape, blobs[-1], fmaps, kernel_size=[3], weight_std=math.sqrt(2.0/float(run_shape[-1][2]*pow(3,len(run_shape[-1][4])))))
            blobs = blobs + [relu]
        # Else use the SK-Net instead
        else:
            for i in range(0, netconf.sknet_conv_depth):
                # TODO: Not implemented yet (fixme)
                run_shape = run_shape
        
        if netconf.unet_depth > 0:
            # U-Net upsampling; Upconvolution+MergeCrop+2*Convolution
            for i in range(0, netconf.unet_depth):
                deconv, conv = self.upconv(run_shape, blobs[-1], fmaps, netconf.unet_fmap_dec_rule(fmaps), kernel_size=netconf.unet_downsampling_strategy[i], stride=netconf.unet_downsampling_strategy[i], weight_std=math.sqrt(2.0/float(run_shape[-1][2]*pow(3,len(run_shape[-1][4])))))
                blobs = blobs + [conv]
                fmaps = netconf.unet_fmap_dec_rule(fmaps)
                # Here, layer (2 + 3 * i) with reversed i (high to low) is picked
                mergec = self.mergecrop(run_shape, blobs[-1], blobs[-1 + 3 * (netconf.unet_depth - i)])
                blobs = blobs + [mergec]
                conv, relu = self.conv_relu(run_shape, blobs[-1], fmaps, kernel_size=[3], weight_std=math.sqrt(2.0/float(run_shape[-1][2]*pow(3,len(run_shape[-1][4])))))
                blobs = blobs + [relu]
                conv, relu = self.conv_relu(run_shape, blobs[-1], fmaps, kernel_size=[3], weight_std=math.sqrt(2.0/float(run_shape[-1][2]*pow(3,len(run_shape[-1][4])))))
                blobs = blobs + [relu]
                
            conv = self.convolution(run_shape, blobs[-1], fmaps_end, kernel_size=[1], weight_std=math.sqrt(2.0/float(run_shape[-1][2]*pow(3,len(run_shape[-1][4])))))
            blobs = blobs + [conv]
        
        # Return the last blob of the network (goes to error objective)
        return blobs[-1]


def caffenet(netconf, netmode):
    # Start Caffe proto net
    net = caffe.NetSpec()
    # Specify input data structures
    
    dims = len(netconf.input_shape)
    
    if netmode == caffe_pb2.TEST:
        netgen = NetworkGenerator(netconf, netmode);
        
        net.data, net.datai = netgen.data_layer([1]+[netconf.fmap_input]+netconf.input_shape)
        net.silence = L.Silence(net.datai, ntop=0)
        
        # Shape specs:
        # 00.    Convolution buffer size
        # 01.    Weight memory size
        # 02.    Num. channels
        # 03.    [d] parameter running value
        # 04.    [w] parameter running value
        run_shape_in = [[0,0,1,[1 for i in range(0,dims)],netconf.input_shape]]
        run_shape_out = run_shape_in
        
        last_blob = netgen.implement_usknet(netconf, netmode, net, run_shape_out, netconf.fmap_start, netconf.fmap_output)

        # Implement the prediction layer
        if netconf.loss_function == 'malis':
            net.prob = L.Sigmoid(last_blob, ntop=1)
        
        if netconf.loss_function == 'euclid':
            net.prob = L.Sigmoid(last_blob, ntop=1)
            
        if netconf.loss_function == 'softmax':
            net.prob = L.Softmax(last_blob, ntop=1)

        for i in range(0,len(run_shape_out)):
            print(run_shape_out[i])
            
        print("Max. memory requirements: %s B" % (netgen.compute_memory_buffers(run_shape_out)+netgen.compute_memory_weights(run_shape_out)+netgen.compute_memory_blobs(run_shape_out)))
        print("Weight memory: %s B" % netgen.compute_memory_weights(run_shape_out))
        print("Max. conv buffer: %s B" % netgen.compute_memory_buffers(run_shape_out))
        
    else:
        netgen = NetworkGenerator(netconf, netmode);
        
        net.data, net.datai = netgen.data_layer([1]+[netconf.fmap_input]+netconf.input_shape)

        if netconf.loss_function == 'malis':
            net.label, net.labeli = netgen.data_layer([1]+[netconf.fmap_output]+netconf.output_shape)
            net.components, net.componentsi = netgen.data_layer([1,1]+netconf.output_shape)
            net.nhood, net.nhoodi = netgen.data_layer([1,1]+[netconf.fmap_output]+[3])
            net.silence = L.Silence(net.datai, net.labeli, net.componentsi, net.nhoodi, ntop=0)
            
        if netconf.loss_function == 'euclid':
            net.label, net.labeli = netgen.data_layer([1]+[netconf.fmap_output]+netconf.output_shape)
            net.scale, net.scalei = netgen.data_layer([1]+[netconf.fmap_output]+netconf.output_shape)
            net.silence = L.Silence(net.datai, net.labeli, net.scalei, ntop=0)

        if netconf.loss_function == 'softmax':
            # Currently only supports binary classification
            net.label, net.labeli = netgen.data_layer([1]+[netconf.fmap_output]+netconf.output_shape)
            net.silence = L.Silence(net.datai, net.labeli, ntop=0)
    
        run_shape_in = [[0,1,1,[1 for i in range(0,dims)],netconf.input_shape]]
        run_shape_out = run_shape_in
    
        # Start the actual network
        last_blob = netgen.implement_usknet(netconf, netmode, net, run_shape_out, netconf.fmap_start, netconf.fmap_output)
        
        for i in range(0,len(run_shape_out)):
            print(run_shape_out[i])
            
        print("Max. memory requirements: %s B" % (netgen.compute_memory_buffers(run_shape_out)+netgen.compute_memory_weights(run_shape_out)+2*netgen.compute_memory_blobs(run_shape_out)))
        print("Weight memory: %s B" % netgen.compute_memory_weights(run_shape_out))
        print("Max. conv buffer: %s B" % netgen.compute_memory_buffers(run_shape_out))
        
        # Implement the loss
        if netconf.loss_function == 'malis':       
            last_blob = L.Sigmoid(last_blob, in_place=True)
            net.loss = L.MalisLoss(last_blob, net.label, net.components, net.nhood, ntop=0)
        
        if netconf.loss_function == 'euclid':
            last_blob = L.Sigmoid(last_blob, in_place=True)
            net.loss = L.EuclideanLoss(last_blob, net.label, net.scale, ntop=0)
            
        if netconf.loss_function == 'softmax':
            net.loss = L.SoftmaxWithLoss(last_blob, net.label, ntop=0)
            
    # Return the protocol buffer of the generated network
    return net.to_proto()


def create_nets(netconf):
    return (caffenet(netconf, caffe_pb2.TRAIN), caffenet(netconf, caffe_pb2.TEST))


