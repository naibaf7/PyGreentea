from __future__ import print_function

import copy
import math

# Are these needed? They aren't used in this file.
import os, sys, inspect
import h5py
import numpy as np
import matplotlib
import random
import multiprocessing
import net_visualizer as nvs
from Crypto.Random.random import randint
from functools import partial


# Import pycaffe
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2

# General variables
# Size of a float variable in bytes
fsize = 4

    
class SKNetConf:
    # SK-Net convolution steps (may change if necessary)
    sknet_conv = [[8],[6],[4]]
    # Feature map increase rule
    sknet_fmap_inc_rule = lambda self,fmaps: int(math.ceil(float(fmaps) * 1.5))
    # Number of 1x1 (IP) Convolution steps
    sknet_ip_depth = 2
    # Feature map increase rule from SK-Convolution to IP
    sknet_fmap_bridge_rule = lambda self,fmaps: int(math.ceil(float(fmaps) * 4))
    # Feature map decrease rule within IP
    sknet_fmap_dec_rule = lambda self,fmaps: int(math.ceil(float(fmaps) / 2.5))
    # Network padding
    sknet_padding = [88, 88, 88]
    
class UNetConf:
    # Number of U-Net Pooling-Convolution downsampling/upsampling steps
    unet_depth = 3
    # Feature map increase rule (downsampling)
    unet_fmap_inc_rule = lambda self,fmaps: int(math.ceil(float(fmaps) * 3))
    # Feature map decrease rule (upsampling)
    unet_fmap_dec_rule = lambda self,fmaps: int(math.ceil(float(fmaps) / 3))
    # Skewed U-Net downsampling strategy
    unet_downsampling_strategy = [[2],[2],[2]]
    # U-Net convolution setup (downsampling path)
    unet_conv_down = [[[3],[3]]]
    # U-Net convolution setup (upsampling path)
    unet_conv_up = [[[3],[3]]]
    # SK-Net configurations
    sk_netconfs = []
    # Upsampling path with deconvolutions instead of convolutions
    use_deconvolution_uppath = False
    # Deep residual network bridging
    bridge = False
    # Bridge operation ('add' or 'stack')
    bridge_op = 'add'

# Create the network we want
class NetConf:
    # 10 GB total memory limit
    mem_global_limit = 10 * 1024 * 1024 * 1024
    # 4 GB single buffer memory limit
    mem_buf_limit = 4 * 1024 * 1024 * 1024
    # Network input dimension
    input_shape = [132, 132, 132]
    # Corresponding output dimension
    output_shape = [44, 44, 44]
    # Number of feature maps in the start
    fmap_start = 16
    # Number of input feature maps
    fmap_input = 1
    # Number of ouput feature maps
    fmap_output = 3
    # Loss function and mode ("malis", "euclid", "softmax")
    loss_function = "euclid"
    malis_split_component_phases = False
    # ReLU negative slope
    relu_slope = 0.005
    # Batch normalization
    use_batchnorm = True
    # Batch normalization moving average fraction
    batchnorm_maf = 0.95
    # Dropout
    dropout = 0.2
    # Ignore convolution buffer in memory computations
    ignore_conv_buffer = False
    # U-Net configurations
    u_netconfs = [UNetConf()]
    # Activation function before loss (sigmoid, relu, tanh)
    loss_activation = 'sigmoid'




class LayerException(Exception):
    pass

class ShapeException(Exception):
    pass

class MemoryLimitException(Exception):
    pass

class ConvolutionBufferException(Exception):
    pass

class RunShapeUpdater():
    def __init__(self):
        self.shape_update = lambda x: [x[i] for i in range(0,len(x))]
        self.dilation_update = lambda x: [x[i] for i in range(0,len(x))]
        self.fmaps_update = lambda x: x
        self.conv_buffer_mem_update = lambda x: 0
        self.weight_mem_update = lambda x: 0
        self.aux_mem_update = lambda x: 0


class RunShape():
    def __init__(self, other_runshape, runshape_updater):
        if (other_runshape is not None and runshape_updater is None):
            self.shape = other_runshape.shape
            self.dilation = other_runshape.dilation
            self.fmaps = other_runshape.fmaps
            self.conv_buffer_mem = other_runshape.conv_buffer_mem
            self.weight_mem = other_runshape.weight_mem
            self.aux_mem = other_runshape.aux_mem
        elif (other_runshape is not None and runshape_updater is not None):
            self.shape = runshape_updater.shape_update(other_runshape.shape)
            self.dilation = runshape_updater.dilation_update(other_runshape.dilation)
            self.fmaps = runshape_updater.fmaps_update(other_runshape.fmaps)
            self.conv_buffer_mem = runshape_updater.conv_buffer_mem_update(other_runshape.conv_buffer_mem)
            self.weight_mem = runshape_updater.weight_mem_update(other_runshape.weight_mem)
            self.aux_mem = runshape_updater.aux_mem_update(other_runshape.aux_mem)
        else:
            # The feature map shape traced (N-dimensional)
            self.shape = []
            # The dilation parameter traced (N-dimensional)
            self.dilation = []
            # The feature maps traced (1-dimensional)
            self.fmaps = 1
            # Convolution buffer memory (only the largest value is considered)
            self.conv_buffer_mem = 0
            # Weight memory (all values are summed up)
            self.weight_mem = 0
            # Auxiliary memory (such as 
            self.aux_mem = 0
            
        if self.fmaps < 1:
            raise ShapeException("Constraint violated: fmaps > 1, value %s" % self.fmaps)
            
        if any(x < 1 for x in self.shape):
            raise ShapeException("Constraint violated: shape[i] > 1, value %s " % self.shape)
        
        if any(x < 1 for x in self.dilation):
            raise ShapeException("Constraint violated: dilation[i] > 1, value %s " % self.shape)
    
    def print(self):
        print("f: %s w: %s d: %s" % (self.fmaps, self.shape, self.dilation))
        print("WM: %s" % (self.weight_mem))
        print("CM: %s" % (self.conv_buffer_mem))
        print("AM: %s" % (self.aux_mem))


class NetworkGenerator:
    def __init__(self, netconf, mode):
        self.netconf = copy.deepcopy(netconf)
        self.mode = copy.deepcopy(mode)
        self.graph = nvs.Graph()
        
    def compute_memory_aux(self, shape_arr):
        memory = 0
        for i in range(0,len(shape_arr)):
            memory += shape_arr[i].aux_mem
        return memory 
    
    def compute_memory_weights(self, shape_arr):
        memory = 0
        for i in range(0,len(shape_arr)):
            memory += shape_arr[i].weight_mem
        return memory
            
    def compute_memory_buffers(self, shape_arr):
        memory = 0
        for i in range(0,len(shape_arr)):
            memory = max(memory, shape_arr[i].conv_buffer_mem)
        return memory
            
    def compute_memory_blobs(self, shape_arr):
        memory = 0
        for i in range(0,len(shape_arr)):
            mem = fsize * shape_arr[i].fmaps
            for j in range(0,len(shape_arr[i].shape)):
                mem *= shape_arr[i].shape[j]
            memory += mem
        return memory
    
    def update_shape(self, shape_arr, update):
        last_shape = shape_arr[-1]
        new_shape = RunShape(last_shape, update);
        shape_arr += [new_shape]
        return shape_arr
    
    def data_layer(self, shape):
        data, label = L.MemoryData(dim=shape, ntop=2)
        return data, label

    def deconv_relu(self, run_shape, bottom, num_output, kernel_size=[3], stride=[1], pad=[0], group=1, weight_std=0.01):
        update = RunShapeUpdater()
        deconv = L.Deconvolution(bottom, convolution_param=dict(kernel_size=kernel_size, stride=stride, dilation=run_shape[-1].dilation,
                                    num_output=num_output, pad=pad, group=group,
                                    weight_filler=dict(type='gaussian', std=weight_std),
                                    bias_filler=dict(type='constant')), param=[dict(lr_mult=1),dict(lr_mult=2)])
        
        relu = L.ReLU(deconv, in_place=True, negative_slope=self.netconf.relu_slope)
        last = relu
        
        if (self.netconf.dropout > 0):
            drop = L.Dropout(last, in_place=True, dropout_ratio=self.netconf.dropout)
            last = drop
        
        if (self.netconf.use_batchnorm == True):
            bnl = L.BatchNorm(last, in_place=True,
                              param=[dict(lr_mult=0,decay_mult=0),dict(lr_mult=0,decay_mult=0),dict(lr_mult=0,decay_mult=0)],
                              batch_norm_param=dict(use_global_stats=(self.mode == caffe_pb2.TEST), moving_average_fraction=self.netconf.batchnorm_maf))
            last = bnl
            # Auxiliary memory consumption here is mean and variance of the input
            update.aux_mem_update = lambda x: fsize * 2 * num_output * reduce(lambda y, z: y*z, [run_shape[-1].shape[i] - (kernel_size[min(i,len(kernel_size)-1)] - 1) * (run_shape[-1].dilation[i]) for i in range(0, len(run_shape[-1].shape))])
        
        
        # The convolution buffer and weight memory
        weight_mem = fsize * num_output * run_shape[-1].fmaps
        conv_buff = fsize * run_shape[-1].fmaps
        for i in range(0,len(run_shape[-1].shape)):        
            conv_buff *= kernel_size[min(i,len(kernel_size)-1)]
            conv_buff *= run_shape[-1].shape[i]
            weight_mem *= kernel_size[min(i,len(kernel_size)-1)]
        
        # Shape update rules
        update.conv_buffer_mem_update = lambda x: conv_buff
        update.weight_mem_update = lambda x: weight_mem
        update.fmaps_update = lambda x: num_output
        update.shape_update = lambda x: [x[i] + (kernel_size[min(i,len(kernel_size)-1)] - 1) * (run_shape[-1].dilation[i]) for i in range(0, len(x))]
        self.update_shape(run_shape, update)
        
        return deconv, last

    # Convolution block. Order of operations:
    # 1. Convolution
    # 2. Shortcut/bridge (IN and OUT) (residual DNN)
    # 3. Dropout
    # 4. Batchnorm
    # 5. ReLU
    # References: http://torch.ch/blog/2016/02/04/resnets.html, https://github.com/KaimingHe/deep-residual-networks
    def conv_relu(self, run_shape, bottom, num_output, bridge_in = None, bridge_in_index = -1, in_place=True, bridge_op='add', kernel_size=[3], stride=[1], pad=[0], group=1, weight_std=0.01):
        update = RunShapeUpdater()
                
        conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride, dilation=run_shape[-1].dilation,
                                    num_output=num_output, pad=pad, group=group,
                                    param=[dict(lr_mult=1),dict(lr_mult=2)],
                                    weight_filler=dict(type='gaussian', std=weight_std),
                                    bias_filler=dict(type='constant'))
        last = conv
        
        if (self.netconf.use_batchnorm == True):
            # Auxiliary memory consumption here is mean and variance of the input
            update.aux_mem_update = lambda x: fsize * 2 * num_output * reduce(lambda y, z: y*z, [run_shape[-1].shape[i] for i in range(0, len(run_shape[-1].shape))])
        
        # The convolution buffer and weight memory
        weight_mem = fsize * num_output * run_shape[-1].fmaps
        conv_buff = fsize * run_shape[-1].fmaps
        for i in range(0,len(run_shape[-1].shape)):        
            conv_buff *= kernel_size[min(i,len(kernel_size)-1)]
            conv_buff *= run_shape[-1].shape[i]
            weight_mem *= kernel_size[min(i,len(kernel_size)-1)]
        
        # Shape update rules
        update.conv_buffer_mem_update = lambda x: conv_buff
        update.weight_mem_update = lambda x: weight_mem
        update.fmaps_update = lambda x: num_output
        update.shape_update = lambda x: [x[i] + 2*pad[min(i,len(pad)-1)] - (kernel_size[min(i,len(kernel_size)-1)] - 1) * (run_shape[-1].dilation[i]) for i in range(0, len(x))]
        self.update_shape(run_shape, update)
        
        # Deep residual / shortcut / bridge (add or stack)
        bridge_out_index = -1
        bridge_out = None
        if (bridge_in != None):
            bridge_bottom = bridge_in
            bridge_bottom_index = bridge_in_index
            # Make the number of feature maps fit for addition: y = f(x) + W*x
            if (run_shape[-1].fmaps != run_shape[bridge_bottom_index].fmaps):
                run_shape_bridge = [run_shape[bridge_bottom_index]]
                bridge_bottom = self.convolution(run_shape_bridge, bridge_bottom, run_shape[-1].fmaps, kernel_size=[1])
                run_shape = run_shape[0:-1] + [run_shape_bridge[-1]] + [run_shape[-1]]
                bridge_bottom_index = len(run_shape) - 1
                
            bridge_out = self.mergecrop(run_shape, bridge_bottom_index, last, bridge_bottom, op=bridge_op)
            bridge_out_index = len(run_shape) - 1
            last = bridge_out
        
        # Dropout
        if (self.netconf.dropout > 0):
            drop = L.Dropout(last, in_place=in_place, dropout_ratio=self.netconf.dropout)
            last = drop
        
        # Batchnorm
        if (self.netconf.use_batchnorm == True):
            bnl = L.BatchNorm(last, in_place=in_place,
                              param=[dict(lr_mult=0,decay_mult=0),dict(lr_mult=0,decay_mult=0),dict(lr_mult=0,decay_mult=0)],
                              batch_norm_param=dict(use_global_stats=(self.mode == caffe_pb2.TEST), moving_average_fraction=self.netconf.batchnorm_maf))
            last = bnl

        # Activation
        relu = L.ReLU(last, in_place=in_place, negative_slope=self.netconf.relu_slope)
        last = relu
        
        return conv, last, bridge_out, bridge_out_index
    
    def convolution(self, run_shape, bottom, num_output, kernel_size=[3], stride=[1], pad=[0], group=1, weight_std=0.01):
        update = RunShapeUpdater()

        # The convolution buffer and weight memory
        weight_mem = fsize * num_output * run_shape[-1].fmaps
        conv_buff = fsize * run_shape[-1].fmaps
        for i in range(0,len(run_shape[-1].shape)):        
            conv_buff *= kernel_size[min(i,len(kernel_size)-1)]
            conv_buff *= run_shape[-1].shape[i]
            weight_mem *= kernel_size[min(i,len(kernel_size)-1)]
        
        # Shape update rules
        update.conv_buffer_mem_update = lambda x: conv_buff
        update.weight_mem_update = lambda x: weight_mem
        update.fmaps_update = lambda x: num_output
        update.shape_update = lambda x: [x[i] + 2*pad[min(i,len(pad)-1)] - (kernel_size[min(i,len(kernel_size)-1)] - 1) * (run_shape[-1].dilation[i]) for i in range(0, len(x))]
        self.update_shape(run_shape, update)
        
        return L.Convolution(bottom, kernel_size=kernel_size, stride=stride, dilation=run_shape[-1].dilation,
                                    num_output=num_output, pad=pad, group=group,
                                    param=[dict(lr_mult=1),dict(lr_mult=2)],
                                    weight_filler=dict(type='gaussian', std=weight_std),
                                    bias_filler=dict(type='constant'))
    
    def max_pool(self, run_shape, bottom, kernel_size=[2], stride=[2], pad=[0]):
        dilation = run_shape[-1].dilation
        update = RunShapeUpdater()

        # Shape update rules
        update.dilation_update = lambda x: [x[i] * ((kernel_size[min(i,len(kernel_size)-1)]) if (stride[min(i,len(stride)-1)]==1) else 1) for i in range(0,len(x))]

        # Strictly speaking this update rule is not complete, but should be sufficient for USK
        if dilation[0] == 1 and kernel_size[0] == stride[0]:
            for i in range(0,len(run_shape[-1].shape)):
                if run_shape[-1].shape[i] % kernel_size[min(i,len(kernel_size)-1)] != 0:
                    raise LayerException("Constraint violated: mod(x[i], k[i])== 0, values %s, %s" % (run_shape[-1].shape[i], kernel_size[min(i,len(kernel_size)-1)]))
            
            update.shape_update = lambda x: [x[i] / (kernel_size[min(i,len(kernel_size)-1)]) for i in range(0,len(run_shape[-1].shape))]
        else:
            update.shape_update = lambda x: [x[i] - (kernel_size[min(i,len(kernel_size)-1)] - 1) * (run_shape[-1].dilation[i]) for i in range(0,len(run_shape[-1].shape))]
        self.update_shape(run_shape, update)
    
        return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=kernel_size, stride=stride, pad=pad, dilation=dilation)
    
    def upconv(self, run_shape, bottom, num_output_dec, num_output_conv, weight_std=0.01, kernel_size=[2], stride=[2]):
        # Shape update rules
        update = RunShapeUpdater()
        update.fmaps_update = lambda x: num_output_dec
        update.shape_update = lambda x: [kernel_size[min(i,len(kernel_size)-1)] * x[i] for i in range(0,len(run_shape[-1].shape))]
        self.update_shape(run_shape, update)
        
        deconv = L.Deconvolution(bottom, convolution_param=dict(num_output=num_output_dec, kernel_size=kernel_size, stride=stride, pad=[0], group=num_output_dec,
                                                                weight_filler=dict(type='constant', value=1), bias_term=False),
                                 param=dict(lr_mult=0, decay_mult=0))
    
        # The convolution buffer and weight memory
        weight_mem = fsize * num_output_conv * num_output_dec
        conv_buff = fsize * run_shape[-1].fmaps
        for i in range(0,len(run_shape[-1].shape)):        
            conv_buff *= 2
            conv_buff *= run_shape[-1].shape[i]
        
        # Shape update rules
        update.conv_buffer_mem_update = lambda x: conv_buff
        update.weight_mem_update = lambda x: weight_mem
        update.fmaps_update = lambda x: num_output_conv
        update.shape_update = lambda x: [x[i] for i in range(0,len(run_shape[-1].shape))]
        self.update_shape(run_shape, update)
    
        conv = L.Convolution(deconv, num_output=num_output_conv, kernel_size=[1], stride=[1], pad=[0], group=1,
                                param=[dict(lr_mult=1),dict(lr_mult=2)],
                                weight_filler=dict(type='gaussian', std=weight_std),
                                bias_filler=dict(type='constant'))
        return deconv, conv
    
    def mergecrop(self, run_shape, run_shape_b_index, bottom_a, bottom_b, op = 'stack'):
        run_shape_a = run_shape[-1]
        run_shape_b = run_shape[run_shape_b_index]
        # Shape update rules
        update = RunShapeUpdater()
        if (op == 'stack'):
            update.fmaps_update = lambda x: run_shape_a.fmaps + run_shape_b.fmaps
        else:
            update.fmaps_update = lambda x: run_shape_a.fmaps
        self.update_shape(run_shape, update)
        return L.MergeCrop(bottom_a, bottom_b, forward=[1,1], backward=[1,1], operation=(0 if (op == 'stack') else 1))
    
    def weight_filler(self, shape, ksizes):
        return math.sqrt(2.0/float(shape.fmaps*reduce(lambda a,b: a * b, [abs(ksizes[min(i, len(ksizes)-1)]) for i in range(0,len(shape.shape))])))
    
    def implement_sknet(self, netconf, sknetconf, net, run_shape, blobs, fmaps_start, uidx=0, skidx=0):
        fmaps = fmaps_start
        sw_shape = [sknetconf.sknet_padding[min(i,len(sknetconf.sknet_padding)-1)] + 1 for i in range(0,len(run_shape[-1].shape))]
        for i in range(0, len(sknetconf.sknet_conv)):
            final_ksize = [sknetconf.sknet_conv[i][min(j, len(sknetconf.sknet_conv[i])-1)] for j in range(0,len(sw_shape))]
            for j in range(0, len(sw_shape)):
                if(not (sw_shape[j] - (final_ksize[j] - 1)) % 2 == 0):
                    final_ksize[j] += 1
                sw_shape[j] = (sw_shape[j] - (final_ksize[j] - 1)) / 2
            conv, relu, _, _ = self.conv_relu(run_shape, blobs[-1], fmaps, kernel_size=final_ksize, weight_std=self.weight_filler(run_shape[-1], final_ksize))
            blobs = blobs + [relu]
            conv.name = 'conv_U-' + str(uidx) + '_SK-' + str(skidx) + '_C-' + str(i)
            node = nvs.Node(conv, run_shape[-1])
            self.graph.add_node(node)
            pool = self.max_pool(run_shape, blobs[-1], kernel_size=[2], stride=[1])
            blobs = blobs + [pool]
            pool.name = 'pool_U-' + str(uidx) + '_SK-' + str(skidx) + '_P-' + str(i)
            node = nvs.Node(pool, run_shape[-1])
            self.graph.add_node(node)
            if (i < len(sknetconf.sknet_conv) - 1):
                fmaps = sknetconf.sknet_fmap_inc_rule(fmaps)

        fmaps = sknetconf.sknet_fmap_bridge_rule(fmaps)
        # 1st IP layer
        conv, relu, _, _ = self.conv_relu(run_shape, blobs[-1], fmaps, kernel_size=sw_shape, weight_std=self.weight_filler(run_shape[-1], sw_shape))
        blobs = blobs + [relu]
        run_shape[-1].dilation = [1 for i in range(0,len(run_shape[-1].dilation))]
        conv.name = 'conv_U-' + str(uidx) + '_SK-' + str(skidx) + '_IP-0'
        node = nvs.Node(conv, run_shape[-1])
        self.graph.add_node(node)
        
        print(run_shape[-1].shape)
        # Remaining IP layers
        for i in range(0, sknetconf.sknet_ip_depth - 1):
            fmaps = sknetconf.sknet_fmap_dec_rule(fmaps)
            conv, relu, _, _ = self.conv_relu(run_shape, blobs[-1], fmaps, kernel_size=[1], weight_std=self.weight_filler(run_shape[-1], [1]))
            conv.name = 'conv_U-' + str(uidx) + '_SK-' + str(skidx) + '_IP-' + str(1 + i)
            blobs = blobs + [relu]
            node = nvs.Node(conv, run_shape[-1])
            self.graph.add_node(node)
        
        return blobs, run_shape
                
    
    def implement_usknet(self, netconf, net, run_shape, blobs, fmaps_start, fmaps_end): 
        if self.mode == caffe_pb2.TEST:
            self.netconf.dropout = 0
            
        # Add the start node to the graph
        net.data.name = 'data'
        node = nvs.Node(net.data, run_shape[-1])
        self.graph.add_node(node)
        
        fmaps = fmaps_start
        
        # At the start of the network, bridge input is network input
        bridge_in = blobs[-1]
        bridge_in_index = len(run_shape) - 1
        
        for uidx in range(0,len(netconf.u_netconfs)):
            unetconf = netconf.u_netconfs[uidx]
            mergecrop_tracker = []
            if unetconf.unet_depth > 0:
                # U-Net downsampling; 2*Convolution+Pooling
                for i in range(0, unetconf.unet_depth):
                    convolution_config = unetconf.unet_conv_down[min(i,len(unetconf.unet_conv_down) - 1)]
                    for j in range(0,len(convolution_config)):
                        conv, relu, _, _ = self.conv_relu(run_shape, blobs[-1], fmaps, kernel_size=convolution_config[j], weight_std=self.weight_filler(run_shape[-1], convolution_config[j]))
                        blobs = blobs + [relu]
                        conv.name = 'conv_U-' + str(uidx) + '_LD-' + str(i) + '_C-' + str(j)
                        node = nvs.Node(conv, run_shape[-1])
                        self.graph.add_node(node)
    
                    mergecrop_tracker += [[len(blobs)-1,len(run_shape)-1]]
                    pool = self.max_pool(run_shape, blobs[-1], kernel_size=unetconf.unet_downsampling_strategy[i], stride=unetconf.unet_downsampling_strategy[i])
                    blobs = blobs + [pool]
                    fmaps = unetconf.unet_fmap_inc_rule(fmaps)
                    pool.name = 'pool_U-' + str(uidx) + '_LD-' + str(i) + '_P-0'
                    node = nvs.Node(pool, run_shape[-1])
                    self.graph.add_node(node)
        
            
            # If there is no SK-Net component, fill with normal convolutions
            if (unetconf.unet_depth > 0 and (len(unetconf.sk_netconfs) - 1 < unetconf.unet_depth or unetconf.sk_netconfs[unetconf.unet_depth] == None)):
                convolution_config = unetconf.unet_conv_down[min(unetconf.unet_depth, len(unetconf.unet_conv_down) - 1)]
                for j in range(0,len(convolution_config)):
                    # Here we are at the bottom, so the second half of the convolutions already belongs to the up-path
                    if (unetconf.use_deconvolution_uppath and j >= len(convolution_config)/2):
                        conv, relu, _, _ = self.conv_relu(run_shape, blobs[-1], fmaps, kernel_size=convolution_config[j], pad=[convolution_config[j][k] - 1 for k in range(0,len(convolution_config[j]))], weight_std=self.weight_filler(run_shape[-1], convolution_config[j]))
                        blobs = blobs + [relu]
                        conv.name = 'conv_U-' + str(uidx) + '_LD-' + str(unetconf.unet_depth) + '_C-' + str(j)
                        node = nvs.Node(conv, run_shape[-1])
                        self.graph.add_node(node)
                        
                    else:
                        conv, relu, _, _ = self.conv_relu(run_shape, blobs[-1], fmaps, kernel_size=convolution_config[j], weight_std=self.weight_filler(run_shape[-1], convolution_config[j]))
                        blobs = blobs + [relu]
                        conv.name = 'conv_U-' + str(uidx) + '_LU-' + str(unetconf.unet_depth) + '_C-' + str(j)
                        node = nvs.Node(conv, run_shape[-1])
                        self.graph.add_node(node)

            else:
                blobs, run_shape = self.implement_sknet(netconf, unetconf.sk_netconfs[unetconf.unet_depth], net, run_shape, blobs, unetconf.sk_netconfs[unetconf.unet_depth].sknet_fmap_inc_rule(fmaps),
                                                        uidx=uidx, skidx=unetconf.unet_depth)
                fmaps = run_shape[-1].fmaps
    
            if unetconf.unet_depth > 0:
                # U-Net upsampling; Upconvolution+MergeCrop+2*Convolution
                for i in range(0, unetconf.unet_depth):
                    deconv, conv = self.upconv(run_shape, blobs[-1], fmaps, unetconf.unet_fmap_dec_rule(fmaps), kernel_size=unetconf.unet_downsampling_strategy[unetconf.unet_depth - i - 1],
                                               stride=unetconf.unet_downsampling_strategy[unetconf.unet_depth - i - 1],
                                               weight_std=self.weight_filler(run_shape[-1],unetconf.unet_downsampling_strategy[unetconf.unet_depth - i - 1]))
                    blobs = blobs + [conv]
                    deconv.name = 'conv_U-' + str(uidx) + '_LU-' + str(unetconf.unet_depth - i - 1) + '_DD-0'
                    conv.name = 'conv_U-' + str(uidx) + '_LU-' + str(unetconf.unet_depth - i - 1) + '_DC-0'
                    fmaps = unetconf.unet_fmap_dec_rule(fmaps)
                    
                    pre_merge_blobs = [blobs[mergecrop_tracker[unetconf.unet_depth - i - 1][0]]]
                    pre_merge_shape_index = mergecrop_tracker[unetconf.unet_depth - i - 1][1]
                    
                    # Insert SK-Net in the mergecrop bridge
                    if (len(unetconf.sk_netconfs) > unetconf.unet_depth - i - 1 and unetconf.sk_netconfs[unetconf.unet_depth - i - 1] != None):
                        sknet_conf = copy.deepcopy(unetconf.sk_netconfs[unetconf.unet_depth - i - 1])
                        sknet_conf.sknet_padding = [run_shape[pre_merge_shape_index].shape[k] - run_shape[-1].shape[k] for k in range(0,len(run_shape[-1].shape))]
                        sk_run_shape = [run_shape[pre_merge_shape_index]]
                        pre_merge_blobs, sk_run_shape = self.implement_sknet(netconf, sknet_conf, net, sk_run_shape, pre_merge_blobs, sknet_conf.sknet_fmap_inc_rule(run_shape[pre_merge_shape_index].fmaps),
                                                                             uidx=uidx, skidx=unetconf.unet_depth - i - 1)
                        new_run_shape = run_shape[0:pre_merge_shape_index]+sk_run_shape
                        new_pre_merge_shape_index = len(new_run_shape) - 1
                        new_run_shape += run_shape[pre_merge_shape_index+1:len(run_shape)]
                        run_shape = new_run_shape
                        pre_merge_shape_index = new_pre_merge_shape_index
    
                    # Here, layer (2 + 3 * i) with reversed i (high to low) is picked
                    mergec = self.mergecrop(run_shape, pre_merge_shape_index, blobs[-1], pre_merge_blobs[-1])
                    blobs = blobs + [mergec]
                    mergec.name = 'conv_U-' + str(uidx) + '_LU-' + str(unetconf.unet_depth - i - 1) + '_M-0'
                    node = nvs.Node(mergec, run_shape[-1])
                    self.graph.add_node(node)
                    
                    convolution_config = unetconf.unet_conv_up[min(unetconf.unet_depth - i - 1, len(unetconf.unet_conv_up) - 1)]
                    for j in range(0,len(convolution_config)):
                        in_place = True
                        # In-place computation disabled if the next U-Net is bridged, due to applying Dropout, Batchnorm, ReLU after the MergeCrop
                        if (j == len(convolution_config)-1 and i == unetconf.unet_depth-1 and len(netconf.u_netconfs) > uidx + 1 and netconf.u_netconfs[uidx + 1].bridge == True):
                            in_place = False
                        pad =  [convolution_config[j][k] - 1 for k in range(0,len(convolution_config[j]))] if (unetconf.use_deconvolution_uppath) else [0]                       
                        conv, relu, bridge, bridge_index = self.conv_relu(run_shape, blobs[-1], fmaps, kernel_size=convolution_config[j],
                                                    in_place = in_place,
                                                    bridge_in=(bridge_in if (unetconf.bridge == True and j == len(convolution_config)-1 and i == unetconf.unet_depth-1) else None),
                                                    bridge_in_index=bridge_in_index,
                                                    pad=pad, weight_std=self.weight_filler(run_shape[-1], convolution_config[j]))
                        conv.name = 'conv_U-' + str(uidx) + '_LU-' + str(i) + '_C-' + str(j)
                        node = nvs.Node(conv, run_shape[-1])
                        self.graph.add_node(node)
                        # Update shortcut / bridge input to jump the next U-Net
                        if (j == len(convolution_config)-1 and i == unetconf.unet_depth-1):
                            if (unetconf.bridge == True and bridge != None):
                                # The current U-Net is bridged, so use the bridge output after the last bridge
                                bridge_in = bridge
                                bridge_in_index = bridge_index
                            else:
                                # The current U-Netis not bridged, so use the output of the last convolution (before ReLU)
                                bridge_in = conv
                                bridge_in_index = len(run_shape) - 1
                        blobs = blobs + [relu]
                               
        conv = self.convolution(run_shape, blobs[-1], fmaps_end, kernel_size=[1], weight_std=self.weight_filler(run_shape[-1], [1]))
        blobs = blobs + [conv]
        conv.name = 'conv_out'
        node = nvs.Node(conv, run_shape[-1])
        self.graph.add_node(node)
        
        if (not netconf.ignore_conv_buffer and self.compute_memory_buffers(run_shape) > netconf.mem_buf_limit):
            raise ConvolutionBufferException("Constraint violated: convolution buffer exceeds limt, %s > %s", self.compute_memory_buffers(run_shape), netconf.mem_buf_limit)
        
        total_memory = 0
        if not netconf.ignore_conv_buffer:
            total_memory += self.compute_memory_buffers(run_shape)
        total_memory += self.compute_memory_aux(run_shape)
        total_memory += self.compute_memory_weights(run_shape)
        total_memory += (1 if (self.mode == caffe_pb2.TEST) else 2) * self.compute_memory_blobs(run_shape)
        
        if (total_memory > netconf.mem_global_limit):
            raise MemoryLimitException("Constraint violated: memory usage exceeds limt, %s > %s", total_memory, netconf.mem_global_limit)
        
        # Return the last blob of the network (goes to error objective)
        return blobs, run_shape
    
    def implement_loss_activation(self, blob, in_place):
        loss_blob = None
        if (self.netconf.loss_activation == 'sigmoid'):
            loss_blob = L.Sigmoid(blob, ntop=1, in_place=in_place)
        elif (self.netconf.loss_activation == 'relu'):
            loss_blob = L.ReLU(blob, ntop=1, in_place=in_place, negative_slope=0.0)
        else:
            loss_blob = None
        return loss_blob        
    
    
def compute_valid_io_shapes(netconf, netmode, min_output_shape, max_output_shape, fmaps_in=1, fmaps_out=1, constraints=None):
    
    valid_in_shapes = []
    valid_out_shapes = []
    
    dims = len(min_output_shape)
    
    for current_dim in range(0, dims):
        filtered_in_shapes = copy.deepcopy(valid_in_shapes)
        
        if not (constraints is None) and len(constraints) > current_dim and not (constraints[current_dim] is None):
            in_shape = [(constraints[i](filtered_in_shapes[0]) if i >= current_dim else filtered_in_shapes[0][i]) for i in range(0,current_dim+1)]
        else:
            in_shape = [(min_output_shape[i] if i >= current_dim else filtered_in_shapes[0][i]) for i in range(0,current_dim+1)]
            
        in_index = 0
        valid_in_shapes = []
        valid_out_shapes = []
        
        while(True):
            net = caffe.NetSpec()
            
            run_shape = RunShape(None, None)
            run_shape.shape = in_shape[0:current_dim+1]
            run_shape.dilation = [1 for i in range(0,dims)]
            run_shape.fmaps = fmaps_in
        
            run_shape_in = [run_shape]
            run_shape_out = run_shape_in
            
            netgen = NetworkGenerator(netconf, netmode);
            
            limit_reached = False
            valid_io_shape = True
            
            try:
                net.data, net.datai = netgen.data_layer([1]+[fmaps_in]+in_shape[0:current_dim+1])
                net.silence = L.Silence(net.datai, ntop=0)
                # Chained blob list to construct the network (forward direction)
                blobs = []
                # All networks start with data
                blobs = blobs + [net.data]
                netgen.implement_usknet(netconf, net, run_shape_out, blobs, 1, fmaps_out)
            except MemoryLimitException:
                limit_reached = True
                valid_io_shape = True
            except ConvolutionBufferException:
                limit_reached = True
                valid_io_shape = True
            except ShapeException:
                limit_reached = False
                valid_io_shape = False
            except LayerException:
                limit_reached = False
                valid_io_shape = False
            
                        
            if (valid_io_shape and not limit_reached and not reduce(lambda a,b: a and b, [run_shape_out[-1].shape[i] >= max_output_shape[i] for i in range(0,current_dim+1)], True)):
                print("++++ Valid: %s => %s" % (run_shape_out[0].shape, run_shape_out[-1].shape))
                valid_in_shapes += [run_shape_out[0].shape]
                valid_out_shapes += [run_shape_out[-1].shape]
            else:
                print("-- Invalid: %s => []" % (run_shape_out[0].shape))
     
            
            incremented = False       
            
            if not incremented and ((valid_io_shape or limit_reached) and len(filtered_in_shapes) > 0):
                if in_index >= len(filtered_in_shapes) - 1:
                    in_index = 0
                    in_shape[0:current_dim] = filtered_in_shapes[in_index]
                else:
                    in_index += 1
                    in_shape[0:current_dim] = filtered_in_shapes[in_index]
                    incremented = True
            
            if not (constraints is None) and len(constraints) > current_dim and not (constraints[current_dim] is None):
                in_shape[current_dim] = constraints[current_dim](in_shape)
                if in_index > 0:
                    incremented = True
            else:
                if not incremented:
                    if in_shape[current_dim] >= max_output_shape[current_dim]:
                        in_shape[current_dim] = min_output_shape[current_dim]
                    else:
                        in_shape[current_dim] += 1
                        incremented = True
         
            if not incremented:
                break
        
        if (len(valid_in_shapes) == 0):
            break
        
        
    max_fmap_counts = []
    for shape_idx in range(0,len(valid_in_shapes)):
        
        incexp = True
        bisect = False
        
        fmaps_start = 1
        lower_limit = 1
        upper_limit = 1
        
        while(True):
            net = caffe.NetSpec()
            
            run_shape = RunShape(None, None)
            run_shape.shape = valid_in_shapes[shape_idx]
            run_shape.dilation = [1 for i in range(0,dims)]
            run_shape.fmaps = 1
        
            run_shape_in = [run_shape]
            run_shape_out = run_shape_in
            
            netgen = NetworkGenerator(netconf, netmode);
            
            limit_reached = False
            valid_io_shape = True
            
            try:
                net.data, net.datai = netgen.data_layer([1]+[1]+valid_in_shapes[shape_idx])
                net.silence = L.Silence(net.datai, ntop=0)
                # Chained blob list to construct the network (forward direction)
                blobs = []
                # All networks start with data
                blobs = blobs + [net.data]
                netgen.implement_usknet(netconf, net, run_shape_out, blobs, fmaps_start, fmaps_out)
            except (MemoryLimitException, ConvolutionBufferException, ShapeException, LayerException):
                limit_reached = True
                
            # Deterministic exit condition (protects against infinite looping)
            if fmaps_start > 100000:
                upper_limit = 100000
                break;
        
            if (not limit_reached and incexp):
                fmaps_start *= 2
            elif (limit_reached and incexp):
                incexp = False
                bisect = True
                lower_limit = fmaps_start / 2
                upper_limit = fmaps_start
            elif (not limit_reached and bisect):
                if(lower_limit >= upper_limit):
                    break;
                lower_limit = fmaps_start + 1
            elif (limit_reached and bisect):
                upper_limit = fmaps_start - 1

            if bisect:
                fmaps_start = (upper_limit + lower_limit) / 2
                
            print("%s in [%s, %s]" % (fmaps_start, lower_limit, upper_limit))
            
        max_fmap_counts += [upper_limit]
        print("Current shape: %s, %s, %s" % (shape_idx, valid_in_shapes[shape_idx], upper_limit))
        
    return valid_in_shapes, valid_out_shapes, max_fmap_counts


def caffenet(netconf, netmode):
    # Start Caffe proto net
    net = caffe.NetSpec()
    # Specify input data structures
    
    dims = len(netconf.input_shape)
    
    run_shape = RunShape(None, None)
    run_shape.shape = netconf.input_shape
    run_shape.dilation = [1 for i in range(0,dims)]
    run_shape.fmaps = 1
    
    run_shape_in = [run_shape]
    run_shape_out = run_shape_in
    
    if netmode == caffe_pb2.TEST:
        netgen = NetworkGenerator(netconf, netmode);
        
        net.data, net.datai = netgen.data_layer([1]+[netconf.fmap_input]+netconf.input_shape)
        net.silence = L.Silence(net.datai, ntop=0)
        # Chained blob list to construct the network (forward direction)
        blobs = []
        # All networks start with data
        blobs = blobs + [net.data]
        blobs, run_shape_out = netgen.implement_usknet(netconf, net, run_shape_out, blobs, netconf.fmap_start, netconf.fmap_output)
        last_blob = blobs[-1]

        # Implement the prediction layer
        if netconf.loss_function == 'malis':
            net.prob = netgen.implement_loss_activation(last_blob, False)
        
        if netconf.loss_function == 'euclid':
            net.prob = netgen.implement_loss_activation(last_blob, False)
            
        if netconf.loss_function == 'softmax':
            net.prob = L.Softmax(last_blob, ntop=1)

        for i in range(0,len(run_shape_out)):
            print("Shape: [%s]" % i)
            run_shape_out[i].print()
            
        print("Max. memory requirements: %s B" % (netgen.compute_memory_buffers(run_shape_out)+netgen.compute_memory_weights(run_shape_out)+netgen.compute_memory_blobs(run_shape_out)))
        print("Weight memory: %s B" % netgen.compute_memory_weights(run_shape_out))
        print("Max. conv buffer: %s B" % netgen.compute_memory_buffers(run_shape_out))
        
    else:
        netgen = NetworkGenerator(netconf, netmode);
        
        net.data, net.datai = netgen.data_layer([1]+[netconf.fmap_input]+netconf.input_shape)

        if netconf.loss_function == 'malis':
            net.label, net.labeli = netgen.data_layer([1]+[netconf.fmap_output]+netconf.output_shape)
            net.components, net.componentsi = netgen.data_layer([1, 2 if netconf.malis_split_component_phases else 1]+netconf.output_shape)
            net.nhood, net.nhoodi = netgen.data_layer([1, 1]+[netconf.fmap_output]+[3])
            net.silence = L.Silence(net.datai, net.labeli, net.componentsi, net.nhoodi, ntop=0)
            
        if netconf.loss_function == 'euclid':
            net.label, net.labeli = netgen.data_layer([1]+[netconf.fmap_output]+netconf.output_shape)
            net.scale, net.scalei = netgen.data_layer([1]+[netconf.fmap_output]+netconf.output_shape)
            net.silence = L.Silence(net.datai, net.labeli, net.scalei, ntop=0)

        if netconf.loss_function == 'softmax':
            net.label, net.labeli = netgen.data_layer([1]+[netconf.fmap_output]+netconf.output_shape)
            net.silence = L.Silence(net.datai, net.labeli, ntop=0)
        
        
        # Start the actual network
        # Chained blob list to construct the network (forward direction)
        blobs = []
        # All networks start with data
        blobs = blobs + [net.data]
        blobs, run_shape_out = netgen.implement_usknet(netconf, net, run_shape_out, blobs, netconf.fmap_start, netconf.fmap_output)
        last_blob = blobs[-1]
        
        for i in range(0,len(run_shape_out)):
            print("Shape: [%s]" % i)
            run_shape_out[i].print()
            
        print("Max. memory requirements: %s B" % (netgen.compute_memory_buffers(run_shape_out)+netgen.compute_memory_weights(run_shape_out)+2*netgen.compute_memory_blobs(run_shape_out)))
        print("Weight memory: %s B" % netgen.compute_memory_weights(run_shape_out))
        print("Max. conv buffer: %s B" % netgen.compute_memory_buffers(run_shape_out))
        
        # Implement the loss
        if netconf.loss_function == 'malis':       
            last_blob = netgen.implement_loss_activation(last_blob, True)
            net.loss = L.MalisLoss(last_blob, net.label, net.components, net.nhood, ntop=0)
        
        if netconf.loss_function == 'euclid':
            last_blob = netgen.implement_loss_activation(last_blob, True)
            net.loss = L.EuclideanLoss(last_blob, net.label, net.scale, ntop=0)
            
        if netconf.loss_function == 'softmax':
            net.loss = L.SoftmaxWithLoss(last_blob, net.label, ntop=0)
            
    # Return the protocol buffer of the generated network
    protonet = net.to_proto()
    protonet.name = 'NET_' + ('TEST' if (netmode == caffe_pb2.TEST) else 'TRAIN')
    netgen.graph.set_netspec(net)
    tikzgraph = netgen.graph.generate_tikz_graph()
    return (protonet, tikzgraph)


def create_nets(netconf):
    trainnet, trainnet_tikzgraph = caffenet(netconf, caffe_pb2.TRAIN)
    testnet, testnet_tikzgraph = caffenet(netconf, caffe_pb2.TEST)
    return (trainnet, testnet, trainnet_tikzgraph, testnet_tikzgraph)
    
    
    
    
    
    
    
