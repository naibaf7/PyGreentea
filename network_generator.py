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
# Size of a float variable in bytes
fsize = 4

# Create the network we want
class NetConf:
    # 10 GB total memory limit
    mem_global_limit = 10 * 1024 * 1024 * 1024
    # 4 GB single buffer memory limit
    mem_buf_limit = 4 * 1024 * 1024 * 1024
    # Desired input dimensions (will select closest possible)
    input_shape = [132,132,132]
    # Desired output dimensions (will select closest possible)
    output_shape = [44, 44, 44]
    # Number of U-Net Pooling-Convolution downsampling/upsampling steps
    unet_depth = 3
    # Number of feature maps in the start
    fmap_start = 32
    # Number of input feature maps
    fmap_input = 1
    # Number of ouput feature maps
    fmap_output = 3
    # Feature map increase rule (downsampling)
    def unet_fmap_inc_rule(self, fmaps):
        return int(math.ceil(fmaps * 4))
    # Feature map decrease rule (upsampling)
    def unet_fmap_dec_rule(self, fmaps):
        return int(math.ceil(fmaps / 4))
    # Skewed U-Net downsampling strategy
    unet_downsampling_strategy = [[2,2,2],[2,2,2],[2,2,2]]
    # Number of SK-Net Pooling-Convolution steps (per U-Net bridge)
    sknet_conv_depth = [0]
    # Feature map increase rule
    def sknet_fmap_inc_rule(self, fmaps):
        return int(math.ceil(fmaps * 1.5))
    # Number of 1x1 (IP) Convolution steps (per U-Net bridge)
    sknet_ip_depth = [0]
    # Feature map increase rule from SK-Convolution to IP
    def sknet_fmap_bridge_rule(self, fmaps):
        return int(math.ceil(fmaps * 4))
    # Feature map decrease rule within IP
    def sknet_fmap_dec_rule(self, fmaps):
        return int(math.ceil(fmaps / 2.5))
    # Loss function and mode ("malis", "euclid", "softmax")
    loss_function = "euclid"
    # ReLU negative slope
    relu_slope = 0.005
    # Batch Normalization
    use_batchnorm = True
    batchnorm_maf = 0.95
    # Dropout
    dropout = 0.2

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
    
    def conv_relu(self, run_shape, bottom, num_output, kernel_size=[3], stride=[1], pad=[0], dilation=[1], group=1, weight_std=0.01):
        update = RunShapeUpdater()
                
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
        update.shape_update = lambda x: [x[i] - (kernel_size[min(i,len(kernel_size)-1)] - 1) * (run_shape[-1].dilation[i]) for i in range(0, len(x))]
        self.update_shape(run_shape, update)
        
        return conv, last
    
    def convolution(self, run_shape, bottom, num_output, kernel_size=[3], stride=[1], pad=[0], dilation=[1], group=1, weight_std=0.01):
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
        update.shape_update = lambda x: [x[i] - (kernel_size[min(i,len(kernel_size)-1)] - 1) * (run_shape[-1].dilation[i]) for i in range(0, len(x))]
        self.update_shape(run_shape, update)
        
        return L.Convolution(bottom, kernel_size=kernel_size, stride=stride, dilation=dilation,
                                    num_output=num_output, pad=pad, group=group,
                                    param=[dict(lr_mult=1),dict(lr_mult=2)],
                                    weight_filler=dict(type='gaussian', std=weight_std),
                                    bias_filler=dict(type='constant'))
    
    def max_pool(self, run_shape, bottom, kernel_size=[2], stride=[2], pad=[0], dilation=[1]):
        update = RunShapeUpdater()

        # Shape update rules
        update.dilation_update = lambda x: [x[i] * dilation[min(i,len(dilation)-1)] for i in range(0,len(x))]

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
        
        deconv = L.Deconvolution(bottom, convolution_param=dict(num_output=num_output_dec, kernel_size=kernel_size, stride=stride, pad=[0], dilation=[1], group=num_output_dec,
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
    
        conv = L.Convolution(deconv, num_output=num_output_conv, kernel_size=[1], stride=[1], pad=[0], dilation=[1], group=1,
                                param=[dict(lr_mult=1),dict(lr_mult=2)],
                                weight_filler=dict(type='gaussian', std=weight_std),
                                bias_filler=dict(type='constant'))
        return deconv, conv
    
    def mergecrop(self, run_shape, run_shape_b_index, bottom_a, bottom_b):
        run_shape_a = run_shape[-1]
        run_shape_b = run_shape[run_shape_b_index]
        # Shape update rules
        update = RunShapeUpdater()
        update.fmaps_update = lambda x: run_shape_a.fmaps + run_shape_b.fmaps
        self.update_shape(run_shape, update)
        return L.MergeCrop(bottom_a, bottom_b, forward=[1,1], backward=[1,1])
    
    def implement_usknet(self, netconf, net, run_shape, fmaps_start, fmaps_end):
        # Chained blob list to construct the network (forward direction)
        blobs = []
    
        if self.mode == caffe_pb2.TEST:
            self.netconf.dropout = 0
    
        # All networks start with data
        blobs = blobs + [net.data]
        
        fmaps = fmaps_start
        
        mergecrop_tracker = []
    
        if netconf.unet_depth > 0:
            # U-Net downsampling; 2*Convolution+Pooling
            for i in range(0, netconf.unet_depth):
                conv, relu = self.conv_relu(run_shape, blobs[-1], fmaps, kernel_size=[3], weight_std=math.sqrt(2.0/float(run_shape[-1].fmaps*pow(3,len(run_shape[-1].shape)))))
                blobs = blobs + [relu]
                conv, relu = self.conv_relu(run_shape, blobs[-1], fmaps, kernel_size=[3], weight_std=math.sqrt(2.0/float(run_shape[-1].fmaps*pow(3,len(run_shape[-1].shape)))))
                blobs = blobs + [relu]  # This is the blob of interest for mergecrop (index 2 + 3 * i)
                mergecrop_tracker += [[len(blobs)-1,len(run_shape)-1]]
                pool = self.max_pool(run_shape, blobs[-1], kernel_size=netconf.unet_downsampling_strategy[i], stride=netconf.unet_downsampling_strategy[i])
                blobs = blobs + [pool]
                fmaps = netconf.unet_fmap_inc_rule(fmaps)
    
        
        # If there is no SK-Net component, fill with 2 convolutions
        if (netconf.unet_depth > 0 and netconf.sknet_conv_depth[0] == 0):
            conv, relu = self.conv_relu(run_shape, blobs[-1], fmaps, kernel_size=[3], weight_std=math.sqrt(2.0/float(run_shape[-1].fmaps*pow(3,len(run_shape[-1].shape)))))
            blobs = blobs + [relu]
            conv, relu = self.conv_relu(run_shape, blobs[-1], fmaps, kernel_size=[3], weight_std=math.sqrt(2.0/float(run_shape[-1].fmaps*pow(3,len(run_shape[-1].shape)))))
            blobs = blobs + [relu]
        # Else use the SK-Net instead
        else:
            for i in range(0, netconf.sknet_conv_depth):
                # TODO: Not implemented yet (fixme)
                run_shape = run_shape
        
        if netconf.unet_depth > 0:
            # U-Net upsampling; Upconvolution+MergeCrop+2*Convolution
            for i in range(0, netconf.unet_depth):
                deconv, conv = self.upconv(run_shape, blobs[-1], fmaps, netconf.unet_fmap_dec_rule(fmaps), kernel_size=netconf.unet_downsampling_strategy[i], stride=netconf.unet_downsampling_strategy[i], weight_std=math.sqrt(2.0/float(run_shape[-1].fmaps*pow(3,len(run_shape[-1].shape)))))
                blobs = blobs + [conv]
                fmaps = netconf.unet_fmap_dec_rule(fmaps)
                # Here, layer (2 + 3 * i) with reversed i (high to low) is picked
                mergec = self.mergecrop(run_shape, mergecrop_tracker[netconf.unet_depth - i - 1][1], blobs[-1], blobs[mergecrop_tracker[netconf.unet_depth - i - 1][0]])
                blobs = blobs + [mergec]
                conv, relu = self.conv_relu(run_shape, blobs[-1], fmaps, kernel_size=[3], weight_std=math.sqrt(2.0/float(run_shape[-1].fmaps*pow(3,len(run_shape[-1].shape)))))
                blobs = blobs + [relu]
                conv, relu = self.conv_relu(run_shape, blobs[-1], fmaps, kernel_size=[3], weight_std=math.sqrt(2.0/float(run_shape[-1].fmaps*pow(3,len(run_shape[-1].shape)))))
                blobs = blobs + [relu]
                
            conv = self.convolution(run_shape, blobs[-1], fmaps_end, kernel_size=[1], weight_std=math.sqrt(2.0/float(run_shape[-1].fmaps*pow(3,len(run_shape[-1].shape)))))
            blobs = blobs + [conv]
        
        
        if (self.compute_memory_buffers(run_shape) > netconf.mem_buf_limit):
            raise ConvolutionBufferException("Constraint violated: convolution buffer exceeds limt, %s > %s", self.compute_memory_buffers(run_shape), netconf.mem_buf_limit)
        
        total_memory = 0
        total_memory += self.compute_memory_buffers(run_shape)
        total_memory += self.compute_memory_aux(run_shape)
        total_memory += self.compute_memory_weights(run_shape)
        total_memory += (1 if (self.mode == caffe_pb2.TEST) else 2) * self.compute_memory_blobs(run_shape)
        
        if (total_memory > netconf.mem_global_limit):
            raise MemoryLimitException("Constraint violated: memory usage exceeds limt, %s > %s", total_memory, netconf.mem_global_limit)
        
        # Return the last blob of the network (goes to error objective)
        return blobs[-1]
    
    
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
                netgen.implement_usknet(netconf, net, run_shape_out, 1, fmaps_out)
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
                netgen.implement_usknet(netconf, net, run_shape_out, fmaps_start, fmaps_out)
            except (MemoryLimitException, ConvolutionBufferException, ShapeException, LayerException):
                limit_reached = True
        
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
        
        last_blob = netgen.implement_usknet(netconf, net, run_shape_out, netconf.fmap_start, netconf.fmap_output)

        # Implement the prediction layer
        if netconf.loss_function == 'malis':
            net.prob = L.Sigmoid(last_blob, ntop=1)
        
        if netconf.loss_function == 'euclid':
            net.prob = L.Sigmoid(last_blob, ntop=1)
            
        if netconf.loss_function == 'softmax':
            net.prob = L.Softmax(last_blob, ntop=1)

        for i in range(0,len(run_shape_out)):
            run_shape_out[i].print()
            
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
        
        
        # Start the actual network
        last_blob = netgen.implement_usknet(netconf, net, run_shape_out, netconf.fmap_start, netconf.fmap_output)
        
        for i in range(0,len(run_shape_out)):
            run_shape_out[i].print()
            
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


