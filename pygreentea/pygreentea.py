from __future__ import print_function

import gc
import inspect
import math, copy
import os
import sys
import threading
import multiprocessing
import concurrent.futures
import time
import random
import pygreentea

# Determine where PyGreentea is
pygtpath = os.path.normpath(os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0])))
rootpath = os.path.dirname(pygtpath)

# Determine where PyGreentea gets called from
cmdpath = os.getcwd()

sys.path.append(pygtpath)
sys.path.append(pygtpath + '/..')
sys.path.append(cmdpath)

print(os.path.dirname(pygreentea.__file__))

import h5py
import numpy as np
import png
from scipy import io

# Load the configuration file
import config

from numpy import float32, int32, uint8

# Import Caffe
pycaffepath = ''
if (os.path.isabs(config.caffe_path)):
    pycaffepath = config.caffe_path + '/python'
else:
    pycaffepath = rootpath + '/' + config.caffe_path + '/python'

sys.path.append(pycaffepath)
   
import caffe as caffe

# Import the network generator
import netgen
from netgen import metalayers
from netgen import fix_input_dims

# Import Malis
if (os.path.isabs(config.malis_path)):
    sys.path.append(config.malis_path)
else:
    sys.path.append(pygtpath+'/'+config.malis_path)
import malis as malis

def minidx(data, index):
    return data[min(len(data) - 1, index)]

# Wrapper around a networks set_input_arrays to prevent memory leaks of locked up arrays
class NetInputWrapper:
    
    def __init__(self, net, input_specs={}, output_specs={}):
        self.net = net
        self.input_specs = input_specs
        self.output_specs = output_specs
        self.inputs = {}
        
        for set_key in self.input_specs.keys():
            shape = self.input_specs[set_key].shape
            # Pre-allocate arrays that will persist with the network
            self.inputs[set_key] = np.zeros(tuple(shape), dtype=float32)
        
    def set_inputs(self, data):      
        for set_key in self.input_specs.keys():
            np.copyto(self.inputs[set_key], np.ascontiguousarray(data[set_key]).astype(float32))
            self.net.set_layer_input_arrays(self.input_specs[set_key].memory_layer, self.inputs[set_key], None)
    
    def get_outputs(self):
        outputs = {}
        for set_key in self.output_specs.keys():
            outputs[set_key] = self.output_specs[set_key].blob.data
        return outputs

# Transfer network weights from one network to another
def net_weight_transfer(dst_net, src_net):
    # Go through all source layers/weights
    for layer_key in src_net.params:
        # Test existence of the weights in destination network
        if (layer_key in dst_net.params):
            # Copy weights + bias
            for i in range(0, min(len(dst_net.params[layer_key]), len(src_net.params[layer_key]))):
                np.copyto(dst_net.params[layer_key][i].data, src_net.params[layer_key][i].data)
        

def normalize(dataset, newmin=-1, newmax=1):
    maxval = dataset
    while len(maxval.shape) > 0:
        maxval = maxval.max(0)
    minval = dataset
    while len(minval.shape) > 0:
        minval = minval.min(0)
    return ((dataset - minval) / (maxval - minval)) * (newmax - newmin) + newmin

def get_solver_states(prefix):
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    print(files)
    solverstates = []
    for file in files:
        if(prefix+'_iter_' in file and '.solverstate' in file):
            solverstates += [(int(file[len(prefix+'_iter_'):-len('.solverstate')]),file)]
    return sorted(solverstates)
            
def get_caffe_models(prefix):
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    print(files)
    caffemodels = []
    for file in files:
        if(prefix+'_iter_' in file and '.caffemodel' in file):
            caffemodels += [(int(file[len(prefix+'_iter_'):-len('.caffemodel')]),file)]
    return sorted(caffemodels)

def scale_errors(data, factor_low, factor_high):
    scaled_data = np.add((data >= 0.5) * factor_high, (data < 0.5) * factor_low)
    return scaled_data

def count_affinity(dataset):
    aff_high = np.sum(dataset >= 0.5)
    aff_low = np.sum(dataset < 0.5)
    return aff_high, aff_low

def border_reflect(dataset, border):
    return np.pad(dataset,((border, border)),'reflect')

def slice_data(data, offsets, sizes):
    """
    data should be of shape [#feature maps (channels), spatial Z, spatial Y, spatial X]
    offsets and sizes should be of shape [spatial Z, spatial Y, spatial X]
    The number of spatial dimensions can vary
    """
    slicing = [slice(0, data.shape[0])] + [slice(offsets[i], offsets[i]+sizes[i]) for i in range(0, min(len(offsets),len(data.shape)-1))]
    return data[slicing]

def set_slice_data(data, insert_data, offsets, sizes):
    slicing = [slice(0, data.shape[0])] + [slice(offsets[i], offsets[i]+sizes[i]) for i in range(0, min(len(offsets),len(data.shape)-1))]
    data[slicing] = insert_data

def sanity_check_net_blobs(net):
    for key in net.blobs.keys():
        dst = net.blobs[key]
        data = np.ndarray.flatten(dst.data[0].copy())
        print('Blob: %s; %s' % (key, data.shape))
        failure = False
        first = -1
        for i in range(0,data.shape[0]):
            if abs(data[i]) > 1000:
                failure = True
                if first == -1:
                    first = i
                print('Failure, location %d; objective %d' % (i, data[i]))
        print('Failure: %s, first at %d, mean %3.5f' % (failure,first,np.mean(data)))
        if failure:
            break
        
def dump_feature_maps(net, folder):
    for key in net.blobs.keys():
        dst = net.blobs[key]
        norm = normalize(dst.data[0], 0, 255)
        # print(norm.shape)
        for f in range(0,norm.shape[0]):
            outfile = open(folder+'/'+key+'_'+str(f)+'.png', 'wb')
            writer = png.Writer(norm.shape[2], norm.shape[1], greyscale=True)
            # print(np.uint8(norm[f,:]).shape)
            writer.write(outfile, np.uint8(norm[f,:]))
            outfile.close()
            
            
def dump_tikzgraph_maps(net, folder):
    xmaps = 2
    ymaps = 2
    padding = 12
    
    for key in net.blobs.keys():
        dst = net.blobs[key]
        norm = normalize(dst.data[0], 0, 255)
        
        width = xmaps*norm.shape[2]+(xmaps-1)*padding
        height = ymaps*norm.shape[2]+(ymaps-1)*padding
        
        mapout = np.ones((width,height))*255
        
        # print(norm.shape)
        for f in range(0,xmaps * ymaps):
            xoff = (norm.shape[2] + padding) * (f % xmaps)
            yoff = (norm.shape[1] + padding) * (f / xmaps)
            
            mapout[xoff:xoff+norm.shape[2],yoff:yoff+norm.shape[1]] = norm[min(f,norm.shape[0]-1),:]
            
            outfile = open(folder+'/'+key+'.png', 'wb')
            writer = png.Writer(width, height, greyscale=True)
            # print(np.uint8(norm[f,:]).shape)
            writer.write(outfile, np.uint8(mapout))
            outfile.close()

class TestNetEvaluator(object):
    def __init__(self, test_net, train_net, data_arrays, options):
        self.options = options
        self.test_net = test_net
        self.train_net = train_net
        self.datasets = data_arrays
        self.thread = None
        input_dims, output_dims, input_padding = get_spatial_io_dims(self.test_net)
        fmaps_in, fmaps_out = get_fmap_io_dims(self.test_net)
        self.shapes = [[1, fmaps_in] + input_dims]
        self.fmaps_out = fmaps_out
        self.n_data_dims = len(output_dims)
        self.net_io = NetInputWrapper(self.test_net, self.shapes)

    def run_test(self, iteration):
        caffe.select_device(self.options.test_device, False)
        for dataset_i in range(len(self.datasets)):
            dataset_to_process = self.datasets[dataset_i]
            if 'name' in dataset_to_process:
                h5_file_name = dataset_to_process['name'] + '.h5'
            else:
                h5_file_name = 'test_out_' + repr(dataset_i) + '.h5'
            temp_file_name = h5_file_name + '.inprogress'
            with h5py.File(temp_file_name, 'w') as h5_file:
                prediction_shape = (self.fmaps_out,) + dataset_to_process['data'].shape[-self.n_data_dims:]
                target_array = h5_file.create_dataset(name='main', shape=prediction_shape, dtype=np.float32)
                output_arrays = process(self.test_net,
                                        data_arrays=[dataset_to_process],
                                        shapes=self.shapes,
                                        net_io=self.net_io,
                                        target_arrays=[target_array])
            os.rename(temp_file_name, h5_file_name)
            print("Just saved {}".format(h5_file_name))

    def evaluate(self, iteration):
        # Test/wait if last test is done
        if self.thread is not None:
            try:
                self.thread.join()
            except:
                self.thread = None
        net_weight_transfer(self.test_net, self.train_net)
        if config.use_one_thread:
            self.run_test(iteration)
        else:
            self.thread = threading.Thread(target=self.run_test, args=[iteration])
            self.thread.start()


def init_solver(solver_config, options):
    caffe.set_mode_gpu()
    caffe.select_device(options.train_device, False)
    solver_inst = caffe.get_solver(solver_config)
    if options.test_net is None:
        return solver_inst, None
    else:
        return solver_inst, init_testnet(options.test_net, test_device=options.test_device, level=options.test_level, stages=options.test_stages)


def init_testnet(test_net, trained_model=None, test_device=0, level=0, stages=None):
    caffe.set_mode_gpu()
    if isinstance(test_device, list):
        # Initialize test network for each device
        networks = []
        for device in test_device:
            caffe.select_device(device, False)
            if trained_model is None:
                networks += [caffe.Net(test_net, caffe.TEST, level=level, stages=stages)]
            else:
                networks += [caffe.Net(test_net, trained_model, caffe.TEST, level=level, stages=stages)]
        return networks
    else:
        # Initialize test network for a single device
        caffe.select_device(test_device, False)
        if trained_model is None:
            return caffe.Net(test_net, caffe.TEST, level=level, stages=stages)
        else:
            return caffe.Net(test_net, trained_model, caffe.TEST, level=level, stages=stages)
        
class InputSpec(object):
    def __init__(self, name, memory_layer, blob, shape, data_offset=[], scale=[1]):
        self.name = name
        self.memory_layer = memory_layer
        self.blob = blob
        self.shape = shape
        self.spatial_offsets = data_offset
        self.scale = scale
    def compute_spatial_offsets(self, max_shape, reset=False):
        if (reset):
            self.spatial_offsets = [] 
        self.spatial_offsets = []
        for i in range(2 + len(self.spatial_offsets), len(self.shape)):
            self.spatial_offsets.append((minidx(self.scale, i - 2) * max_shape[i] - self.shape[i]))
    def slice_data(self, batch_size, dataset_indexes, offsets, dataset_combined_sizes, data_arrays):
        data_slice = np.asarray([slice_data(data_arrays[dataset_indexes[i]][self.name], [((minidx(self.scale, j) * offsets[i][j] + self.spatial_offsets[j]/2) if (data_arrays[dataset_indexes[i]][self.name].shape[j] == minidx(self.scale, j) * dataset_combined_sizes[i][j]) else (minidx(self.scale, j) * offsets[i][j])) for j in range(0, min(len(offsets[i]),len(self.spatial_offsets)))], self.shape[2:]) for i in range(0, batch_size)])
        # print(data_slice.shape)
        return data_slice
    def scaled_shape(self):
        return [self.shape[0], self.shape[1]] + [self.shape[i + 2] / minidx(self.scale, i) for i in range(0, len(self.shape) - 2)]
    
class OutputSpec(object):
    def __init__(self, name, blob, shape, data_offset=[], scale=[1]):
        self.name = name
        self.blob = blob
        self.shape = shape
        self.spatial_offsets = data_offset
        self.scale = scale
    def compute_spatial_offsets(self, max_shape, reset=False):
        if (reset):
            self.spatial_offsets = []
        for i in range(2 + len(self.spatial_offsets), len(self.shape)):
            self.spatial_offsets.append((minidx(self.scale, i - 2) * max_shape[i] - self.shape[i]))
    def set_slice_data(self, dataset_index, offsets, data_arrays, data_slice):
        set_slice_data(data_arrays[dataset_index][self.name], data_slice, [minidx(self.scale, j) * offsets[j] for j in range(0, len(offsets))], self.shape[2:])
    def scaled_shape(self):
        return [self.shape[0], self.shape[1]] + [self.shape[i + 2] / minidx(self.scale, i) for i in range(0, len(self.shape) - 2)]
        
def get_net_input_specs(net, data_offsets={}, scales={}):
    input_specs = {}
    for layer in net.layers:
        if (layer.type == 'MemoryData'):
            for i in range(0, layer.layer_param.top_size):
                blob_name = layer.layer_param.get_top(i)
                data_offset = []
                scale = [1]
                if (blob_name in data_offsets.keys()):
                    data_offset = data_offsets[blob_name]
                if (blob_name in scales.keys()):
                    scale = scales[blob_name]
                blob = net.blobs[blob_name]
                input_spec = InputSpec(blob_name, layer, blob, np.shape(blob.data), data_offset, scale)
                input_specs[input_spec.name] = input_spec
    return input_specs

def get_net_output_specs(net, blob_names, data_offsets={}, scales={}):
    output_specs = {}
    for blob_name in blob_names:
        data_offset = []
        scale = [1]
        if (blob_name in data_offsets.keys()):
            data_offset = data_offsets[blob_name]
        if (blob_name in scales.keys()):
            scale = scales[blob_name]
        output_spec = OutputSpec(blob_name, net.blobs[blob_name], np.shape(net.blobs[blob_name].data), data_offset, scale)
        output_specs[output_spec.name] = output_spec
    return output_specs

class OffsetGenerator:
    def __init__(self, random, net_input_specs={}, net_output_specs={}):
        self.random = random
        self.dataset_index = 0
        self.offsets = []
        self.net_input_specs = net_input_specs
        self.net_output_specs = net_output_specs
    def make_dataset_offsets(self, batch_size, data_arrays, output_arrays=[], max_shape=[], min_shape=[]):
        dataset_indexes = []
        offsets = []
        dataset_combined_sizes = []
        if (self.dataset_index < len(data_arrays)):
            for i in range(0, batch_size):
                dataset_index = random.randint(0, len(data_arrays) - 1)
                data_array_keys = data_arrays[dataset_index].keys()
                dataset_combined_size = []
                for set_key in data_array_keys:
                    shape = [data_arrays[dataset_index][set_key].shape[j] / minidx(self.net_input_specs[set_key].scale, j) for j in range(0,len(data_arrays[dataset_index][set_key].shape))]
                    for j in range(0, len(shape)):
                        if len(dataset_combined_size) <= j:
                            dataset_combined_size.append(shape[j])
                        else:
                            dataset_combined_size[j] = max(dataset_combined_size[j], shape[j]) 
                dataset_combined_sizes.append(dataset_combined_size)
                if (self.random):
                    offset = [random.randint(0, dataset_combined_size[j - 1] - max_shape[j]) for j in range(2, len(max_shape))]
                    dataset_indexes.append(dataset_index)
                    offsets.append(offset)
                else:
                    while (len(self.offsets) < len(max(min_shape, max_shape))):
                        self.offsets.append(0)
                        
                    dataset_indexes.append(min(self.dataset_index, len(data_arrays) - 1))
                    
                    for set_key in self.net_output_specs.keys():
                        if (len(output_arrays) <= dataset_indexes[-1]):
                            output_arrays.append({})
                        if not (set_key in output_arrays[dataset_indexes[-1]].keys()):
                            shape = [self.net_output_specs[set_key].shape[1]] + [dataset_combined_sizes[-1][1 + j] - max_shape[2 + j] + self.net_output_specs[set_key].scaled_shape()[2 + j] for j in range(0, len(self.net_output_specs[set_key].shape) - 2)]
                            output_arrays[dataset_indexes[-1]][set_key] = np.zeros(tuple(shape), dtype=float32)
                    
                    offset = copy.deepcopy(self.offsets)
                    offsets.append(offset)
                    
                    increased = False
                    for j in range(0, len(min_shape) - 2):
                        q = len(min_shape) - 3 - j
                        while (len(self.offsets) <= q):
                            self.offsets.append(0)
                        if (self.offsets[q] + max_shape[2 + q] < dataset_combined_sizes[-1][1 + q]):
                            self.offsets[q] = self.offsets[q] + min_shape[2 + q]
                            if (self.offsets[q] + max_shape[2 + q] >= dataset_combined_sizes[-1][1 + q]):
                                self.offsets[q] = dataset_combined_sizes[-1][1 + q] - max_shape[2 + q]
                            increased = True
                        else:
                            increased = False
                            self.offsets[q] = 0
                            
                        if increased:
                            break
                    if not increased:
                        self.dataset_index = self.dataset_index + 1
                        
        return dataset_indexes, offsets, dataset_combined_sizes


def train(solver, options, train_data_arrays, data_slice_callback,
          test_net, test_data_arrays, test_data_slice_callback,
          data_offsets={}, scales={}, test_data_offsets={}, test_scales={}):
    caffe.select_device(options.train_device, False)

    net = solver.net

    test_eval = None
    if (options.test_net != None):
        test_eval = TestNetEvaluator(options, test_net, net, test_data_arrays, test_data_slice_callback)
    
    # Get the networks input specifications
    input_specs = get_net_input_specs(net, data_offsets=data_offsets, scales=scales)
    max_shape = []
    if (len(train_data_arrays) > 0):
        dataset_for_keys = train_data_arrays[0]
        for set_key in input_specs.keys():
            if (input_specs[set_key].name in dataset_for_keys.keys()):
                shape = input_specs[set_key].scaled_shape()
                for j in range(0, len(shape)):
                    if len(max_shape) <= j:
                        max_shape.append(shape[j])
                    else:
                        max_shape[j] = max(max_shape[j], shape[j]) 
            
        for set_key in input_specs.keys():
            if (input_specs[set_key].name in train_data_arrays[0].keys()):
                input_specs[set_key].compute_spatial_offsets(max_shape)

    batch_size = max_shape[0]
    net_io = NetInputWrapper(net, input_specs=input_specs)
    
    offset_generator = OffsetGenerator(True, net_input_specs=net_io.input_specs)

    # Loop from current iteration to last iteration
    for i in range(solver.iter, solver.max_iter):
        start = time.time()
        if (options.test_net != None and i % options.test_interval == 1):
            test_eval.evaluate(i)
            if config.use_one_thread:
                # after testing finishes, switch back to the training device
                caffe.select_device(options.train_device, False)
                
        
        dataset_indexes, offsets, dataset_combined_sizes = offset_generator.make_dataset_offsets(batch_size, train_data_arrays, max_shape=max_shape)
        
        slices = {}
        
        if (len(train_data_arrays) > 0):
            dataset_for_keys = train_data_arrays[0]           
            for set_key in dataset_for_keys.keys():
                data_slice = input_specs[set_key].slice_data(batch_size, dataset_indexes, offsets, dataset_combined_sizes, train_data_arrays)
                slices[set_key] = data_slice

        data_slice_callback(input_specs, batch_size, dataset_indexes, offsets, dataset_combined_sizes, train_data_arrays, slices)
        

        net_io.set_inputs(slices)
        
        loss = solver.step(1)  # Single step
        while gc.collect():
            pass
        time_of_iteration = time.time() - start
        
        
def process_input_data(net_io, slices):
    net_io.set_inputs(slices)
    net_io.net.forward()
        
def process_core_multithreaded(device_locks, net_io, data_slices, dataset_indexes, offsets, output_arrays):
    # Each thread sets its GPU
    current_device_id = -1
    while (current_device_id == -1):
        for device_list_id in range(0,len(device_locks)):
            if (device_locks[device_list_id].acquire(False)):
                current_device_id = device_list_id
                break
        if current_device_id == -1:
            time.sleep(0.0005)
    if config.debug:
        print("Using device (list ID): ", current_device_id)
    # Note that this is the list ID, not the absolute device ID
    caffe.select_device(current_device_id, True)
    process_core(net_io[current_device_id], data_slices, dataset_indexes, offsets, output_arrays)
    device_locks[device_list_id].release()
    
def process_core(net_io, data_slices, dataset_indexes, offsets, output_arrays):
    process_local_net_io = None
    if isinstance(net_io, list):
        process_local_net_io = net_io[multiprocessing.Process.name]
    else:
        process_local_net_io = net_io
        
    process_input_data(process_local_net_io, data_slices)
    outputs = process_local_net_io.get_outputs()

    for i in range(0, len(dataset_indexes)):
        index = dataset_indexes[i]
        for set_key in outputs.keys():
            net_io.output_specs[set_key].set_slice_data(index, offsets[i], output_arrays, outputs[set_key][i])
    
def process(test_nets, input_arrays, output_blob_names, output_arrays, data_slice_callback, data_offsets={}, scales={}):
    thread_pool = None
    device_locks = None
    nets = []
    net_ios = []
    batch_size = 0
       
    if isinstance(test_nets, list):
        nets.extend(test_nets)
    else:
        nets.append(test_nets)
        
    for net in nets:
        # Get the networks input specifications
        input_specs = get_net_input_specs(net, data_offsets, scales)
        output_specs = get_net_output_specs(net, output_blob_names, data_offsets, scales)
        # Get the rescaled max and min shapes. The min shape will be the processing stride
        max_shape = []
        min_shape = []
        if (len(input_arrays) > 0):
            dataset_for_keys = input_arrays[0]
            for set_key in input_specs.keys():
                if (input_specs[set_key].name in dataset_for_keys.keys()):
                    shape = input_specs[set_key].scaled_shape()
                    for j in range(0, len(shape)):
                        if len(max_shape) <= j:
                            max_shape.append(shape[j])
                        else:
                            max_shape[j] = max(max_shape[j], shape[j]) 
                
            for set_key in input_specs.keys():
                if (input_specs[set_key].name in input_arrays[0].keys()):
                    input_specs[set_key].compute_spatial_offsets(max_shape)
                    
            for set_key in output_specs.keys():
                output_specs[set_key].compute_spatial_offsets(max_shape)
        
        for set_key in output_specs.keys():
            shape = output_specs[set_key].scaled_shape()
            for j in range(0, len(shape)):
                if len(min_shape) <= j:
                    min_shape.append(shape[j])
                else:
                    min_shape[j] = min(min_shape[j], shape[j]) 
                    
        batch_size = max_shape[0]
        net_io = NetInputWrapper(net, input_specs=input_specs, output_specs=output_specs)
        net_ios.append(net_io)
                    
    # Launch 
    if len(nets) > 1:
        thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(nets))
        device_locks = []
        for device_list_id in range(0,len(nets)):
            device_locks += [threading.Lock()]
    
    offset_generator = OffsetGenerator(False, net_input_specs=net_io.input_specs, net_output_specs=net_io.output_specs)

    while True:
        dataset_indexes, offsets, dataset_combined_sizes = offset_generator.make_dataset_offsets(batch_size, input_arrays, output_arrays=output_arrays, max_shape=max_shape, min_shape=min_shape)
        
        # No more offsets to process, terminate:
        if (len(dataset_indexes) == 0):
            break
        
        data_slices = {}
        
        if (len(input_arrays) > 0):
            dataset_for_keys = input_arrays[0]           
            for set_key in dataset_for_keys.keys():
                data_slice = input_specs[set_key].slice_data(batch_size, dataset_indexes, offsets, dataset_combined_sizes, input_arrays)
                data_slices[set_key] = data_slice
        
        data_slice_callback(input_specs, batch_size, dataset_indexes, offsets, dataset_combined_sizes, input_arrays, data_slices)
        
        if len(nets) > 1:
            thread_pool.submit(process_core_multithreaded, device_locks, net_io, data_slices, dataset_indexes, offsets, output_arrays)
        else:
            process_core(net_io, data_slices, dataset_indexes, offsets, output_arrays)



    if not (thread_pool is None):
        thread_pool.shutdown(True)
