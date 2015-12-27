import os, sys, inspect, resource, gc
import h5py
import numpy as np
import random
import math
import multiprocessing
import threading
from Crypto.Random.random import randint
from gtk import input_add


# Determine where PyGreentea is
pygtpath = os.path.normpath(os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0])))

# Determine where PyGreentea gets called from
cmdpath = os.getcwd()

sys.path.append(pygtpath)
sys.path.append(cmdpath)


from numpy import float32, int32, uint8

# Load the configuration file
import config

# Load the setup module
import setup

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Direct call to PyGreentea, set up everything
if __name__ == "__main__":
    if (pygtpath != cmdpath):
        os.chdir(pygtpath)
    
    if (os.geteuid() != 0):
        print(bcolors.WARNING + "PyGreentea setup should probably be executed with root privileges!" + bcolors.ENDC)
    
    if config.install_packages:
        print(bcolors.HEADER + ("==== PYGT: Installing OS packages ====").ljust(80,"=") + bcolors.ENDC)
        setup.install_dependencies()
    
    print(bcolors.HEADER + ("==== PYGT: Updating Caffe/Greentea repository ====").ljust(80,"=") + bcolors.ENDC)
    setup.clone_caffe(config.caffe_path, config.clone_caffe, config.update_caffe)
    
    print(bcolors.HEADER + ("==== PYGT: Updating Malis repository ====").ljust(80,"=") + bcolors.ENDC)
    setup.clone_malis(config.malis_path, config.clone_malis, config.update_malis)
    
    if config.compile_caffe:
        print(bcolors.HEADER + ("==== PYGT: Compiling Caffe/Greentea ====").ljust(80,"=") + bcolors.ENDC)
        setup.compile_caffe(config.caffe_path)
    
    if config.compile_malis:
        print(bcolors.HEADER + ("==== PYGT: Compiling Malis ====").ljust(80,"=") + bcolors.ENDC)
        setup.compile_malis(config.malis_path)
        
    if (pygtpath != cmdpath):
        os.chdir(cmdpath)
    
    print(bcolors.OKGREEN + ("==== PYGT: Setup finished ====").ljust(80,"=") + bcolors.ENDC)
    sys.exit(0)


setup.setup_paths(config.caffe_path, config.malis_path)
setup.set_environment_vars()

# Import Caffe
import caffe as caffe

# Import the network generator
import network_generator as netgen

# Import Malis
import malis as malis


# Wrapper around a networks set_input_arrays to prevent memory leaks of locked up arrays
class NetInputWrapper:
    
    def __init__(self, net, shapes):
        self.net = net
        self.shapes = shapes
        self.dummy_slice = np.ascontiguousarray([0]).astype(float32)
        self.inputs = []
        for i in range(0,len(shapes)):
            # Pre-allocate arrays that will persist with the network
            self.inputs += [np.zeros(tuple(self.shapes[i]), dtype=float32)]
                
    def setInputs(self, data):      
        for i in range(0,len(self.shapes)):
            np.copyto(self.inputs[i], np.ascontiguousarray(data[i]).astype(float32))
            self.net.set_input_arrays(i, self.inputs[i], self.dummy_slice)
                  

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


def getSolverStates(prefix):
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    print files
    solverstates = []
    for file in files:
        if(prefix+'_iter_' in file and '.solverstate' in file):
            solverstates += [(int(file[len(prefix+'_iter_'):-len('.solverstate')]),file)]
    return sorted(solverstates)
            
def getCaffeModels(prefix):
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    print files
    caffemodels = []
    for file in files:
        if(prefix+'_iter_' in file and '.caffemodel' in file):
            caffemodels += [(int(file[len(prefix+'_iter_'):-len('.caffemodel')]),file)]
    return sorted(caffemodels)
            

def error_scale(data, factor_low, factor_high):
    scale = np.add((data >= 0.5) * factor_high, (data < 0.5) * factor_low)
    return scale


def count_affinity(dataset):
    aff_high = np.sum(dataset >= 0.5)
    aff_low = np.sum(dataset < 0.5)
    return aff_high, aff_low


def border_reflect(dataset, border):
    return np.pad(dataset,((border, border)),'reflect')
    
    
def slice_data(data, offsets, sizes):
    if (len(offsets) == 1):
        return data[offsets[0]:offsets[0] + sizes[0]]
    if (len(offsets) == 2):
        return data[offsets[0]:offsets[0] + sizes[0], offsets[1]:offsets[1] + sizes[1]]
    if (len(offsets) == 3):
        return data[offsets[0]:offsets[0] + sizes[0], offsets[1]:offsets[1] + sizes[1], offsets[2]:offsets[2] + sizes[2]]
    if (len(offsets) == 4):
        return data[offsets[0]:offsets[0] + sizes[0], offsets[1]:offsets[1] + sizes[1], offsets[2]:offsets[2] + sizes[2], offsets[3]:offsets[3] + sizes[3]]


def set_slice_data(data, insert_data, offsets, sizes):
    if (len(offsets) == 1):
        data[offsets[0]:offsets[0] + sizes[0]] = insert_data
    if (len(offsets) == 2):
        data[offsets[0]:offsets[0] + sizes[0], offsets[1]:offsets[1] + sizes[1]] = insert_data
    if (len(offsets) == 3):
        data[offsets[0]:offsets[0] + sizes[0], offsets[1]:offsets[1] + sizes[1], offsets[2]:offsets[2] + sizes[2]] = insert_data
    if (len(offsets) == 4):
        data[offsets[0]:offsets[0] + sizes[0], offsets[1]:offsets[1] + sizes[1], offsets[2]:offsets[2] + sizes[2], offsets[3]:offsets[3] + sizes[3]] = insert_data


def sanity_check_net_blobs(net):
    for key in net.blobs.keys():
        dst = net.blobs[key]
        data = np.ndarray.flatten(dst.data[0].copy())
        print 'Blob: %s; %s' % (key, data.shape)
        failure = False
        first = -1
        for i in range(0,data.shape[0]):
            if abs(data[i]) > 1000:
                failure = True
                if first == -1:
                    first = i
                print 'Failure, location %d; objective %d' % (i, data[i])
        print 'Failure: %s, first at %d, mean %3.5f' % (failure,first,np.mean(data))
        if failure:
            break
        
def get_net_input_specs(net, test_blobs = ['data', 'label', 'scale', 'label_affinity', 'affinty_edges']):
    
    shapes = []
    
    # The order of the inputs is strict in our network types
    for blob in test_blobs:
        if (blob in net.blobs):
            shapes += [[blob, np.shape(net.blobs[blob].data)]]
        
    return shapes

def get_spatial_io_dims(net):
    out_primary = 'label'
    
    if ('prob' in net.blobs):
        out_primary = 'prob'
    
    shapes = get_net_input_specs(net, test_blobs=['data', out_primary])
        
    dims = len(shapes[0][1]) - 2
    
    input_dims = list(shapes[0][1])[2:2+dims]
    output_dims = list(shapes[1][1])[2:2+dims]
    padding = [input_dims[i]-output_dims[i] for i in range(0,dims)]
    
    return input_dims, output_dims, padding

def get_fmap_io_dims(net):
    out_primary = 'label'
    
    if ('prob' in net.blobs):
        out_primary = 'prob'
    
    shapes = get_net_input_specs(net, test_blobs=['data', out_primary])
    
    input_fmaps = list(shapes[0][1])[1]
    output_fmaps = list(shapes[1][1])[1]
    
    return input_fmaps, output_fmaps
    
    
def get_net_output_specs(net):
    return np.shape(net.blobs['prob'].data)


def process(net, data_arrays, shapes=None, net_io=None):    
    input_dims, output_dims, input_padding = get_spatial_io_dims(net)
    fmaps_in, fmaps_out = get_fmap_io_dims(net)

    dims = len(output_dims)
    
    if (shapes == None):
        shapes = []
        # Raw data slice input         (n = 1, f = 1, spatial dims)
        shapes += [[1,fmaps_in] + input_dims]
        
    if (net_io == None):
        net_io = NetInputWrapper(net, shapes)
            
    dst = net.blobs['prob']
    dummy_slice = [0]
    
    pred_arrays = []
    
    for i in range(0, len(data_arrays)):
        data_array = data_arrays[i]['data']
        dims = len(data_array.shape)
        
        offsets = []        
        in_dims = []
        out_dims = []
        for d in range(0, dims):
            offsets += [0]
            in_dims += [data_array.shape[d]]
            out_dims += [data_array.shape[d] - input_padding[d]]
            
        pred_array = np.zeros(tuple([fmaps_out] + out_dims))
                
        while(True):
            data_slice = slice_data(data_array, offsets, [output_dims[di] + input_padding[di] for di in range(0, dims)])
            net_io.setInputs([data_slice])
            net.forward()
            output = dst.data[0].copy()
            
            print offsets
            print output.mean()
            
            set_slice_data(pred_array, output, [0] + offsets, [fmaps_out] + output_dims)
            
            incremented = False
            for d in range(0, dims):
                if (offsets[dims - 1 - d] == out_dims[dims - 1 - d] - output_dims[dims - 1 - d]):
                    # Reset direction
                    offsets[dims - 1 - d] = 0
                else:
                    # Increment direction
                    offsets[dims - 1 - d] = min(offsets[dims - 1 - d] + output_dims[dims - 1 - d], out_dims[dims - 1 - d] - output_dims[dims - 1 - d])
                    incremented = True
                    break
            
            # Processed the whole input block
            if not incremented:
                break
            
        pred_arrays += [pred_array]
            
    return pred_arrays
      
        
    # Wrapper around a networks 
class TestNetEvaluator:
    
    def __init__(self, test_net, train_net, data_arrays, options):
        self.options = options
        self.test_net = test_net
        self.train_net = train_net
        self.data_arrays = data_arrays
        self.thread = None
        
        input_dims, output_dims, input_padding = get_spatial_io_dims(self.test_net)
        fmaps_in, fmaps_out = get_fmap_io_dims(self.test_net)       
        self.shapes = []
        self.shapes += [[1,fmaps_in] + input_dims]
        self.net_io = NetInputWrapper(self.test_net, self.shapes)
            
    def run_test(self, iteration):
        caffe.select_device(self.options.test_device, False)
        pred_arrays = process(self.test_net, self.data_arrays, shapes=self.shapes, net_io=self.net_io)
        

    def evaluate(self, iteration):
        # Test/wait if last test is done
        if not(self.thread is None):
            try:
                self.thread.join()
            except:
                self.thread = None
        # Weight transfer
        net_weight_transfer(self.test_net, self.train_net)
        # Run test
        self.thread = threading.Thread(target=self.run_test, args=[iteration])
        self.thread.start()
                

def init_solver(solver_config, options):
    caffe.set_mode_gpu()
    caffe.select_device(options.train_device, False)
   
    solver_inst = caffe.get_solver(solver_config)
    
    if (options.test_net == None):
        return (solver_inst, None)
    else:
        return (solver_inst, init_testnet(options.test_net, test_device=options.test_device))
    
def init_testnet(test_net, trained_model=None, test_device=0):
    caffe.set_mode_gpu()
    caffe.select_device(test_device, False)
    if(trained_model == None):
        return caffe.Net(test_net, caffe.TEST)
    else:
        return caffe.Net(test_net, trained_model, caffe.TEST)

    
def train(solver, test_net, data_arrays, train_data_arrays, options):
    caffe.select_device(options.train_device, False)
    
    net = solver.net
    
    test_eval = None
    if (options.test_net != None):
        test_eval = TestNetEvaluator(test_net, net, train_data_arrays, options)
    
    input_dims, output_dims, input_padding = get_spatial_io_dims(net)
    fmaps_in, fmaps_out = get_fmap_io_dims(net)

    dims = len(output_dims)
    losses = []
    
    shapes = []
    # Raw data slice input         (n = 1, f = 1, spatial dims)
    shapes += [[1,fmaps_in] + input_dims]
    # Label data slice input    (n = 1, f = #edges, spatial dims)
    shapes += [[1,fmaps_out] + output_dims]
    
    if (options.loss_function == 'malis'):
        # Connected components input   (n = 1, f = 1, spatial dims)
        shapes += [[1,1] + output_dims]
    if (options.loss_function == 'euclid'):
        # Error scale input   (n = 1, f = #edges, spatial dims)
        shapes += [[1,fmaps_out] + output_dims]
    # Nhood specifications         (n = #edges, f = 3)
    if (('nhood' in data_arrays[0]) and (options.loss_function == 'malis')):
        shapes += [[1,1] + list(np.shape(data_arrays[0]['nhood']))]

    net_io = NetInputWrapper(net, shapes)
    
    # Loop from current iteration to last iteration
    for i in range(solver.iter, solver.max_iter):
        
        if (options.test_net != None and i % options.test_interval == 0):
            test_eval.evaluate(i)
        
        # First pick the dataset to train with
        dataset = randint(0, len(data_arrays) - 1)

        offsets = []
        for j in range(0, dims):
            offsets.append(randint(0, data_arrays[dataset]['data'].shape[1+j] - (output_dims[j] + input_padding[j])))
                
        # These are the raw data elements
        data_slice = slice_data(data_arrays[dataset]['data'], [0]+offsets, [fmaps_in]+[output_dims[di] + input_padding[di] for di in range(0, dims)])

        label_slice = None
        components_slice = None

        if (options.training_method == 'affinity'):
            if ('label' in data_arrays[dataset]):
                label_slice = slice_data(data_arrays[dataset]['label'], [0] + [offsets[di] + int(math.ceil(input_padding[di] / float(2))) for di in range(0, dims)], [fmaps_out] + output_dims)
                
            if ('components' in data_arrays[dataset]):
                components_slice = slice_data(data_arrays[dataset]['components'][0,:], [offsets[di] + int(math.ceil(input_padding[di] / float(2))) for di in range(0, dims)], output_dims)
                if (label_slice is None or options.recompute_affinity):
                    label_slice = malis.seg_to_affgraph(components_slice, data_arrays[dataset]['nhood']).astype(float32)
            
            if (components_slice is None or options.recompute_affinity):
                components_slice,ccSizes = malis.connected_components_affgraph(label_slice.astype(int32), data_arrays[dataset]['nhood'])

        else:
            label_slice = slice_data(data_arrays[dataset]['label'], [0] + [offsets[di] + int(math.ceil(input_padding[di] / float(2))) for di in range(0, dims)], [fmaps_out] + output_dims)


        if options.loss_function == 'malis':
            # Also recomputing the corresponding labels (connected components)
            net_io.setInputs([data_slice, label_slice, components_slice, data_arrays[0]['nhood']])
            
        if options.loss_function == 'euclid':
            if(options.scale_error == True):
                frac_pos = np.clip(label_slice.mean(),0.05,0.95)
                w_pos = 1.0/(2.0*frac_pos)
                w_neg = 1.0/(2.0*(1.0-frac_pos))
            else:
                w_pos = 1
                w_neg = 1
                      
            net_io.setInputs([data_slice, label_slice, error_scale(label_slice,w_neg,w_pos)])

        if options.loss_function == 'softmax':
            # These are the affinity edge values
            net_io.setInputs([data_slice, label_slice])
        
        # Single step
        loss = solver.step(1)
        # sanity_check_net_blobs(net)
        
        while gc.collect():
            pass


        if options.loss_function == 'euclid' or options.loss_function == 'euclid_aniso':
            print("[Iter %i] Loss: %s, frac_pos=%f, w_pos=%f" % (i,loss,frac_pos,w_pos))
        else:
            print("[Iter %i] Loss: %s" % (i,loss))
        # TODO: Store losses to file
        losses += [loss]
        

