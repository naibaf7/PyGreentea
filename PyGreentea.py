import os, sys, inspect, resource, gc
import h5py
import numpy as np
import random
import math
import multiprocessing
import threading
from Crypto.Random.random import randint


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

# Import Caffe twice (train and test)
import caffe as caf_train
del sys.modules['caffe']
import caffe as caf_test

# Import the network generator
# import network_generator as netgen

# Import Malis
import malis as malis


# Wrapper around a networks set_input_arrays to prevent memory leaks of locked up arrays
class NetInputWrapper:
    
    def __init__(self, net, shapes):
        self.net = net
        self.shapes = shapes
        self.dummy_slice = np.ascontiguousarray([0]).astype(float32);
        self.inputs = []
        for i in range(0,len(shapes)):
            # Pre-allocate arrays that will persist with the network
            self.inputs += [np.zeros(tuple(self.shapes[i]), dtype=float32)];
                
    def setInputs(self, inputs):
        for i in range(0,len(self.shapes)):
            np.copyto(self.inputs[i], np.ascontiguousarray(inputs[i]).astype(float32))
            self.net.set_input_arrays(i, self.inputs[i], self.dummy_slice)
                  

# Transfer network weights from one network to another
def net_weight_transfer(dst_net, src_net):
    # Go through all source layers/weights
    for layer_key in src_net.params:
        # Test existence of the weights in destination network
        if (layer_key in dst_net.params):
            # Copy weights + bias
            for i in range(0, min(len(dst_net.params[layer_key]), len(src_net.params[layer_key]))):
                np.copyto(dst_net.params[layer_key][i], src_net.params[layer_key][i])
        

def normalize(dataset, newmin=-1, newmax=1):
    maxval = dataset
    while len(maxval.shape) > 0:
        maxval = maxval.max(0)
    minval = dataset
    while len(minval.shape) > 0:
        minval = minval.min(0)
    return ((dataset - minval) / (maxval - minval)) * (newmax - newmin) + newmin


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
            if abs(data[i]) > 100000:
                failure = True
                if first == -1:
                    first = i
                print 'Failure, location %d; objective %d' % (i, data[i])
        print 'Failure: %s, first at %d' % (failure,first)
        if failure:
            break;


def process(net, data_arrays, output_folder, input_padding, output_dims):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    net_io = NetInputWrapper(net, [[1,1] + output_dims]);
    
    dst = net.blobs['prob']
    dummy_slice = [0]
    for i in range(0, len(data_arrays)):
        data_array = data_arrays[i]
        dims = len(data_array.shape)
        
        offsets = []        
        in_dims = []
        out_dims = []
        fmaps_out = 11
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
            
            # while(True):
            #    blob = raw_input('Blob:')
            #    fmap = int(raw_input('Enter the feature map:'))
            #    m = volume_slicer.VolumeSlicer(data=np.squeeze(net.blobs[blob].data[0])[fmap,:,:])
            #    m.configure_traits()
            
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

        # Safe the output
        outhdf5 = h5py.File(output_folder+'/'+str(i)+'.h5', 'w')
        outdset = outhdf5.create_dataset('main', tuple([fmaps_out]+out_dims), np.float32, data=pred_array)
        # outdset.attrs['edges'] = np.string_('-1,0,0;0,-1,0;0,0,-1')
        outhdf5.close()
      
        
    # Wrapper around a networks 
class TestNetEvaluator:
    
    def __init__(self, test_net, train_net):
        self.test_net = test_net
        self.train_net = train_net
        
    def run_test(self, iteration):
        return False
        # TODO: Implement evaluation methods
        # TODO: Store result

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
        self.thread = threading.Thread(target=self.run_test, args=(iteration))
        self.thread.start()
                
        
def train(solver, test_net, data_arrays, label_arrays, affinity_arrays, mode, input_padding, output_dims):
    dims = len(output_dims)
    losses = []
    
    net = solver.net
    
    test_eval = TestNetEvaluator(test_net, net)
    
    shapes = []
    # Raw data slice input         (n = 1, f = 1, spatial dims)
    shapes += [[1,1] + [output_dims[di] + input_padding[di] for di in range(0, dims)]]
    # Affinity data slice input    (n = 1, f = #edges, spatial dims)
    shapes += [[1,11] + output_dims[di]]
    # Connected components input   (n = 1, f = 1, spatial dims)
    shapes += [[1,1] + output_dims[di]]
    # Nhood specifications         (n = #edges, f = 3)
    shapes += [[11,3]]

    net_io = NetInputWrapper(net, shapes);
    
    if mode == 'malis' or mode == 'euclid':
        nhood = malis.mknhood3d()
    if mode == 'malis_aniso' or mode == 'euclid_aniso':
        nhood = malis.mknhood3d_aniso()
    
    # Loop from current iteration to last iteration
    for i in range(solver.iter, solver.max_iter):
        
        # TODO: Make this a parameter
        if (i % 20 == 0):
            test_eval.evaluate(i)
        
        # First pick the dataset to train with
        dataset = randint(0, len(data_arrays) - 1)
        data_array = data_arrays[dataset]
        label_array = label_arrays[dataset]
        affinity_array = affinity_arrays[dataset]

        offsets = []
        for j in range(0, dims):
            offsets.append(randint(0, data_array.shape[j] - (output_dims[j] + input_padding[j])))
                
        # These are the raw data elements
        data_slice = slice_data(data_array, offsets, [output_dims[di] + input_padding[di] for di in range(0, dims)])

        # These are the affinity edge values
        aff_slice = slice_data(affinity_array, [0] + [offsets[di] + int(math.ceil(input_padding[di] / float(2))) for di in range(0, dims)], [11] + output_dims)
        
        if mode == 'malis' or mode == 'malis_aniso':
            # These are the labels (connected components)
            label_slice = slice_data(label_array, [offsets[di] + int(math.ceil(input_padding[di] / float(2))) for di in range(0, dims)], output_dims)
            # Also recomputing the corresponding labels (connected components)
            aff_slice_tmp = malis.seg_to_affgraph(label_slice,nhood)
            label_slice,ccSizes = malis.connected_components_affgraph(aff_slice_tmp,nhood)
            net_io.setInputs([data_slice, label_slice, aff_slice, nhood])
            
        # We pass the raw and affinity array only
        if mode == 'euclid' or mode == 'euclid_aniso':
            frac_pos = np.clip(aff_slice.mean(),0.05,0.95)
            w_pos = np.sqrt(1.0/(2.0*frac_pos))
            w_neg = np.sqrt(1.0/(2.0*(1.0-frac_pos)))            
            net_io.setInputs([data_slice, aff_slice, error_scale(aff_slice,w_neg,w_pos)])

        if mode == 'softmax':
            net_io.setInputs([data_slice, aff_slice])
        
        # Single step
        loss = solver.step(1)
        
        while gc.collect():
            pass


        if mode == 'euclid' or mode == 'euclid_aniso':
            print("[Iter %i] Loss: %s, frac_pos=%f, w_pos=%f" % (i,loss,frac_pos,w_pos))
        else:
            print("[Iter %i] Loss: %s" % (i,loss))
        # TODO: Store losses to file
        losses += [loss]
        

