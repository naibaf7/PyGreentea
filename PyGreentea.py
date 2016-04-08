import os, sys, inspect, gc
import h5py
import numpy as np
from scipy import io
import math
import threading
import png
from Crypto.Random.random import randint
import numpy.random
import time


# set this to True after importing this module to prevent multithreading
USE_ONE_THREAD = False

DEBUG = False

# Determine where PyGreentea is
pygtpath = os.path.normpath(os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0])))

# Determine where PyGreentea gets called from
cmdpath = os.getcwd()

sys.path.append(pygtpath)
sys.path.append(cmdpath)


from numpy import float32, int32, uint8

# Load the configuration file
import config

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
    # Load the setup module
    import setup

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
else: 
    import data_queue


# Import Caffe
caffe_parent_path = os.path.dirname(os.path.dirname(__file__))
caffe_path = os.path.join(caffe_parent_path, 'caffe_gt', 'python')
sys.path.append(caffe_path)
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
        self.input_keys = ['data', 'label', 'scale', 'components', 'nhood']
        self.input_layers = []
        
        for i in range(0, len(self.input_keys)):
            if (self.input_keys[i] in self.net.layers_dict):
                self.input_layers += [self.net.layers_dict[self.input_keys[i]]]
        
        print (len(self.input_layers))
        
        for i in range(0,len(shapes)):
            # Pre-allocate arrays that will persist with the network
            self.inputs += [np.zeros(tuple(self.shapes[i]), dtype=float32)]
                
        print (len(shapes))
        
    def setInputs(self, data):      
        for i in range(0,len(self.shapes)):
            np.copyto(self.inputs[i], np.ascontiguousarray(data[i]).astype(float32))
            self.net.set_layer_input_arrays(self.input_layers[i], self.inputs[i], self.dummy_slice)
                  

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


def scale_errors(data, factor_low, factor_high):
    scaled_data = np.add((data >= 0.5) * factor_high, (data < 0.5) * factor_low)
    return scaled_data


def count_affinity(dataset):
    aff_high = np.sum(dataset >= 0.5)
    aff_low = np.sum(dataset < 0.5)
    return aff_high, aff_low


def border_reflect(dataset, border):
    return np.pad(dataset,((border, border)),'reflect')


def augment_data_simple(dataset):
    nset = len(dataset)
    for iset in range(nset):
        for reflectz in range(2):
            for reflecty in range(2):
                for reflectx in range(2):
                    for swapxy in range(2):

                        if reflectz==0 and reflecty==0 and reflectx==0 and swapxy==0:
                            continue

                        dataset.append({})
                        dataset[-1]['name'] = dataset[iset]['name']+'_x'+str(reflectx)+'_y'+str(reflecty)+'_z'+str(reflectz)+'_xy'+str(swapxy)



                        dataset[-1]['nhood'] = dataset[iset]['nhood']
                        dataset[-1]['data'] = dataset[iset]['data'][:]
                        dataset[-1]['components'] = dataset[iset]['components'][:]

                        if reflectz:
                            dataset[-1]['data']         = dataset[-1]['data'][::-1,:,:]
                            dataset[-1]['components']   = dataset[-1]['components'][::-1,:,:]

                        if reflecty:
                            dataset[-1]['data']         = dataset[-1]['data'][:,::-1,:]
                            dataset[-1]['components']   = dataset[-1]['components'][:,::-1,:]

                        if reflectx:
                            dataset[-1]['data']         = dataset[-1]['data'][:,:,::-1]
                            dataset[-1]['components']   = dataset[-1]['components'][:,:,::-1]

                        if swapxy:
                            dataset[-1]['data']         = dataset[-1]['data'].transpose((0,2,1))
                            dataset[-1]['components']   = dataset[-1]['components'].transpose((0,2,1))

                        dataset[-1]['label'] = malis.seg_to_affgraph(dataset[-1]['components'],dataset[-1]['nhood'])

                        dataset[-1]['reflectz']=reflectz
                        dataset[-1]['reflecty']=reflecty
                        dataset[-1]['reflectx']=reflectx
                        dataset[-1]['swapxy']=swapxy
    return dataset

    
def augment_data_elastic(dataset,ncopy_per_dset):
    dsetout = []
    nset = len(dataset)
    for iset in range(nset):
        for icopy in range(ncopy_per_dset):
            reflectz = np.random.rand()>.5
            reflecty = np.random.rand()>.5
            reflectx = np.random.rand()>.5
            swapxy = np.random.rand()>.5

            dataset.append({})
            dataset[-1]['reflectz']=reflectz
            dataset[-1]['reflecty']=reflecty
            dataset[-1]['reflectx']=reflectx
            dataset[-1]['swapxy']=swapxy

            dataset[-1]['name'] = dataset[iset]['name']
            dataset[-1]['nhood'] = dataset[iset]['nhood']
            dataset[-1]['data'] = dataset[iset]['data'][:]
            dataset[-1]['components'] = dataset[iset]['components'][:]

            if reflectz:
                dataset[-1]['data']         = dataset[-1]['data'][::-1,:,:]
                dataset[-1]['components']   = dataset[-1]['components'][::-1,:,:]

            if reflecty:
                dataset[-1]['data']         = dataset[-1]['data'][:,::-1,:]
                dataset[-1]['components']   = dataset[-1]['components'][:,::-1,:]

            if reflectx:
                dataset[-1]['data']         = dataset[-1]['data'][:,:,::-1]
                dataset[-1]['components']   = dataset[-1]['components'][:,:,::-1]

            if swapxy:
                dataset[-1]['data']         = dataset[-1]['data'].transpose((0,2,1))
                dataset[-1]['components']   = dataset[-1]['components'].transpose((0,2,1))

            # elastic deformations

            dataset[-1]['label'] = malis.seg_to_affgraph(dataset[-1]['components'],dataset[-1]['nhood'])

    return dataset

    
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
    print(dims)
    
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

    using_queue = data_queue.data_queue_should_be_used_with(data_arrays)
    dataset_offsets_to_process = dict()
    if using_queue:
        processing_data_queue = data_queue.DatasetQueue(
            size=5,
            datasets=data_arrays,
            input_shape=tuple(input_dims),
            output_shape=None,  # ignore labels
            n_workers=3
        )

    pred_arrays = []
    for i in range(0, len(data_arrays)):
        data_array = data_arrays[i]['data']
        data_dims = len(data_array.shape)
        
        offsets = []
        in_dims = []
        out_dims = []
        for d in range(0, dims):
            offsets += [0]
            in_dims += [data_array.shape[data_dims-dims+d]]
            out_dims += [data_array.shape[data_dims-dims+d] - input_padding[d]]

        if not using_queue:
            pred_array = np.zeros(tuple([fmaps_out] + out_dims))

        list_of_offsets_to_process = []

        while(True):
            # print("In while loop. offsets = {o}".format(o=offsets))
            if using_queue:
                # print("Appending offsets value {o}".format(o=offsets))
                offsets_to_append = list(offsets)  # make a copy. important!
                list_of_offsets_to_process.append(offsets_to_append)
                # print("Just appended. list_of_offsets_to_process is now ",list_of_offsets_to_process)
            else:
                # process the old-fashioned way
                if DEBUG:
                    print("Processing offsets ", offsets)
                data_slice = slice_data(data_array, [0] + offsets, [fmaps_in] + [output_dims[di] + input_padding[di] for di in range(0, dims)])
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

        if using_queue:
            dataset_offsets_to_process[i] = list_of_offsets_to_process
        else:
            pred_arrays += [pred_array]

    if using_queue:
        for source_dataset_index in dataset_offsets_to_process:
            list_of_offsets_to_process = dataset_offsets_to_process[source_dataset_index]
            if DEBUG:
                print("source_dataset_index = ",source_dataset_index)
                print("Processing source volume #{i} with offsets list {o}"
                      .format(i=source_dataset_index, o=list_of_offsets_to_process))
            # make a copy of that list for enqueueing purposes
            offsets_to_enqueue = list(list_of_offsets_to_process)
            data_array = data_arrays[source_dataset_index]['data']
            data_dims = len(data_array.shape)
            in_dims = []
            out_dims = []
            for d in range(0, dims):
                in_dims += [data_array.shape[data_dims-dims+d]]
                out_dims += [data_array.shape[data_dims-dims+d] - input_padding[d]]
            pred_array = np.zeros(tuple([fmaps_out] + out_dims))
            # start pre-populating queue
            for shared_dataset_index in range(min(processing_data_queue.size, len(list_of_offsets_to_process))):
                # fill shared-memory datasets with an offset
                offsets = offsets_to_enqueue.pop(0)
                offsets = tuple([int(o) for o in offsets])
                # print("Pre-populating processing queue with data at offset {}".format(offsets))
                print("Pre-populating data loader's dataset #{i}/{size}"
                      .format(i=shared_dataset_index, size=processing_data_queue.size))
                shared_dataset_index, async_result = processing_data_queue.start_refreshing_shared_dataset(
                    shared_dataset_index,
                    offsets,
                    source_dataset_index,
                    transform=False,
                    wait=True
                )
            # process each offset
            for i_offsets in range(len(list_of_offsets_to_process)):
                dataset, index_of_shared_dataset = processing_data_queue.get_dataset()
                data_slice = dataset['data']
                assert data_slice.shape == (fmaps_in,) + tuple(input_dims)
                if DEBUG:
                    print("Processing next dataset in processing queue, which has offset {o}". format(o=dataset['offset']))
                # process the chunk
                net_io.setInputs([data_slice])
                net.forward()
                output = dst.data[0].copy()
                offsets_of_this_batch = list(dataset['offset'])  # convert tuple to list
                print offsets_of_this_batch
                print output.mean()
                set_slice_data(pred_array, output, [0] + offsets_of_this_batch, [fmaps_out] + output_dims)
                if len(offsets_to_enqueue) > 0:
                    # start adding the next slice to the queue with index_of_shared_dataset
                    new_offsets = offsets_to_enqueue.pop(0)
                    processing_data_queue.start_refreshing_shared_dataset(
                        index_of_shared_dataset,
                        new_offsets,
                        source_dataset_index,
                        transform=False
                    )
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

        for i in range(0, len(pred_arrays)):
            if ('name' in self.data_arrays[i]):
                h5file = self.data_arrays[i]['name'] + '.h5'
            else:
                h5file = 'test_out_' + repr(i) + '.h5'
            outhdf5 = h5py.File(h5file, 'w')
            outdset = outhdf5.create_dataset('main', pred_arrays[i].shape, np.float32, data=pred_arrays[i])
            # outdset.attrs['nhood'] = np.string_('-1,0,0;0,-1,0;0,0,-1')
            outhdf5.close()
            print("Just saved {}".format(h5file))
        

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
        if USE_ONE_THREAD:
            self.run_test(iteration)
        else:
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

    if data_queue.data_queue_should_be_used_with(data_arrays):
        using_asynchronous_queue = True
        # and initialize queue!
        if DEBUG:
            queue_size = 3
            n_workers = 2
        else:
            queue_size = 20
            n_workers = 10
        queue_initialization_kwargs = dict(
            size=queue_size,
            datasets=data_arrays,
            input_shape=tuple(input_dims),
            output_shape=tuple(output_dims),
            n_workers=n_workers
        )
        print("creating queue with kwargs {}".format(queue_initialization_kwargs))
        training_data_queue = data_queue.DatasetQueue(**queue_initialization_kwargs)
        # start populating the queue
        for i in range(queue_size):
            which_dataset = randint(0, len(data_arrays) - 1)
            offsets = []
            for j in range(0, dims):
                offsets.append(randint(0, data_arrays[which_dataset]['data'].shape[j] - (output_dims[j] + input_padding[j])))
            offsets = tuple([int(x) for x in offsets])
            # print("offsets = ", offsets)
            print("Pre-populating data loader's dataset #{i}/{size}"
                  .format(i=i, size=training_data_queue.size))
            shared_dataset_index, async_result = \
                training_data_queue.start_refreshing_shared_dataset(i, offsets, which_dataset, wait=True)
    else:
        using_asynchronous_queue = False

    # Loop from current iteration to last iteration
    time_counter = 0
    total_time = 0
    for i in range(solver.iter, solver.max_iter):
        
        if (options.test_net != None and i % options.test_interval == 1):
            test_eval.evaluate(i)
            if USE_ONE_THREAD:
                # after testing finishes, switch back to the training device
                caffe.select_device(options.train_device, False)

        start = time.time()

        if not using_asynchronous_queue:
            print("Using data_arrays directly. No queue.")
            # First pick the dataset_index to train with
            dataset_index = randint(0, len(data_arrays) - 1)
            dataset = data_arrays[dataset_index]
            offsets = []
            for j in range(0, dims):
                offsets.append(randint(0, dataset['data'].shape[1 + j] - (output_dims[j] + input_padding[j])))
            # These are the raw data elements
            data_slice = slice_data(dataset['data'], [0] + offsets, [fmaps_in] + [output_dims[di] + input_padding[di] for di in range(0, dims)])
            label_slice = slice_data(dataset['label'], [0] + [offsets[di] + int(math.ceil(input_padding[di] / float(2))) for di in range(0, dims)], [fmaps_out] + output_dims)
            # transform the input
            # this code assumes that the original input pixel values are scaled between (0,1)
            if 'transform' in dataset:
                # print('Pre:',(data_slice.min(),data_slice.mean(),data_slice.max()))
                lo, hi = dataset['transform']['scale']
                data_slice = 0.5 + (data_slice-0.5)*np.random.uniform(low=lo,high=hi)
                lo, hi = dataset['transform']['shift']
                data_slice = data_slice + np.random.uniform(low=lo,high=hi)
                # print('Post:',(data_slice.min(),data_slice.mean(),data_slice.max()))
        else:
            dataset, index_of_shared_dataset = training_data_queue.get_dataset()
            data_slice = dataset['data']
            assert data_slice.shape == (fmaps_in,) + tuple(input_dims)
            label_slice = dataset['label']
            assert label_slice.shape == (fmaps_out,) + tuple(output_dims)
            if DEBUG:
                print("Training with next dataset in queue, which has offset {o}". format(o=dataset['offset']))
            mask_slice = None
            if 'mask' in dataset:
                mask_slice = dataset['mask']

        if DEBUG:
            print("data_slice stats: data_slice.min() {}, data_slice.mean() {}, "
                  "data_slice.max() {}".format(data_slice.min(), data_slice.mean(), data_slice.max()))

        if options.loss_function == 'malis':
            components_slice,ccSizes = malis.connected_components_affgraph(label_slice.astype(int32), dataset['nhood'])
            # Also recomputing the corresponding labels (connected components)
            net_io.setInputs([data_slice, label_slice, components_slice, data_arrays[0]['nhood']])
            
        if options.loss_function == 'euclid':
            label_slice_mean = label_slice.mean()
            if 'mask' in dataset:
                label_slice = label_slice * mask_slice
                label_slice_mean = label_slice.mean() / mask_slice.mean()
            w_pos = 1.0
            w_neg = 1.0
            if options.scale_error:
                frac_pos = np.clip(label_slice_mean, 0.05, 0.95)
                w_pos = w_pos / (2.0 * frac_pos)
                w_neg = w_neg / (2.0 * (1.0 - frac_pos))
            error_scale_slice = scale_errors(label_slice, w_neg, w_pos)
            net_io.setInputs([data_slice, label_slice, error_scale_slice])

        if options.loss_function == 'softmax':
            # These are the affinity edge values
            net_io.setInputs([data_slice, label_slice])
        
        # Single step
        loss = solver.step(1)
        # sanity_check_net_blobs(net)

        if using_asynchronous_queue:
            new_dataset_index = randint(0, len(data_arrays) - 1)
            full_3d_shape_of_new_dataset = data_arrays[new_dataset_index]['data'].shape[-3:]
            offsets = tuple([
                int(randint(0, full_3d_shape_of_new_dataset[j] - input_dims[j]))
                for j in range(dims)
                ])
            # for j in range(0, dims):
            #     offsets.append(randint(0, dataset['data'].shape[1 + j] - (output_dims[j] + input_padding[j])))
            # offsets = tuple(offsets)
            # offsets = (0,0,0)
            # print("refreshing shared dataset #{i} with dataset #{j} with offset {o}"
            #       .format(i=index_of_shared_dataset, j=new_dataset_index, o=offsets))
            training_data_queue.start_refreshing_shared_dataset(
                shared_dataset_index=index_of_shared_dataset,
                offset=offsets,
                dataset_index=new_dataset_index
            )

        while gc.collect():
            pass


        if options.loss_function == 'euclid' or options.loss_function == 'euclid_aniso':
            print("[Iter %i] Loss: %f, frac_pos=%f, w_pos=%f" % (i,loss,frac_pos,w_pos))
        else:
            print("[Iter %i] Loss: %f" % (i,loss))
        # TODO: Store losses to file
        losses += [loss]

        if hasattr(options, 'loss_snapshot') and ((i % options.loss_snapshot) == 0):
            io.savemat('loss.mat',{'loss':losses})

        time_counter += 1
        total_time += time.time() - start
        if DEBUG:
            print("taking {} on average per iteration".format(total_time/time_counter))


