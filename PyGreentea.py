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
    import data_io


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


def process_input_data(net_io, input_data):
    net_io.setInputs([input_data])
    net_io.net.forward()
    net_outputs = net_io.net.blobs['prob']
    output = net_outputs.data[0].copy()
    return output


def generate_dataset_offsets_for_processing(net, data_arrays, process_borders):
    input_dims, output_dims, input_padding = get_spatial_io_dims(net)
    dims = len(output_dims)
    dataset_offsets_to_process = dict()
    for i in range(len(data_arrays)):
        data_array = data_arrays[i]['data']
        data_dims = len(data_array.shape)
        if process_borders:
            border_widths = [int(math.ceil(pad / float(2))) for pad in input_padding]
            origin = [-border_width for border_width in border_widths]
        else:
            origin = [0 for _ in input_padding]
        offsets = list(origin)
        in_dims = []
        out_dims = []
        for d in range(dims):
            in_dims += [data_array.shape[data_dims-dims+d]]
            out_dims += [data_array.shape[data_dims-dims+d] - input_padding[d]]
        list_of_offsets_to_process = []
        while True:
            offsets_to_append = list(offsets)  # make a copy. important!
            list_of_offsets_to_process.append(offsets_to_append)
            incremented = False
            for d in range(dims):
                if process_borders:
                    maximum_offset = in_dims[dims - 1 - d] - output_dims[dims - 1 - d] - border_widths[dims - 1 - d]
                else:
                    maximum_offset = out_dims[dims - 1 - d] - output_dims[dims - 1 - d]
                if offsets[dims - 1 - d] == maximum_offset:
                    # Reset direction
                    offsets[dims - 1 - d] = origin[dims - 1 - d]
                else:
                    # Increment direction
                    next_potential_offset = offsets[dims - 1 - d] + output_dims[dims - 1 - d]
                    offsets[dims - 1 - d] = min(next_potential_offset, maximum_offset)
                    incremented = True
                    break
            if not incremented:
                break
        dataset_offsets_to_process[i] = list_of_offsets_to_process
    return dataset_offsets_to_process


def process(net, data_arrays, shapes=None, net_io=None, zero_pad_source_data=True, target_arrays=None):
    input_dims, output_dims, input_padding = get_spatial_io_dims(net)
    fmaps_in, fmaps_out = get_fmap_io_dims(net)
    dims = len(output_dims)
    if target_arrays is not None:
        assert len(data_arrays) == len(target_arrays)
        for data_array, target in zip(data_arrays, target_arrays):
            prediction_shape = (fmaps_out,) + data_array['data'].shape[-dims:]
            assert prediction_shape == target.shape, \
                "Target array for dname {} is the wrong shape. {} should be {}"\
                    .format(data_array['name'], target.shape, prediction_shape)
    pred_arrays = []
    if shapes is None:
        # Raw data slice input         (n = 1, f = 1, spatial dims)
        shapes = [[1, fmaps_in] + input_dims]
    if net_io is None:
        net_io = NetInputWrapper(net, shapes)
    using_data_loader = data_io.data_loader_should_be_used_with(data_arrays)
    if using_data_loader:
        processing_data_loader = data_io.DataLoader(
            size=5,
            datasets=data_arrays,
            input_shape=tuple(input_dims),
            output_shape=None,  # ignore labels
            n_workers=3
        )
    dataset_offsets_to_process = generate_dataset_offsets_for_processing(
        net, data_arrays, process_borders=zero_pad_source_data)
    for source_dataset_index in dataset_offsets_to_process:
        list_of_offsets_to_process = dataset_offsets_to_process[source_dataset_index]
        if DEBUG:
            print("source_dataset_index = ", source_dataset_index)
            print("Processing source volume #{i} with offsets list {o}"
                  .format(i=source_dataset_index, o=list_of_offsets_to_process))
        # make a copy of that list for enqueueing purposes
        offsets_to_enqueue = list(list_of_offsets_to_process)
        data_array = data_arrays[source_dataset_index]['data']
        if target_arrays is not None:
            pred_array = target_arrays[source_dataset_index]
        else:
            prediction_shape = (fmaps_out,) + data_array.shape[-dims:]
            pred_array = np.zeros(shape=prediction_shape, dtype=np.float32)
        if using_data_loader:
            # start pre-populating queue
            for shared_dataset_index in range(min(processing_data_loader.size, len(list_of_offsets_to_process))):
                # fill shared-memory datasets with an offset
                offsets = offsets_to_enqueue.pop(0)
                offsets = tuple([int(o) for o in offsets])
                # print("Pre-populating processing data loader with data at offset {}".format(offsets))
                print("Pre-populating data loader's dataset #{i}/{size} with dataset #{d} and offset {o}"
                      .format(i=shared_dataset_index, size=processing_data_loader.size,
                              d=source_dataset_index, o=offsets))
                shared_dataset_index, async_result = processing_data_loader.start_refreshing_shared_dataset(
                    shared_dataset_index,
                    offsets,
                    source_dataset_index,
                    transform=False,
                    wait=True
                )
        # process each offset
        for i_offsets in range(len(list_of_offsets_to_process)):
            if using_data_loader:
                dataset, index_of_shared_dataset = processing_data_loader.get_dataset()
                offsets = list(dataset['offset'])  # convert tuple to list
                data_slice = dataset['data']
                if DEBUG:
                    print("Processing next dataset in processing data loader, which has offset {o}"
                          .format(o=dataset['offset']))
            else:
                offsets = list_of_offsets_to_process[i_offsets]
                if zero_pad_source_data:
                    data_slice = data_io.util.get_zero_padded_slice_from_array_by_offset(
                        array=data_array,
                        origin=[0] + offsets,
                        shape=[fmaps_in] + [output_dims[di] + input_padding[di] for di in range(dims)]
                    )
                else:
                    data_slice = slice_data(
                        data_array,
                        [0] + offsets,
                        [fmaps_in] + [output_dims[di] + input_padding[di] for di in range(dims)]
                    )
            # process the chunk
            output = process_input_data(net_io, data_slice)
            print(offsets)
            print(output.mean())
            pads = [int(math.ceil(pad / float(2))) for pad in input_padding]
            offsets_for_pred_array = [0] + [offset + pad for offset, pad in zip(offsets, pads)]
            set_slice_data(pred_array, output, offsets_for_pred_array, [fmaps_out] + output_dims)
            if using_data_loader and len(offsets_to_enqueue) > 0:
                # start adding the next slice to the loader with index_of_shared_dataset
                new_offsets = offsets_to_enqueue.pop(0)
                processing_data_loader.start_refreshing_shared_dataset(
                    index_of_shared_dataset,
                    new_offsets,
                    source_dataset_index,
                    transform=False
                )
        pred_arrays.append(pred_array)
        if using_data_loader:
            processing_data_loader.destroy()
    return pred_arrays


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
        if USE_ONE_THREAD:
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
        return solver_inst, init_testnet(options.test_net, test_device=options.test_device)


def init_testnet(test_net, trained_model=None, test_device=0):
    caffe.set_mode_gpu()
    caffe.select_device(test_device, False)
    if trained_model is None:
        return caffe.Net(test_net, caffe.TEST)
    else:
        return caffe.Net(test_net, trained_model, caffe.TEST)


class MakeDatasetOffset(object):
    def __init__(self, dims, output_dims, input_padding):
        self.dims = dims
        self.output_dims = output_dims
        self.input_padding = input_padding

    def __call__(self, data_array_list):
        which_dataset = randint(0, len(data_array_list) - 1)
        offsets = []
        for j in range(0, self.dims):
            offsets.append(randint(0, data_array_list[which_dataset]['data'].shape[j] - (self.output_dims[j] + self.input_padding[j])))
        offsets = tuple([int(x) for x in offsets])
        return which_dataset, offsets


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

    if data_io.data_loader_should_be_used_with(data_arrays):
        using_data_loader = True
        # and initialize queue!
        loader_size = 20
        n_workers = 10
        make_dataset_offset = MakeDatasetOffset(dims, output_dims, input_padding)
        loader_kwargs = dict(
            size=loader_size,
            datasets=data_arrays,
            input_shape=tuple(input_dims),
            output_shape=tuple(output_dims),
            n_workers=n_workers,
            dataset_offset_func=make_dataset_offset
        )
        print("creating queue with kwargs {}".format(loader_kwargs))
        training_data_loader = data_io.DataLoader(**loader_kwargs)
        # start populating the queue
        for i in range(loader_size):
            if DEBUG:
                print("Pre-populating data loader's dataset #{i}/{size}"
                      .format(i=i, size=training_data_loader.size))
            shared_dataset_index, async_result = \
                training_data_loader.start_refreshing_shared_dataset(i)
    else:
        using_data_loader = False

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

        if not using_data_loader:
            print("Using data_arrays directly. No data loader.")
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
            dataset, index_of_shared_dataset = training_data_loader.get_dataset()
            data_slice = dataset['data']
            assert data_slice.shape == (fmaps_in,) + tuple(input_dims)
            label_slice = dataset['label']
            assert label_slice.shape == (fmaps_out,) + tuple(output_dims)
            if DEBUG:
                print("Training with next dataset in data loader, which has offset {o}". format(o=dataset['offset']))
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

        # def make_offset_for_dataset(source_dataset_index):
        #     full_3d_shape_of_new_dataset = data_arrays[source_dataset_index]['data'].shape[-3:]
        #     offsets = tuple([int(randint(0, full_3d_shape_of_new_dataset[j] - input_dims[j]))
        #                      for j in range(dims)])
        #     return offsets

        if using_data_loader:
            training_data_loader.start_refreshing_shared_dataset(index_of_shared_dataset)

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
    
    if using_data_loader:
        training_data_loader.destroy()
