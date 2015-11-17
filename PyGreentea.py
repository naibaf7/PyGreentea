import os
import h5py
import numpy as np
import random
import math
import multiprocessing
from Crypto.Random.random import randint
import malis as malis
import gc
import resource

# Visualization
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
# from mayavi import mlab
# from mayavi.core.ui.mayavi_scene import MayaviScene
# import volume_slicer

from numpy import float32, int32, uint8


# Load the configuration file
import config
# Import pycaffe
import caffe


# General preparations
colorsr = np.random.rand(5000)
colorsg = np.random.rand(5000)
colorsb = np.random.rand(5000)


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

def inspect_2D_hdf5(hdf5_file):
    print 'HDF5 keys: %s' % hdf5_file.keys()
    dset = hdf5_file[hdf5_file.keys()[0]]
    print 'HDF5 shape: X: %s Y: %s' % dset.shape
    print 'HDF5 data type: %s' % dset.dtype
    print 'Max/Min: %s' % [np.asarray(dset).max(0).max(0), np.asarray(dset).min(0).min(0)]

def inspect_3D_hdf5(hdf5_file):
    print 'HDF5 keys: %s' % hdf5_file.keys()
    dset = hdf5_file[hdf5_file.keys()[0]]
    print 'HDF5 shape: X: %s Y: %s Z: %s' % dset.shape
    print 'HDF5 data type: %s' % dset.dtype
    print 'Max/Min: %s' % [np.asarray(dset).max(0).max(0).max(0), np.asarray(dset).min(0).min(0).min(0)]
    
def inspect_4D_hdf5(hdf5_file):
    print 'HDF5 keys: %s' % hdf5_file.keys()
    dset = hdf5_file[hdf5_file.keys()[0]]
    print 'HDF5 shape: T: %s X: %s Y: %s Z: %s' % dset.shape
    print 'HDF5 data type: %s' % dset.dtype
    print 'Max/Min: %s' % [np.asarray(dset).max(0).max(0).max(0).max(0), np.asarray(dset).min(0).min(0).min(0).min(0)]
    
def display_raw(raw_ds, index):
    slice = raw_ds[0:raw_ds.shape[0], 0:raw_ds.shape[1], index]
    minval = np.min(np.min(slice, axis=1), axis=0)
    maxval = np.max(np.max(slice, axis=1), axis=0)   
    img = Image.fromarray((slice - minval) / (maxval - minval) * 255)
    img.show()
    
def display_con(con_ds, index):
    slice = con_ds[0:con_ds.shape[0], 0:con_ds.shape[1], index]
    rgbArray = np.zeros((con_ds.shape[0], con_ds.shape[1], 3), 'uint8')
    rgbArray[..., 0] = colorsr[slice] * 256
    rgbArray[..., 1] = colorsg[slice] * 256
    rgbArray[..., 2] = colorsb[slice] * 256
    img = Image.fromarray(rgbArray, 'RGB')
    img.show()
    
def display_aff(aff_ds, index):
    sliceX = aff_ds[0, 0:520, 0:520, index]
    sliceY = aff_ds[1, 0:520, 0:520, index]
    sliceZ = aff_ds[2, 0:520, 0:520, index]
    img = Image.fromarray((sliceX & sliceY & sliceZ) * 255)
    img.show()
    
def display_binary(bin_ds, index):
    slice = bin_ds[0:bin_ds.shape[0], 0:bin_ds.shape[1], index]
    img = Image.fromarray(np.uint8(slice * 255))
    img.show()
    
    
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
            net.set_input_arrays(0, np.ascontiguousarray(data_slice[None, None, :]).astype(float32), np.ascontiguousarray(dummy_slice).astype(float32))
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
                
        
def train(solver, data_arrays, label_arrays, affinity_arrays, mode, input_padding, output_dims):
    dims = len(output_dims)
    losses = []
    
    net = solver.net
    if mode == 'malis' or mode == 'euclid':
        nhood = malis.mknhood3d()
    if mode == 'malis_aniso' or mode == 'euclid_aniso':
        nhood = malis.mknhood3d_aniso()
    
    # Loop from current iteration to last iteration
    for i in range(solver.iter, solver.max_iter):
        
        # First pick the dataset to train with
        dataset = randint(0, len(data_arrays) - 1)
        data_array = data_arrays[dataset]
        label_array = label_arrays[dataset]
        affinity_array = affinity_arrays[dataset]

        offsets = []
        for j in range(0, dims):
            offsets.append(randint(0, data_array.shape[j] - (output_dims[j] + input_padding[j])))
        
        dummy_slice = [0]
        
        # These are the raw data elements
        data_slice = slice_data(data_array, offsets, [output_dims[di] + input_padding[di] for di in range(0, dims)])

        # These are the affinity edge values
        aff_slice = slice_data(affinity_array, [0] + [offsets[di] + int(math.ceil(input_padding[di] / float(2))) for di in range(0, dims)], [11] + output_dims)
        
        # These are the labels (connected components)
        label_slice = slice_data(label_array, [offsets[di] + int(math.ceil(input_padding[di] / float(2))) for di in range(0, dims)], output_dims)
        # Also recomputing the corresponding labels (connected components)
        aff_slice_tmp = malis.seg_to_affgraph(label_slice,nhood)
        label_slice,ccSizes = malis.connected_components_affgraph(aff_slice_tmp,nhood)
        

        print (data_slice[None, None, :]).shape
        print (label_slice[None, None, :]).shape
        print (aff_slice[None, :]).shape
        print (nhood[None, None, :]).shape
        
        if mode == 'malis' or mode == 'malis_aniso':
            net.set_input_arrays(0, np.ascontiguousarray(data_slice[None, None, :]).astype(float32), np.ascontiguousarray(dummy_slice).astype(float32))
            net.set_input_arrays(1, np.ascontiguousarray(label_slice[None, None, :]).astype(float32), np.ascontiguousarray(dummy_slice).astype(float32))
            net.set_input_arrays(2, np.ascontiguousarray(aff_slice[None, :]).astype(float32), np.ascontiguousarray(dummy_slice).astype(float32))
            net.set_input_arrays(3, np.ascontiguousarray(nhood[None, None, :]).astype(float32), np.ascontiguousarray(dummy_slice).astype(float32))
            
        # We pass the raw and affinity array only
        if mode == 'euclid' or mode == 'euclid_aniso':
            frac_pos = np.clip(aff_slice.mean(),0.05,0.95)
            w_pos = np.sqrt(1.0/(2.0*frac_pos))
            w_neg = np.sqrt(1.0/(2.0*(1.0-frac_pos)))

            net.set_input_arrays(0, np.ascontiguousarray(data_slice[None, None, :]).astype(float32), np.ascontiguousarray(dummy_slice).astype(float32))
            net.set_input_arrays(1, np.ascontiguousarray(aff_slice[None, :]).astype(float32), np.ascontiguousarray(dummy_slice).astype(float32))
            net.set_input_arrays(2, np.ascontiguousarray(error_scale(aff_slice[None, :],w_neg,w_pos)).astype(float32), np.ascontiguousarray(dummy_slice).astype(float32))
            # net.set_input_arrays(2, np.ascontiguousarray(error_scale(aff_slice[None, :],1.0,0.045)).astype(float32), np.ascontiguousarray(dummy_slice).astype(float32))

        if mode == 'softmax':
            net.set_input_arrays(0, np.ascontiguousarray(data_slice[None, None, :]).astype(float32), np.ascontiguousarray(dummy_slice).astype(float32))
            net.set_input_arrays(1, np.ascontiguousarray(label_slice[None, None, :]).astype(float32), np.ascontiguousarray(dummy_slice).astype(float32))
        
        # Single step
        print "Stepping..."
        loss = solver.step(1)
        print "done"

        # Memory clean up and report
        print("Memory usage (before GC): %d MiB" % ((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024)))
        
        while gc.collect():
            pass

        print("Memory usage (after GC): %d MiB" % ((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024)))


        # m = volume_slicer.VolumeSlicer(data=np.squeeze((net.blobs['Convolution18'].data[0])[0,:,:]))
        # m.configure_traits()

        if mode == 'euclid' or mode == 'euclid_aniso':
            print("[Iter %i] Loss: %s, frac_pos=%f, w_pos=%f" % (i,loss,frac_pos,w_pos))
        else:
            print("[Iter %i] Loss: %s" % (i,loss))
        losses += [loss]
        

