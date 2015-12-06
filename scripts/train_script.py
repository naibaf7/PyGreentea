import os, sys
import numpy as np
from numpy import float32, int32, uint8
import h5py
import config
import caffe
import PyGreentea as pygt

# Specify the device to use
device_id = 0

# Specify the solver file
solver_proto = "net/solver.prototxt"

# Specify values for testing
test_net = "net/anisonet_test.prototxt"
# trained_model = "anisonet1e5_iter_9150.caffemodel"
# trained_model = "anisonet1e5malis_iter_1000.caffemodel"
trained_model = "anisonet1e5malis1e5_iter_23000.caffemodel"


output_folder = "processed"

hdf5_raw_file = 'fibsem_medulla_7col/tstvol-520-1-h5/img_normalized.h5'
hdf5_gt_file = 'fibsem_medulla_7col/tstvol-520-1-h5/groundtruth_seg_thick.h5'
hdf5_aff_file = 'fibsem_medulla_7col/tstvol-520-1-h5/groundtruth_aff.h5'


output_dims = [16, 44, 44]
input_padding = [28, 88, 88]

border_reflect = False

# Select "train" or "process"
# mode = "process"
mode = "train"
loss = "euclid_aniso"



hdf5_raw = h5py.File(hdf5_raw_file, 'r')
hdf5_gt = h5py.File(hdf5_gt_file, 'r')
hdf5_aff = h5py.File(hdf5_aff_file, 'r')

#inspect_3D_hdf5(hdf5_raw)
#inspect_3D_hdf5(hdf5_gt)
#inspect_4D_hdf5(hdf5_aff)

# Make the dataset ready for the network
hdf5_raw_ds = pygt.normalize(np.asarray(hdf5_raw[hdf5_raw.keys()[0]]).astype(float32), -1, 1)
hdf5_gt_ds = np.asarray(hdf5_gt[hdf5_gt.keys()[0]]).astype(float32)
hdf5_aff_ds = np.asarray(hdf5_aff[hdf5_aff.keys()[0]])

#display_aff(hdf5_aff_ds, 1)
#display_con(hdf5_gt_ds, 0)
#display_raw(hdf5_raw_ds, 0)
#display_binary(hdf5_gt_ds, 0)

#Initialize caffe
caffe.set_mode_gpu()
caffe.set_device(device_id)


if(mode == "train"):
    solver = caffe.get_solver(solver_proto)
    solver.restore("anisonet1e5_iter_9000.solverstate")
    net = solver.net
    print "Done"
    pygt.train(solver, [hdf5_raw_ds], [hdf5_gt_ds], [hdf5_aff_ds], loss, input_padding, output_dims)
    
if(mode == "process"):
    net = caffe.Net(test_net, trained_model, caffe.TEST)
    pygt.process(net, [hdf5_raw_ds], output_folder, input_padding, output_dims)
