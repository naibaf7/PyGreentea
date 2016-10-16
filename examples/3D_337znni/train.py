from __future__ import print_function
import sys, os, math
import numpy as np
from numpy import float32, int32, uint8, dtype
from PIL import Image
import glob

# Load PyGreentea
# Relative path to where PyGreentea resides
pygt_path = '../..'
sys.path.append(pygt_path)
import pygreentea.pygreentea as pygt
from pygreentea.pygreentea import malis

# Load the datasets - individual tiff files in a directory
raw_dir = '../../../project_data/dataset_01/train/raw'
label_dir = '../../../project_data/dataset_01/train/labels'

raw_path = sorted(glob.glob(raw_dir+'/*.tif'))
label_path = sorted(glob.glob(label_dir+'/*.png'))
num_images = len(raw_path)

raw_ds = [np.array(Image.open(raw_path[i]).convert('L'), 'f') for i in range(0, num_images)]
gt_ds = [np.array(Image.open(label_path[i]).convert('L'), 'f') for i in range(0,num_images)]
gt_ds_scaled = [np.floor(label/31) for label in gt_ds]


datasets = []
test_datasets = []
for i in range(0,1):
    dataset = {}
    dataset['data'] = np.expand_dims(pygt.normalize(np.asarray(raw_ds, float32)), 0)
    dataset['label'] = np.expand_dims(np.asarray(gt_ds_scaled, float32), 0)
    datasets += [dataset]

# Custom callback function to generate slices
def data_slice_callback(input_specs, batch_size, dataset_indexes, offsets, dataset_combined_sizes, data_arrays, slices):
    pass

def test_data_slice_callback(dataset_idx, offsets, datasets, slices):
    pass

# Set train options
class TrainOptions:
    loss_snapshot = 100
    test_interval = 4000
    train_device = 0
    test_device = 0
    test_net = None
    test_level = 0
    test_stages = None

options = TrainOptions()

# Set solver options
solver_config = pygt.caffe.SolverParameter()
solver_config.train_net = 'net.prototxt'
solver_config.base_lr = 0.00001
solver_config.momentum = 0.99
solver_config.weight_decay = 0.000005
solver_config.lr_policy = 'inv'
solver_config.gamma = 0.0001
solver_config.power = 0.75
solver_config.max_iter = 6000
solver_config.snapshot = 2000
solver_config.snapshot_prefix = 'net'
solver_config.type = 'Adam'
solver_config.display = 1


# Set devices
pygt.caffe.enumerate_devices(False)
pygt.caffe.set_devices((options.train_device,))

solverstates = pygt.get_solver_states(solver_config.snapshot_prefix)

# First training stage (softmax + euclid)
if (len(solverstates) == 0 or solverstates[-1][0] < solver_config.max_iter):
    solver, test_net = pygt.init_solver(solver_config, options)
    if (len(solverstates) > 0):
        solver.restore(solverstates[-1][1])
    pygt.train(solver, options, datasets, data_slice_callback,
               test_net, test_datasets, test_data_slice_callback)