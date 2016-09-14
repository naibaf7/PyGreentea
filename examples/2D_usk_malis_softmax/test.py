from __future__ import print_function
import sys, os, math
import numpy as np
import h5py
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

raw_path = sorted(glob.glob(raw_dir+'/*.tif'))
num_images = len(raw_path)

raw_ds = [np.expand_dims(pygt.normalize(np.array(Image.open(raw_path[i]).convert('L'), 'f')),0) for i in range(0, num_images)]

datasets = []
for i in range(0,len(raw_ds)):
    dataset = {}
    dataset['data'] = raw_ds[i]
    datasets += [dataset]

test_net_file = 'net.prototxt'
test_device = 0

pygt.caffe.set_devices((test_device,))

caffemodels = pygt.get_caffe_models('net')

test_net = pygt.init_testnet(test_net_file, trained_model=caffemodels[-1][1], test_device=test_device)

def process_data_slice_callback(input_specs, batch_size, dataset_indexes, offsets, dataset_combined_sizes, data_arrays, slices):
    # Nothing to process here
    pass

output_arrays = []

pygt.process(test_net, datasets, ['aff_pred', 'smax_pred'], output_arrays, process_data_slice_callback)

for i in range(0, len(output_arrays)):
    for key in output_arrays[i].keys():
        outhdf5 = h5py.File('output/' + key + str(i) + '.h5', 'w')
        outdset = outhdf5.create_dataset('main', np.shape(output_arrays[i][key]), np.float32, data=output_arrays[i][key])
        outhdf5.close()