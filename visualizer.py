import numpy as np

# Visualization
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
# from mayavi import mlab
# from mayavi.core.ui.mayavi_scene import MayaviScene
# import volume_slicer

# General preparations
colorsr = np.random.rand(5000)
colorsg = np.random.rand(5000)
colorsb = np.random.rand(5000)

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