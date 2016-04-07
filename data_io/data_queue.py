from __future__ import print_function

import multiprocessing
import time
from operator import mul
from os.path import join
import warnings

import h5py
import numpy as np

import PyGreentea as pygt

''' where this will be used:
* train()
  * getting unordered batches of a dataset, specified with offset, input size and output size
  * behavior should be like a queue. at the end of each training iteration, train() specifies a replacement
* process()
  * getting batches from a dataset, specified with offset and input size
'''


def update_shared_dataset(index_of_shared, index_of_which_dataset, input_slice, output_slice, transform=True):
    # print("in process id {}".format(os.getpid()))
    shared_dataset = shared_datasets[index_of_shared]
    original_dataset = datasets[index_of_which_dataset]
    # print("original_dataset: ", [
    #     (key,
    #      type(original_dataset[key]),
    #      original_dataset[key])
    #     for key in original_dataset.keys()])
    dataset_numpy = dict()
    # load inputs
    # print("original_dataset['data']: ", original_dataset['data'].shape, original_dataset['data'].dtype)
    data_slice = np.array(original_dataset['data'][input_slice], dtype=np.float32) / (2. ** 8)
    if transform:
        if 'transform' in original_dataset:
            # print('Pre:',(data_slice.min(),data_slice.mean(),data_slice.max()))
            lo, hi = original_dataset['transform']['scale']
            data_slice = 0.5 + (data_slice-0.5)*np.random.uniform(low=lo,high=hi)
            lo, hi = original_dataset['transform']['shift']
            data_slice = data_slice + np.random.uniform(low=lo,high=hi)
        else:
            print("WARNING: source data doesn't have 'transform' attribute.")
    dataset_numpy['data'] = data_slice
    # load outputs if desired
    if output_slice is not None:
        dataset_numpy['components'] = np.array(original_dataset['components'][output_slice])
        if 'label' in original_dataset:
            label_slice = (slice(None),) + output_slice
            dataset_numpy['label'] = np.array(original_dataset['label'][label_slice])
        else:
            # compute affinities from components
            dataset_numpy['label'] = pygt.malis.seg_to_affgraph(dataset_numpy['components'], original_dataset['nhood'])
            warnings.warn("Computing affinity labels because 'label' wasn't provided in data source.", UserWarning)
        if 'mask' in original_dataset:
            dataset_numpy['mask'] = np.array(original_dataset['mask'][output_slice], dtype=np.uint8)
        else:
            # assume no masking
            dataset_numpy['mask'] = np.ones_like(dataset_numpy['components'], dtype=np.uint8)
            warnings.warn("No mask provided. Setting to 1 everywhere.", UserWarning)
    for key in shared_dataset:
        source_array = dataset_numpy[key].astype(dtypes[key])
        target_mp_array = shared_dataset[key]
        if pygt.DEBUG:
            print("dataset_numpy['{0}']: dtype {1} and shape {2}".format(key, source_array.dtype, source_array.shape))
        # print("dataset_numpy['{0}'].flatten().size: {1}".format(key, source_array.flatten().size))
        # print(dir(target_mp_array))
        # print(target_mp_array._length_, target_mp_array._objects, target_mp_array._type_, target_mp_array._b_base_)
        target_mp_array[:] = source_array.flatten()
    return


class DatasetQueue(object):
    def __init__(self, size, datasets, input_shape, output_shape=None, n_workers=1):
        self.size = size
        self.datasets = datasets
        self.input_shape = input_shape
        self.outputs_are_ignored = output_shape is None
        self.output_shape = output_shape or (0, 0, 0)
        self._list = list()
        self.shapes = {
            'data': (1,) + self.input_shape,
            'components': (1,) + self.output_shape,
            'label': (3,) + self.output_shape,
            'mask': (1,) + self.output_shape,
        }
        self.dtypes = {
            'data': np.float32,
            'components': np.int32,
            'label': np.int32,
            'mask': np.uint8,
        }
        self.keys_to_ignore = []
        if self.outputs_are_ignored:
            # then ignore all outputs. (e.g. for test processing)
            self.keys_to_ignore = ['label', 'components', 'mask']
            for output_key in self.keys_to_ignore:
                self.dtypes.pop(output_key)
                self.shapes.pop(output_key)
        sizes = dict()
        for key, shape in self.shapes.iteritems():
            sizes[key] = reduce(mul, shape)
        if pygt.DEBUG:
            print("sizes: ", sizes)
        self.shared_datasets = []
        for n in range(size):
            shared_dataset = dict()
            for key in self.dtypes:
                size = sizes[key]
                dtype = self.dtypes[key]
                ctype = type(np.ctypeslib.as_ctypes(dtype(0)))
                if pygt.DEBUG:
                    print("creating {key}'s multiprocessing.Array with "
                          "ctype {c} and size {s}".format(key=key, c=ctype, s=size))
                shared_dataset[key] = multiprocessing.Array(ctype, size, lock=False)
            self.shared_datasets.append(shared_dataset)
        self.pool = multiprocessing.Pool(
            processes=n_workers,
            initializer=self._initialize_pool,
            initargs=(),
            maxtasksperchild=10
        )
        self.ready_shared_datasets = []
        return

    def _initialize_pool(self):
        global shared_datasets
        shared_datasets = self.shared_datasets
        global datasets
        datasets = self.datasets
        global dtypes
        dtypes = self.dtypes

    def get_dataset(self, copy=False):
        while len(self.ready_shared_datasets) < 1:
            print('\n', 'waiting for an available shared dataset', '\n')
            time.sleep(1)
            continue
        dataset_metadata = self.ready_shared_datasets.pop(0)
        index_of_shared_dataset = dataset_metadata['shared']
        index_of_given_dataset = dataset_metadata['real']
        new_dataset = dict()
        new_dataset['offset'] = dataset_metadata['offset']
        shared_dataset = self.shared_datasets[index_of_shared_dataset]
        given_dataset = self.datasets[index_of_given_dataset]
        for key in shared_dataset:
            # print("loading shared_dataset['{}']".format(key))
            # print("{}'s desired shape: {}".format(key, self.shapes[key]))
            # if key is 'data':
            #     dtype = np.float32
            # else:
            #     dtype = np.int32
            dtype = self.dtypes[key]
            new_dataset[key] = np.frombuffer(shared_dataset[key], dtype)
            if pygt.DEBUG:
                print(key, new_dataset[key].shape, self.shapes[key])
            new_dataset[key] = new_dataset[key].reshape(self.shapes[key])
            if copy:
                new_dataset[key] = new_dataset[key].copy()
        for key in given_dataset:
            # print("processing given_dataset['{}']".format(key))
            if key in shared_dataset or key in self.keys_to_ignore:
                # we already loaded it, or we want to ignore it
                # print("ignoring given_dataset['{key}']".format(key=key))
                pass
            else:
                # get the value from the original dataset dict
                new_dataset[key] = given_dataset[key]
        return new_dataset, index_of_shared_dataset

    def start_refreshing_shared_dataset(self, shared_dataset_index, offset, dataset_index, transform=True, wait=False):
        if self.outputs_are_ignored:
            output_slice = None
        else:
            borders = tuple([(in_ - out_) / 2 for (in_, out_) in zip(self.input_shape, self.output_shape)])
            # print("borders: ", borders)
            output_slice = tuple([slice(offset[i] + borders[i], offset[i] + borders[i] + self.output_shape[i])
                            for i in range(len(offset))])
        # print("output_slice: {}".format(output_slice))
        input_slice = tuple([slice(offset[i], offset[i] + self.input_shape[i])
                       for i in range(len(offset))])
        # print("input_slice: {}".format(input_slice))

        dataset_metadata = dict(real=dataset_index, shared=shared_dataset_index, offset=offset)
        def pool_callback(*args, **kwargs):
            # print(args, kwargs)
            return self.ready_shared_datasets.append(dataset_metadata)
        async_result = self.pool.apply_async(
            func=update_shared_dataset,
            kwds=dict(
                index_of_shared=shared_dataset_index,
                index_of_which_dataset=dataset_index,
                input_slice=input_slice,
                output_slice=output_slice,
                transform=transform
            ),
            callback=pool_callback
        )
        if wait:
            final_result = async_result.get()
            if final_result is not None:
                # probably an error
                print(final_result)
        return shared_dataset_index, async_result


if __name__ == '__main__':
    path = '/groups/turaga/home/turagas/data/FlyEM/fibsem_medulla_7col/'
    datasets = []
    for dname in ['trvol-250-1-h5', 'trvol-250-2-h5']:
        datasets.append(
            dict({
                'name': dname,
                'data': h5py.File(join(path, dname, 'im_uint8.h5'), 'r')['main'],
                'components': h5py.File(join(path, dname, 'groundtruth_seg_thick.h5'), 'r')['main'],
                'nhood': pygt.malis.mknhood3d(),
                'transform': dict({'scale': (0.8, 1.2), 'shift': (-0.2, 0.2)})
            })
        )
    queue_size = 1
    q = DatasetQueue(queue_size,
                     datasets=datasets,
                     input_shape=(80, 80, 80),
                     output_shape=(60, 60, 60),
                     n_workers=1
                     )
    for j in range(len(datasets)):
        i = 0  # index of shared dataset to use
        shared_dataset_index, async_result = q.start_refreshing_shared_dataset(i, (15, 25, 35), j, wait=True)
        print("{}'s async_result.get(): {}".format(datasets[j]['name'], async_result.get()))
        dataset_result, index_of_shared_dataset = q.get_dataset(copy=False)
        print('start - ********************************************************************************')
        for key, value in dataset_result.iteritems():
            try:
                print(key, value.dtype, value.shape, type(value), value[0, 5, 50, 20:30], np.mean(value))
            except:
                print(key, value)
        print('end   - ********************************************************************************')
