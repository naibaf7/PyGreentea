import os
import unittest

import h5py
import numpy as np

from PyGreentea.data_io.util import get_zero_padded_slice_from_array_by_offset, get_zero_padded_array_slice


class TestZeroPaddedSlicing(unittest.TestCase):
    def setUp(self):
        self.test_h5_file_name = 'test.h5'
        with h5py.File(self.test_h5_file_name, 'w') as h5_file:
            h5_file.create_dataset(
                name='test dataset',
                data=[[1]]
            )

    def tearDown(self):
        os.remove(self.test_h5_file_name)

    def test_works_with_interior_slice(self):
        X = np.ones(shape=(5,))
        origin = (1,)
        slice_shape = (3,)
        result = get_zero_padded_slice_from_array_by_offset(X, origin, shape=slice_shape)
        expected_result = np.array([1, 1, 1], dtype=X.dtype)
        assert np.allclose(result, expected_result)
        assert result.dtype == expected_result.dtype, "{}, {}".format(result.dtype, expected_result.dtype)
        slices = [slice(o, o + s, 1) for o, s in zip(origin, slice_shape)]
        result = get_zero_padded_array_slice(X, slices)
        assert np.allclose(result, expected_result)
        assert result.dtype == expected_result.dtype

    def test_works_with_slice_bigger_than_source(self):
        X = np.ones(shape=(1,))
        origin = (-1,)
        slice_shape = (3,)
        result = get_zero_padded_slice_from_array_by_offset(X, origin, shape=slice_shape)
        expected_result = np.array([0, 1, 0], dtype=X.dtype)
        assert np.allclose(result, expected_result)
        assert result.dtype == expected_result.dtype
        slices = [slice(o, o + s, 1) for o, s in zip(origin, slice_shape)]
        result = get_zero_padded_array_slice(X, slices)
        assert np.allclose(result, expected_result)
        assert result.dtype == expected_result.dtype

    def test_works_with_negative_origin(self):
        X = np.ones(shape=(5,))
        origin = (-1,)
        slice_shape = (3,)
        result = get_zero_padded_slice_from_array_by_offset(X, origin, shape=slice_shape)
        expected_result = np.array([0, 1, 1], dtype=X.dtype)
        assert np.allclose(result, expected_result)
        assert result.dtype == expected_result.dtype
        slices = [slice(o, o + s, 1) for o, s in zip(origin, slice_shape)]
        result = get_zero_padded_array_slice(X, slices)
        assert np.allclose(result, expected_result)
        assert result.dtype == expected_result.dtype
        X = np.ones(shape=(5, 5))
        origin = (-1, -1)
        slice_shape = (3, 3)
        result = get_zero_padded_slice_from_array_by_offset(X, origin, shape=slice_shape)
        expected_result = np.array([[0, 0, 0],
                                    [0, 1, 1],
                                    [0, 1, 1]], dtype=X.dtype)
        assert np.allclose(result, expected_result)
        assert result.dtype == expected_result.dtype
        slices = [slice(o, o + s, 1) for o, s in zip(origin, slice_shape)]
        result = get_zero_padded_array_slice(X, slices)
        assert np.allclose(result, expected_result)
        assert result.dtype == expected_result.dtype

    def test_works_with_too_positive_origin_1d(self):
        X = np.ones(shape=(5,))
        slice_shape = (3,)
        origin = (3,)
        result = get_zero_padded_slice_from_array_by_offset(X, origin, shape=slice_shape)
        expected_result = np.array([1, 1, 0], dtype=X.dtype)
        assert np.allclose(result, expected_result)
        assert result.dtype == expected_result.dtype
        slices = [slice(o, o + s, 1) for o, s in zip(origin, slice_shape)]
        result = get_zero_padded_array_slice(X, slices)
        assert np.allclose(result, expected_result)
        assert result.dtype == expected_result.dtype
        X = np.ones(shape=(5, 5))
        slice_shape = (3, 3)
        origin = (3, 3)
        result = get_zero_padded_slice_from_array_by_offset(X, origin, shape=slice_shape)
        expected_result = np.array([[1, 1, 0],
                                    [1, 1, 0],
                                    [0, 0, 0]], dtype=X.dtype)
        assert np.allclose(result, expected_result)
        assert result.dtype == expected_result.dtype
        slices = [slice(o, o + s, 1) for o, s in zip(origin, slice_shape)]
        result = get_zero_padded_array_slice(X, slices)
        assert np.allclose(result, expected_result)
        assert result.dtype == expected_result.dtype

    def test_works_with_hdf5_file(self):
        with h5py.File(self.test_h5_file_name, 'r') as h5_file:
            X = h5_file['test dataset']
            origin = (-1, -1)
            slice_shape = (3, 3)
            result = get_zero_padded_slice_from_array_by_offset(X, origin, slice_shape)
            expected_result = np.array([[0, 0, 0],
                                        [0, 1, 0],
                                        [0, 0, 0]], dtype=X.dtype)
            assert np.allclose(result, expected_result)
            assert result.dtype == expected_result.dtype
            slices = [slice(o, o + s, 1) for o, s in zip(origin, slice_shape)]
            result = get_zero_padded_array_slice(X, slices)
            assert np.allclose(result, expected_result)
            assert result.dtype == expected_result.dtype
