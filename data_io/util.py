import numpy as np


def data_queue_should_be_used_with(data_arrays):
    # if 'data' is a numpy array, we assume data_arrays's contents are already in-memory
    data_is_in_memory = isinstance(data_arrays[0]['data'], np.ndarray)
    return not data_is_in_memory
