from libcpp.list cimport list
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
import numpy as np
import os
cimport numpy as np

cdef extern from "main2.h":
    map[string,vector[double]] eval_c(int dx, int dy, int dz, int dcons, np.uint32_t* gt, np.float32_t* affs, list[int] *threshes, list[string] *funcs, int save_seg, string* out)

def eval(np.ndarray[np.uint32_t,ndim=3] gt,np.ndarray[np.float32_t,ndim=4] affs, list[int] threshes, list[string] funcs, int save_seg, string out='out/'):
    dims = affs.shape
    dirs = [out,out+'linear',out+'square',out+'threshold',out+'watershed',out+'lowhigh']
    for i in range(len(dirs)):
        if not os.path.exists(dirs[i]):
            os.makedirs(dirs[i])
    # list contains six args ('linear','square','fel','thresh','watershed','lowhigh')
    map = eval_c(dims[0],dims[1],dims[2],dims[3],&gt[0,0,0],&affs[0,0,0,0],&threshes,&funcs,save_seg,&out)
    return map