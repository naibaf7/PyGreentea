from __future__ import print_function
import sys, os, math
import time
sys.path.append('/groups/turaga/home/singhc/caffe_v1/PyGreentea') # Relative path to where PyGreentea resides
sys.path.append('src_cython')
from evaluateFile import evaluateFile, averageAndEvaluateFiles
from processFile import processFile

# gt file
hdf5_gt_file = '/groups/turaga/home/turagas/data/FlyEM/fibsem_medulla_7col/tstvol-520-2-h5/groundtruth_seg_thick.h5' #groundtruth_aff.h5

# input models
model_base_folder = '/groups/turaga/home/turagas/research/caffe_v1/pygt_models/fibsem'
fibsemFolders = ['2','3','4','5','6']
iters = [30000,70000]

# output folders
train = False # which dataset to evaluate
h5OutputFilenames = ["data_tier2/test/output_"+str(iters[j])+"/"+"tstvol-2_"+fibsemFolders[i] for j in range(len(iters)) for i in range(len(fibsemFolders))]
randOutputFolder = ['data_tier2/test/out/fibsem' +fibsemFolders[i]+ '_'+str(iters[j])+'/' for j in range(len(iters)) for i in range(len(fibsemFolders))]

# settings
threshes = [i*2000 for i in range(1,6)]+[i*20000 for i in range(2,16)] # default: 100...1,000...100,000
funcs = ['square'] #'linear','threshold','watershed','lowhigh'
save_segs = False


for iter_idx in range(len(iters)):
	for fibsem_idx in range(len(fibsemFolders)):
		start = time.clock()
		processFile(model_base_folder+fibsemFolders[fibsem_idx]+'/',iters[iter_idx],h5OutputFilenames[iter_idx*len(fibsemFolders)+fibsem_idx],train)
		evaluateFile([hdf5_gt_file,h5OutputFilenames[fibsem_idx]+'.h5',threshes,funcs,save_segs,randOutputFolder[iter_idx*len(fibsemFolders)+fibsem_idx]])
		#averageAndEvaluateFiles([hdf5_gt_file,h5OutputFilenames, threshes,funcs,save_segs,randOutputFolder[iter_idx*len(fibsemFolders)+fibsem_idx]]) # for averaging
		print("time elapsed ",time.clock()-start)


