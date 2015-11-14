import os, sys, inspect
import multiprocessing

# Specify the path caffe here
caffe_path = "../../caffe_gt"

# Specify wether or not to compile caffe
library_compile = True




cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
if cmd_folder not in sys.path:
    sys.path.append(cmd_folder)
    
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], caffe_path + "/python")))
if cmd_subfolder not in sys.path:
    sys.path.append(cmd_subfolder)

sys.path.append(caffe_path + "/python")

# Ensure correct compilation of Caffe and Pycaffe
if library_compile:
    cpus = multiprocessing.cpu_count()
    cwd = os.getcwd()
    os.chdir(caffe_path)
    result = os.system("make all -j %s" % cpus)
    if result != 0:
        sys.exit(result)
    result = os.system("make pycaffe -j %s" % cpus)
    if result != 0:
        sys.exit(result)
    os.chdir(cwd)

# Fix up OpenCL variables. Can interfere with the
# frame buffer if the GPU is also a display driver
os.environ["GPU_MAX_ALLOC_PERCENT"] = "100"
os.environ["GPU_SINGLE_ALLOC_PERCENT"] = "100"
os.environ["GPU_MAX_HEAP_SIZE"] = "100"
os.environ["GPU_FORCE_64BIT_PTR"] = "1"
