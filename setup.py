import inspect
import subprocess
import os
import platform
import sys
import multiprocessing

import config

# Determine where PyGreentea is
pygtpath = os.path.normpath(os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0])))

# Determine where PyGreentea gets called from
cmdpath = os.getcwd()

sys.path.append(pygtpath)
sys.path.append(cmdpath)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Direct call to PyGreentea, set up everything
if __name__ == "__main__":
    # Load the setup module
    import setup

    if (pygtpath != cmdpath):
        os.chdir(pygtpath)
    
    if (os.geteuid() != 0):
        print(bcolors.WARNING + "PyGreentea setup should probably be executed with root privileges!" + bcolors.ENDC)
    
    if config.install_packages:
        print(bcolors.HEADER + ("==== PYGT: Installing OS packages ====").ljust(80,"=") + bcolors.ENDC)
        setup.install_dependencies()
    
    print(bcolors.HEADER + ("==== PYGT: Updating Caffe/Greentea repository ====").ljust(80,"=") + bcolors.ENDC)
    setup.clone_caffe(config.caffe_path, config.clone_caffe, config.update_caffe)
    
    print(bcolors.HEADER + ("==== PYGT: Updating Malis repository ====").ljust(80,"=") + bcolors.ENDC)
    setup.clone_malis(config.malis_path, config.clone_malis, config.update_malis)
    
    if config.compile_caffe:
        print(bcolors.HEADER + ("==== PYGT: Compiling Caffe/Greentea ====").ljust(80,"=") + bcolors.ENDC)
        setup.compile_caffe(config.caffe_path)
    
    if config.compile_malis:
        print(bcolors.HEADER + ("==== PYGT: Compiling Malis ====").ljust(80,"=") + bcolors.ENDC)
        setup.compile_malis(config.malis_path)
        
    if (pygtpath != cmdpath):
        os.chdir(cmdpath)
    
    print(bcolors.OKGREEN + ("==== PYGT: Setup finished ====").ljust(80,"=") + bcolors.ENDC)
    sys.exit(0)


def setup_paths(caffe_path, malis_path):
    cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
    if cmd_folder not in sys.path:
        sys.path.append(cmd_folder)
        
    cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], caffe_path + "/python")))
    if cmd_subfolder not in sys.path:
        sys.path.append(cmd_subfolder)
        
    cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], malis_path)))
    if cmd_subfolder not in sys.path:
        sys.path.append(cmd_subfolder)
    
    sys.path.append(caffe_path + "/python")
    sys.path.append(malis_path + "/python")


def linux_distribution():
    try:
        return platform.linux_distribution()
    except:
        return "N/A"


def sys_info():
    print("""Python version: %s
    dist: %s
    linux_distribution: %s
    system: %s
    machine: %s
    platform: %s
    uname: %s
    version: %s
    mac_ver: %s
    """ % (
           sys.version.split('\n'),
           str(platform.dist()),
           linux_distribution(),
           platform.system(),
           platform.machine(),
           platform.platform(),
           platform.uname(),
           platform.version(),
           platform.mac_ver(),
           ))


def install_dependencies():
    # We support Fedora (22/23/24) and Ubuntu (14.04/15.04/16.04)
    if (linux_distribution()[0].lower() == "fedora"):
        # TODO: Add missing Fedora packages
        packages = ['git', 'gcc', 'protobuf-python', 'protobuf-c',
                    'protobuf-compiler', 'boost-system', 'boost-devel',
                    'boost-python', 'glog', 'glog-devel', 'gflags',
                    'gflags-devel', 'python', 'python-devel', 'python-pip',
                    'atlas', 'atlas-sse2', 'atlas-sse3', 'openblas',
                    'openblas-devel', 'openblas-openmp64', 'openblas-openmp',
                    'openblas-threads64', 'openblas-threads', 'opencl-headers']
        # for package in packages:
        #     subprocess.call(['dnf', 'install', '-y', package])
        subprocess.call(['dnf', 'install', '-y'] + packages)
    if (linux_distribution()[0].lower() == "ubuntu"):
        # TODO: Add missing Ubuntu packages
        packages = ['git', 'gcc', 'libprotobuf-dev', 'libleveldb-dev',
                    'libsnappy-dev', 'libopencv-dev', 'libboost-all-dev',
                    'libhdf5-serial-dev', 'protobuf-compiler', 'gfortran',
                    'libjpeg62', 'libfreeimage-dev', 'libatlas-base-dev',
                    'libopenblas-base', 'libopenblas-dev', 'libgoogle-glog-dev',
                    'libbz2-dev', 'libxml2-dev', 'libxslt-dev', 'libffi-dev',
                    'libssl-dev', 'libgflags-dev', 'liblmdb-dev', 'python-dev',
                    'python-pip', 'python-yaml', 'libviennacl-dev', 'opencl-headers']
        subprocess.call(['apt-get', 'update' '-y'])
        # for package in packages:
        #     subprocess.call(['apt-get', 'install', '-y', package])
        subprocess.call(['apt-get', 'install', '-y'] + packages)
    
    subprocess.call(['pip', 'install', '--upgrade', 'pip'])
    subprocess.call(['pip', 'install', 'cython'])
   
def compile_malis(path):
    cwd = os.getcwd()
    os.chdir(path)
    subprocess.call(['sh', 'make.sh'])
    os.chdir(cwd)

def compile_caffe(path):
    cpus = multiprocessing.cpu_count()
    cwd = os.getcwd()
    os.chdir(path)
    # Copy the default Caffe configuration if not existing
    subprocess.call(['cp', '-n', 'Makefile.config.example', 'Makefile.config'])
    result = subprocess.call(['make', 'all', '-j' + str(cpus)])
    if result != 0:
        sys.exit(result)
    result = subprocess.call(['make', 'pycaffe', '-j' + str(cpus)])
    if result != 0:
        sys.exit(result)
    os.chdir(cwd)
    
def clone_malis(path, clone, update):
    if clone:
        subprocess.call(['git', 'clone', 'https://github.com/srinituraga/malis.git', path])
    if update:
        cwd = os.getcwd()
        os.chdir(path)
        subprocess.call(['git', 'pull'])
        os.chdir(cwd)

def clone_caffe(path, clone, update):
    if clone:
        subprocess.call(['git', 'clone', 'https://github.com/naibaf7/caffe.git', path])
    if update:
        cwd = os.getcwd()
        os.chdir(path)
        subprocess.call(['git', 'pull'])
        os.chdir(cwd)
        

def set_environment_vars():
    # Fix up OpenCL variables. Can interfere with the
    # frame buffer if the GPU is also a display driver
    os.environ["GPU_MAX_ALLOC_PERCENT"] = "100"
    os.environ["GPU_SINGLE_ALLOC_PERCENT"] = "100"
    os.environ["GPU_MAX_HEAP_SIZE"] = "100"
    os.environ["GPU_FORCE_64BIT_PTR"] = "1"
