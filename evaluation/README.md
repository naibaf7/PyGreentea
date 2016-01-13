# Evaluation scripts to calculate rand merge error, rand split error in python.  
Most code is python wrappers around code from https://bitbucket.org/poozh/watershed with small modifications.  For use in https://github.com/naibaf7/PyGreentea.

Build with src_cython/cythonBuild.sh.  To use, change settings and filenames in processAndEval.py then run processAndEval.py.  This generates .dat files containing the rand merge and rand split scores.  These can be plotted with plot_rand and plot_f in the visualization folder.