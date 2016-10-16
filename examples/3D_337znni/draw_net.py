from __future__ import print_function
import sys, os
import subprocess
from tornado.process import Subprocess

pygt_path = '../..'
sys.path.append(pygt_path)
import pygreentea.pygreentea as pygt

cmdpath = os.getcwd()

os.chdir(pygt.pycaffepath)
subprocess.call(['python', 'draw_net.py', cmdpath + '/net.prototxt', cmdpath + '/net.ps',
                 '--rankdir', 'TB',
                 '--margin', '0, 0',
                 '--page', '5, 8',
                 '--pagesize', '5, 8',
                 '--size', '5, 8'])

os.chdir(cmdpath)
subprocess.call(['ps2pdf', '-g3600x5760', 'net.ps', 'net.pdf'])

