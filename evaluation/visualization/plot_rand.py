import numpy as np
import pylab as plt
import sys
import os.path as op

def formatAndSave(ax,outputFile):
    plt.xlabel('Rand Split')
    plt.ylabel('Rand Merge')
    #plt.xlim([.5,1])
    #plt.ylim([.5,1])
    plt.legend(bbox_to_anchor=(.4,.4),bbox_transform=plt.gcf().transFigure)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(outputFile)
    ax.grid()
    plt.show()

target_folders = ["null"]
if len(sys.argv)>=3:
    output_file = sys.argv[1]       # /path/to/outputFile.png
    target_folders = sys.argv[2:]   # list of /path/to/folder containing square.dat
else:
    print "not enough args!"
    exit(0)
print "target folders: ",target_folders
print "output file: ",output_file+'.png'
try:
    fig = plt.figure(figsize=(15,8))
    ax = plt.subplot(111)
    for i in range(len(target_folders)):
        target_folder = target_folders[i]
        #outfolder = 'data_tier2/'+train_or_test+'out/'+target_folder
        outfolder = target_folder
        try:
            a=np.fromfile( op.join(outfolder, 'square.dat') )
            data=a.reshape(len(a)/2,2)
            plt.plot( data[:,0], data[:,1], 'o-', label=target_folder)
            plt.hold(True)
        except:
            continue
        '''
        try:
            a=np.fromfile( op.join(outfolder , 'linear.dat') )
            linear=a.reshape(len(a)/2,2)
            plt.plot( linear[:,0], linear[:,1], 'o-', label='linear'+'_'+target_folder) #ms=10
        except:
            continue
        try:
            a=np.fromfile( op.join(outfolder, 'threshold.dat') )
            linear=a.reshape(len(a)/2,2)
            plt.plot( linear[:,0], linear[:,1], 'o-', label='threshold'+'_'+target_folder)
        except:
            continue
        '''
    formatAndSave(ax,output_file)
except:
    formatAndSave(ax,output_file)


