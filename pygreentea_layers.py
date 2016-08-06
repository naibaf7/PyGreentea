import copy, math

# Import pycaffe
import caffe
import caffe.net_spec as net_spec

from collections import OrderedDict, Counter

from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2


class MetaLayers(object):
    def __getattr__(self, name):
        def metalayer_fn(*args, **kwargs):
            fn = None
            netconf = NetConf()
            netconf.parse(kwargs)
            if (name == 'UNet'):
                unetconf = UNetConf()
                unetconf.parse(kwargs)
                fn = implement_usknet(args[0], netconf, unetconf)
            elif (name == 'SKNet'):
                sknetconf = SKNetConf()
                sknetconf.parse(kwargs)
                fn = implement_sknet(args[0], netconf, sknetconf)
            elif (name == 'USKNet'):
                unetconf = UNetConf()
                unetconf.parse(kwargs)
                fn = implement_usknet(args[0], netconf, unetconf)
            return fn
        return metalayer_fn
    
class SKNetConf:
    # SK-Net convolution steps (may change if necessary)
    conv = [[8],[6],[4]]
    # Feature map increase rule
    fmap_inc_rule = lambda self,fmaps: int(math.ceil(float(fmaps) * 1.5))
    # Number of 1x1 (IP) Convolution steps
    ip_depth = 2
    # Feature map increase rule from SK-Convolution to IP
    fmap_bridge_rule = lambda self,fmaps: int(math.ceil(float(fmaps) * 4))
    # Feature map decrease rule within IP
    fmap_dec_rule = lambda self,fmaps: int(math.ceil(float(fmaps) / 2.5))
    # Network padding
    padding = [44]
    
    def parse(self, params):
        if ('conv' in params):
            self.conv = params['conv']
        if ('fmap_inc_rule' in params):
            self.fmap_inc_rule = params['fmap_inc_rule']
        if ('fmap_dec_rule' in params):
            self.fmap_dec_rule = params['fmap_dec_rule']
        if ('ip_depth' in params):
            self.ip_depth = params['ip_depth']
        if ('fmap_bridge_rule' in params):
            self.fmap_bridge_rule = params['fmap_bridge_rule']
        if ('padding' in params):
            self.padding = params['padding']
    
class UNetConf:
    # Number of U-Net Pooling-Convolution downsampling/upsampling steps
    depth = 3
    # Feature map increase rule (downsampling)
    fmap_inc_rule = lambda self,fmaps: int(math.ceil(float(fmaps) * 3))
    # Feature map decrease rule (upsampling)
    fmap_dec_rule = lambda self,fmaps: int(math.ceil(float(fmaps) / 3))
    # Skewed U-Net downsampling strategy
    downsampling_strategy = [[2],[2],[2]]
    # U-Net convolution setup (downsampling path)
    conv_down = [[[3],[3]]]
    # U-Net convolution setup (upsampling path)
    conv_up = [[[3],[3]]]
    # SK-Net configurations
    sknetconfs = []
    # Upsampling path with deconvolutions instead of convolutions
    use_deconv_uppath = False
    
    def parse(self, params):
        if ('depth' in params):
            self.depth = params['depth']
        if ('fmap_inc_rule' in params):
            self.fmap_inc_rule = params['fmap_inc_rule']
        if ('fmap_dec_rule' in params):
            self.depth = params['fmap_dec_rule']
        if ('downsampling_strategy' in params):
            self.downsampling_strategy = params['downsampling_strategy']
        if ('conv_down' in params):
            self.conv_down = params['conv_down']
        if ('conv_up' in params):
            self.conv_up = params['conv_up']
        if ('use_deconv_uppath' in params):
            self.use_deconv_uppath = params['use_deconv_uppath']
        if ('sknetconfs' in params):
            for sknetconf_dict in params['sknetconfs']:
                if (sknetconf_dict != None):
                    self.sknetconfs += [SKNetConf()]
                    self.sknetconfs[-1].parse(sknetconf_dict)
            
class NetConf:
    # Number of feature maps in the start
    fmap_start = 16
    # ReLU negative slope
    relu_slope = 0.005
    # Batch normalization
    use_batchnorm = False
    # Batch normalization moving average fraction
    batchnorm_maf = 0.95
    # Dropout
    dropout = 0.2
    
    def parse(self, params):
        if ('fmap_start' in params):
            self.fmap_start = params['fmap_start']
        if ('relu_slope' in params):
            self.relu_slope = params['relu_slope']
        if ('use_batchnorm' in params):
            self.use_batchnorm = params['use_batchnorm']
        if ('batchnorm_maf' in params):
            self.batchnorm_maf = params['batchnorm_maf']
        if ('dropout' in params):
            self.dropout = params['dropout']


def deconv_relu(netconf, bottom, num_output, kernel_size=[3], stride=[1], pad=[0], dilation=[1], group=1):
    deconv = L.Deconvolution(bottom, convolution_param=dict(kernel_size=kernel_size, stride=stride, dilation=dilation,
                                num_output=num_output, pad=pad, group=group,
                                weight_filler=dict(type='msra'),
                                bias_filler=dict(type='constant')), param=[dict(lr_mult=1),dict(lr_mult=2)])
    
    relu = L.ReLU(deconv, in_place=True, negative_slope=netconf.relu_slope)
    last = relu
    
    if (netconf.dropout > 0):
        drop = L.Dropout(last, in_place=True, dropout_ratio=netconf.dropout)
        last = drop
    
    if (netconf.use_batchnorm == True):
        bnltrain = L.BatchNorm(last, in_place=True, include=[dict(phase=0)],
                          param=[dict(lr_mult=0,decay_mult=0),dict(lr_mult=0,decay_mult=0),dict(lr_mult=0,decay_mult=0)],
                          batch_norm_param=dict(use_global_stats=False, moving_average_fraction=netconf.batchnorm_maf))
        bnltest = L.BatchNorm(last, in_place=True, include=[dict(phase=1)],
                          param=[dict(lr_mult=0,decay_mult=0),dict(lr_mult=0,decay_mult=0),dict(lr_mult=0,decay_mult=0)],
                          batch_norm_param=dict(use_global_stats=True, moving_average_fraction=netconf.batchnorm_maf))
        last = {bnltrain, bnltest}  
    return last

# Convolution block. Order of operations:
# 1. Convolution
# 3. Dropout
# 4. Batchnorm
# 5. ReLU
def conv_relu(netconf, bottom, num_output, in_place=False, kernel_size=[3], stride=[1], pad=[0], dilation=[1], group=1):           
    conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride, dilation=dilation,
                                num_output=num_output, pad=pad, group=group,
                                param=[dict(lr_mult=1),dict(lr_mult=2)],
                                weight_filler=dict(type='msra'),
                                bias_filler=dict(type='constant'))
    last = conv
           
    # Dropout
    if (netconf.dropout > 0):
        drop = L.Dropout(last, in_place=in_place, dropout_ratio=netconf.dropout)
        last = drop
    
    # Batchnorm
    if (netconf.use_batchnorm == True):
        bnltrain = L.BatchNorm(last, in_place=in_place, include=[dict(phase=0)],
                          param=[dict(lr_mult=0,decay_mult=0),dict(lr_mult=0,decay_mult=0),dict(lr_mult=0,decay_mult=0)],
                          batch_norm_param=dict(use_global_stats=False, moving_average_fraction=netconf.batchnorm_maf))
        bnltest = L.BatchNorm(last, in_place=in_place, include=[dict(phase=1)],
                          param=[dict(lr_mult=0,decay_mult=0),dict(lr_mult=0,decay_mult=0),dict(lr_mult=0,decay_mult=0)],
                          batch_norm_param=dict(use_global_stats=True, moving_average_fraction=netconf.batchnorm_maf))
        last = {bnltrain, bnltest}

    # Activation
    relu = L.ReLU(last, in_place=in_place, negative_slope=netconf.relu_slope)
    last = relu
    
    return last
    
def convolution(bottom, num_output, kernel_size=[3], stride=[1], pad=[0], dilation=[1], group=1):      
    return L.Convolution(bottom, kernel_size=kernel_size, stride=stride, dilation=dilation,
                                num_output=num_output, pad=pad, group=group,
                                param=[dict(lr_mult=1),dict(lr_mult=2)],
                                weight_filler=dict(type='msra'),
                                bias_filler=dict(type='constant'))
    
def max_pool(netconf, bottom, kernel_size=[2], stride=[2], pad=[0], dilation=[1]):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=kernel_size, stride=stride, pad=pad, dilation=dilation)
    
def upconv(netconf, bottom, num_output_dec, num_output_conv, kernel_size=[2], stride=[2]):    
    deconv = L.Deconvolution(bottom, convolution_param=dict(num_output=num_output_dec, kernel_size=kernel_size, stride=stride, pad=[0], group=num_output_dec,
                                                            weight_filler=dict(type='constant', value=1), bias_term=False),
                             param=dict(lr_mult=0, decay_mult=0))
    
    conv = L.Convolution(deconv, num_output=num_output_conv, kernel_size=[1], stride=[1], pad=[0], group=1,
                            param=[dict(lr_mult=1),dict(lr_mult=2)],
                            weight_filler=dict(type='msra'),
                            bias_filler=dict(type='constant'))
    return conv
    
def mergecrop(bottom_a, bottom_b, op = 'stack'):
    return L.MergeCrop(bottom_a, bottom_b, forward=[1,1], backward=[1,1], operation=(0 if (op == 'stack') else 1))

    
def implement_sknet(bottom, netconf, sknetconf):
    blobs = [bottom]
    fmaps = [netconf.fmap_start]
    dilation = [1 for i in range(0,len(sknetconf.padding))]
    print(sknetconf.padding)
    sw_shape = [sknetconf.padding[min(i,len(sknetconf.padding)-1)] + 1 for i in range(0,len(sknetconf.padding))]
    for i in range(0, len(sknetconf.conv)):
        final_ksize = [sknetconf.conv[i][min(j, len(sknetconf.conv[i])-1)] for j in range(0,len(sw_shape))]
        for j in range(0, len(sw_shape)):
            if(not (sw_shape[j] - (final_ksize[j] - 1)) % 2 == 0):
                final_ksize[j] += 1
            sw_shape[j] = (sw_shape[j] - (final_ksize[j] - 1)) / 2
        conv = conv_relu(netconf, blobs[-1], fmaps[-1], kernel_size=final_ksize, dilation=dilation)
        blobs = blobs + [conv]
        pool = max_pool(netconf, blobs[-1], kernel_size=[2], stride=[1], dilation=dilation)
        dilation = [2 * d for d in dilation]
        blobs = blobs + [pool]
        if (i < len(sknetconf.conv) - 1):
            fmaps = fmaps + [sknetconf.fmap_inc_rule(fmaps[-1])]

    fmaps = fmaps + [sknetconf.fmap_bridge_rule(fmaps[-1])]
    # 1st IP layer
    conv = conv_relu(netconf, blobs[-1], fmaps[-1], kernel_size=sw_shape)
    blobs = blobs + [conv]

    # Remaining IP layers
    for i in range(0, sknetconf.ip_depth - 1):
        fmaps = fmaps + [sknetconf.fmap_dec_rule(fmaps[-1])]
        conv = conv_relu(netconf, blobs[-1], fmaps[-1], kernel_size=[1])
        blobs = blobs + [conv]    
    return blobs[-1]
            

def implement_usknet(bottom, netconf, unetconf): 
    blobs = [bottom]
    mergecrop_tracker = []
    fmaps = [netconf.fmap_start]
    pad_shape = [[0 for k in range(0, len(unetconf.conv_down[0][0]))] for i in range(0, unetconf.depth + 1)]
    if unetconf.depth > 0:
        # U-Net downsampling; 2*Convolution+Pooling
        for i in range(0, unetconf.depth):
            convolution_config = unetconf.conv_down[min(i,len(unetconf.conv_down) - 1)]
            for j in range(0,len(convolution_config)):
                conv = conv_relu(netconf, blobs[-1], fmaps[-1], kernel_size=convolution_config[j])
                blobs = blobs + [conv]
                for k in range(0, len(unetconf.conv_down[0][0])):
                    pad_shape[i][k] += (convolution_config[j][min(k, len(convolution_config[j]) - 1)] - 1)

            mergecrop_tracker += [len(blobs)-1]
            pool = max_pool(netconf, blobs[-1], kernel_size=unetconf.downsampling_strategy[i], stride=unetconf.downsampling_strategy[i])
            blobs = blobs + [pool]
            fmaps = fmaps + [unetconf.fmap_inc_rule(fmaps[-1])]
    
    # If there is no SK-Net component, fill with normal convolutions
    if (unetconf.depth > 0 and (len(unetconf.sknetconfs) - 1 < unetconf.depth or unetconf.sknetconfs[unetconf.depth] == None)):
        convolution_config = unetconf.conv_down[min(unetconf.depth, len(unetconf.conv_down) - 1)]
        for j in range(0,len(convolution_config)):
            # Here we are at the bottom, so the second half of the convolutions already belongs to the up-path
            if (unetconf.use_deconv_uppath and j >= len(convolution_config)/2):
                conv = conv_relu(netconf, blobs[-1], fmaps[-1], kernel_size=convolution_config[j], pad=[convolution_config[j][k] - 1 for k in range(0,len(convolution_config[j]))])
                blobs = blobs + [conv]
            else:
                conv = conv_relu(netconf, blobs[-1], fmaps[-1], kernel_size=convolution_config[j])
                blobs = blobs + [conv]
                for k in range(0, len(unetconf.conv_down[0][0])):
                    pad_shape[unetconf.depth][k] += (convolution_config[j][min(k, len(convolution_config[j]) - 1)] - 1)
    else:
        netconf_sk = copy.deepcopy(netconf)
        netconf_sk.fmap_start = fmaps[-1]
        sknetconf_sk = copy.deepcopy(unetconf.sknetconfs[unetconf.depth])
        sknetconf_sk.padding = [sknetconf_sk.padding[min(i,len(sknetconf_sk.padding) - 1)] for i in range(0,len(pad_shape[unetconf.depth]))]
        blobs = blobs + [implement_sknet(blobs[-1], netconf_sk, sknetconf_sk)]
        for k in range(0, len(unetconf.conv_down[0][0])):
            pad_shape[unetconf.depth][k] += sknetconf_sk.padding[k]
    if unetconf.depth > 0:
        # U-Net upsampling; Upconvolution+MergeCrop+2*Convolution
        for i in range(0, unetconf.depth):
            conv = upconv(netconf, blobs[-1], fmaps[-1], unetconf.fmap_dec_rule(fmaps[-1]), kernel_size=unetconf.downsampling_strategy[unetconf.depth - i - 1],
                                       stride=unetconf.downsampling_strategy[unetconf.depth - i - 1])
            blobs = blobs + [conv]
            fmaps = fmaps + [unetconf.fmap_dec_rule(fmaps[-1])]
            
            pre_merge_blobs = [blobs[mergecrop_tracker[unetconf.depth - i - 1]]]
            
            # Insert SK-Net in the mergecrop bridge
            if (len(unetconf.sknetconfs) > unetconf.depth - i - 1 and unetconf.sknetconfs[unetconf.depth - i - 1] != None):
                netconf_sk = copy.deepcopy(netconf)
                netconf_sk.fmap_start = fmaps[-1]
                sknetconf_sk = copy.deepcopy(unetconf.sknetconfs[unetconf.depth - i - 1])
                sknetconf_sk.padding = [0 for k in range(0, len(unetconf.conv_down[0][0]))]
                for j in range(unetconf.depth - i, unetconf.depth + 1):
                    for k in range(0, len(unetconf.conv_down[0][0])):
                        sknetconf_sk.padding[k] += pad_shape[j][k] * (j - (unetconf.depth - i - 1)) * 2
                pre_merge_blobs += [implement_sknet(pre_merge_blobs[-1], netconf_sk, sknetconf_sk)]

            # Here, layer (2 + 3 * i) with reversed i (high to low) is picked
            mergec = mergecrop(blobs[-1], pre_merge_blobs[-1])
            blobs = blobs + [mergec]
            
            convolution_config = unetconf.conv_up[min(unetconf.depth - i - 1, len(unetconf.conv_up) - 1)]
            for j in range(0,len(convolution_config)):
                pad =  [convolution_config[j][k] - 1 for k in range(0,len(convolution_config[j]))] if (unetconf.use_deconv_uppath) else [0]                       
                conv = conv_relu(netconf, blobs[-1], fmaps[-1], kernel_size=convolution_config[j], pad=pad)
                blobs = blobs + [conv]
                for k in range(0, len(unetconf.conv_up[0][0])):
                    pad_shape[unetconf.depth - i - 1][k] += (convolution_config[j][min(k, len(convolution_config[j]) - 1)] - 1)
    # Return the last blob of the network (goes to error objective)
    return blobs[-1]
    
    

metalayers = MetaLayers()
