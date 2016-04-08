from __future__ import print_function
import os, sys, inspect
import h5py
import numpy as np
import matplotlib
import random
import math, copy
import six
from Crypto.Random.random import randint
from functools import partial
from collections import OrderedDict, Counter


# Import pycaffe
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.net_spec import Function
from caffe.proto import caffe_pb2

def get_tokens(name):
    tokens = OrderedDict()
    tokenstrings = name.split('_')
    for i in range (0, len(tokenstrings)):
        tokenattr = tokenstrings[i].split('-')
        if(len(tokenattr) == 2):
            tokens[tokenattr[0]] = int(tokenattr[1])
    return tokens

def get_token_value(token_dict_array, token):
    for token_dict in token_dict_array:
        if token in token_dict:
            return token_dict[token]
    return None

class Stack:
    def __init__(self):
        self.__storage = []
        
    def __len__(self):
        return len(self.__storage)

    def isEmpty(self):
        return len(self.__storage) == 0

    def push(self,p):
        self.__storage.append(p)

    def pop(self):
        return self.__storage.pop()


class Graph:
    def __init__(self):
        self.nodes = OrderedDict()
        self.cmpxscale = 0.016
        
    def add_node(self, node):
        node.graph = self
        self.nodes[node.layer.name] = node
    
    def set_netspec(self, netspec):
        self.netspec = netspec
        
    def layer_with_head(self, layer, head, connecting_layers):
        if (head == None and len(layer.tops) > 0 and layer.tops[0].name in self.nodes):
            head = layer
        return (layer, head, connecting_layers);
        
    def connect_graph(self):
        stack = Stack()
        for layer_key in self.netspec.tops:
            layer = self.netspec.tops[layer_key]
            if (type(layer) is Function):
                stack.push(self.layer_with_head(layer, None, []))
            else:
                stack.push(self.layer_with_head(layer.fn, None, []))
                
        while (not(stack.isEmpty())):
            layer, head, connecting_layers = stack.pop()
            
            if (not layer is None and len(layer.tops) > 0):
                print('LAYER: ' + str(layer.tops[0].name) + ' NODE: ' + str(layer.tops[0].name in self.nodes))
            if (not head is None and len(head.tops) > 0):
                print('HEAD: ' + str(head.tops[0].name)  + ' NODE: ' + str(head.tops[0].name in self.nodes))
            
            if len(layer.tops) > 0 and layer.tops[0].name in self.nodes:
                node = self.nodes[layer.tops[0].name]
                if (not head is None and not layer == head and len(head.tops) > 0):
                    head_node = self.nodes[head.tops[0].name]
                    print('From: ' + node.layer.name + ' to ' + head_node.layer.name)
                    edge = Edge(node, head_node, connecting_layers[1:])
                    head = layer
                    connecting_layers = []
                    
            for input in layer.inputs:
                connecting_layers_copy = []
                if not(head == None):
                    connecting_layers_copy = [input.fn] + [l for l in connecting_layers]
                stack.push(self.layer_with_head(input.fn, head, connecting_layers_copy))
                
    def fix_sizes_and_offsets(self):
        stack = Stack()
        stack.push(self.nodes['data'])
        while (not(stack.isEmpty())):
            node = stack.pop()
            node.fix_sizes_and_offsets()
            for i in range(0,len(node.edges_out)):
                stack.push(node.edges_out[i].node_to)
        
    def generate_tikz_graph(self):
        # Infer the edge connections between nodes
        self.connect_graph()
        # Infer front to back positioning and sizes of nodes
        self.fix_sizes_and_offsets()
        
        graph =     """
\\documentclass{article}
\\usepackage{incgraph}
\\usepackage{tikz}
\\usetikzlibrary{shapes,arrows,positioning,shadows,calc,fit,arrows,decorations.pathreplacing,decorations.markings,decorations.pathmorphing,patterns,shapes,shapes.multipart}
\\usepackage[OT1]{fontenc}
\\usepackage[british]{babel}
\\usepackage[utf8]{inputenc}
\\usepackage{graphicx}
\\usepackage{pdfpages}
\\usepackage{float}
\\usepackage[export]{adjustbox}
\\usepackage{algorithm}
\\usepackage[noend]{algpseudocode}
\\usepackage[urldate=long,backend=biber,sorting=none,natbib=true,style=numeric]{biblatex}
\\usepackage{eso-pic}
\\usepackage{xspace}
\\usepackage[justification=centering]{caption}
\\usepackage{siunitx}
\\usepackage{calc}
\\usepackage{pgf}
\\usepackage{fp}
\\pgfdeclarelayer{bg}
\\pgfsetlayers{bg,main}

\\begin{document}
    \\begin{inctext}
        {
        
        \\definecolor{convcolor}{RGB}{255, 127, 14}
        \\definecolor{poolcolor}{RGB}{31, 119, 180}
        \\definecolor{upconvcolor}{RGB}{140, 86, 75}
        \\definecolor{othercolor}{RGB}{219, 219, 141}

        \\tikzstyle{convline} = [line width=3.0pt, draw, -latex', color=convcolor]
        \\tikzstyle{poolline} = [line width=3.0pt, draw, -latex', color=poolcolor]
        \\tikzstyle{upconvline} = [line width=3.0pt, draw, -latex', color=upconvcolor]
        \\tikzstyle{otherline} = [line width=3.0pt, draw, -latex', color=othercolor]
        
        \\tikzset{
        incgraphic/.style n args={3}{
        path picture={
            \\pgfmathparse{(#2)}\\let\\vw\\pgfmathresult
            \\pgfmathparse{(#3)}\\let\\vh\\pgfmathresult
            \\node[anchor=north] at (path picture bounding box.north){
                \\includegraphics*[height=\\vh cm, width=\\vw cm]{dump/#1.png}
            };}, draw, inner sep=0.0cm, text centered, minimum height=#3cm, minimum width=(#2cm)}
        }
        
        \\begin{tikzpicture}[node distance = 1cm and 0.5cm, auto]
                    """
                    
        stack = Stack()
        stack.push(self.nodes['data'])
        while (not(stack.isEmpty())):
            node = stack.pop()
            for i in range(0,len(node.edges_out)):
                stack.push(node.edges_out[i].node_to)
            position = ''
            for i in range(0,len(node.edges_in)):
                if (node.edges_in[i].position != None):
                    position = node.edges_in[i].position
            graph +=    """
            \\node[incgraphic={"""+node.name+"""}{"""+str(node.width)+"""}{"""+str(node.height)+"""}, """+position+""", xshift="""+str(node.xshift)+"""cm, yshift="""+str(node.yshift)+"""cm]("""+node.name+"""){};
                        """
            
            if node.display_fmaps:
                graph +=    """
            \\node[rotate=0, anchor=south, xshift="""+str(node.fmaps_xshift)+"""cm, yshift="""+str(node.fmaps_yshift)+"""cm] at ("""+node.name+""".north) {$"""+str(node.run_shape.fmaps)+"""$};
                            """
            if node.display_size:
                graph +=    """
            \\node[rotate=90, anchor=south west, xshift="""+str(node.size_xshift)+"""cm, yshift="""+str(node.size_yshift)+"""cm] at ("""+node.name+""".south west) {$"""+str(node.run_shape.shape[0])+"""^2$};
                            """
                    
                    
        for key, node in six.iteritems(self.nodes):
            for edge in node.edges_in:
                
                graph +=    """
            \\path["""+edge.type+"""] (""" + edge.node_from.layer.name + """) -- (""" + edge.node_to.layer.name + """);
                            """
        
        graph +=    """
        \\end{tikzpicture}
        }
    \\end{inctext}
\\end{document}
                    """
        print(graph)
        return graph

class Node:
    def __init__(self, layer, run_shape):
        self.graph = None
        self.name = layer.name
        self.tokens = get_tokens(self.name)
        self.run_shape = run_shape
        self.layer = layer
        self.display_fmaps = True
        self.fmaps_xshift = 0
        self.fmaps_yshift = 0
        self.display_size = True
        self.size_xshift = 0
        self.size_yshift = 0
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0
        self.xshift = 0
        self.yshift = 0
        self.edges_in = []
        self.edges_out = []
        
    def fix_sizes_and_offsets(self):
        self.width = self.run_shape.shape[1] * self.graph.cmpxscale
        self.height = self.run_shape.shape[0] * self.graph.cmpxscale
        
class Edge:
    def __init__(self, node_from, node_to, connecting_layers):
        self.graph = node_from.graph
        self.connecting_layers = connecting_layers
        self.node_from = node_from
        self.node_to = node_to
        self.position = None
        self.linetype = None
        if not (node_from == node_to or self.edge_exists()):
            self.node_from.edges_out += [self]
            self.node_to.edges_in += [self]
        self.get_type_and_positions()
            
    def edge_exists(self):
        for edge in self.node_from.edges_out:
            if (edge.node_from == self.node_from and edge.node_to == self.node_to):
                return True
        for edge in self.node_from.edges_in:
            if (edge.node_from == self.node_from and edge.node_to == self.node_to):
                return True
            
    def get_type_and_positions(self):
        
        token_dict_array = []
        #print ('Connecting layers from ' + self.node_from.name + ' to ' + self.node_to.name + ':')
        for layer in self.connecting_layers:
            if (len(layer.tops) > 0 and layer.tops[0].name != None):
                tokens = get_tokens(layer.tops[0].name)
                token_dict_array += [tokens]
                
        token_from = get_tokens(self.node_from.name)
        token_to = get_tokens(self.node_to.name)
                
        self.type = 'convline'
        self.position = 'right =of ' + self.node_from.name
        
        if ('P' in token_to and 'SK' in token_to):
            self.type = 'poolline'
            self.position = 'right =of ' + self.node_from.name
            
        if ('SK' in token_from and 'M' in token_to):
            self.type = 'upconvline'
            self.position = 'above =of ' + self.node_from.name
            self.node_from.display_fmaps = False
        
        if ('P' in token_to and 'SK' not in token_to):
            self.type = 'poolline'
            self.position = 'below =of ' + self.node_from.name
            self.node_to.display_fmaps = False
        
        if ('C' in token_from and not get_token_value(token_dict_array, 'DC') == None and 'M' in token_to):
            self.type = 'upconvline'
            self.position = 'above =of ' + self.node_from.name
            self.node_from.display_fmaps = False
        
        if ('C' in token_from and get_token_value(token_dict_array, 'DC') == None and 'M' in token_to):
            self.type = 'otherline'
            self.position = None
                        