#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Author: Armin Masoumian (masoumian.armin@gmail.com)


import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Conv1x1, Conv3x3, CRPBlock, upsample
from .layers import GraphConvolution, GCN
import math
import scipy.sparse as sp
import numpy as np
import networkx as nx
import pandas as pd
from scipy.sparse import identity


def normalize(x):
    rowsum = np.array(x.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    x = r_mat_inv.dot(x)
    return x


nfeat = 256
nhid = 320
nhid2 = 1280
p = 0.7
nclass = 1

G = nx.generators.random_graphs.gnp_random_graph(nhid,p)
adj = nx.adjacency_matrix(G)
xx = identity(320).toarray()
adj = normalize(adj + xx)
adj = torch.from_numpy(adj).float().to('cuda:0')

class DepthDecoder(nn.Module):
    def __init__(self,  num_ch_enc):
        super(DepthDecoder, self).__init__()

        bottleneck = 256
        stage = 4
        self.do = nn.Dropout(p=0.5)

        self.reduce4 = Conv1x1(num_ch_enc[4], 512, bias=False)
        self.reduce3 = Conv1x1(num_ch_enc[3], bottleneck, bias=False)
        self.reduce2 = Conv1x1(num_ch_enc[2], bottleneck, bias=False)
        self.reduce1 = Conv1x1(num_ch_enc[1], bottleneck, bias=False)

        self.iconv4 = Conv3x3(512, bottleneck)
        self.iconv3 = Conv3x3(bottleneck*2+1, bottleneck)
        self.iconv2 = Conv3x3(bottleneck*2+1, bottleneck)
        self.iconv1 = Conv3x3(bottleneck*2+1, bottleneck)

        self.crp4 = self._make_crp(bottleneck, bottleneck, stage)
        self.crp3 = self._make_crp(bottleneck, bottleneck, stage)
        self.crp2 = self._make_crp(bottleneck, bottleneck, stage)
        self.crp1 = self._make_crp(bottleneck, bottleneck, stage)

        self.merge4 = Conv3x3(bottleneck, bottleneck)
        self.merge3 = Conv3x3(bottleneck, bottleneck)
        self.merge2 = Conv3x3(bottleneck, bottleneck)
        self.merge1 = Conv3x3(bottleneck, bottleneck)

        # disp
        self.disp4 = nn.Sequential(Conv3x3(bottleneck, 1), nn.Sigmoid())
        self.disp3 = nn.Sequential(Conv3x3(bottleneck, 1), nn.Sigmoid())
        self.disp2 = nn.Sequential(Conv3x3(bottleneck, 1), nn.Sigmoid())
        self.disp1 = nn.Sequential(Conv3x3(bottleneck, 1), nn.Sigmoid())

        # GCN
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.gc3 = GraphConvolution(nfeat, nhid2)
        self.gc4 = GraphConvolution(nhid2, nclass)


    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes,stages)]
        return nn.Sequential(*layers)

    def forward(self, input_features, frame_id=0):
        self.outputs = {}
        l0, l1, l2, l3, l4 = input_features
        l4 = self.do(l4)
        l3 = self.do(l3)
        x4 = self.reduce4(l4)
        x4 = self.iconv4(x4)
        x4 = F.leaky_relu(x4)
        x4 = self.crp4(x4)
        x4 = self.merge4(x4)
        x4 = F.leaky_relu(x4)
        y4 = x4.view(10*32,-1)
        y4 = self.gc1(y4, adj)
        y3 = self.gc2(y4, adj)
        y4 = y3.view(1, 1, 10, 32)
        y4 = self.do(y4)
        disp4 = upsample(y4)
        x4 = upsample(x4)

 
        z3 = torch.transpose(y3, 0, 1)
        yy = torch.matmul(y3, z3)
        yy = yy.cpu()
        yy = yy.detach().numpy()
        yy = normalize(yy)
        yy = torch.from_numpy(yy).float().to('cuda:0')
        yy = yy.view(1, 1, 320, 320)
        yy = F.interpolate(yy, scale_factor=4, mode="nearest")
        yy = yy.view(1280,-1)

        x3 = self.reduce3(l3)
        x3 = torch.cat((x3, x4, disp4), 1)
        x3 = self.iconv3(x3)
        x3 = F.leaky_relu(x3)
        x3 = self.crp3(x3)
        x3 = self.merge3(x3)
        x3 = F.leaky_relu(x3)
        y5 = x3.view(20*64,-1)
        y5 = self.gc3(y5, yy)
        y5 = self.gc4(y5, yy)
        y5 = y5.view(1, 1, 20, 64)
        y5 = self.do(y5)
        disp3 = upsample(y5)
        x3 = upsample(x3)


        x2 = self.reduce2(l2)
        x2 = torch.cat((x2, x3 , disp3), 1)
        x2 = self.iconv2(x2)
        x2 = F.leaky_relu(x2)
        x2 = self.crp2(x2)
        x2 = self.merge2(x2)
        x2 = F.leaky_relu(x2)
        x2 = upsample(x2)
        disp2 = self.disp2(x2)


        x1 = self.reduce1(l1)
        x1 = torch.cat((x1, x2, disp2), 1)
        x1 = self.iconv1(x1)
        x1 = F.leaky_relu(x1)
        x1 = self.crp1(x1)
        x1 = self.merge1(x1)
        x1 = F.leaky_relu(x1)
        x1 = upsample(x1)
        disp1 = self.disp1(x1)

        self.outputs[("disp", frame_id, 3)] = disp4
        self.outputs[("disp", frame_id, 2)] = disp3
        self.outputs[("disp", frame_id, 1)] = disp2
        self.outputs[("disp", frame_id, 0)] = disp1

        return self.outputs
