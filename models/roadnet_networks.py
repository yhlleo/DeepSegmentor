#! -*- coding: utf-8 -*-
# Author: Yahui Liu <yahui.liu@unitn.it>

"""
Reference:

RoadNet: Learning to Comprehensively Analyze Road Networks in Complex Urban Scenes 
  From High-Resolution Remotely Sensed Images.
  https://ieeexplore.ieee.org/document/8506600
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import get_norm_layer, init_net

class RoadNet(nn.Module):
    def __init__(self, in_nc, out_nc, ngf, norm='batch', use_selu=1):
        super(RoadNet, self).__init__()
        norm_layer = get_norm_layer(norm_type=norm)

        #------------road surface segmentation------------#
        self.segment_conv1 = nn.Sequential(*self._conv_block(in_nc, ngf, norm_layer, use_selu, num_block=2))
        self.side_segment_conv1 = nn.Conv2d(ngf, out_nc, kernel_size=1, stride=1, bias=False)

        self.segment_conv2 = nn.Sequential(*self._conv_block(ngf, ngf*2, norm_layer, use_selu, num_block=2))
        self.side_segment_conv2 = nn.Conv2d(ngf*2, out_nc, kernel_size=1, stride=1, bias=False)

        self.segment_conv3 = nn.Sequential(*self._conv_block(ngf*2, ngf*4, norm_layer, use_selu, num_block=3))
        self.side_segment_conv3 = nn.Conv2d(ngf*4, out_nc, kernel_size=1, stride=1, bias=False)

        self.segment_conv4 = nn.Sequential(*self._conv_block(ngf*4, ngf*8, norm_layer, use_selu, num_block=3))
        self.side_segment_conv4 = nn.Conv2d(ngf*8, out_nc, kernel_size=1, stride=1, bias=False)

        self.segment_conv5 = nn.Sequential(*self._conv_block(ngf*8, ngf*8, norm_layer, use_selu, num_block=3))
        self.side_segment_conv5 = nn.Conv2d(ngf*8, out_nc, kernel_size=1, stride=1, bias=False)

        self.fuse_segment_conv = nn.Conv2d(out_nc*5, out_nc, kernel_size=1, stride=1, bias=False)

        ngf2 = ngf//2
        #------------road edge detection------------#
        self.edge_conv1 = nn.Sequential(*self._conv_block(in_nc+out_nc, ngf2, norm_layer, use_selu, num_block=2))
        self.side_edge_conv1 = nn.Conv2d(ngf2, out_nc, kernel_size=1, stride=1, bias=False) 

        self.edge_conv2 = nn.Sequential(*self._conv_block(ngf2, ngf2*2, norm_layer, use_selu, num_block=2))
        self.side_edge_conv2 = nn.Conv2d(ngf2*2, out_nc, kernel_size=1, stride=1, bias=False)

        self.edge_conv3 = nn.Sequential(*self._conv_block(ngf2*2, ngf2*4, norm_layer, use_selu, num_block=2))
        self.side_edge_conv3 = nn.Conv2d(ngf2*4, out_nc, kernel_size=1, stride=1, bias=False)

        self.edge_conv4 = nn.Sequential(*self._conv_block(ngf2*4, ngf2*8, norm_layer, use_selu, num_block=2))
        self.side_edge_conv4 = nn.Conv2d(ngf2*8, out_nc, kernel_size=1, stride=1, bias=False)

        self.fuse_edge_conv = nn.Conv2d(out_nc*4, out_nc, kernel_size=1, stride=1, bias=False)

        #------------road centerline extraction------------#
        self.centerline_conv1 = nn.Sequential(*self._conv_block(in_nc+out_nc, ngf2, norm_layer, use_selu, num_block=2))
        self.side_centerline_conv1 = nn.Conv2d(ngf2, out_nc, kernel_size=1, stride=1, bias=False) 

        self.centerline_conv2 = nn.Sequential(*self._conv_block(ngf2, ngf2*2, norm_layer, use_selu, num_block=2))
        self.side_centerline_conv2 = nn.Conv2d(ngf2*2, out_nc, kernel_size=1, stride=1, bias=False)

        self.centerline_conv3 = nn.Sequential(*self._conv_block(ngf2*2, ngf2*4, norm_layer, use_selu, num_block=2))
        self.side_centerline_conv3 = nn.Conv2d(ngf2*4, out_nc, kernel_size=1, stride=1, bias=False)

        self.centerline_conv4 = nn.Sequential(*self._conv_block(ngf2*4, ngf2*8, norm_layer, use_selu, num_block=2))
        self.side_centerline_conv4 = nn.Conv2d(ngf2*8, out_nc, kernel_size=1, stride=1, bias=False)

        self.fuse_centerline_conv = nn.Conv2d(out_nc*4, out_nc, kernel_size=1, stride=1, bias=False)

        self.maxpool = nn.MaxPool2d(2, stride=2)

        #self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        #self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        #self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

    def _conv_block(self, in_nc, out_nc, norm_layer, use_selu, num_block=2, kernel_size=3, 
        stride=1, padding=1, bias=True):
        conv = []
        for i in range(num_block):
            cur_in_nc = in_nc if i == 0 else out_nc
            conv += [nn.Conv2d(cur_in_nc, out_nc, kernel_size=kernel_size, stride=stride, 
                               padding=padding, bias=bias)]
            if use_selu:
                conv += [nn.SeLU(True)]
            else:
                conv += [norm_layer(out_nc), nn.ReLU(True)]
        return conv

    def _segment_forward(self, x):
        """
        predict road surface segmentation
        :param: x, image tensor, [N, C, H, W]
        """
        h,w = x.size()[2:]
        # main stream features
        conv1 = self.segment_conv1(x)
        conv2 = self.segment_conv2(self.maxpool(conv1))
        conv3 = self.segment_conv3(self.maxpool(conv2))
        conv4 = self.segment_conv4(self.maxpool(conv3))
        conv5 = self.segment_conv5(self.maxpool(conv4))
        # side output features
        side_output1 = self.side_segment_conv1(conv1)
        side_output2 = self.side_segment_conv2(conv2)
        side_output3 = self.side_segment_conv3(conv3)
        side_output4 = self.side_segment_conv4(conv4)
        side_output5 = self.side_segment_conv5(conv5)
        # upsampling side output features
        side_output2 = F.interpolate(side_output2, size=(h, w), mode='bilinear', align_corners=True) #self.up2(side_output2)
        side_output3 = F.interpolate(side_output3, size=(h, w), mode='bilinear', align_corners=True) #self.up4(side_output3)
        side_output4 = F.interpolate(side_output4, size=(h, w), mode='bilinear', align_corners=True) #self.up8(side_output4)
        side_output5 = F.interpolate(side_output5, size=(h, w), mode='bilinear', align_corners=True) #self.up16(side_output5)

        fused = self.fuse_segment_conv(torch.cat([
            side_output1, 
            side_output2, 
            side_output3,
            side_output4,
            side_output5], dim=1))
        return [side_output1, side_output2, side_output3, side_output4, side_output5, fused]

    def _edge_forward(self, x):
        """
        predict road edge
        :param: x, [image tensor, predicted segmentation tensor], [N, C+1, H, W]
        """
        h, w = x.size()[2:]
        # main stream features
        conv1 = self.edge_conv1(x)
        conv2 = self.edge_conv2(self.maxpool(conv1))
        conv3 = self.edge_conv3(self.maxpool(conv2))
        conv4 = self.edge_conv4(self.maxpool(conv3))
        # side output features
        side_output1 = self.side_edge_conv1(conv1)
        side_output2 = self.side_edge_conv2(conv2)
        side_output3 = self.side_edge_conv3(conv3)
        side_output4 = self.side_edge_conv4(conv4)
        # upsampling side output features
        side_output2 = F.interpolate(side_output2, size=(h, w), mode='bilinear', align_corners=True) #self.up2(side_output2)
        side_output3 = F.interpolate(side_output3, size=(h, w), mode='bilinear', align_corners=True) #self.up4(side_output3)
        side_output4 = F.interpolate(side_output4, size=(h, w), mode='bilinear', align_corners=True) #self.up8(side_output4)
        fused = self.fuse_edge_conv(torch.cat([
            side_output1, 
            side_output2, 
            side_output3,
            side_output4], dim=1))        
        return [side_output1, side_output2, side_output3, side_output4, fused]

    def _centerline_forward(self, x):
        """
        predict road edge
        :param: x, [image tensor, predicted segmentation tensor], [N, C+1, H, W]
        """
        h,w = x.size()[2:]
        # main stream features
        conv1 = self.centerline_conv1(x)
        conv2 = self.centerline_conv2(self.maxpool(conv1))
        conv3 = self.centerline_conv3(self.maxpool(conv2))
        conv4 = self.centerline_conv4(self.maxpool(conv3))
        # side output features
        side_output1 = self.side_centerline_conv1(conv1)
        side_output2 = self.side_centerline_conv2(conv2)
        side_output3 = self.side_centerline_conv3(conv3)
        side_output4 = self.side_centerline_conv4(conv4)
        # upsampling side output features
        side_output2 = F.interpolate(side_output2, size=(h, w), mode='bilinear', align_corners=True) #self.up2(side_output2)
        side_output3 = F.interpolate(side_output3, size=(h, w), mode='bilinear', align_corners=True) #self.up4(side_output3)
        side_output4 = F.interpolate(side_output4, size=(h, w), mode='bilinear', align_corners=True) #self.up8(side_output4)
        fused = self.fuse_centerline_conv(torch.cat([
            side_output1, 
            side_output2, 
            side_output3,
            side_output4], dim=1))
        return [side_output1, side_output2, side_output3, side_output4, fused]

    def forward(self, x):
        segments = self._segment_forward(x)

        x_ = torch.cat([x, segments[-1]], dim=1)
        edges = self._edge_forward(x_)
        centerlines = self._centerline_forward(x_)
        return segments, edges, centerlines

def define_roadnet(in_nc, 
                   out_nc, 
                   ngf, 
                   norm='batch',
                   use_selu=1,
                   init_type='xavier', 
                   init_gain=0.02, 
                   gpu_ids=[]):
    net = RoadNet(in_nc, out_nc, ngf, norm, use_selu)
    return init_net(net, init_type, init_gain, gpu_ids)
