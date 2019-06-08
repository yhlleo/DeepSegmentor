
import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import get_norm_layer, init_net

class DeepCrackNet(nn.Module):
    def __init__(self, in_nc, num_classes, ngf, norm='batch'):
        super(DeepCrackNet, self).__init__()

        norm_layer = get_norm_layer(norm_type=norm)
        conv1 = [nn.Conv2d(in_nc, ngf, kernel_size=3, stride=1, padding=1, bias=False),
                 norm_layer(ngf),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1, bias=False),
                 norm_layer(ngf),
                 nn.ReLU(inplace=True)]
        self.conv1 = nn.Sequential(*conv1)
        self.side_conv1 = nn.Conv2d(ngf, num_classes, kernel_size=1, stride=1, bias=False)

        conv2 = [nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=1, padding=1, bias=False),
                 norm_layer(ngf*2),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(ngf*2, ngf*2, kernel_size=3, stride=1, padding=1, bias=False),
                 norm_layer(ngf*2),
                 nn.ReLU(inplace=True)]
        self.conv2 = nn.Sequential(*conv2)
        self.side_conv2 = nn.Conv2d(ngf*2, num_classes, kernel_size=1, stride=1, bias=False)

        conv3 = [nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=1, padding=1, bias=False),
                 norm_layer(ngf*4),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(ngf*4, ngf*4, kernel_size=3, stride=1, padding=1, bias=False),
                 norm_layer(ngf*4),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(ngf*4, ngf*4, kernel_size=3, stride=1, padding=1, bias=False),
                 norm_layer(ngf*4),
                 nn.ReLU(inplace=True)]
        self.conv3 = nn.Sequential(*conv3)
        self.side_conv3 = nn.Conv2d(ngf*4, num_classes, kernel_size=1, stride=1, bias=False)

        conv4 = [nn.Conv2d(ngf*4, ngf*8, kernel_size=3, stride=1, padding=1, bias=False),
                 norm_layer(ngf*8),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(ngf*8, ngf*8, kernel_size=3, stride=1, padding=1, bias=False),
                 norm_layer(ngf*8),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(ngf*8, ngf*8, kernel_size=3, stride=1, padding=1, bias=False),
                 norm_layer(ngf*8),
                 nn.ReLU(inplace=True)]
        self.conv4 = nn.Sequential(*conv4)
        self.side_conv4 = nn.Conv2d(ngf*8, num_classes, kernel_size=1, stride=1, bias=False)

        conv5 = [nn.Conv2d(ngf*8, ngf*8, kernel_size=3, stride=1, padding=1, bias=False),
                 norm_layer(ngf*8),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(ngf*8, ngf*8, kernel_size=3, stride=1, padding=1, bias=False),
                 norm_layer(ngf*8),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(ngf*8, ngf*8, kernel_size=3, stride=1, padding=1, bias=False),
                 norm_layer(ngf*8),
                 nn.ReLU(inplace=True)]
        self.conv5 = nn.Sequential(*conv5)
        self.side_conv5 = nn.Conv2d(ngf*8, num_classes, kernel_size=1, stride=1, bias=False)

        self.fuse_conv = nn.Conv2d(num_classes*5, num_classes, kernel_size=1, stride=1, bias=False)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

    def forward(self, x):
        h,w = x.size()[2:]
        # main stream features
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.maxpool(conv1))
        conv3 = self.conv3(self.maxpool(conv2))
        conv4 = self.conv4(self.maxpool(conv3))
        conv5 = self.conv5(self.maxpool(conv4))
        # side output features
        side_output1 = self.side_conv1(conv1)
        side_output2 = self.side_conv2(conv2)
        side_output3 = self.side_conv3(conv3)
        side_output4 = self.side_conv4(conv4)
        side_output5 = self.side_conv5(conv5)
        # upsampling side output features
        side_output2 = self.up2(side_output2)
        side_output3 = self.up4(side_output3)
        side_output4 = self.up8(side_output4)
        side_output5 = self.up16(side_output5)

        fused = self.fuse_conv(torch.cat([side_output1, 
                                          side_output2, 
                                          side_output3,
                                          side_output4,
                                          side_output5], dim=1))
        return side_output1, side_output2, side_output3, side_output4, side_output5, fused

def define_deepcrack(in_nc, 
                     num_classes, 
                     ngf, 
                     norm='batch',
                     init_type='xavier', 
                     init_gain=0.02, 
                     gpu_ids=[]):
    net = DeepCrackNet(in_nc, num_classes, ngf, norm)
    return init_net(net, init_type, init_gain, gpu_ids)
