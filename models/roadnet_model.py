# Author: Yahui Liu <yahui.liu@uintn.it>

import torch
import numpy as np
import itertools
from .base_model import BaseModel
import torch.nn.functional as F
from .roadnet_networks import define_roadnet

class RoadNetModel(BaseModel):
    """
    This class implements the RoadNet model.
    RoadNet paper: https://ieeexplore.ieee.org/document/8506600
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options."""
        return parser

    def __init__(self, opt):
        """Initialize the RoadNet class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['segment', 'segment_l2', 'edge', 'edge_l2', 'centerline', 'centerline_l2']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['image', 'label_gt', 'label_pred']
        # specify the models you want to save to the disk. 
        self.model_names = ['G']

        # define networks 
        self.netG = define_roadnet(opt.input_nc, 
                                   opt.output_nc,
                                   opt.ngf, 
                                   opt.norm,
                                   opt.use_selu,
                                   opt.init_type, 
                                   opt.init_gain, 
                                   self.gpu_ids)

        if self.isTrain:
            # define loss functions
            #self.criterionL2 = torch.nn.MSELoss(size_average=True, reduce=True)
            self.weight_segment_side = [0.5, 0.75, 1.0, 0.75, 0.5, 1.0]
            self.weight_others_side = [0.5, 0.75, 1.0, 0.75, 1.0]

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, eps=1e-3, weight_decay=2e-4)
            #self.optimizer = torch.optim.SGD(self.netG.parameters(), lr=opt.lr, momentum=0.9, weight_decay=2e-4)
            self.optimizers.append(self.optimizer)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.image          = input['image'].to(self.device)
        self.segment_gt     = input['segment'].to(self.device)
        self.edge_gt        = input['edge'].to(self.device)
        self.centerline_gt  = input['centerline'].to(self.device)
        self.image_paths    = input['A_paths']

    def _class_balanced_sigmoid_cross_entropy(self, logits, label):
        """
        The class-balanced cross entropy loss, as in `Holistically-Nested Edge Detection
        <http://arxiv.org/abs/1504.06375>`_.
        Args:
            logits: of shape [N,1,H,W].
            label: of the same shape. the ground truth in {0,1}.
        Returns:
            class-balanced cross entropy loss.
        """
        count_neg = torch.sum(1 - label)
        count_pos = torch.sum(label)
        beta = (count_neg/(count_neg+count_pos)).detach()

        #critic = torch.nn.BCEWithLogitsLoss(size_average=True, reduce=True, pos_weight=pos_weight)
        sigmoid_logits = torch.sigmoid(logits)
        loss = -beta*label*sigmoid_logits.log()-(1-beta)*(1-label)*(1-sigmoid_logits).log()
        return loss.mean()

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.segments, self.edges, self.centerlines = self.netG(self.image)

        # for visualization
        segment_gt_viz     = (self.segment_gt-0.5)/0.5
        edge_gt_viz        = (self.edge_gt-0.5)/0.5
        centerline_gt_viz = (self.centerline_gt-0.5)/0.5
        self.label_gt = torch.cat([centerline_gt_viz, edge_gt_viz, segment_gt_viz], dim=1)

        segment_fused      = (torch.sigmoid(self.segments[-1])-0.5)/0.5
        edge_fused         = (torch.sigmoid(self.edges[-1])-0.5)/0.5
        centerlines_fused  = (torch.sigmoid(self.centerlines[-1])-0.5)/0.5
        self.label_pred = torch.cat([centerlines_fused, edge_fused, segment_fused], dim=1)

    def backward(self):
        """Calculate the loss"""

        self.loss_segment = 0.0
        for out, w in zip(self.segments, self.weight_segment_side):
            loss_segment_side = self._class_balanced_sigmoid_cross_entropy(out, self.segment_gt) * w
            print(w, loss_segment_side.date,  out.mean().data, self.segment_gt.mean().data)
            self.loss_segment += loss_segment_side
        self.loss_segment_l2 = torch.mean((torch.sigmoid(self.segments[-1])-self.segment_gt)**2) * 0.5

        self.loss_edge = 0.0
        for out, w in zip(self.edges, self.weight_others_side):
            self.loss_edge += self._class_balanced_sigmoid_cross_entropy(out, self.edge_gt) * w
        self.loss_edge_l2 = torch.mean((torch.sigmoid(self.edges[-1])-self.edge_gt)**2) * 0.5

        self.loss_centerline = 0.0
        for out, w in zip(self.centerlines, self.weight_others_side):
            self.loss_centerline += self._class_balanced_sigmoid_cross_entropy(out, self.centerline_gt) * w
        self.loss_centerline_l2 = torch.mean((torch.sigmoid(self.centerlines[-1])-self.centerline_gt)**2) * 0.5

        self.loss_total = self.loss_segment + self.loss_edge + self.loss_centerline + \
            self.loss_segment_l2 + self.loss_edge_l2 + self.loss_centerline_l2
        self.loss_total.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute predictions.
        self.optimizer.zero_grad()  # set G's gradients to zero
        self.backward()             # calculate gradients for G
        self.optimizer.step()       # update G's weights
