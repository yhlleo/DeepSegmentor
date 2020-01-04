# Author: Yahui Liu <yahui.liu@uintn.it>

import torch
import numpy as np
import itertools
from .base_model import BaseModel
from .deepcrack_networks import define_deepcrack, BinaryFocalLoss

class DeepCrackModel(BaseModel):
    """
    This class implements the DeepCrack model.
    DeepCrack paper: https://www.sciencedirect.com/science/article/pii/S0925231219300566
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options."""
        parser.add_argument('--lambda_side', type=float, default=1.0, help='weight for side output loss')
        parser.add_argument('--lambda_fused', type=float, default=1.0, help='weight for fused loss')
        return parser

    def __init__(self, opt):
        """Initialize the DeepCrack class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['side', 'fused', 'total']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.display_sides = opt.display_sides
        self.visual_names = ['image', 'label_viz', 'fused']
        if self.display_sides:
            self.visual_names += ['side1', 'side2', 'side3', 'side4', 'side5']
        # specify the models you want to save to the disk. 
        self.model_names = ['G']

        # define networks 
        self.netG = define_deepcrack(opt.input_nc, 
                                     opt.num_classes, 
                                     opt.ngf, 
                                     opt.norm,
                                     opt.init_type, 
                                     opt.init_gain, 
                                     self.gpu_ids)

        self.softmax = torch.nn.Softmax(dim=1)

        if self.isTrain:
            # define loss functions
            #self.weight = torch.from_numpy(np.array([0.0300, 1.0000], dtype='float32')).float().to(self.device)
            #self.criterionSeg = torch.nn.CrossEntropyLoss(weight=self.weight)
            if self.opt.loss_mode == 'focal':
                self.criterionSeg = BinaryFocalLoss()
            else: 
                self.criterionSeg = nn.BCEWithLogitsLoss(size_average=True, reduce=True, 
                    pos_weight=torch.tensor(1.0/3e-2).to(self.device))
            self.weight_side = [0.5, 0.75, 1.0, 0.75, 0.5]

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.SGD(self.netG.parameters(), lr=opt.lr, momentum=0.9, weight_decay=2e-4)
            self.optimizers.append(self.optimizer)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.image = input['image'].to(self.device)
        self.label = input['label'].to(self.device)
        #self.label3d = self.label.squeeze(1)
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.outputs = self.netG(self.image)

        # for visualization
        self.label_viz = (self.label.float()-0.5)/0.5
        #self.fused = (self.softmax(self.outputs[-1])[:,1].detach().unsqueeze(1)-0.5)/0.5
        #if self.display_sides:
        #    self.side1 = (self.softmax(self.outputs[0])[:,1].detach().unsqueeze(1)-0.5)/0.5
        #    self.side2 = (self.softmax(self.outputs[1])[:,1].detach().unsqueeze(1)-0.5)/0.5
        #    self.side3 = (self.softmax(self.outputs[2])[:,1].detach().unsqueeze(1)-0.5)/0.5
        #    self.side4 = (self.softmax(self.outputs[3])[:,1].detach().unsqueeze(1)-0.5)/0.5
        #    self.side5 = (self.softmax(self.outputs[4])[:,1].detach().unsqueeze(1)-0.5)/0.5
        self.fused = (torch.sigmoid(self.outputs[-1])-0.5)/0.5
        if self.display_sides:
            self.side1 = (torch.sigmoid(self.outputs[0])-0.5)/0.5
            self.side2 = (torch.sigmoid(self.outputs[1])-0.5)/0.5
            self.side3 = (torch.sigmoid(self.outputs[2])-0.5)/0.5
            self.side4 = (torch.sigmoid(self.outputs[3])-0.5)/0.5
            self.side5 = (torch.sigmoid(self.outputs[4])-0.5)/0.5

    def backward(self):
        """Calculate the loss"""
        lambda_side = self.opt.lambda_side
        lambda_fused = self.opt.lambda_fused

        self.loss_side = 0.0
        for out, w in zip(self.outputs[:-1], self.weight_side):
            #self.loss_side += self.criterionSeg(out, self.label3d) * w
            self.loss_side += self.criterionSeg(out, self.label) * w

        #self.loss_fused = self.criterionSeg(self.outputs[-1], self.label3d)
        self.loss_fused = self.criterionSeg(self.outputs[-1], self.label)
        self.loss_total = self.loss_side * lambda_side + self.loss_fused * lambda_fused
        self.loss_total.backward()

    def optimize_parameters(self, epoch=None):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute predictions.
        self.optimizer.zero_grad()  # set G's gradients to zero
        self.backward()             # calculate gradients for G
        self.optimizer.step()       # update G's weights
