# Author: Yahui Liu <yahui.liu@unitn.it>

import os.path
import random
import cv2
import glob
import numpy as np
from PIL import Image

from data.base_dataset import BaseDataset
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from data.utils import MaskToTensor

class RoadNetDataset(BaseDataset):
    """A dataset class for road dataset.
    """
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.img_paths   = glob.glob(os.path.join(opt.dataroot, '{}_image'.format(opt.phase), '*.png'))
        self.segment_dir = os.path.join(opt.dataroot, '{}_segment'.format(opt.phase))
        self.edge_dir    = os.path.join(opt.dataroot, '{}_edge'.format(opt.phase))
        self.centerline_dir = os.path.join(opt.dataroot, '{}_centerline'.format(opt.phase))

        self.img_transforms = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5),
                                                                       (0.5, 0.5, 0.5))])
        self.lab_transform  = MaskToTensor()
        
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary:
            image (tensor) - - a road image 
            segment (tensor) - - its corresponding surface segmenation
            edge (tensor) - - its corresponding edges
            centerline (tensor) - - its corresponding centerlines
            A_paths (str) - - image paths
        """
        # read a image given a random integer index
        img_path = self.img_paths[index]
        image    = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # set paths of annotation maps
        segment_path    = os.path.join(self.segment_dir, os.path.basename(img_path))
        edge_path       = os.path.join(self.edge_dir, os.path.basename(img_path))
        centerline_path = os.path.join(self.centerline_dir, os.path.basename(img_path))

        # load annotation maps and only use the red channel
        segment    = cv2.imread(segment_path, cv2.IMREAD_UNCHANGED)[:,:,2]
        edge       = cv2.imread(edge_path, cv2.IMREAD_UNCHANGED)[:,:,2]
        centerline = cv2.imread(centerline_path, cv2.IMREAD_UNCHANGED)[:,:,2]
        w, h = self.opt.load_width, self.opt.load_height
        if w > 0 or h > 0:
            image      = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
            segment    = cv2.resize(segment, (w, h), interpolation=cv2.INTER_CUBIC)
            edge       = cv2.resize(edge, (w, h), interpolation=cv2.INTER_CUBIC)
            centerline = cv2.resize(centerline, (w, h), interpolation=cv2.INTER_CUBIC)

        # binarize segmentation
        _, segment    = cv2.threshold(segment, 127, 1, cv2.THRESH_BINARY)
        _, edge       = cv2.threshold(edge, 127, 1, cv2.THRESH_BINARY)
        _, centerline = cv2.threshold(centerline, 127, 1, cv2.THRESH_BINARY)

        # apply flip
        if (not self.opt.no_flip) and random.random() > 0.5:
            if random.random() > 0.5:
                image      = np.fliplr(image)
                segment    = np.fliplr(segment)
                edge       = np.fliplr(edge)
                centerline = np.fliplr(centerline)
            else:
                image      = np.flipup(image)
                segment    = np.flipup(segment)
                edge       = np.flipup(edge)
                centerline = np.flipup(centerline)

        # transfer to Image format
        image      = Image.fromarray(image.copy())
        segment    = Image.fromarray(segment.copy())
        edge       = Image.fromarray(edge.copy())
        centerline = Image.fromarray(centerline.copy())

        # apply affine transform
        if self.opt.use_augment:
            if random.random() > 0.5:
                angle, scale, shift = get_params()
                image      = affine_transform(image, angle, scale, shift, w, h)
                segment    = affine_transform(segment, angle, scale, shift, w, h)
                edge       = affine_transform(edge, angle, scale, shift, w, h)
                centerline = affine_transform(centerline, angle, scale, shift, w, h)

        # apply the transform to both A and B
        image      = self.img_transforms(image)
        segment    = self.lab_transform(segment)
        edge       = self.lab_transform(edge)
        centerline = self.lab_transform(centerline)

        return {'image': image, 
                'segment': segment, 
                'edge': edge, 
                'centerline': centerline, 
                'A_paths': img_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_paths)
