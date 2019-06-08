import os.path
import random
import cv2
from PIL import Image
from data.base_dataset import BaseDataset, get_params, get_transform
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from data.utils import MaskToTensor

class DeepCrackDataset(BaseDataset):
    """A dataset class for crack dataset."""

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.img_paths = make_dataset(os.path.join(opt.dataroot, '{}_img'.format(opt.phase)))
        self.lab_dir = os.path.join(opt.dataroot, '{}_lab'.format(opt.phase))

        self.img_transforms = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5),
                                                                       (0.5, 0.5, 0.5))])
        self.lab_transform = MaskToTensor()
        
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image
            B (tensor) - - its corresponding segmentation
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        img_path = self.img_paths[index]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        lab_path = os.path.join(self.lab_dir, os.path.basename(img_path).split('.')[0]+'.png')
        lab = cv2.imread(lab_path, cv2.IMREAD_UNCHANGED)
        w, h = self.opt.load_width, self.opt.load_height
        if w > 0 or h > 0:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_CUBIC)

        # binarize segmentation
        _, lab = cv2.threshold(lab, 127, 1, cv2.THRESH_BINARY)

        # apply flip
        if (not self.opt.no_flip) and random.random() > 0.5:
            if random.random() > 0.5:
                img = np.fliplr(img)
                lab = np.fliplr(lab)
            else:
                img = np.flipud(img)
                lab = np.flipud(lab)                

        # transfer to Image format
        img = Image.fromarray(img.copy())
        lab = Image.fromarray(lab.copy())

        # apply affine transform
        if self.use_augment:
            params = self._get_params()
            img = self._im_trans(img, params)
            lab = self._im_trans(lab, params)

        # apply the transform to both A and B
        img = self.img_transforms(img)
        lab = self.lab_transform(lab)

        return {'image': img, 'label': lab, 'A_paths': img_path, 'B_paths': lab_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_paths)
