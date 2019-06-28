from PIL import Image
import numpy as np
import torch
from torchvision import transforms


class DataAugmentation():

    def __init__(self, step_rotation, step_scale, device):
        self.step_rotation = step_rotation
        self.step_scale = step_scale
        self.device = device
        self.to_tensor = transforms.ToTensor()

    def compute_border(self, width, height, alpha):
        x_half = width/2
        y_half = height/2
        x = np.array([1, width, width, 1])
        y = np.array([1, 1, height, height])
        xr = x_half + (x-x_half)*np.cos(alpha) + (y-y_half)*np.sin(alpha)  
        yr = y_half - (x-x_half)*np.sin(alpha) + (y-y_half)*np.cos(alpha)  

        x1r, x2r, x3r, x4r = xr
        y1r, y2r, y3r, y4r = yr

        min_x, min_y = (1, 1)
        max_x, max_y = (width, height)

        if alpha > 0:
            py = ((y1r - y2r) + (x1r*y2r - y1r*x2r)) / (x1r - x2r)
            px = (py*(x3r - x2r) + (x2r*y3r - x3r*y2r)) / (y3r - y2r)

            min_x = width - px
            min_y = py
            max_x = px
            max_y = height - py

        elif alpha < 0:
            py = (width*(y1r - y2r) + (x1r*y2r - y1r*x2r)) / (x1r - x2r)
            px = (py*(x1r - x4r) - (x1r*y4r - y1r*x4r)) / (y1r - y4r)

            min_x = px
            min_y = py
            max_x = width - px
            max_y = height - py

        return (
            np.floor(min_x) - 1, np.floor(min_y) - 1,
            np.floor(max_x), np.floor(max_y))

    def rotate_crop_resize(self, image_target, target_copies):
        for i_r in [-1,0,1]:
            alpha = self.step_rotation * i_r 
            min_x, min_y, max_x, max_y = self.compute_border(
                image_target.width, image_target.height, alpha)
            image_target_rt = image_target.rotate(np.degrees(alpha))
            image_target_rt = image_target_rt.crop(
                (min_x, min_y, max_x, max_y))
            
            for i_s in [-1,0,1]:
                max_dim = np.max(
                    [image_target_rt.height, image_target_rt.width]) 
                max_sz = np.floor(max_dim * np.power(self.step_scale, i_s))
                factor = max_sz / max_dim

                target_image_rt_s = image_target_rt.resize(
                    (int(image_target_rt.width*factor), 
                    int(image_target_rt.height*factor)),
                    resample=Image.BILINEAR)
                target_copies.append(target_image_rt_s)

    def atom_resize(self, image, stand_imageSize_example, stand_atom):
        max_dim = np.max(image.size)
        scale = stand_imageSize_example / max_dim
        new_dim_x = np.floor((image.width*scale) / stand_atom)
        new_dim_x *= stand_atom
        new_dim_y = np.floor((image.height*scale) / stand_atom)
        new_dim_y *= stand_atom

        image = image.resize(
            (int(new_dim_x), int(new_dim_y)), resample=Image.BILINEAR)
        return image

    def transform_tensor(self, image, transform=True):
        image = self.to_tensor(image)
        image = image.to(self.device)
        if transform:
            image.mul_(2).sub_(1)
            #image = image.requires_grad_()
        return image

    def augment_data(self, image_path, stand_imageSize_example,
                     stand_atom, flip=True, transform=True):
        image_target = Image.open(image_path)

        image_target = self.atom_resize(
            image_target, stand_imageSize_example, stand_atom)
        
        target_copies = []
        self.rotate_crop_resize(image_target, target_copies)
        if flip:
            image_target = image_target.transpose(Image.FLIP_LEFT_RIGHT)
            self.rotate_crop_resize(image_target, target_copies)

        for i, target_copie in enumerate(target_copies):
            target_copies[i] = self.transform_tensor(target_copie, transform)

        return target_copies
