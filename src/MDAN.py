import os
import time

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import utils
from torchvision.models import vgg19

from AG import DataAugmentation
from utils.TVLoss import TVLoss
from utils.neural_patches import NeuralPatches


def weights_init(m):
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)

    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight, mean=1.0, std=0.02)
        nn.init.constant_(m.bias, 0)


def zero_bias(m):
    if type(m) == nn.Conv2d:
        nn.init.constant_(m.bias, 0)


class MDAN():
    
    def __init__(self, params):
        self.params = params
        self.target_folder = params['data'] + 'Style/'
        self.serialized_model = (
            self.params['data'] + self.params['model_folder'] + 'netS.pth')

        # Network
        self.netS_vgg_Outputlayer = 13
        self.netS_vgg_nOutputPlane = 256
        self.netS_vgg_Outputpatchsize = 8
        self.netS_vgg_Outputpatchstep = 4

        self.netC_weights = 0.5
        self.netC_vgg_Outputlayer = 31

        # Optimization
        self.netD_lr = 0.02
        self.netG_lr = 0.02
        self.netD_beta1 = 0.5
        self.netG_beta1 = 0.5

        self.device = params['device']

        # Load pretrained model
        vgg = vgg19(pretrained=True)

        # VGGD model
        netVGGD = self.create_VGGD(vgg)

        # SVGG model 
        self.netSVGG = self.create_SVGG(netVGGD)
        print('Created netSVGG.')
        print(self.netSVGG)

        # C model
        self.netC = self.create_C(netVGGD)
        print('Created netC.')
        print(self.netC)

        # Criteria
        self.netS_criterion = nn.HingeEmbeddingLoss()
        self.netC_criterion = nn.MSELoss()
        print(self.netC_criterion, self.netS_criterion)

    def create_VGGD(self, vgg):
        netVGGD = nn.Sequential(TVLoss(), *list(vgg.features.children())[:-1])
        netVGGD = netVGGD.to(device=self.device)
        return netVGGD

    def create_SVGG(self, netVGGD):
        netSVGG = nn.Sequential(
            *list(netVGGD.children())[:self.netS_vgg_Outputlayer])
        netSVGG = netSVGG.to(device=self.device)
        return netSVGG

    def create_C(self, netVGGD):
        netC = nn.Sequential(
            *list(netVGGD.children())[:self.netC_vgg_Outputlayer])
        netC = netC.to(device=self.device)
        return netC

    def create_S(self, flag_pretrained):
        if flag_pretrained:
            netS = torch.load(self.serialized_model)
        else:
            netS = nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Conv2d(
                    self.netS_vgg_nOutputPlane, self.netS_vgg_nOutputPlane,
                    3, 1, 1),
                nn.BatchNorm2d(self.netS_vgg_nOutputPlane),
                nn.LeakyReLU(0.2),
                nn.Conv2d(
                    self.netS_vgg_nOutputPlane, 1,
                    self.netS_vgg_Outputpatchsize))
            netS.apply(weights_init)
        netS = netS.to(device=self.device)
        return netS

    def run(self, input_content_folder, output_style_folder,
            flag_pretrained):
        # S model
        self.netS = self.create_S(flag_pretrained)
        print('Created netS.')
        print(self.netS)

        # Build data
        target_image = self.target_folder + 'style.png'
        data_augmentation = DataAugmentation(np.pi/18, 1.1, self.device)
        target_copies = data_augmentation.augment_data(
            target_image, 
            self.params['stand_imageSize_example'],
            self.params['stand_atom'],
            True)

        # Target Neural Patches
        neural_patche = NeuralPatches(
            self.netS_vgg_Outputpatchsize, self.netS_vgg_Outputpatchstep,
            self.netS_vgg_nOutputPlane, True, self.device)
        netS_patches = neural_patche.compute_patches(
            target_copies, self.netSVGG)

        # Optimize
        self.source_folder = self.params['data'] + input_content_folder
        self.source_files = os.listdir(self.source_folder)

        print('Num of source images: {}'.format(len(self.source_files)))
        print('*****************************************************')
        print('Synthesis: ');
        print('*****************************************************') 
        self.optimize(
            data_augmentation, neural_patche,
            netS_patches, output_style_folder)
        # Save trained model
        torch.save(self.netS, self.serialized_model)

    def optimize(self, data_augmentation, neural_patche, netS_patches, 
                 output_style_folder):
        for source_file in self.source_files:
            image_source = Image.open(self.source_folder+source_file)
            image_source = data_augmentation.atom_resize(
                image_source, 
                self.params['stand_imageSize_syn'],
                self.params['stand_atom'])
            image_source = data_augmentation.transform_tensor(image_source)
            
            # Extract neural patches
            source_x, source_y, _ = neural_patche.target_per_image(
                [image_source], self.netSVGG)
            source_x, source_y = source_x[0], source_y[0]
            batch_size = len(source_x)
            print('Batch Size: {}'.format(batch_size))
            
            # Real/fake labels and original feature_map created 
            label = torch.Tensor(batch_size, 1).to(device=self.device)
            netC_feature_map_source = self.netC(image_source.unsqueeze(0)).detach()
            
            # Generated image to be optimized
            image_G = image_source.data.clone()
            image_G.requires_grad_()
            
            # Optimizers
            self.optimS = Adam(
                self.netS.parameters(), 
                lr=self.netD_lr, betas=(self.netD_beta1,0.999))
            self.optimG = Adam(
                [image_G],
                lr=self.netG_lr, betas=(self.netG_beta1,0.999))

            # Start training
            for epoch in range(self.params['epochs']):
                epoch_tm = time.time()

                for i_iter in range(self.params['epoch_iter']):
                    tm = time.time()

                    self.train_iteration(
                        image_G, source_x, source_y, batch_size, 
                        netS_patches, netC_feature_map_source, label)

                source_name = output_style_folder+source_file.split('.')[0]
                self.save_image(image_G.data.clone(), source_name, epoch)
                
    def collect_patches(self, source_x, source_y, batch_size, 
                        netS_feature_map_G):
        netS_patches_G = torch.Tensor(
            batch_size,
            self.netS_vgg_nOutputPlane,
            self.netS_vgg_Outputpatchsize,
            self.netS_vgg_Outputpatchsize).to(device=self.device)

        for count_patch, i_patch in enumerate(range(len(source_y))):
            y_start = int(source_y[i_patch].item())
            y_end = y_start + self.netS_vgg_Outputpatchsize
            x_start = int(source_x[i_patch].item())
            x_end = x_start + self.netS_vgg_Outputpatchsize

            netS_patches_G[count_patch] = netS_feature_map_G[
                0, :, y_start:y_end, x_start:x_end]
        return netS_patches_G

    def train_iteration(self, image_G, source_x, source_y, batch_size, 
                        netS_patches, netC_feature_map_source, label):
        # Collect generated patches
        netS_feature_map_G = self.netSVGG(image_G.unsqueeze(0))
        netS_patches_G = self.collect_patches(
            source_x, source_y, batch_size, netS_feature_map_G)
        netS_patches_G_copy = netS_patches_G.data.clone()

        # Randomly select real patches
        random_patch = torch.randint(
            len(netS_patches), [batch_size], dtype=torch.long)
        netS_patches_real = netS_patches[random_patch].data.clone()

        # Train netS
        self.discriminator_step(
            label, netS_patches_real, netS_patches_G_copy)

        # Update synthesis
        self.generator_step(
            netC_feature_map_source,
            image_G,
            netS_patches_G, label)
        print('\n')

    def discriminator_step(self, label, netS_patches_real, netS_patches_G):
        self.netS.apply(zero_bias)
        self.optimS.zero_grad()

        # Train with real images
        label.fill_(-1)
        StyleScore_real = self.netS(netS_patches_real).view(-1, 1)
        err_real = self.netS_criterion(StyleScore_real, label)
        print('Desc real', err_real.item())

        # Train with generated images
        label.fill_(-1)
        StyleScore_G = self.netS(netS_patches_G).view(-1,1)
        err_fake = self.netS_criterion(StyleScore_G*label, label)
        print('Desc fake', err_fake.item())

        total_loss = err_real + err_fake
        total_loss.backward()
        self.optimS.step()
        print('Desc Error:', total_loss.item())

    def generator_step(self, netC_feature_map_source, image_G,
                       netS_patches_G, label):
        # Predict with images
        self.netS.apply(zero_bias)
        self.optimS.zero_grad()

        label.fill_(-1)
        StyleScore_G = self.netS(netS_patches_G).view(-1, 1)
        errG_style = self.netS_criterion(StyleScore_G, label)
        print('Gen style', errG_style.item())
 
        # Grad from content loss
        self.optimG.zero_grad()
        
        netC_feature_map_G = self.netC(image_G.unsqueeze(0))
        errG_content = self.netC_criterion(
            netC_feature_map_G, netC_feature_map_source)
        errG_content *= self.netC_weights
        print('Gen content', errG_content.item())

        errG = errG_content + errG_style
        errG.backward()
        self.optimG.step()
        print('Generator Loss:', errG.item())

    def save_image(self, im_disp, source_name, epoch):
        # Save image
        im_disp.add_(-torch.mean(im_disp))

        contrast = self.params['contrast_std']
        std_contrast = torch.std(im_disp) * contrast
        im_disp[torch.lt(im_disp, -std_contrast)] = -std_contrast
        im_disp[torch.gt(im_disp, std_contrast)] = std_contrast

        min_im_disp = torch.min(im_disp)
        max_im_disp = torch.max(im_disp)
        im_disp.add_(-min_im_disp).div_(max_im_disp - min_im_disp) 
        
        utils.save_image(
            im_disp,
            #self.params['data']+source_name+'_e'+ str(epoch)+'.png')
            self.params['data']+source_name+'.png')
