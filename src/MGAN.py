import os

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import utils
from torchvision.models import vgg19
from torchvision.transforms import ToTensor


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


class MGAN():
    
    def __init__(self, params):
        self.params = params
        self.block_size = params['pixel_blockSize']

        # Data
        self.target_folder = params['data'] + 'StyleTrainPatch128'
        self.source_folder = params['data'] + 'ContentTrainPatch128'
        self.test_source_folder = params['data'] + 'ContentTestPatch128'
        self.test_target_folder = params['data'] + 'StyleTestPatch128'
        self.model_folder = params['data'] + params['model_folder']

        self.nf = 64 
        self.pixel_weights = 1
        self.tv_weight = 1e-4

        self.netEnco_vgg_Outputlayer = 21
        self.netEnco_vgg_nOutputPlane = 512
        self.netEnco_vgg_Outputblocksize = int(self.block_size / 8)

        # Discriminator
        self.netS_weights = params['netS_weight']
        self.netS_vgg_Outputlayer = 12
        self.netS_vgg_nOutputPlane = 256
        self.netS_blocksize = int(self.block_size / 16)
        self.netS_flag_mask = True
        self.netS_border = 1

        # Optimization
        self.batch_size = 64
        self.netD_lr = 0.02
        self.netG_lr = 0.02
        self.netD_beta1 = 0.5
        self.netG_beta1 = 0.5

        # Misc
        self.save_iterval_image = 100
        self.device = params['device']    

        weight_sum = self.pixel_weights + self.netS_weights
        self.pixel_weights = self.pixel_weights / weight_sum
        self.netS_weights = self.netS_weights / weight_sum

        # Load pretrained model
        vgg = vgg19(pretrained=True)

        # VGG model
        netVGG = self.create_VGG(vgg)

        # Encoder model
        self.netEncoder = self.create_Encoder(netVGG)
        print('Created Encoder.')
        print(self.netEncoder)

        # Generator model
        self.netG = self.create_Generator()
        print('Created Generator.')
        print(self.netG)

        # Discriminator model
        self.netSVGG = self.create_SVGG(netVGG)
        print('Created netSVGG.')
        print(self.netSVGG)
        self.netS = self.create_S()
        print('Created netS.')
        print(self.netS)

        # Criteria
        self.pixel_criterion = nn.MSELoss()
        self.netS_criterion = nn.HingeEmbeddingLoss()
        print(self.pixel_criterion, self.netS_criterion)


    def create_VGG(self, vgg):
        netVGG = nn.Sequential(*list(vgg.features.children())[:-1])
        netVGG = netVGG.to(device=self.device)
        return netVGG


    def create_Encoder(self, netVGG):
        netEncoder = nn.Sequential(
            *list(netVGG.children())[:self.netEnco_vgg_Outputlayer])
        netEncoder = netEncoder.to(device=self.device)
        return netEncoder


    def create_Generator(self):
        netG = nn.Sequential(
            nn.ConvTranspose2d(self.netEnco_vgg_nOutputPlane, 512, 3, 1, 1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh())
        netG.apply(weights_init)
        netG = netG.to(device=self.device)
        return netG


    def create_SVGG(self, netVGG):
        netSVGG = nn.Sequential(
            *list(netVGG.children())[:self.netS_vgg_Outputlayer])
        netSVGG = netSVGG.to(device=self.device)
        return netSVGG


    def create_S(self):
        netS = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.netS_vgg_nOutputPlane, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 1, 1))
        netS.apply(weights_init)
        netS = netS.to(device=self.device)
        return netS


    def run(self):
        # Build data
        num_source_images = len(os.listdir(self.source_folder))
        num_target_images = len(os.listdir(self.target_folder))
        num_test_images = len(os.listdir(self.test_source_folder))
        print('Source images: {}'.format(num_source_images))
        print('Target images: {}'.format(num_target_images))
        print('Test images: {}'.format(num_test_images))
        
        to_tensor = ToTensor()

        BlockPixel_test_source = torch.Tensor(
            self.batch_size, 3, self.block_size, self.block_size)
        BlockPixel_test_source = BlockPixel_test_source.to(device=self.device)

        BlockPixel_test_target = torch.Tensor(
            self.batch_size, 3, self.block_size, self.block_size)
        BlockPixel_test_target = BlockPixel_test_target.to(device=self.device)

        list_image_test = torch.randint(num_test_images,(1, self.batch_size))
        list_image_test = list_image_test.long()[0]

        test_files = os.listdir(self.test_source_folder)
        for i_img in range(self.batch_size):
            file_name = test_files[list_image_test[i_img].item()]
            BlockPixel_test_source[i_img] = to_tensor(
                Image.open(self.test_source_folder+'/'+file_name))
            BlockPixel_test_target[i_img] = to_tensor(
                Image.open(self.test_target_folder+'/'+file_name))
        
        BlockPixel_test_source.mul_(2).sub_(1)
        BlockPixel_test_target.mul_(2).sub_(1)
        # BlockInterface_test = self.netEncoder(BlockPixel_test_source).clone()


        out_size = self.batch_size * self.netS_blocksize * self.netS_blocksize
        label = torch.Tensor(out_size, 1).to(device=self.device)

        netS_mask = torch.ones(
            self.batch_size, 1, self.netS_blocksize, self.netS_blocksize)
        netS_mask = netS_mask.byte().to(device=self.device)
        if self.netS_flag_mask:
            netS_mask[
                :,:,
                self.netS_border:self.netS_blocksize-self.netS_border,
                self.netS_border:self.netS_blocksize-self.netS_border].fill_(0)
        netS_mask = netS_mask.view(out_size, 1)



        BlockPixel_target = torch.Tensor(
            self.batch_size, 3, self.block_size, self.block_size)
        BlockPixel_target = BlockPixel_target.to(device=self.device)

        BlockPixel_source = torch.Tensor(
            self.batch_size, 3, self.block_size, self.block_size)
        BlockPixel_source = BlockPixel_source.to(device=self.device)

        print('*****************************************************')
        print('Training Loop: ');
        print('*****************************************************') 

        # Optimizers
        self.optimS = Adam(
            self.netS.parameters(), 
            lr=self.netD_lr, betas=(self.netD_beta1,0.999))
        self.optimG = Adam(
            self.netG.parameters(),
            lr=self.netG_lr, betas=(self.netG_beta1,0.999))

        for epoch in range(self.params['epochs']):
            for i_iter in range(0, num_source_images, self.batch_size):

                # Randomly select images to train
                list_image = torch.randint(num_source_images,(1, self.batch_size))
                list_image = list_image.long()[0]
                # Read PatchPixel_real and PatchPixel_photo
                source_files = os.listdir(self.source_folder)
                for i_img in range(self.batch_size):
                    file_name = source_files[list_image[i_img].item()]
                    BlockPixel_target[i_img] = to_tensor(
                        Image.open(self.target_folder+'/'+file_name))
                    BlockPixel_source[i_img] = to_tensor(
                        Image.open(self.source_folder+'/'+file_name))
                

                BlockPixel_target.mul_(2).sub_(1)
                BlockVGG_target = self.netSVGG(BlockPixel_target).detach()

                BlockPixel_source.mul_(2).sub_(1)
                BlockInterface = self.netEncoder(BlockPixel_source).detach()
                BlockPixel_G = self.netG(BlockInterface)
                BlockVGG_G = self.netSVGG(BlockPixel_G)

                # Train netS
                self.discriminator_step(
                    label, BlockVGG_target,
                    BlockVGG_G.detach(), netS_mask)

                # Train netG
                self.generator_step(
                    label, BlockPixel_target,
                    BlockPixel_G, BlockVGG_G, netS_mask)
                print('\n')

            utils.save_image(
                BlockPixel_G.data[0].clone(),
                self.model_folder+'e_'+str(epoch)+'.png')

            torch.save(
                self.netG, self.model_folder+'epoch_'+str(epoch)+'_netG.pth')
            torch.save(
                self.netS, self.model_folder+'epoch_'+str(epoch)+'_netS.pth')


    def discriminator_step(self, label, BlockVGG_target, 
                           BlockVGG_G, netS_mask):
        self.netS.apply(zero_bias)
        self.optimS.zero_grad()

        # Train with real images    
        label.fill_(-1)
        StyleScore_real = self.netS(BlockVGG_target).view(-1, 1)
        if self.netS_flag_mask:
            StyleScore_real[netS_mask] = 1
        errS = self.netS_criterion(StyleScore_real, label)
    
        # Train with generated images
        label:fill(-1)
        StyleScore_G = self.netS(BlockVGG_G).view(-1, 1)
        if self.netS_flag_mask:
            StyleScore_G[netS_mask] = -1
        errS = errS + self.netS_criterion(StyleScore_G*label, label)

        errS = errS.div(self.batch_size)
        print('Discriminator Loss:', errS.item())

        errS.backward()
        self.optimS.step()



    def generator_step(self, label, BlockPixel_target, 
                       BlockPixel_G, BlockVGG_G, netS_mask):
        self.netG.apply(zero_bias)
        self.optimG.zero_grad()

        # Pixel loss
        errG_Pixel = self.pixel_criterion(BlockPixel_G, BlockPixel_target)
        print('G Pixel Loss:', errG_Pixel.item())

        # Style loss
        label.fill_(-1)
        StyleScore_G = self.netS(BlockVGG_G).view(-1, 1)
        if self.netS_flag_mask:
          StyleScore_G[netS_mask] = 1
        errG_Style = self.netS_criterion(StyleScore_G, label)
        print('G Style Loss:', errG_Style.item())

        errG = errG_Pixel*self.pixel_weights + errG_Style*self.netS_weights
        print('Generator Loss:', errG.item())
        errG.backward()
        self.optimG.step()
