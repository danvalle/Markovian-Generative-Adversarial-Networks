import os

from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import utils
from torchvision.models import vgg19

from AG import DataAugmentation


model_name = '../data/imnet/MGAN/epoch_4_netG.pth'
input_folder = '../data/imnet/Testset/'
output_folder = '../data/imnet/test_ans/'
noise_name = 'noise.jpg'

max_length = 512
stand_atom = 8
noise_weight = 0.0
netEnco_vgg_Outputlayer = 21

device = torch.device('cuda')


# Create complete network
vgg = vgg19(pretrained=True)
net_deco = torch.load(model_name)

net_release = nn.Sequential(
    *list(vgg.features.children())[:netEnco_vgg_Outputlayer],
    *list(net_deco.children())) 
net_release = net_release.to(device=device)
print(net_release)


# Load data
if not os.path.exists(output_folder):
	os.makedirs(output_folder)

noise_image = Image.open(noise_name)
data_augmentation = DataAugmentation(0, 0, device)
to_tensor = transforms.ToTensor()

print('*****************************************************')
print('Testing: ');
print('*****************************************************') 
for image_file in os.listdir(input_folder):
    # Resize the image image
    image_input = Image.open(input_folder + image_file).convert('RGB') 
    image_input = data_augmentation.atom_resize(
        image_input, max_length, stand_atom)
    image_input = to_tensor(image_input).to(device=device)
    
    # Add noise to the image (improve background quality)
    input_size = image_input.size()  
    shaped_noise = noise_image.resize(
        (input_size[2], input_size[1]), resample=Image.BILINEAR)
    shaped_noise = to_tensor(shaped_noise).to(device=device)

    image_input.add_(shaped_noise.mul(noise_weight))
    image_input.mul_(2).sub_(1)

    # Decode image
    image_syn = net_release(image_input.unsqueeze(0))

    # Save image
    image_syn.add_(-torch.mean(image_syn))

    std_contrast = torch.std(image_syn)*2
    image_syn[torch.lt(image_syn, -std_contrast)] = -std_contrast
    image_syn[torch.gt(image_syn, std_contrast)] = std_contrast
    min_im = torch.min(image_syn)
    max_im = torch.max(image_syn)
    image_syn.add_(-min_im).div_(max_im - min_im)

    image_name = image_file[:-4]
    print('Saving image', image_name)
    utils.save_image(
        image_syn, output_folder + image_name + '_MGANs.jpg')
