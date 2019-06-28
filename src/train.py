import os

import numpy as np
import torch
from torchvision import utils

from MDAN import MDAN
from AG import DataAugmentation
from MGAN import MGAN
from utils.arg_parser import parse_args


def create_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def run_MDAN(args, device):
    MDAN_params = {
        'data': args.data,
        'stand_imageSize_syn': args.stand_imageSize_syn,
        'stand_imageSize_example': args.stand_imageSize_example,
        'stand_atom': args.stand_atom,
        'model_folder': 'MDAN/',
        'epochs': args.MDAN_epochs,
        'epoch_iter': args.MDAN_iterations,
        'contrast_std': args.MDAN_contrast_std,
        'device': device
    }

    input_content_folder = ['ContentInitial/', 'ContentTrain/', 'ContentTest/']
    output_style_folder = ['StyleInitial/', 'StyleTrain/', 'StyleTest/']
    flag_pretrained = [False, True, True]
    create_folder(MDAN_params['data']+MDAN_params['model_folder'])

    model = MDAN(MDAN_params)
    for i in range(len(input_content_folder)):
        create_folder(MDAN_params['data']+output_style_folder[i])

        model.run(
            input_content_folder[i],
            output_style_folder[i],
            flag_pretrained[i])


def run_AG(args, device):
    data_augmentaion = DataAugmentation(
        args.AG_step_rotation, args.AG_step_scale, device)

    size = str(args.image_size)
    folders = [
        ('ContentTest/', 'ContentTestPatch'),
        ('StyleTest/', 'StyleTestPatch'),
        ('ContentTrain/', 'ContentTrainPatch'),
        ('StyleTrain/', 'StyleTrainPatch')
    ]
    
    for folder_name, patch_name in folders:
        folder_source = args.data + folder_name
        folder_source_patch = args.data + patch_name + size + '/'
        create_folder(folder_source_patch)

        count = 0
        for image_file in os.listdir(folder_source):
            
            copies = data_augmentaion.augment_data(
                folder_source+image_file, args.stand_imageSize_syn,
                args.stand_atom, flip=args.AG_flag_flip, transform=False)

            for image in copies:
                row_len = image.size()[1] - args.image_size + 1
                col_len = image.size()[2] - args.image_size + 1

                for i_row in range(0, row_len, args.AG_sampleStep):
                    for i_col in range(0, col_len, args.AG_sampleStep):
                        row_end = i_row + args.image_size
                        col_end = i_col + args.image_size

                        utils.save_image(
                            image[:, i_row:row_end, i_col:col_end],
                            folder_source_patch+str(count)+'.png')
                        count += 1


def run_MGAN(args, device):
    MGAN_params = {
        'data': args.data,
        'epochs': args.MGAN_epochs,
        'pixel_blockSize': args.image_size,
        'netS_weight': args.MGAN_netS_weight,
        'model_folder': 'MGAN/',
        'device': device
    }
    create_folder(MGAN_params['data']+MGAN_params['model_folder'])

    model = MGAN(MGAN_params)
    model.run()


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(args.device)

    if args.MDAN:
        run_MDAN(args, device)

    if args.AG:
        run_AG(args, device)

    if args.MGAN:
        run_MGAN(args, device)
