import argparse
import numpy as np


def handle_bool(arg):
    """Handle boolean arguments, chosing between true and false"""
    if arg.lower() == 'true':
        return True
    return False


def parse_args():
    parser = argparse.ArgumentParser(description='MGAN for Style Transfer')

    # Required params
    parser.add_argument(
        '-data', required=True, 
        help='Path to data folder. Must contain subfolders ContentInitial,'
            ' ContentTrain and ContentTest in it')

    # Network params
    parser.add_argument(
        '-MDAN', default=True, type=handle_bool,
        help='Apply MDAN optimization')
    parser.add_argument(
        '-AG', default=True, type=handle_bool,
        help='Apply data augmentation')
    parser.add_argument(
        '-MGAN', default=True, type=handle_bool,
        help='Apply MGAN optimization')
    parser.add_argument(
        '-device', default='cuda', help='cpu or cuda device')

    # Image params
    parser.add_argument(
        '-stand_imageSize_syn', default=384, type=int, 
        help='')
    parser.add_argument(
        '-stand_imageSize_example', default=384, type=int,
        help='')
    parser.add_argument(
        '-stand_atom', default=8, type=int,
        help='')
    parser.add_argument(
        '-image_size', default=128, type=int,
        help='')

    # MDAN params
    parser.add_argument(
        '-MDAN_epochs', default=5, type=int,
        help='Number of epochs in MDAN')
    parser.add_argument(
        '-MDAN_iterations', default=25, type=int,
        help='Number of iterations in each epoch in MDAN')
    parser.add_argument(
        '-MDAN_contrast_std', default=2, type=int,
        help='Standard contrast applied when saving image')

    # Augmentation params
    parser.add_argument(
        '-AG_sampleStep', default=64, type=int,
        help='AG param')
    parser.add_argument(
        '-AG_step_rotation', default=np.pi/18, type=float,
        help='AG param')
    parser.add_argument(
        '-AG_step_scale', default=1.1, type=float,
        help='AG param')
    parser.add_argument(
        '-AG_flag_flip', default=True, type=handle_bool,
        help='AG param')

    # MGAN params
    parser.add_argument(
        '-MGAN_netS_weight', default=1e-2, type=float,
        help='Weight that gives more focus on texture and less in content')
    parser.add_argument(
        '-MGAN_epochs', default=5, type=int,
        help='Number of epochs in MDAN')

    args = parser.parse_args()
    return args