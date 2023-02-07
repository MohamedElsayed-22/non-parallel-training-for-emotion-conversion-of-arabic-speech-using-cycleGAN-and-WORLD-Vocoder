import argparse
import os
import numpy as np
from model import CycleGAN
from utils import *
from conversion_function import *



parser = argparse.ArgumentParser(description = 'Convert voices using pre-trained EmoCycleGAN model.')

model_dir_default = r'.\model\ang2neu'
model_name_default = r'ang2neu.ckpt'
data_dir_default = r'.\Database\Emotion\ang_neu\val_ang'
conversion_direction_default = 'A2B'
output_dir_default = r'.\converted_voices'

parser.add_argument('--model_dir', type = str, help = 'Directory for the pre-trained model.', default = model_dir_default)
parser.add_argument('--model_name', type = str, help = 'Filename for the pre-trained model.', default = model_name_default)
parser.add_argument('--data_dir', type = str, help = 'Directory for the voices for conversion.', default = data_dir_default)
parser.add_argument('--conversion_direction', type = str, help = 'Conversion direction for CycleGAN. A2B or B2A. The first object in the model file name is A, and the second object in the model file name is B.', default = conversion_direction_default)
parser.add_argument('--output_dir', type = str, help = 'Directory for the converted voices.', default = output_dir_default)

argv = parser.parse_args()

model_dir = argv.model_dir
model_name = argv.model_name
data_dir = argv.data_dir
conversion_direction = argv.conversion_direction
output_dir = argv.output_dir

conversion(model_dir = model_dir, model_name = model_name, data_dir = data_dir, conversion_direction = conversion_direction, output_dir = output_dir)


