import numpy as np
import argparse
from utils import *
from WORLD_utils import *
from train_function import train



parser = argparse.ArgumentParser(description = 'Train CycleGAN model for datasets.')

train_A_dir_default = r'.\\Database\\Emotion\\ang_neu\\ang'
train_B_dir_default = r'.\\Database\\Emotion\\ang_neu\\neu'
model_dir_default = r'.\\model\\ang2neu'
model_name_default = r'ang2neu.ckpt'
random_seed_default = 0
validation_A_dir_default = r'.\\Database\\Emotion\\ang_neu\\val_ang'
validation_B_dir_default = r'.\\Database\\Emotion\\ang_neu\\val_neu'
output_dir_default = r'.\\validation_output'
tensorboard_log_dir_default = '.\\log'

parser.add_argument('--train_A_dir', type = str, help = 'Directory for A.', default = train_A_dir_default)
parser.add_argument('--train_B_dir', type = str, help = 'Directory for B.', default = train_B_dir_default)
parser.add_argument('--model_dir', type = str, help = 'Directory for saving models.', default = \
                    model_dir_default)
parser.add_argument('--model_name', type = str, help = 'File name for saving model.', default = \
                    model_name_default)
parser.add_argument('--random_seed', type = int, help = 'Random seed for model training.',\
                        default = random_seed_default)
parser.add_argument('--validation_A_dir', type = str, help = 'Convert validation A after each training \
                    epoch. If set none, no conversion would be done during the training.', default = \
                    validation_A_dir_default)
parser.add_argument('--validation_B_dir', type = str, help = 'Convert validation B after each training \
                    epoch. If set none, no conversion would be done during the training.', default = \
                    validation_B_dir_default)
parser.add_argument('--output_dir', type = str, help = 'Output directory for converted validation voices.',\
                        default = output_dir_default)
parser.add_argument('--tensorboard_log_dir', type = str, help = 'TensorBoard log directory.', default = \
                    tensorboard_log_dir_default)

argv = parser.parse_args()

train_A_dir = argv.train_A_dir
train_B_dir = argv.train_B_dir
model_dir = argv.model_dir
model_name = argv.model_name
random_seed = argv.random_seed
validation_A_dir = None if argv.validation_A_dir == 'None' or \
    argv.validation_A_dir == 'none' else argv.validation_A_dir
validation_B_dir = None if argv.validation_B_dir == 'None' or \
    argv.validation_B_dir == 'none' else argv.validation_B_dir
output_dir = argv.output_dir
tensorboard_log_dir = argv.tensorboard_log_dir

train(train_A_dir = train_A_dir, train_B_dir = train_B_dir, model_dir = \
        model_dir, model_name = model_name, random_seed = random_seed, validation_A_dir = \
        validation_A_dir, validation_B_dir = validation_B_dir, output_dir = output_dir, \
        tensorboard_log_dir = tensorboard_log_dir)
