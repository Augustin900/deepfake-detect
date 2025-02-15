# System library imports
from os import listdir, system, name, environ, makedirs
from os.path import join
from sys import exit

# Colored text import
from colored import fg, attr

from numpy import expand_dims
from cv2 import imread, cvtColor, COLOR_BGR2RGB, resize

# Flask import
from flask import *

environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# TensorFlow reqirements
import tensorflow as tf
import tensorflow.keras                                                    # type: ignore
from tensorflow.keras.models import Model, load_model                      # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout # type: ignore
from tensorflow.keras.applications import EfficientNetB0                   # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator        # type: ignore
from tensorflow.keras.optimizers import Adam                               # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping      # type: ignore