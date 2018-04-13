import vgg
import tensorflow as tf
import numpy as np 

from sys import stderr

from PIL import Image

CONTENT_LAYERS = ('relu4_2','relu5_2')
STYLE_LAYERS = ('relu1_1','relu2_1','relu3_1','relu4_1','relu5_1')