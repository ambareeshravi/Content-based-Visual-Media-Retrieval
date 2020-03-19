import pickle as pkl
from glob import glob
import numpy as np
import cv2
from keras.models import Model

def resize_image(im_array, resolution = (32,32)):
    return cv2.resize(im_array, resolution)

def get_files_list(folder):
    return sorted(glob(folder))

def cosine_similarity(a,b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def snip_model(m, layer_name = "global_average_pooling2d_1"):
    return Model(inputs=m.inputs, outputs=m.get_layer(layer_name).output)

def save_pickle(file_name, content):
    try:
        pkl.dump(content, open(file_name, "wb"))
        return True
    except:
        return False
    
def load_pickle(file_name):
    try:
        return pkl.load(open(file_name, "rb"))
    except:
        return False