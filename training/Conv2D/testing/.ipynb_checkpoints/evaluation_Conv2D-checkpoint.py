# imports
import pickle, json
import pandas as pd
import numpy as np

from glob import glob
import cv2
from random import shuffle
from tqdm import tqdm

from keras.models import load_model
from keras.models import Model

from sklearn import metrics
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import label_binarize

def snip_model(m, layer_name = "global_average_pooling2d_1"):
    return Model(inputs=m.inputs, outputs=m.get_layer(layer_name).output)

def cosine_similarity(a,b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def get_class(file_name):
    for k, v in class_id.items():
        if v in file_name:
            return k
        
def cosine_similarity(a,b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def get_class(file_name):
    for k, v in class_id.items():
        if v in file_name:
            return k
        
def get_file_features(fl, clr):
    return np.squeeze(snip_fc1.predict(np.expand_dims((np.array(resize_image(cv2.imread(fl, 1)))), axis = 0 ) / 255))
        
                 
clr_model = load_model('clr-gs/CIFAR_clr_v0.h5')
gs_model = load_model('clr-gs/CIFAR_gs.h5')
with open("clr_gs_results.pkl", "rb") as f:
    results = pickle.load(f)
results_df = pd.DataFrame.from_dict(results).transpose()
y_act, y_clr, y_gs = results_df['actual_label'].tolist(), results_df['clr_label'].tolist(), results_df['gs_label'].tolist()
print(metrics.classification_report(y_act, y_clr))
print(metrics.classification_report(y_act, y_gs))
metrics.confusion_matrix(y_act, y_clr)
metrics.confusion_matrix(y_act, y_gs)

threshold = 0.85
        
class_id = {'airplane' : 0, 'automobile' : 1, 'bird' : 2, 'cat' : 3, 'deer' : 4, 'dog' : 5, 'frog' : 6, 'horse' : 7, 'ship' : 8, 'truck' : 9}

clr_accuracy_list = list()

# actual_label 	clr_features 	clr_label 	gs_features 	gs_label 	img_array

for key1 in results.keys():
    result_class_ids = list()
    file_1_features = results[key1]["clr_features"]
    file_1_label = results[key1]["actual_label"]
    for key2 in results.keys():
        if key1 != key2:
            file_2_features = results[key2]["clr_features"]
            if cosine_similarity(file_1_features, file_2_features) > threshold:
                result_class_ids.append(results[key1]["clr_label"])
    if len(result_class_ids) > 2:
        accuracy = (result_class_ids.count(file_1_label) / len(result_class_ids))
        clr_accuracy_list.append(accuracy)

print("Overall accuracy for colour ", sum(clr_accuracy_list)/len(clr_accuracy_list))

gs_accuracy_list = list()

for key1 in results.keys():
    result_class_ids = list()
    file_1_features = results[key1]["gs_features"]
    file_1_label = results[key1]["actual_label"]
    for key2 in results.keys():
        if key1 != key2:
            file_2_features = results[key2]["gs_features"]
            if cosine_similarity(file_1_features, file_2_features) > threshold:
                result_class_ids.append(results[key1]["gs_label"])
    if len(result_class_ids) > 2:
        accuracy = (result_class_ids.count(file_1_label) / len(result_class_ids))
        gs_accuracy_list.append(accuracy)

print("Overall accuracy for grey ", sum(gs_accuracy_list)/len(gs_accuracy_list))

def resize_image(im_array, resolution = (224,224)):
    return cv2.resize(im_array, resolution)

imlist = sorted(glob("paris6k_miniset/*.jpg"))

print(len(imlist))

threshold = 0.95

class_id = {0:'invalides', 1:'louvre' ,2:'museedorsay',3:'triomphe'}


accuracy_list = list()
train_file_1=list()
train_file_2=list()

for file_1 in (imlist):
    similar_images = list()
    file_1_features = get_file_features(file_1)
    for file_2 in imlist:
    file_2_features = get_file_features(file_2)
    if cosine_similarity(file_1_features, file_2_features) > threshold:
        similar_images.append(file_2)
    result_class_ids = [get_class(file_) for file_ in similar_images]
    if len(result_class_ids) > 2:
        accuracy = (result_class_ids.count(get_class(file_1)) - 1) / len(result_class_ids)
        print(accuracy, result_class_ids.count(get_class(file_1)), len(result_class_ids))
        accuracy_list.append(accuracy)

print("Overall accuracy for grey = ", sum(accuracy_list)/len(accuracy_list))