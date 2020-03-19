def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import cv2
import numpy as np
import sklearn
from glob import glob
from tqdm import tqdm
import pickle, json
import os
from random import shuffle
from matplotlib import pyplot as plt
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from keras.applications.mobilenet import MobileNet
from keras.models import Model, Sequential
from keras.layers import LSTM, Dense, InputLayer

def get_submodel(main_model, output_layer = "fc1"):
    return Model(inputs=main_model.inputs, output=main_model.get_layer(output_layer).output)

def read_image(image_path):
    return cv2.imread(image_path)

def resize_image(im_array, resolution = (224,224)):
    return cv2.resize(im_array, resolution)

def load_images(files_list):
    return np.array([read_image(file) for file in files_list])

def get_frames_video(video_path, resize_to = (224,224), output_frames = 16):
    cap = cv2.VideoCapture(video_path)
    read_count = 1
    frames_list = list()
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while cap.isOpened():
        isRead, frame = cap.read()
        if not isRead: break
        if read_count % int(frame_total/output_frames) == 0:
            frame = cv2.resize(frame, resize_to)
            frames_list.append(frame)
        read_count += 1
        if len(frames_list) == 16: break
    return np.array(frames_list)

def get_frame_features(frames_list):
    return np.array([np.squeeze(mobile_net_submodel.predict(np.expand_dims(frame, axis=0))) for frame in frames_list])

def get_category(idx):
    a = np.zeros(3)
    a[idx] = 1
    return a

def prep_data(data_folder  = "DataSets/KTH/"):
    for folder in os.listdir(data_folder):
        with open(folder+".pkl", "wb") as f:
            pickle.dump({'folder': np.array([get_frame_features(get_frames_video(os.path.join(os.path.join(data_folder, folder), file))) for file in os.listdir(os.path.join(data_folder, folder))])}, f)
    
    classes = ['walking', 'boxing', 'handwaving']
    class_id = dict([(cls, get_category(idx)) for idx, cls in enumerate(classes)])
    
    with open("class.json", "w") as f:
        json.dump({"classes": classes, "class_id": class_id}, f)
        
    trainX, trainY = list(), list()
    for cls, label in class_id.items():
        with open(cls+".pkl", "rb") as f:
            data_list = pickle.load(f)['folder']
        for dl in data_list:
            if dl.shape == (16,1024):
                trainX.append(dl)
                trainY.append(label)
    trainX, trainY = np.array(trainX), np.array(trainY)
    return classes, class_id, trainX, trainY

def get_lstm_model(no_classes = 3, lstm_hidden_units = 100, image_features_size = 1024, step_size = 16, print_summary = False):
    lstm_model = Sequential()
    lstm_model.add(InputLayer(input_shape = (step_size, image_features_size)))
    lstm_model.add(LSTM(lstm_hidden_units))
    lstm_model.add(Dense(no_classes))
    if print_summary: print(lstm_model.summary())
    return lstm_model

def train_model(lstm_model, epochs = 20, batch_size = 2, validation_split = 0.1, model_name = "model_run2.hdf5"):
    lstm_model.compile(loss='mean_squared_error', optimizer='adam') #other losses
    lstm_model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=validation_split, shuffle=True, verbose=2)
    lstm_model.save(model_name)
    
def load_model(model_path = "model_run_1.h5"):
    lstm_model = get_lstm_model()
    lstm_model.load_weights(model_path)
    return lstm_model

def model_predict(lstm_model, file_path, get_class = False):
    if get_class:
        return classes[np.argmax(lstm_model.predict(np.expand_dims(get_frame_features(get_frames_video(file_path)), axis=0)))]
    else: return np.squeeze(lstm_model.predict(np.expand_dims(get_frame_features(get_frames_video(file_path)), axis=0))    )

def testfile_class(file_name):
    return os.path.split(file_name)[-1].split("_")[1]

def squeeze_array(array):
    return np.array([np.squeeze(i) for i in array])

if __name__ == '__main__':
    mobile_net = MobileNet()
    mobile_net_submodel = get_submodel(mobile_net, "global_average_pooling2d_1")
    lstm_model = load_model()

    # testing path
    # creating test set
    test_path = "DataSets/KTH/test/"
    test_set = list()
    lstm_sub_model = get_submodel(lstm_model, "lstm_1")

    test_folders = dict([(idx, folder) for idx, folder in enumerate(os.listdir(test_path))])
    rev_test_folders = dict([(v,k) for k,v in test_folders.items()])

    for file in glob(test_path + "*/*"):
        test_set.append(np.array([rev_test_folders[file.split("/")[3]], np.squeeze(model_predict(lstm_sub_model, file))]))

    test_set = np.array(test_set)

    # wtf is this?!!
    with open("test_set.pkl", "wb") as f:
        pickle.dump(test_set, f)

    with open("test_set.pkl", "rb") as f:
        test_set = pickle.load(f)

    np.random.shuffle(test_set)
    train_features, train_labels, test_features, test_labels = squeeze_array(test_set[:150][:,1]), squeeze_array(test_set[:150][:,0]), squeeze_array(test_set[150:][:,1]), squeeze_array(test_set[150:][:,0])